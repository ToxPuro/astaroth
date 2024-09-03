%{
//#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <libgen.h> // dirname
#include <sys/stat.h>

#include "ast.h"
#include "codegen.h"
#include <ctype.h>
#include "tinyexpr.h"
#include <dirent.h>
#include <math.h>

#define YYSTYPE ASTNode*

ASTNode* root = NULL;

extern FILE* yyin;
extern char* yytext;

int yylex();
int yyparse();
int yyerror(const char* str);
int yyget_lineno();
#define AC_INCLUDE "PREPROCESSOR_AC_INCLUDE" 


//These are used to generate better error messages in case of errors
FILE* yyin_backup;
const char* stage4_name_backup;
const char* dir_backup;

//These are used to evaluate constant int expressions 
static string_vec const_ints;
static string_vec const_int_values;


void
cleanup(void)
{
    if (root)
        astnode_destroy(root); // Frees all children and itself
}
ASTNode*
astnode_hostdefine(const char* buffer, const int token)
{
        ASTNode* res = astnode_create(NODE_HOSTDEFINE, NULL, NULL);
        astnode_set_buffer(buffer,res);
        res->token = 255 + token;

        astnode_set_prefix("#",res); 

        // Ugly hack
        const char* def_in = "hostdefine";
        const char* def_out = "define";
        assert(strlen(def_in) > strlen(def_out));
        assert(!strncmp(res->buffer, def_in, strlen(def_in)));

        for (size_t i = 0; i < strlen(def_in); ++i)
            res->buffer[i] = ' ';
        strcpy(res->buffer, def_out);
        res->buffer[strlen(def_out)] = ' ';

        astnode_set_postfix("\n", res);
	return res;
}


void set_identifier_type(const NodeType type, ASTNode* curr);
void set_identifier_prefix(const char* prefix, ASTNode* curr);
void set_identifier_infix(const char* infix, ASTNode* curr);
ASTNode* get_node(const NodeType type, ASTNode* node);
ASTNode* get_node_by_token(const int token, const ASTNode* node);
static inline int eval_int(ASTNode* node);
static ASTNode* create_type_declaration(const char* tqual, const char* tspec, const int token);
static ASTNode* create_type_qualifiers(const char* tqual,int token);
static ASTNode* create_type_qualifier(const char* tqual, const int token);

bool is_directory(const char *path) {
    if(!path) return false;
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) {
        // If stat fails, return false
        return false;
    }
    return S_ISDIR(statbuf.st_mode);
}

void
expand_macros(const char* file_in, const char* file_out)
{
          const size_t cmdlen = 4096;
	  char* cmd = malloc(cmdlen*sizeof(char));
          snprintf(cmd, cmdlen, "gcc -x c -E %s > %s", file_in, file_out);
          const int retval = system(cmd);
	  free(cmd);
          if (retval == -1) {
              fprintf(stderr, "Catastrophic error: preprocessing failed.\n");
              assert(retval != -1);
          }
}

void
process_hostdefines(const char* file_in, const char* file_out)
{
  FILE* in = fopen(file_in, "r");
  assert(in);

  FILE* out = fopen(file_out, "w");
  assert(out);

  const size_t  len = 4096;
  char* buf = malloc(len*sizeof(char));
  while (fgets(buf, len, in)) {
    fprintf(out, "%s", buf);

    char* line = buf;
    while (strlen(line) > 0 && line[0] == ' ') // Remove whitespace
      ++line;

    if (!strncmp(line, "hostdefine", strlen("hostdefine"))) {
      while (strlen(line) > 0 && line[0] != ' ') // Until whitespace
        ++line;

      fprintf(out, "#define%s", line);
    }
  }

  free(buf);
  fclose(in);
  fclose(out);
}


FILE*
get_preprocessed_file(const char* filename, char* file_buf)
{
	const char* stage0 = "AC_INTERNAL_TO_GCC_STAGE0";
	const char* stage1 = "AC_INTERNAL_TO_GCC_STAGE1";
	const char* stage2 = "AC_INTERNAL_FROM_GCC";
	FILE* in = fopen(filename,"r");
	if(!in) return NULL;
	FILE* out = fopen(stage0,"w");
  	const size_t  len = 4096;
  	char* buf = malloc(len*sizeof(char));
  	while (fgets(buf, len, in)) {
  	  char* line = buf;
	  // Remove whitespace
  	  while (strlen(line) > 0 && line[0] == ' ') ++line;
	  remove_substring(line,";");
          if (!strncmp(line, "#include", strlen("#include")))
    	  	replacestr(line,"#include",AC_INCLUDE);
	  fprintf(out,"%s",line);
	}
        fclose(out);
	fclose(in);
	process_hostdefines(stage0,stage1);
	//expand_macros(stage1,stage2);
	in = fopen(stage1,"r");
	size_t size = 0;
	out = open_memstream(&file_buf,&size);
  	while (fgets(buf, len, in)) {
  	  char* line = buf;
	  fprintf(out,"%s",line);
	}
	fclose(in);
	return out;
}

void
process_includes(const size_t depth, const char* dir, const char* file, FILE* out, const bool log)
{
  if(is_directory(file)) 
  {
	DIR* d = opendir(file);
	struct dirent *dir_entry;
	while((dir_entry = readdir(d)) != NULL)
		if(strcmp(dir_entry->d_name,".") && strcmp(dir_entry->d_name,".."))
		{
		        char* file_path = malloc((strlen(file) + strlen(dir_entry->d_name) + 1000)*sizeof(char));
			sprintf(file_path,"%s/%s",file,dir_entry->d_name);
		        process_includes(depth+1,dir,file_path,out,log);
			free(file_path);
		}

	return;
  }
  const size_t max_nests = 64;
  if (depth >= max_nests) {
    fprintf(stderr, "CRITICAL ERROR: Max nests %lu reached when processing includes. Aborting to avoid thrashing the disk. Possible reason: circular includes.\n", max_nests);
    exit(EXIT_FAILURE);
  }

  if(log) printf("Building AC object %s\n", file);
  char* file_buf = NULL;
  FILE* in = get_preprocessed_file(file,file_buf);
  if (!in) {
    fprintf(out,"AC_FATAL_ERROR: could not open include file '%s'\n",file);
    return;
  }

  const size_t  len = 4096;
  char* buf = malloc(len*sizeof(char));
  while (fgets(buf, len, in)) {
    char* line = buf;
    if (!strncmp(line, AC_INCLUDE, strlen(AC_INCLUDE))) {

      char incl[len];
      sscanf(line, AC_INCLUDE" \"%[^\"]\"\n", incl);

      char path[len];
      sprintf(path, "%s/%s", dir, incl);

      fprintf(out, "// Include file %s start\n", path);
      process_includes(depth+1, dir, path, out,log);
      fprintf(out, "// Included file %s end\n", path);

    } else {
      fprintf(out, "%s", buf);
    }
  }
  free(buf);
  fclose(in);
  free(file_buf);
}


void
check_file(const FILE* fp, const char* filename)
{
	if(!fp)
	{
	    fprintf(stderr,"Fatal error did not found file: %s\n",filename);
		assert(fp);
	    exit(EXIT_FAILURE);
	}
}
bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}
void
make_dir(const char* dirname)
{
	if(is_directory(dirname))
	{
		char command[4098];
		sprintf(command, "rm -rf %s\n",dirname);
		int res = system(command);
		if(res)
		{
			fprintf(stderr,"Fatal error: could not remove dir: %s\n",dirname),
			assert(res == 0);
			exit(EXIT_FAILURE);
		}
	}
	int res = mkdir(dirname,0777);
	if(res)
	{
		fprintf(stderr,"Fatal error: could not create dir: %s\n",dirname),
		assert(res == 0);
		exit(EXIT_FAILURE);
	}
}
void
reset_diff_files()
{
		const char* files[] = {};
          	//for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); ++i) {
          	//  FILE* fp = fopen(files[i], "w");
	  	//  check_file(fp,files[i]);
          	//  fclose(fp);
          	//}
}
void
reset_extra_files()
{
          const char* files[] = { ACC_GEN_PATH"/extra_dfuncs.h",ACC_GEN_PATH"/boundcond_kernels.h"};
          for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); ++i) {
	    if(!file_exists(files[i])) continue;
            FILE* fp = fopen(files[i], "w");
	    check_file(fp,files[i]);
            fclose(fp);
          }
}
void
reset_all_files()
{
          const char* files[] = {"kernel_reduce_outputs.h","user_declarations.h", "user_defines.h", "user_kernels.h", "user_kernel_declarations.h",  "user_input_typedefs.h", "user_typedefs.h","user_kernel_ifs.h",
		 "device_mesh_info_decl.h",  "array_decl.h", "comp_decl.h","comp_loaded_decl.h", "input_decl.h","get_device_array.h","get_config_arrays.h","get_config_param.h",
		 "get_arrays.h","dconst_decl.h","dconst_accesses_decl.h","get_address.h","load_dconst_arrays.h","store_dconst_arrays.h","dconst_arrays_decl.h","memcpy_to_gmem_arrays.h","memcpy_from_gmem_arrays.h",
		  "array_types.h","scalar_types.h","scalar_comp_types.h","array_comp_types.h","get_num_params.h","gmem_arrays_decl.h","get_gmem_arrays.h","vtxbuf_is_communicated_func.h",
		 "load_and_store_uniform_overloads.h","load_and_store_uniform_funcs.h","load_and_store_uniform_header.h","get_array_info.h","get_from_comp_config.h","get_param_name.h","to_str_funcs.h","get_default_value.h",
		 "user_kernel_ifs.h", "user_dfuncs.h","user_kernels.h.raw","user_loaders.h", "user_taskgraphs.h","user_loaders.h","user_read_fields.bin","user_written_fields.bin","user_field_has_stencil_op.bin",
		  "fields_info.h",
		  };
          for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); ++i) {
	    if(!file_exists(files[i])) continue;
            FILE* fp = fopen(files[i], "w");
	    check_file(fp,files[i]);
            fclose(fp);
          }
}


int code_generation_pass(const char* stage0, const char* stage1, const char* stage2, const char* stage3, const char* stage4, const char* dir, const bool gen_mem_accesses, const bool optimize_conditionals, const bool gen_extra_dfuncs)
{
	init_str_vec(&const_ints);
	init_str_vec(&const_int_values);
	if(!file_exists(ACC_GEN_PATH))
	  make_dir(ACC_GEN_PATH);
        // Stage 1: Preprocess includes
        {
	  const bool log = !gen_extra_dfuncs;
          FILE* out = fopen(stage1, "w");
          assert(out);
	  fprintf(out,"#define AC_LAGRANGIAN_GRID (%d)\n",AC_LAGRANGIAN_GRID);
	  fprintf(out,"#define TWO_D (%d)\n",TWO_D);
       	  process_includes(1, dir, ACC_BUILTIN_VARIABLES, out,log);
       	  process_includes(1, dir, ACC_BUILTIN_TYPEDEFS, out,log);
       	  process_includes(1, dir, ACC_BUILTIN_FUNCS, out,log);
	  if(file_exists(ACC_GEN_PATH"/extra_dfuncs.h"))
       	  	process_includes(1, dir, ACC_GEN_PATH"/extra_dfuncs.h", out,log);
	  if(file_exists(ACC_OVERRIDES_PATH) && !AC_RUNTIME_COMPILATION)
       	  	process_includes(1, dir, ACC_OVERRIDES_PATH, out,log);
	  //the actual includes
          process_includes(0, dir, stage0, out,log);

	  if(file_exists(ACC_GEN_PATH"/boundcond_kernels.h"))
       	  	process_includes(1, dir, ACC_GEN_PATH"/boundcond_kernels.h", out,log);
       	  process_includes(1, dir, ACC_BUILTIN_KERNELS, out,log);
       	  process_includes(1, dir, ACC_BUILTIN_DEFAULT_VALUES, out,log);
          fclose(out);
        }

        // Stage 2: Preprocess everything else
        {
	  expand_macros(stage1,stage2);
	  FILE* f_check = fopen(stage2,"r");
          char line[4098];
 	  while (fgets(line, sizeof(line), f_check) != NULL) {
          	if (!strncmp(line, "AC_FATAL_ERROR", strlen("AC_FATAL_ERROR")))
		{
			const size_t len = 4098;
      			char message[len];
      			sscanf(line, "AC_FATAL_ERROR: %[^\"]\n", message);
		        printf("%s %s\n",FATAL_ERROR_MESSAGE,message);	
			exit(EXIT_FAILURE);
		}
	  }
          fclose(f_check);
        }

        // Generate code
        yyin = fopen(stage2, "r");

	stage4_name_backup = stage2;
        yyin_backup = fopen(stage2, "r");
        if (!yyin)
            return EXIT_FAILURE;

        int error = yyparse();
        if (error)
            return EXIT_FAILURE;

        fclose(yyin);
	if(gen_extra_dfuncs)
	{
		FILE* fp = fopen(ACC_GEN_PATH"/extra_dfuncs.h","w");
		gen_extra_funcs(root, fp);
		fclose(fp);
		return EXIT_SUCCESS;
	}
        // generate(root, stdout);

        // Stage 0: Clear all generated files to ensure acc failure can be detected later
	ASTNode* new_root = astnode_dup(root,NULL);
	preprocess(new_root, optimize_conditionals);

	reset_all_files();
  	gen_output_files(new_root);

	if(OPTIMIZE_MEM_ACCESSES)
	{
	
        	FILE* fp_cpu = fopen("user_kernels.h.raw", "w");
        	assert(fp_cpu);
        	generate(new_root, fp_cpu, true, optimize_conditionals);
		fclose(fp_cpu);
     		generate_mem_accesses(); // Uncomment to enable stencil mem access checking
        	system("cp user_kernels.h user_cpu_kernels.h");
	}
        FILE* fp = fopen("user_kernels.h.raw", "w");
        assert(fp);
        generate(new_root, fp, gen_mem_accesses, optimize_conditionals);

	astnode_destroy(root);
	root = NULL;
	
        fclose(fp);


        // Stage 4: Format
        format_source("user_kernels.h.raw", "user_kernels.h");


        return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    atexit(&cleanup);
    string_vec filenames;
    init_str_vec(&filenames);
    char* file = NULL;
    if(argc > 2)
    {
	file = malloc(sizeof(char)*(strlen(argv[1]) + strlen(argv[argc-1])));
	sprintf(file,"%s/%s",dirname(strdup(argv[1])), argv[argc-1]);
    }
    else if (argc == 2)
	file = argv[1];
    else {
        puts("Usage: ./acc [source file]");
        return EXIT_FAILURE;
    }
    //HACK to find if multiple .ac files without DSL_MODULE_FILE
    if(strstr(file,"//"))
    {
      fprintf(stderr, "Error multiple .ac files passed to acc, can only process one at a time. Ensure that DSL_MODULE_DIR contains only one .ac file or specify the file with DSL_MODULE_FILE.\n");
	return EXIT_FAILURE;
    }
	
    bool found_file = false;
    for(int i = 1; i < argc; ++i)
	found_file |= !strcmp(argv[i], file);
    if(!found_file)
    {
	fprintf(stderr,"Did not find file: %s in %s\n",argv[argc-1],dirname(argv[1]));
        return EXIT_FAILURE;
    }
    char stage0[strlen(file)+1];
    strcpy(stage0, file);
    const char* stage1 = "user_kernels.ac.pp_stage1";
    const char* stage2 = "user_kernels.ac.pp_stage2";
    const char* stage3 = "user_kernels.ac.pp_stage3";
    const char* stage4 = "user_kernels.ac.pp_stage4";
    const char* dir = dirname(file); // WARNING: dirname has side effects!
    dir_backup = dir;
    
    reset_extra_files();
    code_generation_pass(stage0, stage1, stage2, stage3, stage4, dir, false, false, true); 
    code_generation_pass(stage0, stage1, stage2, stage3, stage4,  dir, false, OPTIMIZE_CONDITIONALS, false);
    

    return EXIT_SUCCESS;
}
%}

%token IDENTIFIER STRING NUMBER REALNUMBER DOUBLENUMBER
%token IF ELIF ELSE WHILE FOR RETURN IN BREAK CONTINUE
%token BINARY_OP ASSIGNOP QUESTION UNARY_OP
%token INT UINT REAL MATRIX FIELD FIELD3 STENCIL WORK_BUFFER BOOL INTRINSIC 
%token KERNEL INLINE ELEMENTAL BOUNDARY_CONDITION SUM MAX COMMUNICATED AUXILIARY DCONST_QL CONST_QL SHARED DYNAMIC_QL CONSTEXPR RUN_CONST GLOBAL_MEMORY_QL OUTPUT VTXBUFFER COMPUTESTEPS BOUNDCONDS INPUT
%token HOSTDEFINE
%token STRUCT_NAME STRUCT_TYPE ENUM_NAME ENUM_TYPE 
%token STATEMENT_LIST_HEAD STATEMENT
%token REAL3 INT3
%token RANGE

%nonassoc QUESTION
%nonassoc ':'
%left '-'
%left '+'
%left '&'
%left BINARY_OP
%%


root: program { root = astnode_create(NODE_UNKNOWN, $1, NULL); }
    ;

program: /* Empty*/                  { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); }
       | program variable_definitions {
            $$ = astnode_create(NODE_UNKNOWN, $1, $2);

            ASTNode* variable_definition = $$->rhs;
            assert(variable_definition);

            ASTNode* declaration = get_node(NODE_DECLARATION, variable_definition);
            assert(declaration);

            ASTNode* declaration_list = declaration->rhs;
            assert(declaration_list);


	    ASTNode* declaration_list_head = declaration_list;
	    const bool are_arrays = (get_node(NODE_ARRAY_ACCESS,declaration) != NULL) ||
				    (get_node(NODE_ARRAY_INITIALIZER,declaration) != NULL);
	    ASTNode* type_specifier= get_node(NODE_TSPEC, declaration);
            ASTNode* assignment = get_node(NODE_ASSIGNMENT, variable_definition);
	
            //if (assignment) {
	    //    
            //    fprintf(stderr, "FATAL ERROR: Device constant assignment is not supported. Load the value at runtime with ac[Grid|Device]Load[Int|Int3|Real|Real3]Uniform-type API functions or use #define.\n");
            //    assert(!assignment);
            //}
            if (get_node_by_token(FIELD, variable_definition)) {
                variable_definition->type |= NODE_VARIABLE;
                set_identifier_type(NODE_VARIABLE_ID, declaration_list);
		if(are_arrays)
		{
			const int num_fields = eval_int(declaration_list_head->lhs->rhs);
			ASTNode* copy = astnode_dup(declaration_list_head->lhs,declaration_list_head);
			char* index = malloc(100*sizeof(char));
			ASTNode* field_name = declaration_list_head->lhs->lhs->lhs;
                        const char* field_name_str = strdup(field_name->buffer);
			sprintf(index,"_%d",num_fields-1);
			strcat(field_name->buffer,index);
			char* array_val     = malloc(4098*sizeof(char));
			char* array_val_tmp = malloc(4098*sizeof(char));
			sprintf(array_val,"%s",field_name->buffer);
			for(int i=num_fields-2; i>=0;--i)
			{
				ASTNode* new_head = astnode_create(NODE_UNKNOWN,NULL,NULL);
				new_head->lhs  = astnode_dup(copy,new_head);
				new_head->parent=declaration_list_head;
				declaration_list_head ->rhs = declaration_list_head->lhs;
				declaration_list_head ->lhs=new_head;
				declaration_list_head  = new_head;
				ASTNode* field_name_inner = declaration_list_head->lhs->lhs->lhs;
				sprintf(index,"_%d",i);
				strcat(field_name_inner->buffer,index);
				sprintf(array_val_tmp,"%s,%s",field_name_inner->buffer,array_val);
				//swap
				char* swap_tmp;
				swap_tmp = array_val;
				array_val = array_val_tmp;
				array_val_tmp = swap_tmp;
				
			}
			ASTNode* tmp = $$;

			ASTNode* type_declaration = create_type_declaration("const","VertexBufferHandle*",CONST_QL);

			ASTNode* var_identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
			var_identifier->buffer = malloc(sizeof(char)*4098);
			snprintf(var_identifier->buffer,4098,"%s",field_name_str);
			var_identifier->token = IDENTIFIER;
			var_identifier->type = NODE_VARIABLE_ID;
			

			ASTNode* expression_identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
			expression_identifier->buffer = strdup(array_val);
			expression_identifier->token = IDENTIFIER;
			ASTNode* expression = astnode_create(NODE_UNKNOWN,expression_identifier,NULL);
			ASTNode* assign_expression = astnode_create(NODE_UNKNOWN,expression,NULL);

			ASTNode* assign_leaf = astnode_create(NODE_ASSIGNMENT, var_identifier,assign_expression);

			ASTNode* assignment_list = astnode_create(NODE_UNKNOWN, assign_leaf,NULL);

			ASTNode* var_definitions = astnode_create(NODE_ASSIGN_LIST,type_declaration,assignment_list);
			var_definitions -> type |= NODE_NO_OUT;
			var_definitions -> type |= NODE_DECLARATION;
			var_definitions->postfix = strdup(";");

			$$ = astnode_create(NODE_UNKNOWN,tmp,var_definitions);
			tmp = $$;
			free(index);
			free(array_val);
			free(array_val_tmp);
		}
            } 
            else if(get_node_by_token(WORK_BUFFER, variable_definition)) {
                variable_definition->type |= NODE_VARIABLE;
                set_identifier_type(NODE_VARIABLE_ID, declaration_list);
            }
	    else if(are_arrays)
	    {
                variable_definition->type |= NODE_VARIABLE;
                set_identifier_type(NODE_VARIABLE_ID, declaration_list);
		//make it an array type i.e. pointer
		strcat(type_specifier->lhs->buffer,"*");

		//if dconst or runtime array evaluate the dimension to a single integer to make further transformations easier
		const ASTNode* tqual = get_node(NODE_TQUAL,variable_definition);
		if(!tqual || has_qualifier(variable_definition,"dconst") || has_qualifier(variable_definition,"run_const"))
		{
			
			char* tmp = malloc(1000*sizeof(char));
			const ASTNode* base = variable_definition->lhs->rhs->lhs;
			while(base->rhs)
			{
			  ASTNode* array_len_node = base->rhs;
			  const int array_len = eval_int(array_len_node);
			  set_buffers_empty(array_len_node);
			  array_len_node -> buffer = itoa(array_len);
			  base = base->lhs;
			}
			free(tmp);
		}
				
	    }
	    //assume is a dconst var
	    else if (assignment)
	    {
		if(!has_qualifier($$->rhs,"const"))
		{
                  fprintf(stderr, FATAL_ERROR_MESSAGE"assignment to a global variable only allowed for constant values\n");
                  assert(!has_qualifier($$->rhs,"const"));
		  exit(EXIT_FAILURE);
		}
                variable_definition->type |= NODE_VARIABLE;
                set_identifier_type(NODE_VARIABLE_ID, declaration_list);
		const char* spec = get_node(NODE_TSPEC,$$->rhs)->lhs->buffer;
		if(!strcmp(spec,"int"))
		{	

			char* assignment_val = malloc(4098*sizeof(char));
			ASTNode* def_list_head = get_node(NODE_ASSIGN_LIST,$$->rhs)->rhs;
			while(def_list_head->rhs)
			{
				ASTNode* def = def_list_head->rhs;
				char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
				int val = eval_int(def->rhs);
				push(&const_ints,name);
				push(&const_int_values,itoa(val));
				def_list_head = def_list_head->lhs;
			}
			ASTNode* def = def_list_head->lhs;
			char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
			int val = eval_int(def->rhs);
			push(&const_ints,name);
			push(&const_int_values,itoa(val));
			free(assignment_val);
		}
	
	    }

	    else if(has_qualifier($$->rhs,"run_const") || has_qualifier($$->rhs,"output"))
	    {
                variable_definition->type |= NODE_VARIABLE;
                set_identifier_type(NODE_VARIABLE_ID, declaration_list);
	    }
            else {
                variable_definition->type |= NODE_DCONST;
                set_identifier_type(NODE_VARIABLE_ID, declaration_list);
            }

         }
       | program intrinsic_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program function_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program stencil_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program hostdefine           { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program struct_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program steps_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program boundconds_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program enum_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       ;
/*
 * =============================================================================
 * Terminals
 * =============================================================================
 */
struct_name : STRUCT_NAME { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; $$->token = IDENTIFIER;};
enum_name: ENUM_NAME { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
identifier: IDENTIFIER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
number: NUMBER         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
      | REALNUMBER     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_prefix("AcReal(", $$); astnode_set_postfix(")", $$); }
      | DOUBLENUMBER   {
            $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken;
            astnode_set_prefix("double(", $$); astnode_set_postfix(")", $$);
            $$->buffer[strlen($$->buffer) - 1] = '\0'; // Drop the 'd' postfix
        }
      ;
string: STRING         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
if: IF                 { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
elif: ELIF             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
else: ELSE             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ",$$); };
while: WHILE           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
for: FOR               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
in: IN                 { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
communicated: COMMUNICATED { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
dconst_ql: DCONST_QL   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
const_ql: CONST_QL     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
shared: SHARED { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
dynamic: DYNAMIC_QL { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
constexpr: CONSTEXPR     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
run_const: RUN_CONST   { 						
	 								$$ = astnode_create(NODE_UNKNOWN, NULL, NULL); 
									if(AC_RUNTIME_COMPILATION)
	 								      astnode_set_buffer("run_const", $$); 
									else
	 									astnode_set_buffer("dconst", $$); 
									$$->token = 255 + yytoken; astnode_set_postfix(" ", $$); 
		       };
output: OUTPUT         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
input:  INPUT          { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
global_ql: GLOBAL_MEMORY_QL{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
auxiliary: AUXILIARY   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
int: INT               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
uint: UINT             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
real: REAL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcReal", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
bool: BOOL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("bool", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
matrix: MATRIX         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcMatrix", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
field: FIELD           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
field3: FIELD3         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
work_buffer: WORK_BUFFER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
stencil: STENCIL       { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("Stencil", $$); /*astnode_set_buffer(yytext, $$);*/ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
return: RETURN         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("return", $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$);};
kernel: KERNEL         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("Kernel", $$); $$->token = 255 + yytoken; };
vtxbuffer: VTXBUFFER   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("VertexBufferHandle", $$); $$->token = 255 + yytoken; };
computesteps: COMPUTESTEPS { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("ComputeSteps", $$); $$->token = 255 + yytoken; };
boundconds: BOUNDCONDS{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("BoundConds", $$); $$->token = 255 + yytoken; };
intrinsic: INTRINSIC{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("intrinsic", $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$);};
inline: INLINE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("inline", $$); $$->token = 255 + yytoken; };
elemental: ELEMENTAL { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("elemental", $$); $$->token = 255 + yytoken; };
boundary_condition: BOUNDARY_CONDITION { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("boundary_condition", $$); $$->token = 255 + yytoken;};
sum: SUM               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("sum", $$); $$->token = 255 + yytoken; };
max: MAX               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("max", $$); $$->token = 255 + yytoken; };
struct_type: STRUCT_TYPE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
enum_type: ENUM_TYPE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
hostdefine: HOSTDEFINE { $$ = astnode_hostdefine(yytext,yytoken);};

/*
 * =============================================================================
 * Structure Definitions 
 * =============================================================================
*/

function_call_list: function_call {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
		  | function_call_list  function_call {$$ = astnode_create(NODE_UNKNOWN,$1,$2);}
		  ;

steps_definition_call: identifier '(' identifier ')' {$$ = astnode_create(NODE_UNKNOWN,$1,$3);}
		     ;
boundconds_definition: boundconds identifier '{' function_call_list '}' {$$ = astnode_create(NODE_BOUNDCONDS_DEF, $2, $4);}
	       ;
steps_definition: computesteps steps_definition_call '{' function_call_list '}' {$$ = astnode_create(NODE_TASKGRAPH_DEF, $2, $4);}
	       ;


struct_definition:     struct_name'{' declarations '}' {
                        $$ = astnode_create(NODE_STRUCT_DEF,$1,$3);
			remove_substring($$->lhs->buffer,"struct");
			strip_whitespace($$->lhs->buffer);
                 }
		 ;
enum_definition: enum_name '{' expression_list '}'{
                        $$ = astnode_create(NODE_ENUM_DEF,$1,$3);
		        remove_substring($1->buffer,"enum");
		        strip_whitespace($1->buffer);
		}
		//| enum_name '{' expression_list '}' enum_type {
                //        $$ = astnode_create(NODE_ENUM_DEF,$1,$3);
		//	remove_substring_parser($1->buffer,"typedef");
		//        remove_substring_parser($1->buffer,"enum");
		//        strip_whitespace($1->buffer);
		//}
		;

/*
 * =============================================================================
 * Types
 * =============================================================================
*/
scalar_type_specifier: 
		int            { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | uint         { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | real         { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | bool         { $$ = astnode_create(NODE_TSPEC, $1, NULL); }

type_specifier: 
	        scalar_type_specifier {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	      | scalar_type_specifier '[' ']' {
				$$ = astnode_create(NODE_UNKNOWN,$1,NULL); 
				const char* old_tspec = $1->lhs->buffer;
				char tmp[strlen(old_tspec) + 1];
				sprintf(tmp,"%s*",old_tspec);
				free($1->lhs->buffer);
				$1->lhs->buffer = strdup(tmp);
			}
              | matrix       { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | matrix '[' expression ']'
		{ 
 		   char* express_str = malloc(sizeof(char)*4098);
		   combine_all($3, express_str);
		   $$ = astnode_create(NODE_TSPEC, $1, NULL); 
		   char* tmp = malloc(sizeof(char)*(strlen($1->buffer)+ strlen(express_str) + 100));  
		   sprintf(tmp,"AcMatrixN<%s>",express_str);
		   astnode_set_buffer(tmp,$1);
		   free(tmp);
		   free(express_str);
  		}
              | field        { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | field3       { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | work_buffer  { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | stencil      { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | struct_type  { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | enum_type    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | vtxbuffer    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
	      | kernel       { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              ;

type_specifiers: type_specifiers ',' type_specifier {$$ = astnode_create(NODE_UNKNOWN,$1,$3); }
	       | type_specifier  {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	       ;

type_qualifier: sum          { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | max          { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | communicated { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | dconst_ql    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | const_ql     { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | shared       { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | dynamic      { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | constexpr    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | run_const    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | global_ql    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | output       { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | input        { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | auxiliary    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | inline       { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | elemental    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | boundary_condition     { $$ = astnode_create(NODE_TQUAL, $1, NULL);}
              ;

type_qualifiers: type_qualifiers type_qualifier {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
	       | type_qualifier {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	       ;




/*
 * =============================================================================
 * Operators
 * =============================================================================
*/

//Plus and minus have to be in the parser since based on context they are unary or binary ops
binary_op: '+'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("+", $$);    $$->token = BINARY_OP; astnode_set_prefix(" ",$$); astnode_set_postfix(" ",$$);}
         | '-'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("-", $$);    $$->token = BINARY_OP; astnode_set_prefix(" ",$$); astnode_set_postfix(" ",$$);}
         | '&'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("&", $$);    $$->token = BINARY_OP; astnode_set_prefix(" ",$$); astnode_set_postfix(" ",$$);}
         | BINARY_OP   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = BINARY_OP; astnode_set_prefix(" ",$$); astnode_set_postfix(" ",$$);}
         ;

unary_op: '-'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = UNARY_OP; }
        | '+'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = UNARY_OP; }
        | '!'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = UNARY_OP; }
        | '&'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = UNARY_OP; }
        | UNARY_OP  { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$);  $$->token = UNARY_OP; }
        ;

assignment_op: ASSIGNOP    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
             ;

/*
 * =============================================================================
 * Expressions
 * =============================================================================
*/

primary_expression: identifier         { $$ = astnode_create(NODE_PRIMARY_EXPRESSION, $1, NULL); }
                  | number             { $$ = astnode_create(NODE_PRIMARY_EXPRESSION, $1, NULL); }
                  | string             { $$ = astnode_create(NODE_PRIMARY_EXPRESSION, $1, NULL); }
                  ;

struct_initializer: 
                  '{' expression_list '}' { $$ = astnode_create(NODE_STRUCT_INITIALIZER, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
		  ;

base_identifier:
		  identifier         { $$ = astnode_create(NODE_PRIMARY_EXPRESSION, $1, NULL); }
                  | base_identifier '[' expression ']' { $$ = astnode_create(NODE_ARRAY_ACCESS, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                  | base_identifier '.' identifier     { $$ = astnode_create(NODE_STRUCT_EXPRESSION, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
		  ;

postfix_expression: primary_expression                         { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | base_identifier '[' expression ']'      { $$ = astnode_create(NODE_ARRAY_ACCESS, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                  | base_identifier '.' identifier          { $$ = astnode_create(NODE_STRUCT_EXPRESSION, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                  | postfix_expression '(' ')'                 { $$ = astnode_create(NODE_FUNCTION_CALL, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
                  | postfix_expression '(' expression_list ')' { $$ = astnode_create(NODE_FUNCTION_CALL, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); } 
                  | type_specifier '(' expression_list ')'     { $$ = astnode_create(NODE_UNKNOWN, $1, $3); char tmp[10000]; combine_all($$->lhs,tmp);  $$->expr_type = strdup(tmp); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); $$->lhs->type ^= NODE_TSPEC; /* Unset NODE_TSPEC flag, casts are handled as functions */ }
                  | '(' type_specifier ')'  struct_initializer { 
						$$ = astnode_create(NODE_UNKNOWN, $2, $4); astnode_set_prefix("(",$$); 
						astnode_set_postfix(")",$$); 
						char tmp[1000]; 
						combine_all($$->lhs,tmp); 
						astnode_set_buffer(strdup(tmp),$$); 
						//$$->rhs->expr_type = strdup(tmp);
						}
                  | '[' expression_list ']' { $$ = astnode_create(NODE_ARRAY_INITIALIZER, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
		  | struct_initializer {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
                  | '(' expression ')' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("(", $$); astnode_set_postfix(")", $$); }


unary_expression: postfix_expression          { $$ = astnode_create(NODE_EXPRESSION, $1, NULL); }
                | unary_op postfix_expression { $$ = astnode_create(NODE_EXPRESSION, $1, $2);}
                ;

binary_expression: binary_op unary_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                 ;


choose: QUESTION expression ':' expression {$$ = astnode_create(NODE_UNKNOWN,$2,$4);  astnode_set_prefix("? ",$$->lhs);  astnode_set_prefix(": ",$$->rhs);}
      ;
expression: unary_expression             { $$ = astnode_create(NODE_EXPRESSION, $1, NULL); }
	  | expression choose            { $$ = astnode_create(NODE_TERNARY_EXPRESSION,$1,$2); } 
          | expression binary_expression { $$ = astnode_create(NODE_BINARY_EXPRESSION, $1, $2); }


expression_list: expression                     { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
               | expression_list ',' expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
               ;

/*
 * =============================================================================
 * Definitions and Declarations
 * =============================================================================
*/
variable_definition: declaration { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
                   | assignment  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
                   ;

intrinsic_body: identifier {$$ = astnode_create(NODE_UNKNOWN, $1,NULL); }
	      | identifier '(' type_specifiers ')' {  $$ = astnode_create(NODE_UNKNOWN,$1,$3); }
	      ;

intrinsic_type_declaration: type_declaration intrinsic
			    {
				$$ = astnode_create(NODE_UNKNOWN,$1,$2);  
			    }
intrinsic_definition:  
		    intrinsic_type_declaration intrinsic_body {
		    		$$ = astnode_create(NODE_DECLARATION, $1, $2);
				$$ -> type |= NODE_NO_OUT;
                		set_identifier_type(NODE_FUNCTION_ID, $2);
				if($$->lhs->lhs)
				{
					if(count_num_of_nodes_in_list($$->lhs->lhs) == 1 && !strcmp(get_node(NODE_TSPEC,$$->lhs->lhs)->lhs->buffer,"AcReal"))
					{
						ASTNode* base = astnode_create(NODE_UNKNOWN,create_type_qualifier("intrinsic",INTRINSIC),NULL);
						ASTNode* tqualifiers = astnode_create(NODE_UNKNOWN,base,create_type_qualifier("AcReal",REAL));
						$$->rhs->lhs->rhs = tqualifiers;
						tqualifiers->parent = $$->rhs->lhs->rhs;
					}
				}
		    }
		     ;
variable_definitions: non_null_declaration { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
                   |  type_declaration assignment_list  
				{ 
				  if(!get_node(NODE_TSPEC,$1))
				  {
				  	fprintf(stderr,"Fatal error: all global variables have to have a type specifier\n");
				  	fprintf(stderr,"Offending variable: %s\n",get_node_by_token(IDENTIFIER,$2)->buffer);
				  	exit(EXIT_FAILURE);
				  }
				  $$ = astnode_create(NODE_ASSIGN_LIST, $1, $2); $$->type |= NODE_DECLARATION; astnode_set_postfix(";", $$); 
				  //if list assignment make the type a pointer type
				  ASTNode* assignment = get_node(NODE_ASSIGNMENT, $2);
				  if(get_node(NODE_ARRAY_INITIALIZER,assignment->rhs))
				  {
					ASTNode* tspec = get_node(NODE_TSPEC,$1);
					strcat(tspec->lhs->buffer,"*");
				  }
				  if(!$$->lhs->rhs)
				  {
					const char* tspec = get_node(NODE_TSPEC,$$->lhs)->lhs->buffer;
					if(strcmps(tspec,"Field","Field3"))
					{
						ASTNode* tqualifiers = create_type_qualifiers("dconst",DCONST_QL);
						tqualifiers -> parent = $$->lhs;
						$$->lhs->rhs = tqualifiers;
					}
				  }
				}
                   ;


assignment_list_leaf: identifier assignment_op expression {
		    							$$ = astnode_create(NODE_ASSIGNMENT,$1,$3);
								 }
		    ;
assignment_list: assignment_list ',' assignment_list_leaf {$$ = astnode_create(NODE_UNKNOWN,$1,$3);}
	       | assignment_list_leaf {$$ = astnode_create(NODE_UNKNOWN,$1,NULL);}

declarations: declarations declaration {$$ = astnode_create(NODE_UNKNOWN, $1,$2); }
	    | declaration {$$ = astnode_create(NODE_UNKNOWN, $1,NULL); }
	    ;
non_null_declaration: type_declaration declaration_list { 

				if(!get_node(NODE_TSPEC,$1))
				{
					fprintf(stderr,"Fatal error: all global variables have to have a type specifier\n");
					fprintf(stderr,"Offending variable: %s\n",get_node_by_token(IDENTIFIER,$2)->buffer);
					exit(EXIT_FAILURE);
				}

		    		$$ = astnode_create(NODE_DECLARATION, $1, $2); 
				if(!$$->lhs->rhs)
				{
				    const char* tspec = get_node(NODE_TSPEC,$$->lhs)->lhs->buffer;
				    if(strcmps(tspec,"Field","Field3"))
				    {
				    	ASTNode* tqualifiers = create_type_qualifiers("dconst",DCONST_QL);
				    	tqualifiers -> parent = $$->lhs;
				    	$$->lhs->rhs = tqualifiers;
				    }
				}
			}
           ;
declaration: type_declaration  declaration_list { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
	   //| type_declaration  '{' declaration_list '}' { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
           ;



declaration_list: base_identifier { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | declaration_list ',' base_identifier { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); /* Note ';' infix */ }
                ;

parameter: type_declaration identifier { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
         ;

parameter_list: parameter                    { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | parameter_list ',' parameter { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
              ;

type_declaration: /* Empty */                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL);}
                | type_specifier                 { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | type_qualifiers                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | type_qualifiers type_specifier { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                ;


assignment: declaration assignment_body { 
	  		$$ = astnode_create(NODE_ASSIGNMENT, $1, $2); 
			//Convert C-arrays to AcArrays to standardize all arrays to AcArrays such that each DSL array has the same functionality.
			//If the std::array implementation was better on HIP could also not require type specifier since it could be inferred
			if($2->prefix && !strcmp($2->prefix,"[]"))
			{
				ASTNode* tspec = get_node(NODE_TSPEC,$1);
				if(tspec)
				{
					astnode_set_prefix("",$2);
					const size_t old_size = strlen(tspec->lhs->buffer);
					char* tmp = malloc(sizeof(char)*(old_size + 1000));
					sprintf(tmp,"AcArray<%s,%d>",tspec->lhs->buffer,count_num_of_nodes_in_list($2->rhs));
					free(tspec->lhs->buffer);
					tspec->lhs->buffer = tmp;
				}
			}
			if(get_node(NODE_ARRAY_INITIALIZER,$2))
			{
				ASTNode* tspec = get_node(NODE_TSPEC,$1);
				if(tspec)
				{
					const size_t old_size = strlen(tspec->lhs->buffer);
					char* tmp = malloc(sizeof(char)*(old_size + 1000));
					sprintf(tmp,"AcArray<%s,%d>",tspec->lhs->buffer,count_num_of_nodes_in_list(get_node(NODE_ARRAY_INITIALIZER,$2)->lhs));
					free(tspec->lhs->buffer);
					tspec->lhs->buffer = tmp;
				}
			}
			const int num_of_elems = count_num_of_nodes_in_list($1->rhs);
			if(num_of_elems > 1)
			{
				astnode_set_prefix("auto [",$1);
				astnode_set_postfix("]",$1);
				add_no_auto($1, NULL);
			}
		}
          ;


assignment_body: assignment_op expression_list
	       {
                    $$ = astnode_create(NODE_UNKNOWN, $1, $2);
                    if ($$->rhs && $$->rhs->rhs) {
                        astnode_set_prefix("[]", $$);
                        astnode_set_infix("{", $$);
                        astnode_set_postfix("}", $$);
                    }

                }
               ;

/*
 * =============================================================================
 * Statements
 * =============================================================================
*/

basic_statement: 
	   variable_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
         | return expression    { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_postfix(";", $$); }
         | function_call        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
         ;

non_selection_statement: 
	   basic_statement { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
         | iteration_statement  { $$ = astnode_create(NODE_BEGIN_SCOPE, $1, NULL); }
         ;

statement: selection_statement     { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
	 | non_selection_statement {$$ = astnode_create(NODE_UNKNOWN, $1, NULL);  }
         ;


statement_list: statement                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | statement_list statement { $$ = astnode_create(NODE_STATEMENT_LIST_HEAD, $1, $2);}
              ;


compound_statement: '{' '}'                { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
                  | '{' statement_list '}' { $$ = astnode_create(NODE_BEGIN_SCOPE, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
                  ;

selection_statement: if if_statement        { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   ;

if_statement: expression if_root  {$$ = astnode_create(NODE_IF, $1, $2); astnode_set_prefix("(",$$); astnode_set_infix(")",$$); }
	    
if_root: compound_statement else_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2);}
       | non_selection_statement else_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2);}
       | compound_statement elif_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2);}
       | non_selection_statement elif_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2);}       
       | compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); 
				if($$->lhs->lhs)
				{
					const int n_statements = count_num_of_nodes_in_list($$->lhs->lhs);
					if(n_statements == 1)
					{
						$$->lhs->type = NODE_UNKNOWN;
					}
				}
			    }
       | statement 	    { $$ = astnode_create(NODE_UNKNOWN, $1, NULL);}
         ;

elif_statement: elif if_statement                 { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

else_statement: else compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
	      | else statement          { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

iteration_statement: while_statement compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   | for_statement compound_statement   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   | for_statement variable_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   | for_statement function_call        { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_postfix(";",$$); }
                   ;

while_statement: while expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
               ;

for_statement: for for_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
             ;

for_expression: identifier in expression {
    			$$ = astnode_create(NODE_UNKNOWN, $1, $3);
	      }
	      | identifier in range {
    			$$ = astnode_create(NODE_UNKNOWN, $1, $3);
    			astnode_set_infix("=", $$);

    			const size_t padding = 32;
    			char* tmp = malloc(strlen($1->buffer) + padding);
    			sprintf(tmp, ";%s<", $1->buffer);
    			astnode_set_buffer(tmp, $$->rhs);

    			sprintf(tmp, ";++%s", $1->buffer);
    			astnode_set_postfix(tmp, $$);
    			free(tmp);
	      }
	      ;

range: expression ':' expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->token = RANGE;}
     ;

/*
 * =============================================================================
 * Functions
 * =============================================================================
*/
function_definition: declaration function_body {
                        $$ = astnode_create(NODE_UNKNOWN, $1, $2);
			if(get_node(NODE_TSPEC,$$->lhs) && strcmp(get_node(NODE_TSPEC,$$->lhs)->lhs->buffer,"Kernel"))
			{
				fprintf(stderr,"%s","Fatal error: can't specify the type of functions, since they will be deduced\n");
				fprintf(stderr,"%s","Consider instead leaving a code comment telling the return type\n");
				fprintf(stderr,"Offending function: %s\n",get_node_by_token(IDENTIFIER,$$->lhs)->buffer);
				assert(false);
				exit(EXIT_FAILURE);
			}

			if(has_qualifier($$,"extern"))
			{
				$$->type |= NODE_NO_OUT;
			}

                        ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, $$->lhs);
                        assert(fn_identifier);
                        set_identifier_type(NODE_FUNCTION_ID, fn_identifier);

                        const ASTNode* is_kernel = get_node_by_token(KERNEL, $$);
			if(is_kernel)
				astnode_set_prefix("__global__ void \n#if MAX_THREADS_PER_BLOCK\n__launch_bounds__(MAX_THREADS_PER_BLOCK)\n#endif\n",$$);
                        ASTNode* compound_statement = $$->rhs->rhs;
                        if (is_kernel) {
                            $$->type |= NODE_KFUNCTION;
                            // Set kernel built-in variables
                            const char* default_param_list=  "(const int3 start, const int3 end, VertexBufferArray vba";
                            astnode_set_prefix(default_param_list, $$->rhs);

                            assert(compound_statement);
                            astnode_set_prefix("{", compound_statement);
                            astnode_set_postfix(
                              //"\n#pragma unroll\n"
                              //"for (int field = 0; field < NUM_FIELDS; ++field)"
                              //"if (!isnan(out_buffer[field]))"
                              //"vba.out[field][idx] = out_buffer[field];"
                              "}", compound_statement);
                        } else {
                            astnode_set_infix(" __attribute__((unused)) =[&]", $$);
                            astnode_set_postfix(";", $$);
                            $$->type |= NODE_DFUNCTION;
                            set_identifier_type(NODE_DFUNCTION_ID, fn_identifier);

                            // Pass device function parameters by const reference
                            if ($$->rhs->lhs) {
                                set_identifier_prefix("const ", $$->rhs->lhs);
                                set_identifier_infix("&", $$->rhs->lhs);
                            }

                        }
                    }
                   ;

function_body: '(' ')' compound_statement                { $$ = astnode_create(NODE_BEGIN_SCOPE, NULL, $3); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
             | '(' parameter_list ')' compound_statement { $$ = astnode_create(NODE_BEGIN_SCOPE, $2, $4); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
             ;

function_call: declaration '(' ')'                 { $$ = astnode_create(NODE_FUNCTION_CALL, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
             | declaration '(' expression_list ')' { $$ = astnode_create(NODE_FUNCTION_CALL, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$);   }
             ;

/*
 * =============================================================================
 * Stencils
 * =============================================================================
*/
assignment_body_designated: assignment_op expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("\"", $$); astnode_set_postfix("\"", $$); }
          ;

stencilpoint: stencil_index_list assignment_body_designated { 
	    								$$ = astnode_create(NODE_UNKNOWN, $1, $2); 
									const int num_of_indexes = count_num_of_nodes_in_list($1);
									if(TWO_D && num_of_indexes != 2)
									{
										fprintf(stderr,"Can only use 2D stencils when building for 2D simulation\n");
										assert(num_of_indexes == 2);
										exit(EXIT_FAILURE);
									}
									else if(!TWO_D && num_of_indexes < 3)
										fprintf(stderr,"Specify indexes for all three dimensions\n");
									else if(!TWO_D && num_of_indexes > 3)
										fprintf(stderr,"Only three dimensional stencils for 3D simulation\n");
									if(!TWO_D && num_of_indexes != 3)	
									{
										assert(num_of_indexes == 3);
										exit(EXIT_FAILURE);
									}
							    }
            ;

stencil_index: '[' expression ']' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("[STENCIL_ORDER/2 +", $$); astnode_set_postfix("]", $$); }
     ;

stencil_index_list: stencil_index            { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
          | stencil_index_list stencil_index { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
          ;

stencilpoint_list: stencilpoint                       { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                 | stencilpoint_list ',' stencilpoint { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
                 ;

stencil_body: '{' stencilpoint_list '}' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("},", $$); }
            ;

stencil_definition: declaration stencil_body { $$ = astnode_create(NODE_STENCIL, $1, $2); set_identifier_type(NODE_VARIABLE_ID, $$->lhs); 
		  }
                  ;
%%

void
print(void)
{
    printf("%s\n", yytext);
}

int
yyerror(const char* str)
{
    int line_num = yyget_lineno();
    fprintf(stderr, "\n%s on line %d when processing char %d: [%s]\n", str, line_num, *yytext, yytext);
    char* line;
    size_t len = 0;
    for(int i=0;i<line_num;++i)
	getline(&line,&len,yyin_backup);
    fprintf(stderr, "erroneous line: %s", line);
    fprintf(stderr, "in file: %s/%s\n\n",ACC_OUTPUT_DIR"/../api",stage4_name_backup);
    exit(EXIT_FAILURE);
}

void
set_identifier_type(const NodeType type, ASTNode* curr)
{
    assert(curr);
    if (curr->token == IDENTIFIER) {
        curr->type |= type;
        return;
    }

    if (curr->rhs)
        set_identifier_type(type, curr->rhs);
    if (curr->lhs)
        set_identifier_type(type, curr->lhs);
}


void
set_identifier_prefix(const char* prefix, ASTNode* curr)
{
    assert(curr);
    if (curr->token == IDENTIFIER) {
        astnode_set_prefix(prefix, curr);
        return;
    }

    if (curr->rhs)
      set_identifier_prefix(prefix, curr->rhs);
    if (curr->lhs)
      set_identifier_prefix(prefix, curr->lhs);
}

void
set_identifier_infix(const char* infix, ASTNode* curr)
{
    assert(curr);
    if (curr->token == IDENTIFIER) {
        astnode_set_infix(infix, curr);
        return;
    }

    if (curr->rhs)
      set_identifier_infix(infix, curr->rhs);
    if (curr->lhs)
      set_identifier_infix(infix, curr->lhs);
}

ASTNode*
get_node(const NodeType type, ASTNode* node)
{
  assert(node);

  if (node->type & type)
    return node;
  else if (node->lhs && get_node(type, node->lhs))
    return get_node(type, node->lhs);
  else if (node->rhs && get_node(type, node->rhs))
    return get_node(type, node->rhs);
  else
    return NULL;
}

void
replace_const_ints(ASTNode* node, const string_vec values, const string_vec names)
{
	if(node->lhs)
		replace_const_ints(node->lhs,values,names);
	if(node->rhs)
		replace_const_ints(node->rhs,values,names);
	if(node->token != IDENTIFIER || !node->buffer) return;
	const int index = str_vec_get_index(names,node->buffer);
	if(index == -1) return;
	node->buffer = strdup(values.data[index]);
}

static inline int eval_int(ASTNode* node)
{
	replace_const_ints(node,const_int_values,const_ints);
	char* copy = malloc(sizeof(char)*10000);
	combine_all(node,copy);
	strip_whitespace(copy);
        double* vals = malloc(sizeof(double)*const_ints.size);
        int err;
        te_expr* expr = te_compile(copy, NULL, 0, &err);
        if(!expr)
        {
                fprintf(stderr,"Parse error at tinyexpr\n");
		fprintf(stderr,"Was not able to parse: %s\n",copy);
		fprintf(stderr,"Was not able to parse: %s\n",const_ints.data[0]);
		fprintf(stderr,"Was not able to parse: %s\n",const_ints.data[1]);
		fprintf(stderr,"Was not able to parse: %s\n",const_ints.data[2]);
		fprintf(stderr,"Was not able to parse: %s\n",const_ints.data[3]);
                exit(EXIT_FAILURE);
        }
        int res = (int) round(te_eval(expr));
        te_free(expr);
	free(copy);
        return res;
}
static ASTNode*
create_type_qualifier(const char* tqual, const int token)
{
	if(!tqual) return NULL;
	ASTNode* tqual_identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
	tqual_identifier -> buffer = strdup(tqual);
	tqual_identifier -> token = token;
	ASTNode* type_qualifier  = astnode_create(NODE_TQUAL,tqual_identifier,NULL);
	return type_qualifier;
}
static ASTNode*
create_type_qualifiers(const char* tqual, const int token)
{
	ASTNode* type_qualifiers = astnode_create(NODE_UNKNOWN,create_type_qualifier(tqual,token),NULL);
	return type_qualifiers;
}
static ASTNode*
create_tspec(const char* tspec_str)
{
	ASTNode* tspec_identifier  = astnode_create(NODE_UNKNOWN,NULL,NULL);
	tspec_identifier -> buffer = strdup(tspec_str);
	tspec_identifier -> token = IDENTIFIER;
	ASTNode* tspec  = astnode_create(NODE_TSPEC,tspec_identifier,NULL);
	return tspec;
}
static ASTNode*
create_type_declaration(const char* tqual, const char* tspec, const int token)
{

	ASTNode* type_declaration = astnode_create(NODE_UNKNOWN,create_tspec(tspec),create_type_qualifiers(tqual,token));
	return type_declaration;
}
