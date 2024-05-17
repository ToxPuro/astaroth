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

#define YYSTYPE ASTNode*

ASTNode* root = NULL;

extern FILE* yyin;
extern char* yytext;

int yylex();
int yyparse();
int yyerror(const char* str);
int yyget_lineno();

static const char* global_func_declaration = "__global__ void \n#if MAX_THREADS_PER_BLOCK\n__launch_bounds__(MAX_THREADS_PER_BLOCK)\n#endif\n";
static const char* user_structs_filename = "user_structs.h";
static const char* user_kernel_ifs       = "user_kernel_ifs.h";
static char user_kernel_params_struct_str[10000]; 
static string_vec added_params_to_stencil_accesses;
static const char* stencil_accesses_params_filename= "user_stencil_accesses_params.h";
static char stencil_accesses_default_params[10000];
static int_vec tinyexpr_cache;
static string_vec tinyexpr_cache_str;

//These are used to generate better error messages in case of errors
FILE* yyin_backup;
const char* stage4_name_backup;
const char* dir_backup;
static string_vec const_ints;
static string_vec const_int_values;



void
cleanup(void)
{
    if (root)
        astnode_destroy(root); // Frees all children and itself
}
void strprepend(char* dst, const char* src)
{	
    memmove(dst + strlen(src), dst, strlen(dst)+ 1); // Move existing data including null terminator
    memcpy(dst, src, strlen(src)); // Copy src to the beginning of dst
}
void remove_substring_parser(char *str, const char *sub) {
	int len = strlen(sub);
	char *found = strstr(str, sub); // Find the first occurrence of the substring

	while (found) {
		memmove(found, found + len, strlen(found + len) + 1); // Shift characters to overwrite the substring
		found = strstr(found, sub); // Find the next occurrence of the substring
	}
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
static inline int eval_int(const char* str);



static inline void
mark_as_input(ASTNode* node, const char* name, char* type)
{
	if(node->lhs)
		mark_as_input(node->lhs,name,type);
	if(node->rhs)
		mark_as_input(node->rhs,name,type);
	if(node->buffer && !strcmp(node->buffer,name) && !node->lhs)
	{
		node->type |= NODE_INPUT;
		node->lhs = astnode_create(NODE_CODEGEN_INPUT, NULL,NULL);
		node->lhs->type |= NODE_TSPEC;
		node->lhs->buffer=strdup(type);
	}
	
}
void
process_param(ASTNode* kernel_root, const ASTNode* param, char* struct_params)
{
				char* param_type = malloc(4096*sizeof(char));
                                combine_buffers(param->lhs, param_type);
				char* param_str = malloc(4096*sizeof(char));
				param_str[0] = '\0';
                              	sprintf(param_str,"%s %s;",param_type, param->rhs->buffer);
				mark_as_input(kernel_root,param->rhs->buffer,param_type);
				strprepend(struct_params,param_str);
				if(str_vec_contains(added_params_to_stencil_accesses,param->rhs->buffer))
					return;
				push(&added_params_to_stencil_accesses,strdup(param->rhs->buffer));
				char* default_param = malloc(4096*sizeof(char));
                                if(!strcmp(param_type,"int"))
			          sprintf(default_param,"0");
                                else if(!strcmp(param_type,"AcReal"))
			          sprintf(default_param,"0.0");
				//we assume it is a user specified struct
				else
			          sprintf(default_param,"{}");
				char* tmp = malloc(4096*2*sizeof(char));
				sprintf(tmp," %s %sAC_INTERNAL_INPUT = %s;",param_type,param->rhs->buffer,default_param);
				strcat(stencil_accesses_default_params,tmp);
				free(param_type);
				free(tmp);
				free(param_str);
				free(default_param);

}

void
process_includes(const size_t depth, const char* dir, const char* file, FILE* out)
{
  const size_t max_nests = 64;
  if (depth >= max_nests) {
    fprintf(stderr, "CRITICAL ERROR: Max nests %lu reached when processing includes. Aborting to avoid thrashing the disk. Possible reason: circular includes.\n", max_nests);
    exit(EXIT_FAILURE);
  }

  printf("Building AC object %s\n", file);
  FILE* in = fopen(file, "r");
  if (!in) {
    fprintf(stderr, "FATAL ERROR: could not open include file '%s'\n", file);
    assert(in);
  }

  const size_t  len = 4096;
  char* buf = malloc(len*sizeof(char));
  while (fgets(buf, len, in)) {
    char* line = buf;
    while (strlen(line) > 0 && line[0] == ' ') // Remove whitespace
      ++line;

    if (!strncmp(line, "#include", strlen("#include"))) {

      char incl[len];
      sscanf(line, "#include \"%[^\"]\"\n", incl);

      char path[len];
      sprintf(path, "%s/%s", dir, incl);

      fprintf(out, "// Include file %s start\n", path);
      process_includes(depth+1, dir, path, out);
      fprintf(out, "// Included file %s end\n", path);

    } else {
      fprintf(out, "%s", buf);
    }
  }
  free(buf);
  fclose(in);
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

void
format_source(const char* file_in, const char* file_out)
{
   FILE* in = fopen(file_in, "r");
  assert(in);

  FILE* out = fopen(file_out, "w");
  assert(out);

  while (!feof(in)) {
    const char c = fgetc(in);
    if (c == EOF)
      break;

    fprintf(out, "%c", c);
    if (c == ';')
      fprintf(out, "\n");
  }

  fclose(in);
  fclose(out);
}
bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

int code_generation_pass(const char* stage0, const char* stage1, const char* stage2, const char* stage3, const char* stage4, const char* dir, const bool gen_mem_accesses, const bool optimize_conditionals)
{
    	init_str_vec(&added_params_to_stencil_accesses);
	init_str_vec(&const_ints);
	init_str_vec(&const_int_values);
        // Stage 0: Clear all generated files to ensure acc failure can be detected later
        {
          const char* files[] = {"user_declarations.h", "user_defines.h", "user_kernels.h", "user_kernel_declarations.h", stencil_accesses_params_filename, user_structs_filename, user_kernel_ifs, "user_dfuncs.h"};
          for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); ++i) {
            FILE* fp = fopen(files[i], "w");
            assert(fp);
            fclose(fp);
          }
        }
	user_kernel_params_struct_str[0] = '\0';
	sprintf(user_kernel_params_struct_str,"typedef struct acKernelInputParams {\nunion {\n");

        // Stage 1: Preprocess includes
        {
          FILE* out = fopen(stage1, "w");
          assert(out);
        
          process_includes(0, dir, stage0, out);

          fclose(out);
        }

        // Stage 2: Preprocess hostdefines
        {
          process_hostdefines(stage1, stage2);
        }

        // Stage 3: Preprocess everything else
        {
          const size_t cmdlen = 4096;
	  char* cmd = malloc(cmdlen*sizeof(char));
          snprintf(cmd, cmdlen, "gcc -x c -E %s > %s", stage2, stage3);
          const int retval = system(cmd);
	  free(cmd);
          if (retval == -1) {
              fprintf(stderr, "Catastrophic error: preprocessing failed.\n");
              assert(retval != -1);
          }
        }
	FILE* f_in = fopen(stage3,"r");
	FILE* f_out = fopen(stage4,"w");
        char* line = malloc(10000*sizeof(char));	

	fprintf(f_out,"\n%s\n","Stencil value {[0][0][0] =1}");
        fprintf(f_out,"\n%s\n","vecvalue(v) {\nreturn real3(value(Field(v.x)), value(Field(v.y)), value(Field(v.z)))\n}");
        fprintf(f_out,"\n%s\n","vecprevious(v) {\nreturn real3(previous(Field(v.x)), previous(Field(v.y)), previous(Field(v.z)))\n}");
        fprintf(f_out,"\n%s\n","vecwrite(dst,src) {write(Field(dst.x),src.x)\n write(Field(dst.y),src.y)\n write(Field(dst.z),src.z)}");

 	while (fgets(line, sizeof(line), f_in) != NULL) {
		remove_substring_parser(line,";");
		fprintf(f_out,"%s",line);
    	}
	free(line);
	fclose(f_in);
	fprintf(f_out,"\nKernel AC_BUILTIN_RESET() {\n"
		"for field in 0:NUM_FIELDS {\n"
			"write(Field(field), 0.0)\n"
                "}\n"
	"}\n");
	fclose(f_out);

        // Generate code
        yyin = fopen(stage4, "r");

	stage4_name_backup = stage4;
        yyin_backup = fopen(stage4, "r");
        if (!yyin)
            return EXIT_FAILURE;

        int error = yyparse();
        if (error)
            return EXIT_FAILURE;

	strcat(user_kernel_params_struct_str,"};\n} acKernelInputParams;\n");


	FILE* fp_structs = fopen(user_structs_filename,"a");
	fprintf(fp_structs,"\n%s\n",user_kernel_params_struct_str);
	fclose(fp_structs);


        // generate(root, stdout);
        FILE* fp = fopen("user_kernels.h.raw", "w");
        assert(fp);
	if(gen_mem_accesses)
		fprintf(fp,"%s\n",stencil_accesses_default_params);
        generate(root, fp, gen_mem_accesses, optimize_conditionals);
        fclose(fp);

        fclose(yyin);

        // Stage 4: Format
        format_source("user_kernels.h.raw", "user_kernels.h");


        return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    init_int_vec(&tinyexpr_cache);
    init_str_vec(&tinyexpr_cache_str);
    atexit(&cleanup);

    if (argc > 2) {
      fprintf(stderr, "Error multiple .ac files passed to acc, can only process one at a time. Ensure that DSL_MODULE_DIR contains only one .ac file.\n");
      return EXIT_FAILURE;
    }

    if (argc == 2) {

        char stage0[strlen(argv[1])];
        strcpy(stage0, argv[1]);
        const char* stage1 = "user_kernels.ac.pp_stage1";
        const char* stage2 = "user_kernels.ac.pp_stage2";
        const char* stage3 = "user_kernels.ac.pp_stage3";
        const char* stage4 = "user_kernels.ac.pp_stage4";
        const char* dir = dirname(argv[1]); // WARNING: dirname has side effects!
	dir_backup = dir;

        if (OPTIMIZE_MEM_ACCESSES) {
          code_generation_pass(stage0, stage1, stage2, stage3, stage4, dir, true, OPTIMIZE_CONDITIONALS); // Uncomment to enable stencil mem access checking
          generate_mem_accesses(); // Uncomment to enable stencil mem access checking
        }
        code_generation_pass(stage0, stage1, stage2, stage3, stage4,  dir, false, OPTIMIZE_CONDITIONALS);
        

        return EXIT_SUCCESS;
    } else {
        puts("Usage: ./acc [source file]");
        return EXIT_FAILURE;
    }
}
%}

%token IDENTIFIER STRING NUMBER REALNUMBER DOUBLENUMBER
%token IF ELIF ELSE WHILE FOR RETURN IN BREAK CONTINUE
%token BINARY_OP ASSIGNOP
%token INT UINT INT3 REAL REAL3 MATRIX FIELD STENCIL WORK_BUFFER COMPLEX BOOL
%token KERNEL INLINE SUM MAX COMMUNICATED AUXILIARY DCONST_QL CONST_QL GLOBAL_MEMORY_QL OUTPUT
%token HOSTDEFINE
%token STRUCT_NAME STRUCT_TYPE ENUM_NAME ENUM_TYPE

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
	    bool are_arrays = false;
	    ASTNode* type_specifier= get_node(NODE_TSPEC, declaration);
            ASTNode* assignment = get_node(NODE_ASSIGNMENT, variable_definition);
	    if(type_specifier && assignment == NULL)
	    {
		    while(declaration_list_head->rhs)
		    {
			const ASTNode* 	declaration_postfix_expression = declaration_list_head->rhs;
			if(declaration_postfix_expression->rhs && declaration_postfix_expression->rhs->type != NODE_MEMBER_ID)
				are_arrays = true;
			declaration_list_head = declaration_list_head->lhs;
		    }	

		    const ASTNode* 	declaration_postfix_expression = declaration_list_head->lhs;
		    if(declaration_postfix_expression->rhs && declaration_postfix_expression->rhs->type != NODE_MEMBER_ID)
			are_arrays = true;
		//    if(are_arrays)
		//	assert(!strcmp(type_specifier->lhs->buffer,"int") || !strcmp(type_specifier->lhs->buffer,"AcReal"));
	    }
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
			char* num_fields_str = malloc(1000*sizeof(char));
			combine_all(declaration_list_head->lhs->rhs, num_fields_str);
			const int num_fields = eval_int(num_fields_str);
			ASTNode* copy = astnode_dup(declaration_list_head->lhs,declaration_list_head);
			ASTNode* field_name = declaration_list_head->lhs->lhs->lhs;
			const char* field_name_str = strdup(field_name->buffer);
			char* index = malloc(100*sizeof(char));
			sprintf(index,"_%d",num_fields);
			strcat(field_name->buffer,index);
			for(int i=num_fields-1; i>=0;--i)
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
			}
			ASTNode* tmp = $$;
			char* host_definition = malloc(1000*sizeof(char));
			sprintf(host_definition,"hostdefine N%s_ROWS (%d)",field_name_str,num_fields);
			ASTNode* hostdefine =astnode_hostdefine(host_definition,HOSTDEFINE);
			tmp = astnode_create(NODE_UNKNOWN,tmp,hostdefine);
			sprintf(host_definition,"hostdefine N%s_COLS (%d)",field_name_str,1);
			hostdefine =astnode_hostdefine(host_definition,HOSTDEFINE);
			tmp = astnode_create(NODE_UNKNOWN,tmp,hostdefine);
			ASTNode* field_size_node = astnode_create(NODE_UNKNOWN, NULL, NULL);
			field_size_node->buffer = strdup(itoa(num_fields));
			ASTNode* input_to_codegen = astnode_create(NODE_CODEGEN_INPUT,field_size_node,NULL);
			input_to_codegen->buffer = strdup(field_name_str);
			$$ = astnode_create(NODE_UNKNOWN,tmp,input_to_codegen);
			free(num_fields_str);
			free(host_definition);
			free(index);
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

		//if dconst array evaluate the dimension to a single integer to make further transformations easier
		const ASTNode* tqual = get_node(NODE_TQUAL,variable_definition);
		if(!tqual || !strcmp(tqual->lhs->buffer,"donst"))
		{
			
			char* tmp = malloc(1000*sizeof(char));
			ASTNode* array_len_node = variable_definition->lhs->rhs->lhs->rhs;
			combine_all(array_len_node,tmp);
			const int array_len = eval_int(tmp);
			set_buffers_empty(array_len_node);
			array_len_node -> buffer = itoa(array_len);
			free(tmp);
		}
				
	    }
	    //assume is a dconst var
	    else if (assignment)
	    {
		ASTNode* tqual = get_node(NODE_TQUAL,$$->rhs);
		const bool is_const = !strcmp(tqual->lhs->buffer,"const");
		if(!is_const)
		{
                  fprintf(stderr, "FATAL ERROR: assigment to a global variable only allowed for constant values\n");
                  assert(is_const);
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
				combine_all(def->rhs,assignment_val);
				if(!strstr(assignment_val,","))
				{
				  int val = eval_int(assignment_val);
				  push(&const_ints,name);
				  push(&const_int_values,itoa(val));
				  def_list_head = def_list_head->lhs;
				}
			}
			ASTNode* def = def_list_head->lhs;
			char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
			combine_all(def->rhs,assignment_val);
			if(!strstr(assignment_val,","))
			{
				int val = eval_int(assignment_val);
				push(&const_ints,name);
				push(&const_int_values,itoa(val));
			}
			free(assignment_val);
		}
	
	    }
            else {
                variable_definition->type |= NODE_DCONST;
                set_identifier_type(NODE_DCONST_ID, declaration_list);
            }

         }
       | program function_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program stencil_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program hostdefine          { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       //for now simply discard the struct definition info since not needed
       | program struct_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program enum_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       ;
/*
 * =============================================================================
 * Terminals
 * =============================================================================
 */
struct_name : STRUCT_NAME { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
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
else: ELSE             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
while: WHILE           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
for: FOR               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
in: IN                 { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
communicated: COMMUNICATED { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
dconst_ql: DCONST_QL   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
const_ql: CONST_QL     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
output: OUTPUT         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
global_ql: GLOBAL_MEMORY_QL{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
auxiliary: AUXILIARY   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
int: INT               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
uint: UINT             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
int3: INT3             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
real: REAL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcReal", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
real3: REAL3           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcReal3", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
complex: COMPLEX       { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcComplex", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
bool: BOOL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("bool", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
matrix: MATRIX         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcMatrix", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
field: FIELD           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
work_buffer: WORK_BUFFER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
stencil: STENCIL       { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("", $$); /*astnode_set_buffer(yytext, $$);*/ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
return: RETURN         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$);};
kernel: KERNEL         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(global_func_declaration, $$); $$->token = 255 + yytoken; };
inline: INLINE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("inline", $$); $$->token = 255 + yytoken; };
sum: SUM               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("sum", $$); $$->token = 255 + yytoken; };
max: MAX               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("max", $$); $$->token = 255 + yytoken; };
struct_type: STRUCT_TYPE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
enum_type: ENUM_TYPE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
hostdefine: HOSTDEFINE {
	$$ = astnode_hostdefine(yytext,yytoken);
    };

/*
 * =============================================================================
 * Structure Definitions 
 * =============================================================================
*/
                 
struct_definition:     struct_name '{' declarations '}' {
                        $$ = astnode_create(NODE_STRUCT_DEF,$1,$3);
			remove_substring_parser($1->buffer,"typedef");
			remove_substring_parser($1->buffer,"struct");
			strip_whitespace($1->buffer);
                 }
		 ;
enum_definition: enum_name '{' expression_list '}'{
                        $$ = astnode_create(NODE_ENUM_DEF,$1,$3);
			remove_substring_parser($1->buffer,"typedef");
		        remove_substring_parser($1->buffer,"enum");
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
type_specifier: int     { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | uint    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | int3    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | real    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | real3   { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | complex { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | bool    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | matrix  { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | field   { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | work_buffer { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | stencil     { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | struct_type { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | enum_type { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              ;

type_qualifier: kernel       { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | sum          { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | max          { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | communicated { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | dconst_ql    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | const_ql     { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | global_ql    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | output       { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | auxiliary    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | inline       { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
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
binary_op: '+'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | '-'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | BINARY_OP   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         ;

unary_op: '-'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        | '!'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        | '+'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        ;

assignment_op: ASSIGNOP    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
             ;

/*
 * =============================================================================
 * Expressions
 * =============================================================================
*/
primary_expression: identifier         { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | number             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | string             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | '(' expression ')' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("(", $$); astnode_set_postfix(")", $$); }
                  ;

postfix_expression: primary_expression                         { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | postfix_expression '[' expression ']'      { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                  | postfix_expression '(' ')'                 { $$ = astnode_create(NODE_FUNCTION_CALL, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
                  | postfix_expression '(' expression_list ')' { $$ = astnode_create(NODE_FUNCTION_CALL, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); } 
                  | postfix_expression '.' identifier          { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                  | type_specifier '(' expression_list ')'     { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); $$->lhs->type ^= NODE_TSPEC; /* Unset NODE_TSPEC flag, casts are handled as functions */ }
                  ;

declaration_postfix_expression: identifier                                        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                              | declaration_postfix_expression '[' expression ']' { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                              | declaration_postfix_expression '.' identifier     { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                              ;

unary_expression: postfix_expression          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | unary_op postfix_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                ;

binary_expression: binary_op unary_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                 ;

expression: unary_expression             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
          | expression binary_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }

assign_expression: expression                     { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
               |  '{' expression_list '}' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL);}
               ;

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
variable_definitions: declaration { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
                   |  type_declaration assignment_list  { $$ = astnode_create(NODE_ASSIGN_LIST, $1, $2); $$->type |= NODE_DECLARATION; astnode_set_postfix(";", $$); }
                   ;

assignment_list_leaf: identifier assignment_op assign_expression {$$ = astnode_create(NODE_ASSIGNMENT,$1,$3);}
		    ;
assignment_list: assignment_list ',' assignment_list_leaf {$$ = astnode_create(NODE_UNKNOWN,$1,$3);}
	       | assignment_list_leaf {$$ = astnode_create(NODE_UNKNOWN,$1,NULL);}

declarations: declarations declaration {$$ = astnode_create(NODE_UNKNOWN, $1,$2); }
	    | declaration {$$ = astnode_create(NODE_UNKNOWN, $1,NULL); }
	    ;
declaration: type_declaration declaration_list { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
           ;

declaration_list: declaration_postfix_expression                      { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | declaration_list ',' declaration_postfix_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(";", $$); /* Note ';' infix */ }
                ;

parameter: type_declaration identifier { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
         ;

parameter_list: parameter                    { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | parameter_list ',' parameter { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
              ;

type_declaration: /* Empty */                   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL);}
                | type_qualifiers               { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | type_specifier                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | type_qualifiers type_specifier { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                ;

assignment: declaration assignment_body { $$ = astnode_create(NODE_ASSIGNMENT, $1, $2); }
          ;

assignment_body: assignment_op expression_list 
	       {
                    $$ = astnode_create(NODE_UNKNOWN, $1, $2);

                    // If more than one expression, it's an array declaration
                    if ($$->rhs && $$->rhs->rhs) {
                        astnode_set_prefix("[]", $$);
                        astnode_set_infix("{", $$);
                        astnode_set_postfix("}", $$);
                    }
                }
	       |
	       assignment_op '{' expression_list '}'
	       {
                    $$ = astnode_create(NODE_UNKNOWN, $1, $3);

                    // If more than one expression, it's an array declaration
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
statement: variable_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
         | selection_statement  { $$ = astnode_create(NODE_BEGIN_SCOPE, $1, NULL); }
         | iteration_statement  { $$ = astnode_create(NODE_BEGIN_SCOPE, $1, NULL); }
         | return expression    { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_postfix(";", $$); }
         | function_call        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
         ;

statement_list: statement                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | statement_list statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

compound_statement: '{' '}'                { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
                  | '{' statement_list '}' { $$ = astnode_create(NODE_BEGIN_SCOPE, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
                  ;

selection_statement: if if_statement        { $$ = astnode_create(NODE_SELECTION_STATEMENT, $1, $2); }
                   ;

if_statement: expression compound_statement { $$ = astnode_create(NODE_IF, $1, $2); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
            | expression elif_statement     { $$ = astnode_create(NODE_IF, $1, $2); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
            | expression else_statement     { $$ = astnode_create(NODE_IF, $1, $2); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
            ;

elif_statement: compound_statement elif_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              | elif if_statement                 { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

else_statement: compound_statement else_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              | else compound_statement           { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

iteration_statement: while_statement compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   | for_statement compound_statement   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   ;

while_statement: while expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
               ;

for_statement: for for_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
             ;

for_expression: identifier range_expression {
    $$ = astnode_create(NODE_UNKNOWN, $1, $2);

    if ($$->rhs->rhs->type & NODE_RANGE) {
        astnode_set_infix("=", $$);

        const size_t padding = 32;
        char* tmp = malloc(strlen($1->buffer) + padding);
        sprintf(tmp, ";%s<", $1->buffer);
        astnode_set_buffer(tmp, $$->rhs->rhs);

        sprintf(tmp, ";++%s", $1->buffer);
        astnode_set_postfix(tmp, $$);
        free(tmp);
    }
};

range_expression: in expression { $$ = astnode_create(NODE_UNKNOWN, NULL, $2); astnode_set_infix(":", $$); } // Note: in keyword skipped
                | in range      { $$ = astnode_create(NODE_UNKNOWN, NULL, $2); }
                ;

range: expression ':' expression { $$ = astnode_create(NODE_RANGE, $1, $3); }
     ;

/*
 * =============================================================================
 * Functions
 * =============================================================================
*/
function_definition: declaration function_body {
                        $$ = astnode_create(NODE_FUNCTION, $1, $2);

                        ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, $$->lhs);
                        assert(fn_identifier);
                        set_identifier_type(NODE_FUNCTION_ID, fn_identifier);

                        const ASTNode* is_kernel = get_node_by_token(KERNEL, $$);
			char* struct_params = malloc(4096*sizeof(char));
			struct_params[0] = '\0';
                        ASTNode* compound_statement = $$->rhs->rhs;
                        if (is_kernel) {
                            $$->type |= NODE_KFUNCTION;
                            set_identifier_type(NODE_KFUNCTION_ID, fn_identifier);

                            // Kernel function parameters
                            //if has parameter list
                            if ($$->rhs->lhs)
                            {
                              ASTNode* param_list_head = $$->rhs->lhs;
                              param_list_head->type |= NODE_CODEGEN_INPUT;
                              param_list_head->type |= NODE_INPUT;
                              while(param_list_head->rhs)
                              {
				process_param(compound_statement,param_list_head->rhs,struct_params);
                                param_list_head = param_list_head->lhs;
                              }

			      process_param(compound_statement,param_list_head->lhs,struct_params);
			      //Done since we don't want to codegen the param list in
                            }
			      
                            // Set kernel built-in variables
                            const char* default_param_list=  "(const int3 start, const int3 end, VertexBufferArray vba";
                            astnode_set_prefix(default_param_list, $$->rhs);

				
			    char* kernel_params_struct = malloc(4096*sizeof(char));
			    sprintf(kernel_params_struct,"typedef struct %sInputParams {%s} %sInputParams;\n",fn_identifier->buffer,struct_params,fn_identifier->buffer);

			    FILE* fp_structs= fopen(user_structs_filename,"a");
			    fprintf(fp_structs,"%s\n",kernel_params_struct);
			    fclose(fp_structs);
			    free(struct_params);
			    free(kernel_params_struct);

			    char* tmp = malloc(4096*sizeof(char));
			    sprintf(tmp,"%sInputParams %s;\n", fn_identifier->buffer,fn_identifier->buffer);
			    strcat(user_kernel_params_struct_str,tmp);
			    free(tmp);

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

stencilpoint: stencil_index_list assignment_body_designated { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
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

stencil_definition: declaration stencil_body { $$ = astnode_create(NODE_STENCIL, $1, $2); set_identifier_type(NODE_STENCIL_ID, $$->lhs); }
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
    fprintf(stderr, "in file: %s/%s\n\n",dir_backup,stage4_name_backup);
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

bool
is_number(const char* str)
{
	const size_t n = strlen(str);
	bool res = true;
	for(size_t i = 0; i < n; ++i)
		res &= (isdigit(str[i]) > 0);
	return res;
}
// Function to check if a substring is a standalone word
bool is_whole_word(const char *str, const char *sub, int pos) {
    int sub_len = strlen(sub);

    // Check if the preceding character is not alphanumeric or at the start of the string
    if (pos > 0 && (isalnum(str[pos - 1]) || str[pos - 1] == '_')) {
        return false;
    }

    // Check if the following character is not alphanumeric or at the end of the string
    if (isalnum(str[pos + sub_len]) || str[pos + sub_len] == '_') {
        return false;
    }

    return true;
}

// Function to replace whole word substrings
void change(char *str, const char *old, const char *new_str) {
    int old_len = strlen(old);
    int new_len = strlen(new_str);
    int len_diff = new_len - old_len;

    char *result;
    char *temp = malloc(strlen(str) + 1);
    if (temp == NULL) {
        free(result);
        printf("Memory allocation failed\n");
        return;
    }
    int i, j = 0, found = 0;

    for (i = 0; str[i] != '\0'; i++) {
        // Check for substring match and if it's a whole word
        if (strncmp(&str[i], old, old_len) == 0 && is_whole_word(str, old, i)) {
            found = 1;
            result = temp;
            strcpy(&result[j], new_str);
            j += new_len;
            i += old_len - 1;
        } else {
            result[j++] = str[i];
        }
    }
    result[j] = '\0';

    if (found) {
        strcpy(str, result);
    }

    free(result);
}

static inline int eval_int(const char* str)
{
	if(is_number(str))
		return atoi(str);
	const int index = str_vec_get_index(tinyexpr_cache_str,str);
	if(index > -1)
		return tinyexpr_cache.data[index];
	char* copy = strdup(str);
	strip_whitespace(copy);
        double* vals = malloc(sizeof(double)*const_ints.size);
	bool is_included[const_ints.size];
	size_t final_vars_size = 0;
        for(size_t i = 0; i < const_ints.size; ++i)
        {
                vals[i] = (double)atoi(const_int_values.data[i]);
		is_included[i] = strstr(copy, const_ints.data[i]) != NULL;
                final_vars_size += is_included[i];
		change(copy,const_ints.data[i], const_int_values.data[i]);
        }
        te_variable* final_vars = malloc(sizeof(te_variable)*final_vars_size);
	int j = 0;
        for(size_t i = 0; i < const_ints.size; ++i)
        {
		if(is_included[i])
		{
                      final_vars[j].name = const_ints.data[i];
                      final_vars[j].address = vals +i;
		      final_vars[j].context = NULL;
		      ++j;
		}
        }
        int err;
        te_expr* expr = te_compile(copy, final_vars, (int)final_vars_size, &err);
        if(!expr)
        {
                fprintf(stderr,"Parse error at tinyexpr\n");
		fprintf(stderr,"Was not able to parse: %s\n",str);
		fprintf(stderr,"place %d\n",err);
		fprintf(stderr,"symbol %c\n",str[err]);
		fprintf(stderr,"strlen: %d\n",strlen(copy));
                exit(EXIT_FAILURE);
        }
        int res = (int) te_eval(expr);
        te_free(expr);
	free(copy);
	free(vals);
	free(final_vars);
	push_int(&tinyexpr_cache,res);
	push(&tinyexpr_cache_str,str);
        return res;
}
