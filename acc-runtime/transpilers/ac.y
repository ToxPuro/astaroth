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
static ASTNode* create_type_declaration(const char* tqual, const char* tspec);



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
process_includes(const size_t depth, const char* dir, const char* file, FILE* out)
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
		        process_includes(depth+1,dir,file_path,out);
			free(file_path);
		}

	return;
  }
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
    exit(EXIT_FAILURE);
  }

  const size_t  len = 4096;
  char* buf = malloc(len*sizeof(char));
  while (fgets(buf, len, in)) {
    char* line = buf;
    while (strlen(line) > 0 && line[0] == ' ') // Remove whitespace
      ++line;

    if (!strncmp(line, "include", strlen("include"))) {

      char incl[len];
      sscanf(line, "include '%[^']'\n", incl);

      char path[len];
      sprintf(path, "%s/%s", dir, incl);

      fprintf(out, "!! Include file %s start\n", path);
      process_includes(depth+1, dir, path, out);
      fprintf(out, "!! Included file %s end\n", path);

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

// Function to trim trailing whitespace and & character
void trim_line(char *line) {
    int len = strlen(line);
    int n_single = 0;
    int n_double = 0;
    for(int i = 0; i < len; ++i)
    {
	n_single += line[i] == '\'';
	n_double += line[i] == '"';
	if(line[i] == '!' && n_single %2 == 0 && n_double %2 == 0) 
	{
		line[i] = '\0';
		len = i;
		break;
	}
    }
    while (len > 0 && (line[len - 1] == ' ' || line[len - 1] == '\t' || line[len - 1] == '\n') ) {
        line[--len] = '\0';
    }
}


const bool
is_where_line(const char* line)
{
        int i = 0; 
	while(line[i] == ' ') ++i;
	return
		line[i + 0] == 'w' &&
		line[i + 1] == 'h' &&
		line[i + 2] == 'e' &&
		line[i + 3] == 'r' &&
		line[i + 4] == 'e'; 
}
const bool
is_for_all_line(const char* line)
{
        int i = 0; 
	while(line[i] == ' ') ++i;
	return
		line[i + 0] == 'f' &&
		line[i + 1] == 'o' &&
		line[i + 2] == 'r' &&
		line[i + 3] == 'a' &&
		line[i + 4] == 'l' &&
		line[i + 4] == 'l'; 
}

static int
has_assignment(const char* combined_line)
{
		int i = 0;
	        while(combined_line[i] != '(') ++i;
		int lhs = 1;
	        int rhs = 0;
		++i;
		while(lhs != rhs)
	        {
			lhs += combined_line[i] == '(';
			rhs += combined_line[i] == ')';
			++i;
		}
		bool has_assignment = false;
		const int j = i;
		while(i < strlen(combined_line))
		{
			has_assignment |= (combined_line[i] == '=');
			++i;
		}
		return has_assignment ? j : 0;
}
const bool
is_if_line(const char* line)
{
        int i = 0; 
	while(line[i] == ' ') ++i;
	return
		line[i + 0] == 'i' &&
		line[i + 1] == 'f';
}
#define MAX_LINE_LENGTH (4000)
// Function to read and process Fortran code
void process_fortran_code(FILE *input, FILE *output) {
    char line[MAX_LINE_LENGTH];
    char combined_line[MAX_LINE_LENGTH*2];

    combined_line[0] = '\0'; // Initialize combined line

    while (fgets(line, sizeof(line), input)) {
        trim_line(line);

        if (line[0] == '\0') continue;
        if (line[strlen(line) - 1] == '&')  {
	    line[strlen(line)-1] = '\0';	
            // Continuation line
            strcat(combined_line, line); // Append current line without newline
            trim_line(combined_line); // Remove trailing &
        } else {
            // Regular line
            strcat(combined_line, line); // Append current line
	    char* tmp =  get_replaced_substring(combined_line,".or."," .or. ");
	    char* res =  get_replaced_substring(tmp,".and."," .and. ");
	    remove_substring(res,"(KIND=ikind8)");
	    free(tmp);
	    if(is_if_line(res))
	    {
		const int i = has_assignment(res);
		if(i)
		{
			char new_tmp[4000];
			strcpy(&new_tmp,&res[i]);
			printf("HMM: %s|%s\n",res,new_tmp);
			res[i] = '\0';
            		fprintf(output, "%s then\n", res); // Write combined line to output
            		fprintf(output, "%s\n", new_tmp); // Write combined line to output
	    		if(is_where_line(new_tmp) && has_assignment(new_tmp))
            		fprintf(output, "endwhere\n"); // Write combined line to output
            		fprintf(output, "endif\n"); // Write combined line to output
		}
	    	else if(!strstr(combined_line,"namelist") && !strstr(combined_line,"print*") && !strstr(combined_line,"format ("))
            		fprintf(output, "%s\n", res); // Write combined line to output
	    }
	    else if(!strstr(combined_line,"namelist") && !strstr(combined_line,"print*") && !strstr(combined_line,"format ("))
            	fprintf(output, "%s\n", res); // Write combined line to output
	    free(res);
	    if(is_where_line(combined_line) && has_assignment(combined_line))
			fprintf(output,"endwhere\n");
	    if(is_for_all_line(combined_line) && has_assignment(combined_line))
			fprintf(output,"end forall\n");
            combined_line[0] = '\0'; // Reset combined line for next lines
        }
    }
    // Handle any leftover combined line
    if (combined_line[0] != '\0') {
        fprintf(output, "%s\n", combined_line);
    }
}
void
remove_lines(FILE* input, FILE * output)
{
}

void
transpile(const char* filename)
{
	FILE* stage_1 = fopen("stage1","w");
        process_includes(0, ".", filename, stage_1);
	fclose(stage_1);
	stage_1 = fopen("stage1","r");
	FILE* stage_2 = fopen("stage2","w");
	process_fortran_code(stage_1, stage_2);
	fclose(stage_2);
        yyin = fopen("stage2", "r");
        yyin_backup = fopen("stage2", "r");
        int error = yyparse();
        if (error)
	    exit(EXIT_FAILURE);
}


int
main(int argc, char** argv)
{
    if(argc != 2)
    {
	printf("Usage: ./fortran-transpiler filename\n");
	return EXIT_FAILURE;
    }
    transpile(argv[1]);
    printf("DONE\n");
    return EXIT_SUCCESS;
}
%}

%token DIMENSION COMMA ARR_START ARR_END ALLOCATABLE PARAMETER TARGET VOLATILE POINTER
%token KIND REAL_KIND LEN DO FORALL END_FORALL ENDDO SELECT_CASE END_SELECT CASE DEFAULT_CASE WHERE ENDWHERE ELSEWHERE
%token MODULE USE NAMELIST INTERFACE ENDINTERFACE MODULE_PROCEDURE CONTAINS 
%token IMPLICIT NONE PUBLIC ENDMODULE ONLY PRIVATE
%token SUBROUTINE FUNCTION END_FUNCTION ENDSUBROUTINE INTRINSIC_FUNC EXIT PRINT WRITE OPEN READ CLOSE
%token INTENT OUT INOUT CYCLE OPTIONAL OR TYPE ACCESS_TOKEN


%token IDENTIFIER STRING NUMBER REALNUMBER DOUBLENUMBER
%token IF ENDIF THEN ELIF ELSE WHILE FOR RETURN IN BREAK CONTINUE CALL
%token BINARY_OP ASSIGNOP QUESTION UNARY_OP
%token INT COMPLEX UINT REAL MATRIX FIELD FIELD3 STENCIL WORK_BUFFER BOOL INTRINSIC CHAR 
%token KERNEL INLINE ELEMENTAL BOUNDARY_CONDITION SUM MAX COMMUNICATED AUXILIARY DCONST_QL CONST_QL SHARED DYNAMIC_QL CONSTEXPR RUN_CONST GLOBAL_MEMORY_QL OUTPUT INPUT VTXBUFFER COMPUTESTEPS BOUNDCONDS EXTERN
%token HOSTDEFINE
%token STRUCT_NAME STRUCT_TYPE ENUM_NAME ENUM_TYPE

%left '-' 
%left '+' 
%left '&'
%left BINARY_OP
%%


root: program { root = astnode_create(NODE_UNKNOWN, $1, NULL); }
    ;

end_subroutine:
	      ENDSUBROUTINE identifier {}
	     | ENDSUBROUTINE {}
	     | END_FUNCTION identifier {}
	     | END_FUNCTION {}
program: /* Empty*/                   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); }
       | program variable_definitions { $$ = astnode_create(NODE_UNKNOWN,$1,$2); }
       | program module_declaration   {$$ = astnode_create(NODE_BEGIN_SCOPE,$1,$2); }
       | program use_statement        {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
       | program module_statements    {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
       | program end_module           {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
       | program SUBROUTINE function_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $3); }
       | program FUNCTION   function_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $3); }
       //| program stencil_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       //| program hostdefine          { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       ////for now simply discard the struct definition info since not needed
       //| program struct_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       //| program steps_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       //| program boundconds_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       //| program enum_definition   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       ;
/*
 * =============================================================================
 * Terminals
 * =============================================================================
 */

procedure:
	MODULE_PROCEDURE identifier  {$$ = astnode_create(NODE_UNKNOWN,$2,NULL);} 
procedures:
	    procedure {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	  | procedures procedure {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
namelist : NAMELIST {}
module_statements: IMPLICIT NONE  {}
		 | PUBLIC         {}
		 | PRIVATE        {}
		 | PUBLIC ':' ':' expression_list   {$$ = astnode_create(NODE_UNKNOWN, $4, NULL); }
		 | INTERFACE identifier  procedures ENDINTERFACE identifier  {$$ = astnode_create(NODE_UNKNOWN,$2,$3); }
		 | CONTAINS  {} 
		 ;
struct_name : STRUCT_NAME { $$ = astnode_create(NODE_TSPEC, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
enum_name: ENUM_NAME { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
identifier: IDENTIFIER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
number: NUMBER         {  $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
      | REALNUMBER     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_prefix("AcReal(", $$); astnode_set_postfix(")", $$); }
      | DOUBLENUMBER   {
            $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken;
            astnode_set_prefix("double(", $$); astnode_set_postfix(")", $$);
            $$->buffer[strlen($$->buffer) - 1] = '\0'; // Drop the 'd' postfix
        }
      ;
string: STRING         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken;}
if: IF                 { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken;};
elif: ELIF             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
else: ELSE             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
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
allocatable: ALLOCATABLE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
optional: OPTIONAL { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
volatile: VOLATILE       { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
dimension: DIMENSION{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
return: RETURN{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
case : CASE{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
default_case : DEFAULT_CASE{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
select_case: SELECT_CASE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
do: DO{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
forall: FORALL{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
intent:    INTENT{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
in: IN { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
inout: INOUT{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
out: OUT{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
pointer: POINTER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
module: MODULE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
use:    USE { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
parameter:   PARAMETER   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
target:      TARGET{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
input:  INPUT          { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
global_ql: GLOBAL_MEMORY_QL{ $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
auxiliary: AUXILIARY   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
extern: EXTERN { $$ = astnode_create(NODE_NO_OUT, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
int: INT               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
complex: COMPLEX { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
uint: UINT             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
real: REAL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcReal", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
bool: BOOL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("bool", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
char: CHAR             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("char", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
field: FIELD           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
field3: FIELD3         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
work_buffer: WORK_BUFFER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
stencil: STENCIL       { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("Stencil", $$); /*astnode_set_buffer(yytext, $$);*/ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
return: RETURN         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("return", $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$);};
kernel: KERNEL         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("Kernel", $$); $$->token = 255 + yytoken; };
vtxbuffer: VTXBUFFER   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("Field", $$); $$->token = 255 + yytoken; };
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

char_len:
	'(' LEN assignment_op expression ')' {$$ = astnode_create(NODE_UNKNOWN,$4,NULL); }

scalar_type_specifier: 
		int            { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | uint         { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | real         { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | bool         { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | char '(' LEN assignment_op expression ')'       { $$ = astnode_create(NODE_TSPEC, $1, $2); }
              | char { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | complex { $$ = astnode_create(NODE_TSPEC, $1, NULL); }

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
              | intrinsic    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
	      | kernel       { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
	      | TYPE '(' identifier ')' {$$ = astnode_create(NODE_TSPEC,$3,NULL); }
              ;

type_qualifier: dimension '(' expression_list ')' {$$ = astnode_create(NODE_TQUAL, $1, $3); }
	      | allocatable              {$$ = astnode_create(NODE_TQUAL,$1,NULL); }
	      | optional                 {$$ = astnode_create(NODE_TQUAL,$1,NULL); }
	      | intent '(' inout')'     {$$ = astnode_create(NODE_TQUAL,$1,$3); }
	      | intent '(' in ')'     {$$ = astnode_create(NODE_TQUAL,$1,$3); }
	      | intent '(' out')'     {$$ = astnode_create(NODE_TQUAL,$1,$3); }
	      | volatile                 {$$ = astnode_create(NODE_TQUAL,$1,NULL); }
	      | pointer                  {$$ = astnode_create(NODE_TQUAL,$1,NULL); }
	      | parameter                {$$ = astnode_create(NODE_TQUAL,$1,NULL); }
	      | target                   {$$ = astnode_create(NODE_TQUAL,$1,NULL); }

type_qualifiers: type_qualifiers COMMA type_qualifier {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
	       | type_qualifier {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	       ;

module_declaration:
		  module identifier  {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
end_module:
	    ENDMODULE {$$ = astnode_create(NODE_UNKNOWN,NULL,NULL); }
	  | ENDMODULE identifier {$$ = astnode_create(NODE_UNKNOWN,NULL,NULL); }

use_info:
	  identifier {$$ = astnode_create(NODE_USE_STATEMENT, $1, NULL); }
	  | identifier COMMA ONLY ':' expression_list  {$$ = astnode_create(NODE_USE_STATEMENT,$1,$5); }
use_statement:
	     use use_info {$$ = astnode_create(NODE_USE_STATEMENT, $1, $2); }




/*
 * =============================================================================
 * Operators
 * =============================================================================
*/

//Plus and minus have to be in the parser since based on context they are unary or binary ops
binary_op: '+'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("+", $$); $$->token = BINARY_OP; }
         | '-'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("-", $$); $$->token = BINARY_OP; }
         | '&'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("&", $$); $$->token = BINARY_OP; }
         | BINARY_OP   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = BINARY_OP;}
         ;

unary_op: '-'{  $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        | '+' { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        | '&'          { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        | UNARY_OP     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
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
                  | ':'                { $$ = astnode_create(NODE_PRIMARY_EXPRESSION, $1, NULL); }
                  ;

struct_field:
	     identifier {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
           | identifier '(' param_list ')' { $$ = astnode_create(NODE_UNKNOWN,$1,$2); }

postfix_expression: primary_expression                         { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | postfix_expression '(' param_list ')'      { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
                  | postfix_expression '(' ')'                 { $$ = astnode_create(NODE_FUNCTION_CALL, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
                  | postfix_expression ACCESS_TOKEN struct_field { $$ = astnode_create(NODE_STRUCT_EXPRESSION, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                  | '(' expression ')' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("(", $$); astnode_set_postfix(")", $$); }
		  | ARR_START expression_list ARR_END {$$ = astnode_create(NODE_ARRAY_INITIALIZER, $3, NULL); }
		  | type_specifier '(' expression ')'{$$ = astnode_create(NODE_FUNCTION_CALL,$1,$3); }

declaration_postfix_expression: identifier                                        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                              | declaration_postfix_expression '[' expression ']' { $$ = astnode_create(NODE_ARRAY_ACCESS, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                              | declaration_postfix_expression '.' identifier     { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                              ;

unary_expression: postfix_expression          { $$ = astnode_create(NODE_EXPRESSION, $1, NULL); }
                | unary_op postfix_expression { $$ = astnode_create(NODE_EXPRESSION, $1, $2); }
                ;

binary_expression: binary_op unary_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                 ;


//choose: QUESTION expression ':' expression {$$ = astnode_create(NODE_UNKNOWN,$2,$4);  astnode_set_prefix("? ",$$->lhs);  astnode_set_prefix(": ",$$->rhs);}
//      ;
expression: unary_expression             { $$ = astnode_create(NODE_EXPRESSION, $1, NULL); }
	  //| expression choose            { $$ = astnode_create(NODE_TERNARY_EXPRESSION,$1,$2); } 
          | expression binary_expression { $$ = astnode_create(NODE_BINARY_EXPRESSION, $1, $2); }

param: 
      assignment  {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
     | expression {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
     | expression ':' expression {$$ = astnode_create(NODE_UNKNOWN,$1,$3); }
     ;

param_list: param { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
            | param_list COMMA param { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }

range: 
      expression {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
     | expression ':' expression {$$ = astnode_create(NODE_UNKNOWN,$1,$3); }
     ;


expression_list: expression                     { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
               | expression_list COMMA expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
               ;

/*
 * =============================================================================
 * Definitions and Declarations
 * =============================================================================
*/
//variable_definition: declaration { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
//                   | assignment  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
//                   ;
var_declaration:
		  identifier {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
		| identifier assignment_body  {$$ = astnode_create(NODE_UNKNOWN, $1, $2); }
		| identifier '(' expression_list ')' {$$ = astnode_create(NODE_UNKNOWN, $1, $2); }

var_declarations: 
		  var_declaration {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
		| var_declarations COMMA var_declaration {$$ = astnode_create(NODE_UNKNOWN, $1, $3);}
		;
		
variable_definitions: 
		    type_declaration ':' ':' var_declarations  {$$ = astnode_create(NODE_DECLARATION, $1,$2); }




type_declaration:
                  type_specifier                 { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
		| type_specifier COMMA type_qualifiers { $$ = astnode_create(NODE_UNKNOWN,$1,$2); } 
                ;


array_access: identifier '(' param_list')' {$$ = astnode_create(NODE_ARRAY_ACCESS,$1,$3); }
array_assignment:  array_access assignment_body {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
struct_assignment: identifier ACCESS_TOKEN identifier assignment_body {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
	         | identifier ACCESS_TOKEN identifier '(' param_list ')' assignment_body {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
assignment: identifier assignment_body  { 
	  		$$ = astnode_create(NODE_ASSIGNMENT, $1, $2); 
		}
          ;


assignment_body:  assignment_op expression
	       {
                    $$ = astnode_create(NODE_UNKNOWN, $1, $2);
               }
               ;

do_iterator:
	   assignment COMMA expression {$$ = astnode_create(NODE_UNKNOWN,$1,$3); }
forall_iterator:
	    '(' identifier assignment_op param ')' {$$ = astnode_create(NODE_UNKNOWN,$2,$4); }

loop_start:
	    do do_iterator {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
          | forall forall_iterator {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }

select_case:
	   select_case '(' expression ')' {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }

case:
      default_case statement_list {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
    | case '(' expression_list ')' statement_list {$$ = astnode_create(NODE_UNKNOWN,$2,$3); }
    | case '(' expression_list ')' {}
cases:
       cases case {$$ = astnode_create(NODE_UNKNOWN,$1,$2); }
     | case {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }

print_statement:
           PRINT COMMA expression_list {}
         | PRINT number COMMA expression_list {}
write_statement:
          WRITE '(' expression COMMA binary_op ')' expression {}
	| WRITE '(' identifier COMMA assignment')' {}
open_statement:
         OPEN '('param_list')' {}
close_statement:
         CLOSE '('param_list')' {}
read_statement:
          READ '('param COMMA binary_op COMMA param ')' expression {}
	| READ '('number COMMA binary_op ')' expression {}
	| READ '('number COMMA binary_op COMMA param')' expression_list {}
	| READ '('param COMMA param COMMA param')'{}
exit_statement:
	      EXIT {}
intent_statement:
		  INTENT '(' IN ')' ':' ':' expression_list {}
		| INTENT '(' INOUT ')' ':' ':' expression_list {}


discarded_statement: print_statement
		    | write_statement
  		    | open_statement
		    | read_statement
		    | close_statement 
		    | exit_statement

cycle_statement: CYCLE {}

safe_statement:  

	    assignment              {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  array_assignment        {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  struct_assignment       {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
         ;


statement:
	    safe_statement {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  loop_start statement_list ENDDO {$$ = astnode_create(NODE_BEGIN_SCOPE, $1,$2); }
	 |  loop_start statement_list END_FORALL{$$ = astnode_create(NODE_BEGIN_SCOPE, $1,$2); }
	 |  loop_start ENDDO {}
	 |  select_case cases END_SELECT {$$ = astnode_create(NODE_BEGIN_SCOPE, $1,$2); }
	 |  cycle_statement       {}
	 |  intent_statement      {}
	 |  function_call          {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
         |  use_statement          {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  where_statement {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  discarded_statement    {}
	 |  variable_definitions   {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  return                 {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }
	 |  selection_statement     {$$ = astnode_create(NODE_UNKNOWN,$1,NULL); }






statement_list: statement                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | statement_list statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;
safe_statement_list: safe_statement { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | safe_statement_list safe_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;
assignment_list: assignment{ $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | assignment_list assignment{ $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;



selection_statement: if if_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   ;
elses:

rest_if:
        statement_list else_part ENDIF {}
      | else_part ENDIF {} 
      | statement_list elses ENDIF {}
if_statement:
	      '(' expression')'    safe_statement{ $$ = astnode_create(NODE_UNKNOWN, $2, $4); }
	      | '(' expression')'  cycle_statement  {}
	      | '(' expression')'  discarded_statement{}
	      | '(' expression')'  function_call  { $$ = astnode_create(NODE_UNKNOWN, $2, $4); }
              | '(' expression ')' THEN  statement_list else_part ENDIF {}
              | '(' expression ')' THEN  else_part ENDIF {}
              | '(' expression ')' THEN  statement_list else_ifs else_part ENDIF {}
              ;


where_start:
	   WHERE '(' expression ')' {}
where_body:
	  safe_statement_list elsewhere 
where_statement:
	        where_start where_body ENDWHERE{}
elsewhere:
	  | ELSEWHERE  safe_statement_list{}

else_if:
        elif '('expression ')' THEN statement_list {}

else_ifs:
	  else_ifs else_if {}
	| else_if {}
else_part:
    ELSE  statement_list {}
    | /* empty */ {}
    |ELSE  /* empty */ {}
    ;

//elif_statement: elif if_statement                 { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//              ;

//else_statement: else stat{ $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//	      | else statement          { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//              ;

//iteration_statement: while_statement compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//                   | for_statement compound_statement   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//                   | for_statement variable_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//                   | for_statement function_call        { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_postfix(";",$$); }
//                   ;
//
//while_statement: while expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
//               ;
//
//for_statement: for for_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
//             ;
//
//for_expression: identifier in expression {
//    			$$ = astnode_create(NODE_UNKNOWN, $1, $3);
//	      }
//	      | identifier in range {
//    			$$ = astnode_create(NODE_UNKNOWN, $1, $3);
//    			astnode_set_infix("=", $$);
//
//    			const size_t padding = 32;
//    			char* tmp = malloc(strlen($1->buffer) + padding);
//    			sprintf(tmp, ";%s<", $1->buffer);
//    			astnode_set_buffer(tmp, $$->rhs);
//
//    			sprintf(tmp, ";++%s", $1->buffer);
//    			astnode_set_postfix(tmp, $$);
//    			free(tmp);
//	      }
//	      ;
//
//range: expression ':' expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); }
//     ;

/*
 * =============================================================================
 * Functions
 * =============================================================================
*/
function_definition: identifier  function_body end_subroutine { $$ = astnode_create(NODE_UNKNOWN,$1,$3); } 
		   | identifier  end_subroutine {}
//function_definition: declaration function_body {
//                        $$ = astnode_create(NODE_UNKNOWN, $1, $2);
//			if(get_node(NODE_TSPEC,$$->lhs) && strcmp(get_node(NODE_TSPEC,$$->lhs)->lhs->buffer,"Kernel"))
//			{
//				fprintf(stderr,"%s","Fatal error: can't specify the type of functions, since they will be deduced\n");
//				fprintf(stderr,"%s","Consider instead leaving a code comment telling the return type\n");
//				fprintf(stderr,"Offending function: %s\n",get_node_by_token(IDENTIFIER,$$->lhs)->buffer);
//				assert(false);
//				exit(EXIT_FAILURE);
//			}
//
//			if(has_qualifier($$,"extern"))
//			{
//				$$->type |= NODE_NO_OUT;
//			}
//
//                        ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, $$->lhs);
//                        assert(fn_identifier);
//                        set_identifier_type(NODE_FUNCTION_ID, fn_identifier);
//
//                        const ASTNode* is_kernel = get_node_by_token(KERNEL, $$);
//			if(is_kernel)
//				astnode_set_prefix("__global__ void \n#if MAX_THREADS_PER_BLOCK\n__launch_bounds__(MAX_THREADS_PER_BLOCK)\n#endif\n",$$);
//                        ASTNode* compound_statement = $$->rhs->rhs;
//                        if (is_kernel) {
//                            $$->type |= NODE_KFUNCTION;
//                            // Set kernel built-in variables
//                            const char* default_param_list=  "(const int3 start, const int3 end, VertexBufferArray vba";
//                            astnode_set_prefix(default_param_list, $$->rhs);
//
//                            assert(compound_statement);
//                            astnode_set_prefix("{", compound_statement);
//                            astnode_set_postfix(
//                              //"\n#pragma unroll\n"
//                              //"for (int field = 0; field < NUM_FIELDS; ++field)"
//                              //"if (!isnan(out_buffer[field]))"
//                              //"vba.out[field][idx] = out_buffer[field];"
//                              "}", compound_statement);
//                        } else {
//                            astnode_set_infix(" __attribute__((unused)) =[&]", $$);
//                            astnode_set_postfix(";", $$);
//                            $$->type |= NODE_DFUNCTION;
//                            set_identifier_type(NODE_DFUNCTION_ID, fn_identifier);
//
//                            // Pass device function parameters by const reference
//                            if ($$->rhs->lhs) {
//                                set_identifier_prefix("const ", $$->rhs->lhs);
//                                set_identifier_infix("&", $$->rhs->lhs);
//                            }
//
//                        }
//                    }
//                   ;

function_body: 
	       '(' ')' statement_list { $$ = astnode_create(NODE_BEGIN_SCOPE, NULL, $3); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
	     |         statement_list { $$ = astnode_create(NODE_BEGIN_SCOPE, NULL, $1); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
             | '(' expression_list ')' statement_list { $$ = astnode_create(NODE_BEGIN_SCOPE, $2, $4); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
             ;

function_call: CALL identifier '(' param_list')' {$$ = astnode_create(NODE_FUNCTION_CALL, $2,$4); }
	     | CALL identifier                   {$$ = astnode_create(NODE_FUNCTION_CALL, $2,NULL); }
	     | INTRINSIC_FUNC '(' param_list')'  {$$ = astnode_create(NODE_FUNCTION_CALL, $1,$3); }
//function_call: declaration '(' ')'                 { $$ = astnode_create(NODE_FUNCTION_CALL, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
//             | declaration '(' expression_list ')' { $$ = astnode_create(NODE_FUNCTION_CALL, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$);   }
//             ;
//
///*
// * =============================================================================
// * Stencils
// * =============================================================================
//*/
//assignment_body_designated: assignment_op expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("\"", $$); astnode_set_postfix("\"", $$); }
//          ;

//stencilpoint: stencil_index_list assignment_body_designated { 
//	    								$$ = astnode_create(NODE_UNKNOWN, $1, $2); 
//									const int num_of_indexes = count_num_of_nodes_in_list($1);
//									if(TWO_D && num_of_indexes != 2)
//									{
//										fprintf(stderr,"Can only use 2D stencils when building for 2D simulation\n");
//										assert(num_of_indexes == 2);
//										exit(EXIT_FAILURE);
//									}
//									else if(!TWO_D && num_of_indexes < 3)
//										fprintf(stderr,"Specify indexes for all three dimensions\n");
//									else if(!TWO_D && num_of_indexes > 3)
//										fprintf(stderr,"Only three dimensional stencils for 3D simulation\n");
//									if(!TWO_D && num_of_indexes != 3)	
//									{
//										assert(num_of_indexes == 3);
//										exit(EXIT_FAILURE);
//									}
//							    }
//            ;
//
//stencil_index: '[' expression ']' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("[STENCIL_ORDER/2 +", $$); astnode_set_postfix("]", $$); }
//     ;
//
//stencil_index_list: stencil_index            { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
//          | stencil_index_list stencil_index { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
//          ;
//
//stencilpoint_list: stencilpoint                       { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
//                 | stencilpoint_list ',' stencilpoint { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
//                 ;
//
//stencil_body: '{' stencilpoint_list '}' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("},", $$); }
//            ;
//
//stencil_definition: declaration stencil_body { $$ = astnode_create(NODE_STENCIL, $1, $2); set_identifier_type(NODE_VARIABLE_ID, $$->lhs); 
//		  }
//                  ;
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
    //fprintf(stderr, "in file: %s/%s\n\n",dir_backup,stage4_name_backup);
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
create_type_qualifiers(const char* tqual)
{
	if(!tqual) return NULL;
	ASTNode* tqual_identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
	tqual_identifier -> buffer = strdup(tqual);
	tqual_identifier -> token = IDENTIFIER;
	ASTNode* type_qualifier  = astnode_create(NODE_TQUAL,tqual_identifier,NULL);
	ASTNode* type_qualifiers = astnode_create(NODE_UNKNOWN,type_qualifier,NULL);
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
create_type_declaration(const char* tqual, const char* tspec)
{

	ASTNode* type_declaration = astnode_create(NODE_UNKNOWN,create_tspec(tspec),create_type_qualifiers(tqual));
	return type_declaration;
}
