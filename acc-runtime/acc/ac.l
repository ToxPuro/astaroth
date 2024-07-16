%option yylineno
%option noyywrap
%option noinput
%option nounput

D [0-9]
L [a-zA-Z_]
S [\s]

%{
#include "tab.h"
#include <ctype.h>
#include <stdbool.h>
#include "vecs.h"
%}
%{
	string_vec struct_names          = VEC_INITIALIZER;
	string_vec struct_datatype_names = VEC_INITIALIZER;
	string_vec enum_names            = VEC_INITIALIZER;



void strip_whitespace(char *str) {
    char *dest = str; // Destination pointer to overwrite the original string
    char *src = str;  // Source pointer to traverse the original string

    // Skip leading whitespace
    while (isspace((unsigned char)(*src))) {
        src++;
    }

    // Copy non-whitespace characters to the destination
    while (*src) {
        if (!isspace((unsigned char)(*src))) {
            *dest++ = *src;
        }
        src++;
    }

    // Null-terminate the destination string
    *dest = '\0';
}
%}

%%

"int"                   { return INT; } /* Type specifiers */
"uint"                  { return UINT; }
"int3"                  { return INT3; }
"real"                  { return REAL; }
"bool"                  { return BOOL; }
"Matrix"                { return MATRIX; }
"Field"                 { return FIELD; }
"Field3"                { return FIELD3; }
"VtxBuffer"                 { return VTXBUFFER; }
"WorkBuffer"            {return WORK_BUFFER; }
"ComputeSteps"          {return COMPUTESTEPS; }
"BoundConds"          {return BOUNDCONDS; }
"intrinsic"          {return INTRINSIC; }

"Stencil"               { return STENCIL; }

"Kernel"                { return KERNEL; } /* Type qualifiers */
"inline"                { return INLINE; } /* Type qualifiers */
"boundary_condition"    { return BOUNDARY_CONDITION; } /* Type qualifiers */
"Sum"                   { return SUM; }
"Max"                   { return MAX; }
"communicated"          { return COMMUNICATED; }
"dconst"          { return DCONST_QL; }
"const"          { return CONST_QL; }
"constexpr"      { return CONSTEXPR; }
"output"          { return OUTPUT; }
"run_const"          { return RUN_CONST; }
"gmem"          { return GLOBAL_MEMORY_QL; }
"auxiliary"             { return AUXILIARY; }
"extern"             { return EXTERN; }

"if"        { return IF; } /* Keywords */
"if constexpr"        { return IF; } /* Keywords */
"else if"   { return ELIF; }
"else if constexpr"   { return ELIF; }
"else"      { return ELSE; }
"while"     { return WHILE; }
"for"       { return FOR; }
"in"        { return IN; }
"break"     { return BREAK; }
"continue"  { return CONTINUE; }
"return"    { return RETURN; }
[?]         { return QUESTION; }

struct[[:space:]]{L}({L}|{D})* {
					char* name = strdup(yytext);
					remove_substring(name,"struct");
					remove_substring(name,"typedef");
					strip_whitespace(name);
					push(&struct_names, name);
					push(&struct_datatype_names,name);
					return STRUCT_NAME;
				}
typedef[[:space:]]{L}({L}|{D})*[[:space:]]struct[[:space:]]{L}({L}|{D})* {
					strtok(yytext," ");
					char* datatype_name = strtok(NULL," ");
					strtok(NULL, " ");
					const char* dsl_name = strtok(NULL, " ");
					push(&struct_names,dsl_name);
					push(&struct_datatype_names,datatype_name);
					yytext = datatype_name;
					return STRUCT_NAME;
				}
enum[[:space:]]{L}({L}|{D})* {
				char* name = strdup(yytext);
				remove_substring(name,"enum");
				remove_substring(name,"typedef");
				strip_whitespace(name);
				push(&enum_names, name);
				return ENUM_NAME;
			}
typedef[[:space:]]enum[[:space:]]{L}({L}|{D})* {
				char* name = strdup(yytext);
				remove_substring(name,"enum");
				remove_substring(name,"typedef");
				strip_whitespace(name);
				push(&enum_names, name);
				return ENUM_NAME;
			}

"hostdefine".*        { return HOSTDEFINE; }

{L}({L}|{D})*             {  /* Literals */
				const int struct_index = str_vec_get_index(struct_names,yytext);
				if(struct_index >= 0)
				{
					yytext =  strdup(struct_datatype_names.data[struct_index]);
					return STRUCT_TYPE;
				}
				if(str_vec_contains(enum_names,yytext))
					return ENUM_TYPE;
				return IDENTIFIER; 
			  }   
\"(\\.|[^"\\])*\"         { return STRING; }       /* String literal */
[+-/*|&%^]?=              { return ASSIGNOP; }     /* Assignment */
(<<=)|(>>=)               { return ASSIGNOP; }     /* Assignment */
[/*<>|&%^]              { return BINARY_OP; }    /* Operators */
(&&)|(\|\|)|(<<)|(>>)     { return BINARY_OP; }    /* Operators */
(==)|(>=)|(<=)|(!=)|(^=)  { return BINARY_OP; }    /* Operators */
[-+*/:;=\[\]{}(),\.<>|&!] { return yytext[0]; }    /* Characters */
{D}+[.]{D}*               { return REALNUMBER; }
{D}+[.]?{D}*(e[-+]?{D}+)   { return REALNUMBER; }
{D}+[.]?{D}*(e[-+]?{D}+)?[d]  { return DOUBLENUMBER; } /* Numbers */
{D}+[.]?{D}*(e[-+]?{D}+)?[f]?  { return NUMBER; }      /* Numbers */
0[xX][0-9a-fA-F]+[ulUL]*      { return NUMBER; }      /* Hex */



"#".*               { /* Skip regular comments */ }
[ \t\n\v\r]+        { /* Ignore whitespace, tabs and newlines */ }
.                   { fprintf(stderr, "unrecognized char %d: [%c]\n", *yytext, *yytext); }

%%

/*
[.]{D}+                   { return REALNUMBER; }
[.]?{D}+(e[-]?{D}+)       { return REALNUMBER; }
{D}?[.]+{D}*(e[-]?{D}+)?[f]?  { return NUMBER; }
{D}?[.]+{D}*(e[-]?{D}+)?[d]  { return DOUBLENUMBER; }
struct\s{L}({L}|{D})* {printf("Found it\n"); exit(0); }
*/
