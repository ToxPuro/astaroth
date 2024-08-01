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

"print*"                    { return PRINT; }    /* operators */
"print"                     { return PRINT; }    /* operators */
"write"                     { return WRITE; }    /* operators */
"open"                     { return OPEN; }    /* operators */
"close"                     { return CLOSE; }    /* operators */
"read"                     { return READ; }    /* operators */
"case default"              { printf("HI default\n"); return DEFAULT_CASE; }    /* Operators */
";"                 { /* Skip semicolons */ }
"integer"                   { return INT; } /* Type specifiers */
"complex"                   { return COMPLEX; } /* Type specifiers */
"logical"                   { return BOOL; }
"character"                 { return CHAR; }
"!"[^\n]*                   { /* Skip the comment */ }
"dimension"                 { return DIMENSION; }
".false."                   { return IDENTIFIER; }
".true."                    { return IDENTIFIER; }
","                         { return COMMA; }
"(/"                        { return ARR_START; }
"/)"                        { return ARR_END; }
 \n                         {}
[ \t\v\r]+                { /* Ignore whitespace, tabs and newlines */ }
"allocatable"               { return ALLOCATABLE; }
"parameter"                 { return PARAMETER; }
"target"                    { return TARGET; }
"KIND"                      { return KIND; }
"rkind8"                    { return REAL_KIND; }
"len"                       { return LEN; }
"LEN"                       { return LEN; }
"volatile"                  { return VOLATILE; }
"module"                    { return MODULE; }
"MODULE"                    { return MODULE; }
"use"                       { return USE; }
"USE"                       { return USE; }
"implicit"                  { return IMPLICIT; }
"none"                      { return NONE; }
"public"                    { return PUBLIC; }
"private"                   { return PRIVATE; }
"endmodule"                 { return ENDMODULE; }
"only"                      { return ONLY; }
"pointer"                   { return POINTER; }
"namelist"                  { return NAMELIST; }
"interface"                 { return INTERFACE; }
"endinterface"              { return ENDINTERFACE; }
"module procedure"          { return MODULE_PROCEDURE; }
"contains"                  { return CONTAINS; }
"subroutine"                { return SUBROUTINE; }
"function"                  { return FUNCTION; }
"endfunction"               { return END_FUNCTION; }
"endsubroutine"             { return ENDSUBROUTINE; }
"do"                        {  return DO; }    /* Operators */
"forall"                    {  return FORALL; }    /* Operators */
"end forall"                {  return END_FORALL; }    /* Operators */
"enddo"                     { return ENDDO; }    /* Operators */
"select case"               { return SELECT_CASE; }    /* Operators */
"case"                      { return CASE; }    /* Operators */
"exit"                      { return EXIT; }    /* Operators */
"end select"                { return END_SELECT; }    /* Operators */
"endselect"                 { return END_SELECT; }    /* Operators */

"where"                     { return WHERE; }
"elsewhere"                 { return ELSEWHERE; }
"endwhere"                  { return ENDWHERE; }

"inquire"                 { return INTRINSIC_FUNC; }
"allocate"                { return INTRINSIC_FUNC; }
"aimag"                   { return INTRINSIC_FUNC; }
"intent"                  { return INTENT; }
"inout"                   { return INOUT; }
"INOUT"                   { return INOUT; }

"in"                   { return IN; }
"IN"                   { return IN; }

"out"                   { return OUT; }
"OUT"                   { return OUT; }

"optional"             { return OPTIONAL; }

"cycle"                   { return CYCLE; }



"uint"                  { return UINT; }
"real"                  { return REAL; }
"Matrix"                { return MATRIX; }
"Field"                 { return FIELD; }
"Field3"                { return FIELD3; }
"VtxBuffer"                 { return VTXBUFFER; }
"WorkBuffer"            {return WORK_BUFFER; }
"ComputeSteps"          {return COMPUTESTEPS; }
"BoundConds"          {return BOUNDCONDS; }
"intrinsic"          {return INTRINSIC; }
"__shared__"          {return SHARED; }

"Stencil"               { return STENCIL; }

"Kernel"                { return KERNEL; } /* Type qualifiers */
"inline"                { return INLINE; } /* Type qualifiers */
"elemental"             { return ELEMENTAL; } /* Type qualifiers */
"boundary_condition"    { return BOUNDARY_CONDITION; } /* Type qualifiers */
"Sum"                   { return SUM; }
"Max"                   { return MAX; }
"dconst"          { return DCONST_QL; }
"const"          { return CONST_QL; }
"dynamic"          { return DYNAMIC_QL; }
"constexpr"      { return CONSTEXPR; }
"output"          { return OUTPUT; }
"run_const"          { return RUN_CONST; }
"gmem"          { return GLOBAL_MEMORY_QL; }
"auxiliary"             { return AUXILIARY; }
"extern"             { return EXTERN; }
"type"                     {  return TYPE; }    /* operators */

"then"        { return THEN; } /* Keywords */
"endif"     { return ENDIF; } /* Keywords */
"call"     { return CALL; } /* Keywords */
"if constexpr"        { return IF; } /* Keywords */
"if"        { return IF; } /* Keywords */
"else if"   { return ELIF; }
"elseif"   { return ELIF; }
"else if constexpr"   { return ELIF; }
"else"      { return ELSE; }
"while"     { return WHILE; }
"for"       { return FOR; }
"in"        { return IN; }
"break"     { return BREAK; }
"continue"  { return CONTINUE; }
"return"    { return RETURN; }
[?]         { return QUESTION; }
"%"         { return ACCESS_TOKEN; }

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
\"([^\"\\]|\\.)*\"    { return STRING;}
\'([^\'\\]|\\.)*\'    { return STRING;}
"**"                       { return BINARY_OP; }     /* Assignment */
"/="                       { return BINARY_OP; }     /* Assignment */
"="                       { return ASSIGNOP; }     /* Assignment */
"//"                      {  return BINARY_OP; }     /* Assignment */
[/*<>|^]                 { return BINARY_OP; }      /* Operators */
(&&)|(\|\|)|(<<)|(>>)     { return BINARY_OP; }    /* Operators */
(==)|(>=)|(<=)|(!=)|(^=)  { return BINARY_OP; }    /* Operators */
".not."                   { return UNARY_OP; }    /* Operators */
".and."                   { return BINARY_OP; }    /* operators */
".AND."                   { return BINARY_OP; }    /* operators */
".or."                    { return BINARY_OP; }    /* operators */
".ne."                     { return BINARY_OP; }    /* operators */
[-+*/:;=\[\]{}(),\.<>|&!] { return yytext[0]; }    /* characters */



{D}+[.]{D}*               { return REALNUMBER; }
[.]{D}+               { return REALNUMBER; }
{D}+[.]?{D}*(e[-+]?{D}+)   { return REALNUMBER; }
{D}+[.]?{D}*(e[-+]?{D}+)?[d]  { return DOUBLENUMBER; } /* Numbers */
{D}+[.]?{D}*(e[-+]?{D}+)[f]? { return REALNUMBER; }      /* Numbers */
{D}+[.]?{D}*  		      { return NUMBER; }      /* Numbers */
0[xX][0-9a-fA-F]+[ulUL]*      { return NUMBER; }      /* Hex */



"#".*               { /* Skip regular comments */ }
.                   { fprintf(stderr, "unrecognized char %d: [%c]\n", *yytext, *yytext); }

%%

/*
[.]{D}+                   { return REALNUMBER; }
[.]?{D}+(e[-]?{D}+)       { return REALNUMBER; }
{D}?[.]+{D}*(e[-]?{D}+)?[f]?  { return NUMBER; }
{D}?[.]+{D}*(e[-]?{D}+)?[d]  { return DOUBLENUMBER; }
struct\s{L}({L}|{D})* {printf("Found it\n"); exit(0); }
*/
