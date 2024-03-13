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
%}
%{
	char* struct_names[256];
	size_t num_structs = 0;
	void add_struct_name(const char* struct_name)
	{
		struct_names[num_structs] = strdup(struct_name);
		++num_structs;
	}	
	bool
	is_struct_name(const char* name)
	{	
		for(size_t i = 0; i < num_structs; ++i)
			if(!strcmp(struct_names[i], name))
				return true;
		return false;
	}
        void remove_substring(char *str, const char *sub) {
                int len = strlen(sub); 
                char *found = strstr(str, sub); // Find the first occurrence of the substring

                while (found) {
                        memmove(found, found + len, strlen(found + len) + 1); // Shift characters to overwrite the substring
                        found = strstr(found, sub); // Find the next occurrence of the substring
                }
        }


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
"real3"                 { return REAL3; }
"Matrix"                { return MATRIX; }
"Field"                 { return FIELD; }
"WorkBuffer"            {return WORK_BUFFER; }

"Stencil"               { return STENCIL; }

"Kernel"                { return KERNEL; } /* Type qualifiers */
"Sum"                   { return SUM; }
"Max"                   { return MAX; }
"communicated"          { return COMMUNICATED; }
"dconst"          { return DCONST_QL; }
"auxiliary"             { return AUXILIARY; }

"if"        { return IF; } /* Keywords */
"else if"   { return ELIF; }
"else"      { return ELSE; }
"while"     { return WHILE; }
"for"       { return FOR; }
"in"        { return IN; }
"break"     { return BREAK; }
"continue"  { return CONTINUE; }
"return"    { return RETURN; }

typedef[[:space:]]struct[[:space:]]{L}({L}|{D})* {
					char* name = strdup(yytext);
					remove_substring(name,"struct");
					remove_substring(name,"typedef");
					strip_whitespace(name);
					add_struct_name(name);
					return STRUCT_NAME;
				}

"hostdefine".*        { return HOSTDEFINE; }

{L}({L}|{D})*             {  /* Literals */
				if(is_struct_name(yytext))
					return STRUCT_TYPE;
				return IDENTIFIER; 
			  }   
\"(\\.|[^"\\])*\"         { return STRING; }       /* String literal */
[+-/*|&%^]?=              { return ASSIGNOP; }     /* Assignment */
(<<=)|(>>=)               { return ASSIGNOP; }     /* Assignment */
[/*-+<>|&%^]              { return BINARY_OP; }    /* Operators */
(&&)|(\|\|)|(<<)|(>>)     { return BINARY_OP; }    /* Operators */
(==)|(>=)|(<=)|(!=)|(^=)  { return BINARY_OP; }    /* Operators */
[-+*/:;=\[\]{}(),\.<>|&!] { return yytext[0]; }    /* Characters */
{D}+[.]{D}*               { return REALNUMBER; }
{D}+[.]?{D}*(e[-]?{D}+)   { return REALNUMBER; }
{D}+[.]?{D}*(e[-]?{D}+)?[d]  { return DOUBLENUMBER; } /* Numbers */
{D}+[.]?{D}*(e[-]?{D}+)?[f]?  { return NUMBER; }      /* Numbers */
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
