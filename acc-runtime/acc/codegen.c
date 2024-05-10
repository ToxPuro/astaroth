/*
    Copyright (C) 2021, Johannes Pekkila.
    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "codegen.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ast.h"
#include "tab.h"
#include <string.h>
#include <ctype.h>
static NodeType always_excluded = NODE_CODEGEN_INPUT | NODE_ENUM_DEF | NODE_STRUCT_DEF;
char*
strupr(const char* src)
{
	char* res = strdup(src);
	int index = 0;
	while(res[index] != '\0')
	{
		res[index] = toupper(res[index]);
		++index;
	}
	return res;
}

#define STENCILGEN_HEADER "stencilgen.h"
#define STENCILGEN_SRC ACC_DIR "/stencilgen.c"
#define STENCILGEN_EXEC "stencilgen.out"
#define STENCILACC_SRC ACC_DIR "/stencil_accesses.cpp"
#define STENCILACC_EXEC "stencil_accesses.out"
#define ACC_RUNTIME_API_DIR ACC_DIR "/../api"
//
static string_vec array_fields; 
static int_vec array_field_sizes; 
static string_vec user_enums;
static string_vec user_enum_options[100];

static string_vec user_structs;
static string_vec user_struct_field_types[100];
static string_vec user_struct_field_names[100];

static string_vec user_kernels_with_input_params;
static string_vec user_kernel_combinatorial_params[100];
static string_vec user_kernel_combinatorial_param_options[100][100];

static string_vec user_kernel_combinations[100][1000];
static int user_kernel_num_combinations[100];
static string_vec dfuncs;

static op_vec dfuncs_reduce_ops[1000];
static string_vec dfuncs_reduce_conditions[1000];
static string_vec dfuncs_reduce_outputs[1000];

static string_vec kernel_reduce_outputs[100];
static op_vec kernel_reduce_ops[100];
static string_vec field_names;


static const char* user_kernel_ifs       = "user_kernel_ifs.h";

// Symbols
#define MAX_ID_LEN (256)
typedef struct {
  NodeType type;
  string_vec tqualifiers;
  char tspecifier[MAX_ID_LEN];
  char identifier[MAX_ID_LEN];
  } Symbol;


#define SYMBOL_TABLE_SIZE (65536)
static Symbol symbol_table[SYMBOL_TABLE_SIZE] = {};

char* symbol_var_length[SYMBOL_TABLE_SIZE];
#define MAX_NESTS (32)
static size_t num_symbols[MAX_NESTS] = {};
static size_t current_nest           = 0;


//arrays symbol table
#define MAX_NUM_ARRAYS (256)
int num_of_arrays;

static Symbol*
symboltable_lookup(const char* identifier)
{
  if (!identifier)
    return NULL;

  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(identifier, symbol_table[i].identifier))
      return &symbol_table[i];

  return NULL;
}


static int 
add_symbol(const NodeType type, char* const* tqualifiers, const size_t n_tqualifiers, const char* tspecifier,
           const char* id)
{
  assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);

  symbol_table[num_symbols[current_nest]].type          = type;
  symbol_table[num_symbols[current_nest]].tspecifier[0] = '\0';

  init_str_vec(&symbol_table[num_symbols[current_nest]].tqualifiers);
  if(tspecifier && (!strcmp(tspecifier,"AcReal*") || !strcmp(tspecifier,"int*")) && n_tqualifiers==0)
    push(&symbol_table[num_symbols[current_nest]].tqualifiers,"dconst");
  else
  {
    for(size_t i = 0; i < n_tqualifiers; ++i)
        push(&symbol_table[num_symbols[current_nest]].tqualifiers,tqualifiers[i]);
  }

  if (tspecifier)
    strcpy(symbol_table[num_symbols[current_nest]].tspecifier, tspecifier);

  strcpy(symbol_table[num_symbols[current_nest]].identifier, id);

  ++num_symbols[current_nest];
  //return the index of the lastly added symbol
  return num_symbols[current_nest]-1;
}

static void
symboltable_reset(void)
{
  for(size_t i = 0; i < SYMBOL_TABLE_SIZE ; ++i)
	  free_str_vec(&symbol_table[i].tqualifiers);
  current_nest              = 0;
  num_symbols[current_nest] = 0;

  // Add built-in variables (TODO consider NODE_BUILTIN)
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "print");           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "threadIdx");       // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "blockIdx");        // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "vertexIdx");       // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "globalVertexIdx"); // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "globalGridN");     // TODO REMOVE

  // add_symbol(NODE_UNKNOWN, NULL, NULL, "true");
  // add_symbol(NODE_UNKNOWN, NULL, NULL, "false");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "previous");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "write");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "isnan");  // TODO RECHECK
							 //
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "reduce_sum");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "reduce_min");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "reduce_max");  // TODO RECHECK
  //In develop
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "read_w");
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write_w");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "Field3"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "dot");    // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "cross");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "len");    // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "uint64_t");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "UINT64_MAX"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "rand_uniform");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "exp");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "sin");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "cos");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "sqrt");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "fabs");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "pow");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "multm2_sym");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "diagonal");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "sum");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "log");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "abs");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "atan2"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "AC_REAL_PI");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "NUM_FIELDS");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "NUM_VTXBUF_HANDLES");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "NUM_ALL_FIELDS");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "FIELD_IN");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "FIELD_OUT");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "IDX");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "true");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "false");

  // Astaroth 2.0 backwards compatibility START
  // (should be actually built-in externs in acc-runtime/api/acc-runtime.h)
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_mx");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_my");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_mz");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nx");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_ny");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nz");

  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nxgrid");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nygrid");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nzgrid");

  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nx_min");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_ny_min");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nz_min");

  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nx_max");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_ny_max");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nz_max");

  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_mxy");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nxy");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_nxyz");

  add_symbol(NODE_DCONST_ID, NULL, 0,"int", "AC_xy_plate_bufsize");
  add_symbol(NODE_DCONST_ID, NULL, 0,"int", "AC_xz_plate_bufsize");
  add_symbol(NODE_DCONST_ID, NULL, 0,"int", "AC_yz_plate_bufsize");
  add_symbol(NODE_DCONST_ID, NULL, 0,"int3", "AC_domain_decomposition");
  add_symbol(NODE_DCONST_ID, NULL, 0,"int", "AC_proc_mapping_strategy");
  add_symbol(NODE_DCONST_ID, NULL, 0,"int", "AC_decompose_strategy");
  add_symbol(NODE_DCONST_ID, NULL, 0,"int", "AC_MPI_comm_strategy");
  

  add_symbol(NODE_DCONST_ID, NULL, 0, "int3", "AC_multigpu_offset");

  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsx");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsy");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsz");

  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsx");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsy");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsz");

  //For special reductions
  add_symbol(NODE_DCONST_ID, NULL, 0, "AcReal", "AC_center_x");
  add_symbol(NODE_DCONST_ID, NULL, 0, "AcReal", "AC_center_y");
  add_symbol(NODE_DCONST_ID, NULL, 0, "AcReal", "AC_center_z");
  add_symbol(NODE_DCONST_ID, NULL, 0, "AcReal", "AC_sum_radius");
  add_symbol(NODE_DCONST_ID, NULL, 0, "AcReal", "AC_window_radius");

  // (BC types do not belong here, BCs not handled with the DSL)
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_bc_type_bot_x");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_bc_type_bot_y");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_bc_type_bot_z");

  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_bc_type_top_x");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_bc_type_top_y");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_bc_type_top_z");
  add_symbol(NODE_DCONST_ID, NULL, 0, "int", "AC_init_type");
  // Astaroth 2.0 backwards compatibility END
}

void
print_symbol_table(void)
{
  printf("\n---\n");
  printf("Symbol table:\n");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
    printf("%lu: ", i);
    printf("%s ", symbol_table[i].identifier);

    if (strlen(symbol_table[i].tspecifier) > 0)
      printf("(tspec: %s) ", symbol_table[i].tspecifier);

    if (symbol_table[i].tqualifiers.size)
    {
	    char tqual[4096];
	    tqual[0] = '\0';
	    for(size_t tqi = 0; tqi < symbol_table[i].tqualifiers.size; ++tqi)
		    strcat(tqual,symbol_table[i].tqualifiers.data[tqi]);
      	    printf("(tquals: %s) ", tqual);
    }
	 

    if (symbol_table[i].type & NODE_FUNCTION_ID)
      printf("(%s function)",
             symbol_table[i].type & NODE_KFUNCTION_ID ? "kernel" : "device");

    if (symbol_table[i].type & NODE_DCONST_ID)
      printf("(dconst)");

    if (symbol_table[i].type & NODE_STENCIL_ID)
      printf("(stencil)");

    printf("\n");
  }
  printf("---\n");
}



static const ASTNode*
get_node(const NodeType type, const ASTNode* node)
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
const char*
convert_to_enum_name(const char* name)
{
	if(!strcmp(name,"int"))
		return "AcInt";
	if(!strcmp(name,"int3"))
		return "AcInt3";
	return name;
}
const char*
convert_to_define_name(const char* name)
{
	if(!strcmp(name,"AcReal"))
		return "real";
	if(!strcmp(name,"AcReal3"))
		return "real3";
	return name;
}
bool is_number(const char* str) {
    // Skip leading whitespaces
    while (*str == ' ') {
        str++;
    }

    // Check for optional sign
    if (*str == '+' || *str == '-') {
        str++;
    }

    bool hasDigit = false;
    bool hasDecimal = false;

    while (*str != '\0') {
        if (isdigit(*str)) {
            hasDigit = true;
        } else if (*str == '.') {
            // Check if '.' already encountered or if it's the first character
            if (hasDecimal || !hasDigit) {
                return false;
            }
            hasDecimal = true;
        } else {
            return false; // Character is not a digit or a decimal point
        }
        str++;
    }

    // Check if at least one digit is encountered
    return hasDigit;
}
void
gen_d_offsets(FILE* fp, const char* datatype_scalar, const bool declarations)
{
  char running_offset[4096];
  sprintf(running_offset,"0");
  const char* define_name =  convert_to_define_name(datatype_scalar);
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  if(!declarations)
        fprintf(fp, "static int d_%s_array_offsets[] __attribute__((unused)) = {",define_name);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier,datatype) && str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
    {
            if(declarations)
                fprintf(fp,"\n#ifndef %s_offset\n#define %s_offset (%s)\n#endif\n",symbol_table[i].identifier,symbol_table[i].identifier,running_offset);
            else
                fprintf(fp," %s,\n",running_offset);
            //if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
            sprintf(running_offset,"%s+%s",running_offset,symbol_var_length[i]);
    }
  }
   if(declarations)
        fprintf(fp,"\n#ifndef D_%s_ARRAYS_LEN\n#define D_%s_ARRAYS_LEN (%s)\n#endif\n", strupr(define_name), strupr(define_name),running_offset);
   else
        fprintf(fp, "};");
}

void
gen_array_lengths(FILE* fp, const char* datatype_scalar)
{
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  fprintf(fp, "static const int %s_array_lengths[] __attribute__((unused)) = {", convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
      		fprintf(fp, "%s,", symbol_var_length[i]);
	else if(!str_vec_contains(symbol_table[i].tqualifiers,"const"))
      		fprintf(fp, "(int)%s,", symbol_var_length[i]);
    }
  fprintf(fp, "};");
}
void
gen_array_is_dconst(FILE* fp, const char* datatype_scalar)
{
  fprintf(fp, "static const bool %s_array_is_dconst[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
        if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
                fprintf(fp,"true,");
        else
                fprintf(fp, "false,");
    }
  fprintf(fp, "};");
}
void
gen_enums(FILE* fp, const char* datatype_scalar, const bool for_arrays)
{
  const NodeType type = for_arrays ? NODE_VARIABLE_ID : NODE_DCONST_ID;
  char datatype[1000];
  if(for_arrays)
        sprintf(datatype,"%s*",datatype_scalar);
  else
        sprintf(datatype,"%s",datatype_scalar);
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & type && !strcmp(symbol_table[i].tspecifier, datatype) && !str_vec_contains(symbol_table[i].tqualifiers,"output") && !str_vec_contains(symbol_table[i].tqualifiers,"const"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  if(for_arrays)
  	fprintf(fp, "NUM_%s_ARRAYS} %sArrayParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));
  else
  	fprintf(fp, "NUM_%s_PARAMS} %sParam;",strupr(convert_to_define_name(datatype)),convert_to_enum_name(datatype));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & type && !strcmp(symbol_table[i].tspecifier, datatype) && str_vec_contains(symbol_table[i].tqualifiers,"output") && !str_vec_contains(symbol_table[i].tqualifiers,"const"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  if(for_arrays)
  	fprintf(fp, "NUM_%s_OUTPUT_ARRAYS} %sArrayOutput;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));
  else
  	fprintf(fp, "NUM_%s_OUTPUTS} %sOutput;",strupr(convert_to_define_name(datatype)),convert_to_enum_name(datatype));
}
void
gen_param_names(FILE* fp, const char* datatype_scalar, const bool for_arrays)
{
  const NodeType type = for_arrays ? NODE_VARIABLE_ID : NODE_DCONST_ID;
  if(for_arrays)
  	fprintf(fp, "static const char* %s_array_param_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  else
  	fprintf(fp, "static const char* %sparam_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  char datatype[1000];
  if(for_arrays)
  	sprintf(datatype,"%s*",datatype_scalar);
  else
  	sprintf(datatype,"%s",datatype_scalar);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & type && !strcmp(symbol_table[i].tspecifier, datatype) && !str_vec_contains(symbol_table[i].tqualifiers,"output") && !str_vec_contains(symbol_table[i].tqualifiers,"const"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  if(for_arrays)
  	fprintf(fp, "static const char* %s_array_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  else
  	fprintf(fp, "static const char* %soutput_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));

  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & type && !strcmp(symbol_table[i].tspecifier, datatype) && str_vec_contains(symbol_table[i].tqualifiers,"output") && !str_vec_contains(symbol_table[i].tqualifiers,"const"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");
}

void
gen_field_array_declarations(FILE* fp)
{ 
  //new code make it an actual array.
  //makes making vector field bundles easier
  for(size_t i = 0; i < array_fields.size; ++i)
  {
	  const int size = array_field_sizes.data[i];
	  const char* name = array_fields.data[i];
	  fprintf(fp,"static const Field %s[%d] = {",name,size);
	  for(int j = 0; j < size; ++j)
	  {
	  	fprintf(fp,"%s_%d,\n",name,j);
	  }
	  fprintf(fp,"};\n");
  }
  //old code where you derefence by hand
  //if(node->lhs)
  //  gen_field_array_reads(node->lhs);
  //if(node->rhs)
  //  gen_field_array_reads(node->rhs);
  //if((node->type && !(node->type & NODE_UNKNOWN)) || !node->buffer || !str_vec_contains(array_fields,node->buffer))
  //        return;
  //char index_str[4096];
  //combine_all(node->parent->parent->parent->rhs,index_str);
  //char res[4096];
  //sprintf(res,"%s_0 + %s",node->buffer,index_str);
  //ASTNode* base = node->parent->parent->parent->parent;
  //base->buffer = strdup(res);
  //base->lhs=NULL;
  //base->rhs=NULL;
  //base->prefix=NULL;
  //base->postfix=NULL;
}
void
gen_field_accesses(ASTNode* node)
{
	if(node->lhs)
		gen_field_accesses(node->lhs);
	if(node->rhs)
		gen_field_accesses(node->rhs);
	if(!(node->token == IDENTIFIER))
		return;
	if(!node->buffer)
		return;
	if(!str_vec_contains(field_names,node->buffer))
		return;
	if(!node->parent)
		return;
	if(!node->parent->parent)
		return;
	if(!node->parent->parent->parent)
		return;
	char tmp[1000];
	combine_all(node->parent->parent->parent->parent,tmp);
	const char* search_str = "][";
	const char* substr = strstr(tmp,search_str);
	if(!substr)
		return;
	char x_index[1000];
	char y_index[1000];
	char z_index[1000];
	bool is_assigned = false;
       	const ASTNode* assign_node = get_parent_node(NODE_ASSIGNMENT,node);
	if(assign_node)
	{
		const ASTNode* search = get_node_by_id(node->id,assign_node->lhs);
	      is_assigned = search != NULL;
	}
	const ASTNode* start = is_assigned
				? node->parent->parent
				: node->parent->parent->parent;
	combine_all(start->rhs,x_index);
	combine_all(start->parent->rhs,y_index);
	combine_all(start->parent->parent->rhs,z_index);
	ASTNode* base= start->parent->parent;
	base->lhs = NULL;
	base->rhs = NULL;
	base->infix= NULL;
	base->postfix= NULL;
	char res[4000];
	sprintf(res,"vba.in[%s][IDX(%s,%s,%s)]\n",node->buffer,x_index,y_index,z_index);
	base->buffer = strdup(res);
}
void
gen_array_reads(ASTNode* node, bool gen_mem_accesses, const char* datatype_scalar)
{
  if(node->lhs)
    gen_array_reads(node->lhs,gen_mem_accesses,datatype_scalar);
  if(node->rhs)
    gen_array_reads(node->rhs,gen_mem_accesses,datatype_scalar);
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  if(!node->buffer)
	  return;
  	const int l_current_nest = 0;
  	for (size_t i = 0; i < num_symbols[l_current_nest]; ++i)
	{
    	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
          (!strcmp(symbol_table[i].tspecifier,datatype)) && !strcmp(node->buffer,symbol_table[i].identifier) && node->parent->parent->parent->rhs)
	  {
		if(gen_mem_accesses)
		{
			char big_array_name[1000];
			sprintf(big_array_name,"big_%s_array",convert_to_define_name(datatype_scalar));
			node->buffer = strdup(big_array_name);
			node->type |= NODE_INPUT;
			return;
		}
		char new_name[4096-19];
		new_name[0] = '\0';
		char arrays_name[4096/2];
		sprintf(arrays_name,"%s_arrays",convert_to_define_name(datatype_scalar));
      		if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
		{
			//node->prefix = strdup("DCONST(");
			//node->parent->parent->parent->postfix = strdup("])");
			char index_str[4096/2];
			combine_all(node->parent->parent->parent->rhs,index_str);
			char res[4096];
			sprintf(res,"d_%s[%s_offset+(%s)]\n",arrays_name,node->buffer,index_str);
			ASTNode* base = node->parent->parent->parent->parent;
			base->buffer = strdup(res);
			base->lhs=NULL;
			base->rhs=NULL;
			base->prefix=NULL;
			base->postfix=NULL;
			return;
		}
		sprintf(new_name,"__ldg(&vba.%s[(int)%s]",arrays_name,node->buffer);
		node->parent->parent->parent->postfix = strdup("])");
		node->buffer = strdup(new_name);
	  }
	}
}
bool
is_enum(const char* type)
{
	return str_vec_contains(user_enums,type);
}
bool
is_struct(const char* type)
{
	return str_vec_contains(user_structs,type);
}
string_vec
get_struct_field_types(const char* struct_name)
{
	const int struct_index = str_vec_get_index(user_structs,struct_name);
	return user_struct_field_types[struct_index];
}
string_vec
get_struct_field_names(const char* struct_name)
{
	const int struct_index = str_vec_get_index(user_structs,struct_name);
	return user_struct_field_names[struct_index];
}
string_vec
get_enum_options(const char* enum_name)
{
	const int enum_index = str_vec_get_index(user_enums,enum_name);
	return user_enum_options[enum_index];
}

char* readFile(const char *filename) {
    FILE *file = fopen(filename, "rb"); // Open the file in binary mode

    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Determine the size of the file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Allocate memory for the string
    char *buffer = (char *)malloc(file_size + 1); // Plus one for the null terminator
    if (buffer == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    // Read the entire contents of the file into the allocated memory
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if ((long) bytes_read != file_size) {
        perror("Error reading file");
        free(buffer);
        fclose(file);
        return NULL;
    }

    // Null-terminate the string
    buffer[file_size] = '\0';

    // Close the file
    fclose(file);

    return buffer;
}
void
file_prepend(const char* filename, const char* str_to_prepend)
{
	const char* file_tmp = readFile(filename);
	FILE* fp = fopen(filename,"w");
	fprintf(fp,"%s%s",str_to_prepend,file_tmp);
	fclose(fp);
	free((void*)file_tmp);
}
void
read_user_enums(ASTNode* node)
{
	if(node->type & NODE_ENUM_DEF)
	{
		char tmp[4096];
		combine_all(node->rhs,tmp);
		char enum_def[4096];
		sprintf(enum_def, "typedef enum %s{%s} %s;",node->lhs->buffer,tmp,node->lhs->buffer);
		file_prepend("user_structs.h",enum_def);
		const int enum_index = push(&user_enums,node->lhs->buffer);
		ASTNode* enums_head = node->rhs;
		while(enums_head->rhs)
		{
			push(&user_enum_options[enum_index],get_node_by_token(IDENTIFIER,enums_head->rhs)->buffer);
			enums_head = enums_head->lhs;
		}
		push(&user_enum_options[enum_index],get_node_by_token(IDENTIFIER,enums_head->lhs)->buffer);
	}
	if(node->buffer && str_vec_contains(user_enums,node->buffer))
		node->type |= NODE_ENUM;
	if(node->lhs)
		read_user_enums(node->lhs);
	if(node->rhs)
		read_user_enums(node->rhs);
}
static inline void
process_declaration(const ASTNode* field, char* struct_def, int struct_index)
{
	char type[4096];
	char name[4096];
	type[0] = '\0';
	combine_buffers(get_node(NODE_TSPEC, field),type);
	combine_buffers(field->rhs,name);

	strcat(struct_def,type);
	strcat(struct_def," ");
	strcat(struct_def,name);
	strcat(struct_def,";");
	push(&user_struct_field_types[struct_index], type);
	push(&user_struct_field_names[struct_index], name);
}
void
read_user_structs(ASTNode* node)
{
	if(node->type & NODE_STRUCT_DEF)
	{
		const int struct_index = push(&user_structs,node->lhs->buffer);
		ASTNode* fields_head = node->rhs;
                char struct_def[5000];
		sprintf(struct_def,"typedef struct %s {",node->lhs->buffer);
		while(fields_head->rhs)
		{
			process_declaration(fields_head->rhs,struct_def,struct_index);
			fields_head = fields_head->lhs;
		}
		process_declaration(fields_head->lhs,struct_def,struct_index);
		strcat(struct_def, "} ");
		strcat(struct_def, node->lhs->buffer);
		strcat(struct_def, ";\n");
		file_prepend("user_structs.h",struct_def);
		node->lhs=NULL;
		node->rhs=NULL;
	}
	if(node->lhs)
		read_user_structs(node->lhs);
	if(node->rhs)
		read_user_structs(node->rhs);
}
ASTNode*
get_node_with_buffer(ASTNode* node, NodeType type, const char* buffer)
{
	if(node->type & type && !strcmp(node->lhs->buffer,buffer))
	{
		return node;
	}
	if(node->lhs)
	{
		ASTNode* lhs_res = get_node_with_buffer(node->lhs,type,buffer);
		if(lhs_res)
			return lhs_res;
	}
	if(node->rhs)
	{
		ASTNode* rhs_res = get_node_with_buffer(node->rhs,type,buffer);
		if(rhs_res)
			return rhs_res;
	}
	return NULL;
}
void
add_param_combinations(const char* type, const char* name, const int kernel_index,const char* prefix)
{
	char full_name[4096];
	sprintf(full_name,"%s%s",prefix,name);
	if(is_struct(type))
	{
	  string_vec struct_field_types = get_struct_field_types(type);
	  string_vec struct_field_names = get_struct_field_names(type);
	  for(size_t i=0; i<struct_field_types.size; ++i)
	  {
		  char new_prefix[10000];
		  sprintf(new_prefix, "%s%s.",prefix,name);
		  add_param_combinations(struct_field_types.data[i],struct_field_names.data[i],kernel_index,new_prefix);
	  }
	}
	if(is_enum(type))
	{
		const int param_index = push(&user_kernel_combinatorial_params[kernel_index],full_name);
		string_vec options  = get_enum_options(type);
		for(size_t i = 0; i < options.size; ++i)
		{
			push(&user_kernel_combinatorial_param_options[kernel_index][param_index],options.data[i]);
		}
	}
	if(!strcmp("bool",type))
	{
		const int param_index = push(&user_kernel_combinatorial_params[kernel_index],full_name);
		push(&user_kernel_combinatorial_param_options[kernel_index][param_index],"false");
		push(&user_kernel_combinatorial_param_options[kernel_index][param_index],"true");
	}
}
void
gen_all_posibilities(string_vec res, int kernel_index, size_t my_index)
{
	if(my_index == user_kernel_combinatorial_params[kernel_index].size-1)
	{
		for(size_t i = 0; i<user_kernel_combinatorial_param_options[kernel_index][my_index].size; ++i)
		{

			user_kernel_combinations[kernel_index][user_kernel_num_combinations[kernel_index]] = str_vec_copy(res);
			push(&user_kernel_combinations[kernel_index][user_kernel_num_combinations[kernel_index]], user_kernel_combinatorial_param_options[kernel_index][my_index].data[i]);
			++user_kernel_num_combinations[kernel_index];
			
		}
		return;
	}
	else
	{
		for(size_t i = 0; i<user_kernel_combinatorial_param_options[kernel_index][my_index].size; ++i)
		{
			string_vec copy = str_vec_copy(res);
			push(&copy, user_kernel_combinatorial_param_options[kernel_index][my_index].data[i]);
			gen_all_posibilities(copy,kernel_index,my_index+1);
		}
	}
}
void 
gen_combinations(int kernel_index)
{
	string_vec base;
	init_str_vec(&base);
	if(user_kernel_combinatorial_params[kernel_index].size > 0)
		gen_all_posibilities(base, kernel_index,0);
	free_str_vec(&base);
}
void
gen_kernel_num_of_combinations(const ASTNode* node)
{
	if(node->lhs)
		gen_kernel_num_of_combinations(node->lhs);
	if(node->rhs)
		gen_kernel_num_of_combinations(node->rhs);
	if(node->type & NODE_KFUNCTION && node->rhs->lhs)
	{
	   const int kernel_index = push(&user_kernels_with_input_params,get_node(NODE_KFUNCTION_ID, node)->buffer);
	   ASTNode* param_list_head = node->rhs->lhs;
	   while(param_list_head->rhs)
	   {
	   	char type[4096];
	   	char name[4096];
	        const ASTNode* type_node = get_node(NODE_TSPEC,param_list_head->rhs);
	   	combine_buffers(type_node,type);
	   	combine_buffers(param_list_head->rhs->rhs,name);
	        add_param_combinations(type,name,kernel_index,"");
	        param_list_head = param_list_head->lhs;
	   }
	   char type[4096];
	   char name[4096];
	   const ASTNode* type_node = get_node(NODE_TSPEC,param_list_head->lhs);
	   combine_buffers(type_node,type);
	   combine_buffers(param_list_head->lhs->rhs,name);
	   add_param_combinations(type,name,kernel_index,"");
	   gen_combinations(kernel_index);
	}
}


int 
get_suffix_int(const char *str, const char* suffix_match) {
    const char *optimizedPos = strstr(str, suffix_match);
    if (optimizedPos == NULL) 
        return -1; 

    optimizedPos += strlen(suffix_match);

    int value;
    sscanf(optimizedPos, "%d", &value);
    return value;
}

void
remove_suffix(char *str, const char* suffix_match) {
    char *optimizedPos = strstr(str, suffix_match);
    if (optimizedPos != NULL) {
        *optimizedPos = '\0'; // Replace '_' with null character
    }
}
void
get_reduce_ops(ASTNode* node, ReduceOp* src, int* n)
{
	ReduceOp res = NO_REDUCE;
	if(node->type & NODE_FUNCTION_CALL)
	{
		char func_name[5000];
		combine_buffers(node->lhs,func_name);
		if(!strcmp(func_name,"reduce_sum"))
		{
			src[(*n)] = REDUCE_SUM;
			++(*n);
		}
		if(!strcmp(func_name,"reduce_min"))
		{
			src[(*n)] = REDUCE_MIN;
			++(*n);
		}
		if(!strcmp(func_name,"reduce_max"))
		{
			src[(*n)] = REDUCE_MAX;
			++(*n);
		}
		const int dfunc_index = str_vec_get_index(dfuncs,func_name);
		if(dfunc_index > 0)
			for(size_t i = 0; i < dfuncs_reduce_ops[dfunc_index].size; ++i)
			{
				src[(*n)] = dfuncs_reduce_ops[dfunc_index].data[i];
				++(*n);
			}
	}
	if(node->lhs) 
		get_reduce_ops(node->lhs,src,n);
	if(node->rhs) 
		get_reduce_ops(node->rhs,src,n);
}
void
get_reduce_conditions(ASTNode* node, string_vec* src)
{
	char* res = NULL;
	if(node->type & NODE_FUNCTION_CALL)
	{
		char func_name[5000];
		combine_buffers(node->lhs,func_name);
		if (!strcmp(func_name,"reduce_sum") || !strcmp(func_name,"reduce_min") || !strcmp(func_name,"reduce_max"))
		{
		  char condition[5000];
		  combine_buffers(node->rhs->lhs->lhs,condition);
		  push(src,condition);
		}
		const int dfunc_index = str_vec_get_index(dfuncs,func_name);
		if(dfunc_index > 0)
			for(size_t i = 0; i < dfuncs_reduce_conditions[dfunc_index].size; ++i)
				push(src,dfuncs_reduce_conditions[dfunc_index].data[i]);
	}
	if(node->lhs && res == NULL) 
		get_reduce_conditions(node->lhs,src);
	if(node->rhs && res == NULL) 
		get_reduce_conditions(node->rhs,src);
}
void
get_reduce_outputs(ASTNode* node, string_vec* src)
{
	char* res = NULL;
	const size_t orig_src_size = (*src).size;
	if(node->type & NODE_FUNCTION_CALL)
	{
		char func_name[5000];
		combine_buffers(node->lhs,func_name);
		if (!strcmp(func_name,"reduce_sum") || !strcmp(func_name,"reduce_min") || !strcmp(func_name,"reduce_max"))
		{
		  char output[5000];
		  combine_buffers(node->rhs->rhs,output);
		  push(src,output);
		}
		const int dfunc_index = str_vec_get_index(dfuncs,func_name);
		if(dfunc_index > 0)
			for(size_t i = 0; i < dfuncs_reduce_outputs[dfunc_index].size; ++i)
				push(src,dfuncs_reduce_outputs[dfunc_index].data[i]);

	}
	if(node->lhs && res == NULL) 
		get_reduce_outputs(node->lhs,src);
	if(node->rhs && res == NULL) 
		get_reduce_outputs(node->rhs,src);
}
void
get_dfuncs_reduce_output(ASTNode* node)
{
	if(node->lhs) 
		get_dfuncs_reduce_output(node->lhs);
	if(node->rhs) 
		get_dfuncs_reduce_output(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	char func_name[5000];
	combine_buffers(node->lhs,func_name);
	const int dfunc_index = str_vec_get_index(dfuncs,func_name);
	get_reduce_outputs(node,&dfuncs_reduce_outputs[dfunc_index]);
}
void
get_dfuncs_reduce_condition(ASTNode* node)
{
	if(node->lhs) 
		get_dfuncs_reduce_condition(node->lhs);
	if(node->rhs) 
		get_dfuncs_reduce_condition(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	char func_name[5000];
	combine_buffers(node->lhs,func_name);
	const int dfunc_index = str_vec_get_index(dfuncs,func_name);
	get_reduce_conditions(node,&dfuncs_reduce_conditions[dfunc_index]);
}
void
get_dfuncs_reduce_op(ASTNode* node)
{
	if(node->lhs) 
		get_dfuncs_reduce_op(node->lhs);
	if(node->rhs) 
		get_dfuncs_reduce_op(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	char func_name[5000];
	combine_buffers(node->lhs,func_name);
	ReduceOp reduce_ops[100];
	int n = 0;
	get_reduce_ops(node,reduce_ops,&n);
	const int dfunc_index = str_vec_get_index(dfuncs,func_name);
	for(int i = 0; i < n; ++i)
		push_op(&dfuncs_reduce_ops[dfunc_index], reduce_ops[i]);
}
int
get_kernel_index(const char* kernel_name)
{
  int index = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
    {
	    if(!strcmp(kernel_name,symbol_table[i].identifier))
		    return index;
	    ++index;
    }
	    
  return -1;
}
void
gen_kernel_postfixes(ASTNode* node, const bool gen_mem_accesses)
{
	if(node->lhs)
		gen_kernel_postfixes(node->lhs,gen_mem_accesses);
	if(node->rhs)
		gen_kernel_postfixes(node->rhs,gen_mem_accesses);
	if(!(node->type & NODE_KFUNCTION))
		return;
	ASTNode* compound_statement = node->rhs->rhs;
	char* new_postfix = malloc(sizeof(char)*4000);
	sprintf(new_postfix,"%s",compound_statement->postfix);
	if(gen_mem_accesses)
	{
	  strcat(new_postfix,"}");
	  compound_statement->postfix = strdup(new_postfix);
	  return;
	}
	ReduceOp reduce_ops[100];
	int n_ops = 0;
	get_reduce_ops(node,reduce_ops,&n_ops);
	if(!n_ops)
	{
	  strcat(new_postfix,"}");
	  compound_statement->postfix = strdup(new_postfix);
	  return;
	}
	const int kernel_index = get_kernel_index(get_node(NODE_KFUNCTION_ID,node)->buffer);
	const ASTNode* fn_identifier = get_node(NODE_KFUNCTION_ID,node);
	string_vec conditions; 	
	init_str_vec(&conditions);
	get_reduce_conditions(node,&conditions);
	if(conditions.size == 0) return;
	string_vec reduce_outputs;
	init_str_vec(&reduce_outputs);
	get_reduce_outputs(node,&reduce_outputs);
	if(reduce_outputs.size == 0) return;
	assert((size_t) n_ops == reduce_outputs.size && (size_t) n_ops == conditions.size);
	char* tmp = malloc(sizeof(char)*4096);
	char* res_name   = malloc(sizeof(char)*4096);
	char* output_str = malloc(sizeof(char)*4096);
#if AC_USE_HIP
	const char* shuffle_instruction = "rocprim::warp_shuffle_down(";
	const char* warp_size  = "const size_t warp_size = rocprim::warp_size();\n";
	const char* warp_id= "const size_t warp_id = rocprim::warp_id();\n";
#else
	const char* shuffle_instruction = "__shfl_down_sync(0xffffffff,";
	const char* warp_size  = "constexpr size_t warp_size = 32;\n";
	const char* warp_id= "const size_t warp_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) / warp_size\n;";
#endif

	for(int i = 0; i < n_ops; ++i)
	{
		ReduceOp reduce_op = reduce_ops[i];
		char* condition = conditions.data[i];
		char* output = reduce_outputs.data[i];
		push_op(&kernel_reduce_ops[kernel_index],  reduce_op);
		push(&kernel_reduce_outputs[kernel_index],  output);
	 	//HACK!
	 	if(!strstr(condition,fn_identifier->buffer))
	 	{
	 	        char* ptr = strtok(condition, "==");
	 	        char* ptr2 = strtok(NULL, "==");
	 	        char new_condition[4096];
	 	        sprintf(new_condition, "vba.kernel_input_params.%s.%s == %s",fn_identifier->buffer,ptr,ptr2);
	 	        condition = strdup(new_condition);
	 	}
	 	sprintf(tmp,"if(%s){\n",condition);
	 	strcat(new_postfix,tmp);
		strcat(new_postfix,warp_size);
		strcat(new_postfix,warp_id);
  	 	strcat(new_postfix,"const size_t lane_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) % warp_size\n;");
  	 	strcat(new_postfix,"const int warps_per_block = (blockDim.x*blockDim.y*blockDim.z + warp_size -1)/warp_size;\n");
  	 	strcat(new_postfix,"const int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;\n");
  	 	strcat(new_postfix,"const int out_index =  vba.reduce_offset + warp_id + block_id*warps_per_block;\n");
	 	strcat(new_postfix,"for(int offset = warp_size/2; offset > 0; offset /= 2){ \n");
		char* array_name;
	 	switch(reduce_op)
	 	{
		 	case(REDUCE_SUM):
				array_name = "reduce_sum_res";
				break;
		 	case(REDUCE_MIN):
				array_name = "reduce_min_res";
				break;
		 	case(REDUCE_MAX):
				array_name = "reduce_max_res";
				break;
		 	case(NO_REDUCE):
				printf("WRONG!\n");
				printf("%s\n",fn_identifier->buffer);
      				exit(EXIT_FAILURE);
		}
		sprintf(res_name,"%s[(int)%s]",array_name,output);
	 	switch(reduce_op)
	 	{
		 	case(REDUCE_SUM):

				sprintf(tmp,"%s += %s%s,offset);\n",res_name,shuffle_instruction,res_name);
		 		strcat(new_postfix,tmp); 
				break;
		 	case(REDUCE_MIN):
				sprintf(tmp,"const AcReal shuffle_tmp = %s%s,offset);",shuffle_instruction,res_name);
				strcat(new_postfix,tmp);
				sprintf(tmp,"%s = (shuffle_tmp < %s) ? shuffle_tmp : %s;\n",res_name,res_name,res_name);
				strcat(new_postfix,tmp);
				break;
		 	case(REDUCE_MAX):
				sprintf(tmp,"const AcReal shuffle_tmp = %s%s,offset);",shuffle_instruction,res_name);
				strcat(new_postfix,tmp);
				sprintf(tmp,"%s = (shuffle_tmp > %s) ? shuffle_tmp : %s;\n",res_name,res_name,res_name);
				strcat(new_postfix,tmp);
				break;
		 	case(NO_REDUCE):
				printf("WRONG!\n");
				printf("%s\n",fn_identifier->buffer);
      				exit(EXIT_FAILURE);
	 	}
	 	strcat(new_postfix,"}\n");
		sprintf(output_str, "if(lane_id == 0) {vba.reduce_scratchpads[(int)%s][0][out_index] = %s;}", output, res_name);
	 	strcat(new_postfix,output_str);
		strcat(new_postfix,"}\n");
	}
	strcat(new_postfix,"}");
	compound_statement->postfix = strdup(new_postfix);
	free_str_vec(&conditions);
	free_str_vec(&reduce_outputs);
	free(new_postfix);
	free(tmp);
	free(res_name);
	free(output_str);
}
void
gen_kernel_ifs(ASTNode* node)
{
	if(node->lhs)
		gen_kernel_ifs(node->lhs);
	if(node->rhs)
		gen_kernel_ifs(node->rhs);
	if(!(node->type & NODE_KFUNCTION))
		return;
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,get_node(NODE_KFUNCTION_ID,node)->buffer);
	if(kernel_index == -1)
		return;
	string_vec combination_params = user_kernel_combinatorial_params[kernel_index];
	if(combination_params.size == 0)
		return;
	FILE* fp = fopen(user_kernel_ifs,"a");
	FILE* fp_defs = fopen("user_defines.h","a");
	ASTNode* old_parent = node->parent;
	for(int i = 0; i < user_kernel_num_combinations[kernel_index]; ++i)
	{
		string_vec combinations = user_kernel_combinations[kernel_index][i];
		char res[4096];
		sprintf(res,"if(kernel_enum == KERNEL_%s ",get_node(NODE_KFUNCTION_ID,node)->buffer);
		for(size_t j = 0; j < combinations.size; ++j)
		{
			char tmp[4096];
			sprintf(tmp, " && vba.kernel_input_params.%s.%s ==  %s ",get_node(NODE_KFUNCTION_ID,node)->buffer,combination_params.data[j],combinations.data[j]);
			strcat(res,tmp);
		}
		strcat(res,")\n{\n\t");
		char tmp[4096];
		sprintf(tmp,"return %s_optimized_%d;\n}\n",get_node(NODE_KFUNCTION_ID,node)->buffer,i);
		strcat(res,tmp);
		//sprintf(res,"%sreturn %s_optimized_%d;\n}\n",tmp);
		fprintf(fp_defs,"%s_optimized_%d,",get_node(NODE_KFUNCTION_ID,node)->buffer,i);
		fprintf(fp,"%s",res);
		bool is_left = (old_parent->lhs == node);
		ASTNode* new_parent = astnode_create(NODE_UNKNOWN,node,astnode_dup(node,old_parent));
		char new_name[4096];
		sprintf(new_name,"%s_optimized_%d",get_node(NODE_KFUNCTION_ID,node)->buffer,i);
		((ASTNode*) get_node(NODE_KFUNCTION_ID,new_parent->rhs))->buffer= strdup(new_name);
		new_parent->rhs->parent = new_parent;
		node->parent = new_parent;
		if(is_left)
			old_parent ->lhs = new_parent;
		else
			old_parent ->rhs = new_parent;
		old_parent = new_parent;
	}
	printf("NUM of combinations: %d\n",user_kernel_num_combinations[kernel_index]);
	fclose(fp);
	fprintf(fp_defs,"}\n");
	fclose(fp_defs);
}
void
gen_kernel_input_params(ASTNode* node, bool gen_mem_accesses)
{
	if(node->lhs)
		gen_kernel_input_params(node->lhs,gen_mem_accesses);
	if(node->rhs)
		gen_kernel_input_params(node->rhs,gen_mem_accesses);
	if(!(node->type & NODE_INPUT && node->buffer))
		return;
	if(gen_mem_accesses)
	{
		strcat(node->buffer,"AC_INTERNAL_INPUT");
		return;
	}

	const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
	if(!begin_scope)
		return;
	const ASTNode* fn_declaration= begin_scope->parent->parent->lhs;
	if(!fn_declaration)
		return;
	const ASTNode* fn_identifier = get_node(NODE_KFUNCTION_ID,fn_declaration);
	while(!fn_identifier)
	{
		begin_scope = get_parent_node(NODE_BEGIN_SCOPE,fn_declaration);
		if(!begin_scope)
			return;
		fn_declaration= begin_scope->parent->parent->lhs;
		if(!fn_declaration)
			return;
		fn_identifier = get_node(NODE_KFUNCTION_ID,fn_declaration);
	}
	char* kernel_name = strdup(fn_identifier->buffer);
	const int combinations_index = get_suffix_int(kernel_name,"_optimized_");
	remove_suffix(kernel_name,"_optimized_");
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,kernel_name);

	if(combinations_index == -1)
	{
	  	char res[4096];
	  	sprintf(res,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
	  	node->buffer = strdup(res);
		return;
	}
	const string_vec combinations = user_kernel_combinations[kernel_index][combinations_index];
	char res[4096];
	char full_name[4096];
	if(node->parent->parent->parent->rhs)
	{
		char member_str[4096];
		member_str[0] = '\0';
		combine_buffers(node->parent->parent->parent->rhs,member_str);
		sprintf(full_name,"%s.%s",node->buffer,member_str);
	}
	else
	{
		sprintf(full_name,"%s",node->buffer);
	}
	const int param_index = str_vec_get_index(user_kernel_combinatorial_params[kernel_index],full_name);
	if(param_index >= 0)
	{
	       sprintf(res,"%s",combinations.data[param_index]);
	       node->parent->parent->parent->buffer = strdup(res);
	       node->parent->parent->parent->buffer = strdup(res);

	       node->parent->parent->parent->infix= NULL;
	       node->parent->parent->parent->lhs = NULL;
	       node->parent->parent->parent->rhs = NULL;
	       ASTNode* if_statement = (ASTNode*) get_parent_node(NODE_IF,node);
	       if(if_statement)
		       if_statement->prefix= strdup(" constexpr (");
	}
	else
	{
		sprintf(res,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
		node->buffer = strdup(res);
	}
}

static void
traverse(const ASTNode* node, const NodeType exclude, FILE* stream)
{
  if (node->type & exclude)
	  stream = NULL;

  // Do not translate tqualifiers or tspecifiers immediately
  if (node->parent &&
      (node->parent->type & NODE_TQUAL || node->parent->type & NODE_TSPEC))
    return;

  // Prefix translation
  // Prefix translation
  if (stream)
    if (node->prefix)
      fprintf(stream, "%s", node->prefix);

  // Prefix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest < MAX_NESTS);

    ++current_nest;
    num_symbols[current_nest] = num_symbols[current_nest - 1];
  }

  // Traverse LHS
  if (node->lhs)
    traverse(node->lhs, exclude, stream);

  // Add symbols to symbol table
  if (node->buffer && node->token == IDENTIFIER) {
    const Symbol* symbol = symboltable_lookup(node->buffer);
    if (symbol && node->type & NODE_FUNCTION_PARAM) {
      // Do not allow shadowing.
      //
      // Note that if we want to allow shadowing, then the symbol table must
      // be searched in reverse order
      fprintf(stderr,
              "Error! Symbol '%s' already present in symbol table. Shadowing "
              "is not allowed.\n",
              node->buffer);
      assert(0);
    }
    else if (!symbol) {
      char* tspec = NULL;
      char* tqualifiers[MAX_ID_LEN];
      size_t n_tqualifiers = 0;

      const ASTNode* decl = get_parent_node(NODE_DECLARATION, node);
      if (decl) {
        const ASTNode* tspec_node = get_node(NODE_TSPEC, decl);
        const ASTNode* tqual_node = get_node(NODE_TQUAL, decl);

        if (tspec_node && tspec_node->lhs){
          tspec = tspec_node->lhs->buffer;
        }
        if (tqual_node && tqual_node->lhs)
	{
	  const ASTNode* tqual_list_node = tqual_node->parent;
	  //backtrack to the start of the list
	  while(tqual_list_node->parent && tqual_list_node->parent->rhs && tqual_list_node->parent->rhs->type & NODE_TQUAL)
		  tqual_list_node = tqual_list_node->parent;
	  while(tqual_list_node->rhs)
	  {
		  tqualifiers[n_tqualifiers] = strdup(tqual_list_node->rhs->lhs->buffer);
		  ++n_tqualifiers;
		  tqual_list_node = tqual_list_node->lhs;
	  }
	  tqualifiers[n_tqualifiers] = strdup(tqual_list_node->lhs->lhs->buffer);
	  ++n_tqualifiers;
	}
      }

      if (stream) {
        const ASTNode* is_dconst = get_parent_node(NODE_DCONST, node);
        if (is_dconst)
          fprintf(stream, "__device__ ");

        if (n_tqualifiers)
	  for(size_t i=0; i<n_tqualifiers;++i)
          	fprintf(stream, "%s ", tqualifiers[i]);

        if (tspec){
          fprintf(stream, "%s ", tspec);
        }
        else if (!(node->type & NODE_KFUNCTION_ID) &&
                 !get_parent_node(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID) &&
                 !(node->type & NODE_INPUT) &&
                 !(node->type & NODE_ENUM) &&
                 !strstr(node->buffer, "__ldg") &&
                 !str_vec_contains(array_fields,node->buffer))
	{
          fprintf(stream, "auto ");
	  const ASTNode* assign_node = get_parent_node(NODE_ASSIGNMENT,node);
	  if(assign_node)
	  {
	  	const ASTNode* search = get_node_by_id(node->id,assign_node->rhs);
		if(search)
		{
			fprintf(stderr,"Undeclared variable or function used on the right hand side of an assignment: %s\n",node->buffer);
			exit(EXIT_FAILURE);
		}
	  }
	  const ASTNode* func_call_node = get_parent_node(NODE_FUNCTION_CALL,node);
	  if(func_call_node)
	  {
	  	const ASTNode* search = get_node_by_id(node->id,func_call_node->lhs);
		if(search)
		{
			fprintf(stderr,"Undeclared function used: %s\n",node->buffer);
			exit(EXIT_FAILURE);
		}

	  }
	}
      }
      if (!(node->type & NODE_MEMBER_ID))
      {
        const int symbol_index = add_symbol(node->type, tqualifiers, n_tqualifiers, tspec, node->buffer);
	//get array length
        if (tspec != NULL && (!strcmp(tspec,"AcReal*") || !strcmp(tspec,"int*")))
	{
		char array_length_str[4096];
		combine_all(decl->rhs->lhs->rhs,array_length_str);
		symbol_var_length[symbol_index] =  strdup(array_length_str);
	}
      }
    }
  }

  // Infix translation
  if (stream)
    if (node->infix)
      fprintf(stream, "%s", node->infix);

  // Translate buffer body
  if (stream && node->buffer) {
    const Symbol* symbol = symboltable_lookup(node->buffer);
    if (symbol && symbol->type & NODE_DCONST_ID && !str_vec_contains(symbol->tqualifiers,"output"))
      fprintf(stream, "DCONST(%s)", node->buffer);
    else
      fprintf(stream, "%s", node->buffer);
  }

  // Traverse RHS
  if (node->rhs)
    traverse(node->rhs, exclude, stream);

  // Postfix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest > 0);
    --current_nest;
  }

  // Postfix translation
  if (stream) {
    if (node->postfix)
      fprintf(stream, "%s", node->postfix);
  }
}

void
gen_dconsts(const ASTNode* root, FILE* stream)
{
  symboltable_reset();
  traverse(root, NODE_FUNCTION | NODE_VARIABLE | NODE_STENCIL | NODE_HOSTDEFINE | always_excluded,
           stream);

  /*
  symboltable_reset();
  traverse(root, 0, NULL);

  // Device constants
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!(symbol_table[i].type & NODE_FUNCTION_ID) &&
        !(symbol_table[i].type & NODE_FIELD_ID) &&
        !(symbol_table[i].type & NODE_STENCIL_ID)) {
      fprintf(stream, "__device__ %s %s;", symbol_table[i].tspecifier,
              symbol_table[i].identifier);
    }
    */
}

void
gen_const_variables(const ASTNode* node, FILE* fp)
{
	if(node->lhs)
		gen_const_variables(node->lhs,fp);
	if(node->rhs)
		gen_const_variables(node->rhs,fp);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	const ASTNode* tqual= get_node(NODE_TQUAL,node);
	if(!tqual) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	const bool is_const = !strcmp(tqual->lhs->buffer, "const");
	if(!is_const) return;
	const ASTNode* def_list_head = node->rhs;
	while(def_list_head -> rhs)
	{
		const ASTNode* def = def_list_head->rhs;
		const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
		if(!name) return;
        	const ASTNode* assignment = def->rhs;
		if(!assignment) return;
		char assignment_val[4098];
		combine_all(assignment,assignment_val);
		//char datatype_scalar[1000];
		const char* datatype = tspec->lhs->buffer;
		if(strstr(assignment_val,","))
		{
			fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s[] = {%s};\n#endif\n",datatype, name, assignment_val);
		}
		else
		{
			fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s = %s;\n#endif\n",datatype, name, assignment_val);
		}
		def_list_head = def_list_head -> lhs;
	}
	const ASTNode* def = def_list_head->lhs;
	const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
	if(!name) return;
        const ASTNode* assignment = def->rhs;
	if(!assignment) return;
	char assignment_val[4098];
	combine_all(assignment,assignment_val);
	//char datatype_scalar[1000];
	const char* datatype = tspec->lhs->buffer;
	if(strstr(assignment_val,","))
	{
		fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s[] = {%s};\n#endif\n",datatype, name, assignment_val);
	}
	else
	{
		fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s = %s;\n#endif\n",datatype, name, assignment_val);
	}
}

static int curr_kernel = 0;

static void
gen_kernels(const ASTNode* node, const char* dfunctions,
            const bool gen_mem_accesses)
{
  assert(node);

  if (node->type & NODE_KFUNCTION) {

    const size_t len = 64 * 1024 * 1024;
    char* prefix     = malloc(len);
    assert(prefix);
    prefix[0] = '\0';

    assert(node->rhs);
    assert(node->rhs->rhs);
    ASTNode* compound_statement = node->rhs->rhs;

    strcat(prefix, compound_statement->prefix);

    // Generate stencil FMADs
    char cmdoptions[4096] = "\0";
    if (gen_mem_accesses) {
      sprintf(cmdoptions, "./" STENCILGEN_EXEC " -mem-accesses");
    }
    else {
      sprintf(cmdoptions, "./" STENCILGEN_EXEC " -kernel %d", curr_kernel);
      ++curr_kernel; // HACK TODO better
    }
    FILE* proc = popen(cmdoptions, "r");
    assert(proc);

    char* sdefinitions = malloc(10 * 1024 * 1024);
    assert(sdefinitions);
    sdefinitions[0] = '\0';
    char buf[4096]  = {0};
    while (fgets(buf, sizeof(buf), proc))
      strcat(sdefinitions, buf);

    pclose(proc);

    strcat(prefix, sdefinitions);
    free(sdefinitions);

    strcat(prefix, dfunctions);

    astnode_set_prefix(prefix, compound_statement);
    free(prefix);
  }

  if (node->lhs)
    gen_kernels(node->lhs, dfunctions, gen_mem_accesses);

  if (node->rhs)
    gen_kernels(node->rhs, dfunctions, gen_mem_accesses);
}

// Generate User Defines
static void
gen_user_defines(const ASTNode* root, const char* out)
{
  FILE* fp = fopen(out, "w");
  assert(fp);

  fprintf(fp, "#pragma once\n");

  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | always_excluded, fp);

  symboltable_reset();
  traverse(root, 0, NULL);

  // Stencils
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      fprintf(fp, "stencil_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_STENCILS} Stencil;");

  // Enums
  int num_of_normal_fields=0;
  int num_of_auxiliary_fields=0;
  int num_of_communicated_auxiliary_fields = 0;
  int num_of_fields=0;
  bool field_is_auxiliary[256];
  bool field_is_communicated[256];
  init_str_vec(&field_names);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if(!strcmp(symbol_table[i].tspecifier,"Field")){
      push(&field_names, symbol_table[i].identifier);
      if(str_vec_contains(symbol_table[i].tqualifiers,"auxiliary"))
      {
	      ++num_of_auxiliary_fields;
	      field_is_auxiliary[num_of_fields] = true;
	      if(str_vec_contains(symbol_table[i].tqualifiers,"communicated"))
	      {
		      ++num_of_communicated_auxiliary_fields;
      	      		field_is_communicated[num_of_fields] = true;
	      }
	      else
      	      		field_is_communicated[num_of_fields] = false;

      }
      else
      {
	      ++num_of_normal_fields;
      	      field_is_communicated[num_of_fields] = true;
	      field_is_auxiliary[num_of_fields] = false;
      }
      ++num_of_fields;
    }
  }
  const int num_of_communicated_fields = num_of_normal_fields + num_of_communicated_auxiliary_fields;

  fprintf(fp, "typedef enum {");
  //first communicated fields
  for(int i=0;i<num_of_fields;++i)
	  if(field_is_communicated[i]) fprintf(fp, "%s,",field_names.data[i]);
  for(int i=0;i<num_of_fields;++i)
	  if(!field_is_communicated[i]) fprintf(fp, "%s,",field_names.data[i]);

  fprintf(fp, "NUM_FIELDS=%d,", num_of_fields);
  fprintf(fp, "NUM_COMMUNICATED_FIELDS=%d,", num_of_communicated_fields);
  fprintf(fp, "} Field;\n");

  fprintf(fp, "static const bool vtxbuf_is_auxiliary[] = {");
  for(int i=0;i<num_of_fields;++i)
    if(field_is_auxiliary[i])
    	fprintf(fp, "%s,", "true");
    else
    	fprintf(fp, "%s,", "false");
  fprintf(fp, "};");






  // Enums for work_buffers 
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"WorkBuffer"))
    {
      printf("Workbuffers are under development\n");
      exit(EXIT_FAILURE);
      fprintf(fp, "%s,", symbol_table[i].identifier);
    }
  fprintf(fp, "NUM_WORK_BUFFERS} WorkBuffer;");





  // Kernels
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "KERNEL_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_KERNELS} AcKernel;");

  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  // START---------------------------


  // Enum strings (convenience)
  fprintf(fp, "static const char* stencil_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* field_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"Field"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");





  fprintf(fp, "static const char* work_buffer_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"WorkBuffer"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");




  fprintf(fp, "static const char* kernel_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  const char* scalar_datatypes[] = {"int","AcReal","int3","AcReal3"};
  for (size_t i = 0; i < sizeof(scalar_datatypes)/sizeof(scalar_datatypes[0]); ++i) {
	  gen_param_names(fp,scalar_datatypes[i],false);
	  gen_enums(fp,scalar_datatypes[i],false);
  }

  const char* array_datatypes[] = {"int","AcReal"};
  for (size_t i = 0; i < sizeof(array_datatypes)/sizeof(array_datatypes[0]); ++i) {
  	gen_array_lengths(fp,array_datatypes[i]);
  	gen_array_is_dconst(fp,array_datatypes[i]);
  	gen_d_offsets(fp,array_datatypes[i],false);
  	gen_d_offsets(fp,array_datatypes[i],true);
	gen_param_names(fp,array_datatypes[i],true);
	gen_enums(fp,array_datatypes[i],true);
  }

  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  fprintf(fp, "\n// Redefined for backwards compatibility START\n");
  fprintf(fp, "#define NUM_VTXBUF_HANDLES (NUM_FIELDS)\n");
  fprintf(fp, "typedef Field VertexBufferHandle;\n");
  fprintf(fp, "static const char** vtxbuf_names = field_names;\n");
  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  // END-----------------------------

  // Device constants
  // Would be cleaner to declare dconsts as extern and refer to the symbols
  // directly instead of using handles like above, but for backwards
  // compatibility and user convenience commented out for now
  //
  //for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
  //  if (!(symbol_table[i].type & NODE_FUNCTION_ID) &&
  //      !(symbol_table[i].type & NODE_VARIABLE_ID) &&
  //      !(symbol_table[i].type & NODE_STENCIL_ID)) {
  //    fprintf(fp, "// extern __device__ %s %s;\n", symbol_table[i].tspecifier,
  //            symbol_table[i].identifier);
  //  }
  //}

  // Stencil order
  fprintf(fp, "#ifndef STENCIL_ORDER\n");
  fprintf(fp, "#define STENCIL_ORDER (6)\n");
  fprintf(fp, "#endif\n");
  fprintf(fp, "#define STENCIL_DEPTH (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define STENCIL_HEIGHT (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define STENCIL_WIDTH (STENCIL_ORDER+1)\n");
  char cwd[10004];
  cwd[0] = '\0';
  char* err = getcwd(cwd, sizeof(cwd));
  assert(err != NULL);
  char autotune_path[10004];
  sprintf(autotune_path,"%s/autotune.csv",cwd);
  fprintf(fp,"__attribute__((unused)) static const char* autotune_csv_path= \"%s\";\n",autotune_path);
  fclose(fp);
  //Done to refresh the autotune file when recompiling DSL code
  fp = fopen(autotune_path,"w");
  fclose(fp);

  symboltable_reset();
  fp = fopen("user_constants.h","w");
  gen_const_variables(root,fp);
  fclose(fp);
}


static void
gen_user_kernels(const ASTNode* root, const char* out, const bool gen_mem_accesses)
{
  symboltable_reset();
  traverse(root, always_excluded, NULL);

  FILE* fp = fopen(out, "w");
  assert(fp);
  // fprintf(fp, "#pragma once\n");

  // Kernels
  // for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  //   if (symbol_table[i].type & NODE_KFUNCTION_ID)
  //     fprintf(fp,
  //             "__global__ void %s(const int3 start, const int3 end, "
  //             "VertexBufferArray vba);",
  //             symbol_table[i].identifier);

  // Astaroth 2.0 backwards compatibility START
  // Handles are now used to get optimized kernels for specific input param combinations
  fprintf(fp,"#include \"user_kernel_declarations.h\"\n");
  fprintf(fp, "static const Kernel kernels[] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "%s,", symbol_table[i].identifier); // Host layer handle
  fprintf(fp, "};");

  const char* default_param_list=  "(const int3 start, const int3 end, VertexBufferArray vba";
  FILE* fp_dec = fopen("user_kernel_declarations.h","a");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp_dec, "void __global__ %s %s);\n", symbol_table[i].identifier, default_param_list);
  fclose(fp_dec);

  if(gen_mem_accesses)
  { 
    FILE* fp_cpu = fopen("user_cpu_kernels.h","a");
    fprintf(fp_cpu, "static const Kernel cpu_kernels[] = {");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i)
      if (symbol_table[i].type & NODE_KFUNCTION_ID)
        fprintf(fp_cpu, "%s_cpu,", symbol_table[i].identifier); // Host layer handle
    fprintf(fp_cpu, "};");
    fclose(fp_cpu);
  }
  // Astaroth 2.0 backwards compatibility END

  fclose(fp);

  symboltable_reset();
}

bool is_real_constant(const char* name)
{
  {
  const int l_current_nest = 0;
  for (size_t i = 0; i < num_symbols[l_current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal") && !strcmp(name,symbol_table[i].identifier))
        return true;
  }
  return false;
}
bool is_int_constant(const char* name)
{
  {
  const int l_current_nest = 0;
  for (size_t i = 0; i < num_symbols[l_current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "int") && !strcmp(name,symbol_table[i].identifier))
        return true;
  }
  return false;
}
void
replace_dynamic_coeffs_stencilpoint(ASTNode* node)
{
  if(node->lhs)
    replace_dynamic_coeffs_stencilpoint(node->lhs);
  if(node->buffer)
  {
    if(is_real_constant(node->buffer) || is_int_constant(node->buffer))
    {
      //replace with zero to compile the stencil
      node->buffer = strdup("0.0");
      node->prefix=strdup("AcReal(");
      node->postfix = strdup(")");
    }
  }
  if(node->rhs)
    replace_dynamic_coeffs_stencilpoint(node->rhs);
}
void replace_dynamic_coeffs(ASTNode* node)
{
  if(node->type & NODE_STENCIL)
  {
    ASTNode* list = node->rhs->lhs;
    while(list->rhs)
    {
      ASTNode* stencil_point = list->rhs;
      replace_dynamic_coeffs_stencilpoint(stencil_point->rhs->rhs);
      list = list -> lhs;
    }
    ASTNode* stencil_point = list->lhs;
    replace_dynamic_coeffs_stencilpoint(stencil_point->rhs->rhs);
  }
  if(node->lhs)
    replace_dynamic_coeffs(node->lhs);
  if(node->rhs)
    replace_dynamic_coeffs(node->rhs);
}
void
read_codegen_input(const ASTNode* node)
{
	//currently CODEGEN_INPUT + NODE_VARIABLE_ID = field array name
	if(node->type == (NODE_CODEGEN_INPUT))
	{
		push_int(&array_field_sizes,atoi(node->lhs->buffer));
		push(&array_fields,node->buffer);
	}
	if(node->lhs)
		read_codegen_input(node->lhs);
	if(node->rhs)
		read_codegen_input(node->rhs);
}
void
generate(const ASTNode* root_in, FILE* stream, const bool gen_mem_accesses, const bool optimize_conditionals)
{ 
  //ASTNode* root = astnode_dup(root_in,NULL);
  ASTNode* root = (ASTNode*) root_in;
  init_str_vec(&array_fields);
  init_int_vec(&array_field_sizes);
  init_str_vec(&user_kernels_with_input_params);
  init_str_vec(&user_enums);
  init_str_vec(&user_structs);
  init_str_vec(&dfuncs);
  for(int i = 0; i <1000; ++i)
  {
	  init_str_vec(&dfuncs_reduce_outputs[i]);
	  init_str_vec(&dfuncs_reduce_conditions[i]);
	  init_op_vec(&dfuncs_reduce_ops[i]);
  }
  for(int i=0; i<100;++i)
  {
	  init_str_vec(&user_enum_options[i]);

	  init_str_vec(&user_struct_field_types[i]);
	  init_str_vec(&user_struct_field_names[i]);

	  user_kernel_num_combinations[i] = 0;
	  init_str_vec(&user_kernel_combinatorial_params[i]);

	  for(int j=0;j<100;++j)
	  	  init_str_vec(&user_kernel_combinatorial_param_options[i][j]);
	  init_op_vec(&kernel_reduce_ops[i]);
	  init_str_vec(&kernel_reduce_outputs[i]);
  }
  assert(root);
  read_user_structs(root);
  read_user_enums(root);

  gen_kernel_num_of_combinations(root);
  if(optimize_conditionals)
  {
  	gen_kernel_ifs(root);
  }
  gen_kernel_input_params(root,gen_mem_accesses);
  gen_user_defines(root, "user_defines.h");
  gen_field_accesses(root);
  gen_user_kernels(root, "user_declarations.h", gen_mem_accesses);

  // Fill the symbol table
  traverse(root, always_excluded, NULL);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DFUNCTION_ID)
	    push(&dfuncs, symbol_table[i].identifier);
  get_dfuncs_reduce_output(root);
  get_dfuncs_reduce_condition(root);
  get_dfuncs_reduce_op(root);

  gen_kernel_postfixes(root,gen_mem_accesses);
  // print_symbol_table();

  // Generate user_kernels.h
  fprintf(stream, "#pragma once\n");

  size_t num_stencils = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      ++num_stencils;

  size_t num_fields = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"Field"))
      ++num_fields;

  size_t num_kernels = 0;
  FILE* fp = fopen("user_defines.h","a");

  int num_real_reduce_output = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, "AcReal") && str_vec_contains(symbol_table[i].tqualifiers,"output"))
	    ++num_real_reduce_output;
  fprintf(fp,"%s","static const int kernel_reduce_outputs[NUM_KERNELS][NUM_REAL_OUTPUTS] = { ");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
    {
      fprintf(fp,"%s","{");
      for(int j = 0; j < num_real_reduce_output; ++j)
      {
      	if(kernel_reduce_outputs[num_kernels].size < (size_t) j+1)
	        fprintf(fp,"%d,",-1);
      	else
	      	fprintf(fp,"(int)%s,",kernel_reduce_outputs[num_kernels].data[j]);
      }
      fprintf(fp,"%s","},");
      ++num_kernels;
    }
  fprintf(fp,"%s","};\n");

  fprintf(fp,"%s","typedef enum KernelReduceOp\n{\n\tNO_REDUCE,\n\tREDUCE_MIN,\n\tREDUCE_MAX,\n\tREDUCE_SUM,\n} KernelReduceOp;\n");
  fprintf(fp,"%s","static const KernelReduceOp kernel_reduce_ops[NUM_KERNELS][NUM_REAL_OUTPUTS] = { ");
  int iterator = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
    {
      fprintf(fp,"%s","{");
      for(int j = 0; j < num_real_reduce_output; ++j)
      {

      	if(kernel_reduce_ops[iterator].size < (size_t) i+1)
        	fprintf(fp,"%s,","NO_REDUCE");
	else
	{
      		switch(kernel_reduce_ops[iterator].data[j])
      		{
      		        case(NO_REDUCE):
      		        	fprintf(fp,"%s,","NO_REDUCE");
      		        	break;
      		        case(REDUCE_MIN):
      		        	fprintf(fp,"%s,","REDUCE_MIN");
      		        	break;
      		        case(REDUCE_MAX):
      		        	fprintf(fp,"%s,","REDUCE_MAX");
      		        	break;
      		        case(REDUCE_SUM):
      		        	fprintf(fp,"%s,","REDUCE_SUM");
      		        	break;
      		}
	}
      }
      fprintf(fp,"%s","},");
      ++iterator;
    }
  fprintf(fp,"%s","};\n");
  fclose(fp);



  // Device constants
  // gen_dconsts(root, stream);
  read_codegen_input(root);
  fp = fopen("user_defines.h","a");
  gen_field_array_declarations(fp);
  fclose(fp);
  const char* array_datatypes[] = {"int","AcReal"};
  for (size_t i = 0; i < sizeof(array_datatypes)/sizeof(array_datatypes[0]); ++i)
  	gen_array_reads(root,gen_mem_accesses,array_datatypes[i]);

  // Stencils

  // Stencil generator
  FILE* stencilgen = fopen(STENCILGEN_HEADER, "w");
  assert(stencilgen);

  // Stencil ops
  symboltable_reset();
  traverse(root, always_excluded, NULL);
  { // Unary (non-functional, default string 'val')
    fprintf(stencilgen,
            "static const char* stencil_unary_ops[NUM_STENCILS] = {");
    for (size_t i = 0; i < num_stencils; ++i)
      fprintf(stencilgen, "\"val\",");
    fprintf(stencilgen, "};");
  }

  { // Binary
    fprintf(stencilgen, "static const char* "
                        "stencil_binary_ops[NUM_STENCILS] = {");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
      const Symbol symbol = symbol_table[i];
      if (symbol.type & NODE_STENCIL_ID) {
	      if(symbol.tqualifiers.size)
	      {
		if(symbol.tqualifiers.size > 1)
		{
			fprintf(stderr,"Stencils are supported only with a single type specifier\n");
			exit(EXIT_FAILURE);
		}
        	fprintf(stencilgen, "\"%s\",",symbol.tqualifiers.data[0]);
	      }
	      else
        	fprintf(stencilgen, "\"sum\",");
      }
    }
    fprintf(stencilgen, "};");
  }

  // Stencil coefficients
  symboltable_reset();
  char* stencil_coeffs;
  size_t file_size;
  FILE* stencil_coeffs_fp = open_memstream(&stencil_coeffs, &file_size);
  traverse(root,
           NODE_STENCIL_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
               NODE_HOSTDEFINE | always_excluded,
           stencil_coeffs_fp);
  fflush(stencil_coeffs_fp);

  replace_dynamic_coeffs(root);
  symboltable_reset();
  fprintf(stencilgen, "static char* "
                      "dynamic_coeffs[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = { %s };\n", stencil_coeffs);
  fprintf(stencilgen, "static char* "
                      "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = {");
  traverse(root,
           NODE_STENCIL_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
               NODE_HOSTDEFINE | always_excluded,
           stencilgen);
  fprintf(stencilgen, "};");
  fclose(stencilgen);

  // Compile
  if (gen_mem_accesses || !OPTIMIZE_MEM_ACCESSES) {
    FILE* tmp = fopen("stencil_accesses.h", "w+");
    assert(tmp);
    fprintf(tmp,
            "static int "
            "stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] = {");
    for (size_t i = 0; i < num_kernels; ++i)
      for (size_t j = 0; j < num_fields; ++j)
        for (size_t k = 0; k < num_stencils; ++k)
          fprintf(tmp, "[%lu][%lu][%lu] = 1,", i, j, k);
    fprintf(tmp, "};");
    fprintf(tmp,
            "static int "
            "previous_accessed[NUM_KERNELS][NUM_FIELDS] = {");
    for (size_t i = 0; i < num_kernels; ++i)
      for (size_t j = 0; j < num_fields; ++j)
          fprintf(tmp, "[%lu][%lu] = 1,", i, j);
    fprintf(tmp, "};");
    fclose(tmp);
  }
  /*
  else {
    FILE* tmp = fopen("stencil_accesses.h", "r");
    if (!tmp) {
      tmp = fopen("stencil_accesses.h", "w+");
      assert(tmp);
      fprintf(tmp,
              "static int "
              "stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] = {");
      for (size_t i = 0; i < num_kernels; ++i)
        for (size_t j = 0; j < num_fields; ++j)
          for (size_t k = 0; k < num_stencils; ++k)
            fprintf(tmp, "[%lu][%lu][%lu] = 1,", i, j, k);
      fprintf(tmp, "};");
    }
    fclose(tmp);
  }
  */

  char build_cmd[4096];
  snprintf(build_cmd, 4096,
           "gcc -std=c11 -Wfatal-errors -Wall -Wextra -Wdouble-promotion "
           "-DIMPLEMENTATION=%d "
           "-DMAX_THREADS_PER_BLOCK=%d "
           "-Wfloat-conversion -Wshadow -I. %s -lm "
           "-o %s",
           IMPLEMENTATION, MAX_THREADS_PER_BLOCK, STENCILGEN_SRC,
           STENCILGEN_EXEC);

  const int retval = system(build_cmd);

  if (retval == -1) {
    while (1)
      fprintf(stderr,
              "Catastrophic error: could not compile the stencil generator.\n");
    assert(retval != -1);
    exit(EXIT_FAILURE);
  }

  // Generate stencil definitions
  FILE* proc = popen("./" STENCILGEN_EXEC " -definitions", "r");
  assert(proc);

  char buf[4096] = {0};
  while (fgets(buf, sizeof(buf), proc))
    fprintf(stream, "%s", buf);

  pclose(proc);

  // Device functions
  symboltable_reset();
  char* dfunctions;
  size_t sizeloc;
  FILE* dfunc_fp = open_memstream(&dfunctions, &sizeloc);
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_KFUNCTION |
               NODE_HOSTDEFINE | always_excluded,
           dfunc_fp);
  fflush(dfunc_fp);

  // Kernels
  symboltable_reset();
  gen_kernels(root, dfunctions, gen_mem_accesses);
  fclose(dfunc_fp); // Frees dfunctions also

  symboltable_reset();
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE | always_excluded,
           stream);

  // print_symbol_table();
}

void
generate_mem_accesses(void)
{
  // Generate memory accesses to a header
  printf("Compiling %s...\n", STENCILACC_SRC);
#if AC_USE_HIP
  printf("--- USE_HIP: `%d`\n", AC_USE_HIP);
#else
  printf("--- USE_HIP not defined\n");
#endif
  printf("--- ACC_RUNTIME_API_DIR: `%s`\n", ACC_RUNTIME_API_DIR);
  printf("--- GPU_API_INCLUDES: `%s`\n", GPU_API_INCLUDES);

  char cmd[4096];
  sprintf(cmd, "gcc -Wshadow -I. ");
  strcat(cmd, "-I " ACC_RUNTIME_API_DIR " ");
  if (strlen(GPU_API_INCLUDES) > 0)
    strcat(cmd, " -I " GPU_API_INCLUDES " ");
#if AC_USE_HIP
  strcat(cmd, "-DAC_USE_HIP=1 ");
#endif
  strcat(cmd, STENCILACC_SRC " -lm -lstdc++ -o " STENCILACC_EXEC " ");

  /*
  const char* cmd = "gcc -Wshadow -I. "
#if AC_USE_HIP
                    "-DAC_USE_HIP=1 "
#endif
                    "-I " GPU_API_INCLUDES " "    //
                    "-I " ACC_RUNTIME_API_DIR " " //
      STENCILACC_SRC " -lm "
                    "-o " STENCILACC_EXEC;
  */
  printf("Compile command: %s\n", cmd);
  const int retval = system(cmd);
  if (retval != 0) {
    fprintf(stderr, "Catastrophic error: could not compile the stencil access "
                    "generator.\n");
    fprintf(stderr, "Compiler error code: %d\n",retval);
    assert(retval != 0);
    exit(EXIT_FAILURE);
  }
  printf("%s compilation done\n", STENCILACC_SRC);
  // Generate stencil accesses
  FILE* proc = popen("./" STENCILACC_EXEC " stencil_accesses.h", "r");
  assert(proc);
  pclose(proc);
}
