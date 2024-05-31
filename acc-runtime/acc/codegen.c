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
static inline char*
strupr(const char* src)
{
	char* res = strdup(src);
	int index = -1;
	while(res[++index] != '\0')
		res[index] = toupper(res[index]);
	return res;
}
static int* written_fields = NULL;
static int* read_fields   = NULL;
static int* field_has_stencil_op = NULL;
static size_t num_fields = 0;
static size_t num_kernels = 0;

#define STENCILGEN_HEADER "stencilgen.h"
#define STENCILGEN_SRC ACC_DIR "/stencilgen.c"
#define STENCILGEN_EXEC "stencilgen.out"
#define STENCILACC_SRC ACC_DIR "/stencil_accesses.cpp"
#define STENCILACC_EXEC "stencil_accesses.out"
#define ACC_RUNTIME_API_DIR ACC_DIR "/../api"
//



#define MAX_KERNELS (100)
#define MAX_COMBINATIONS (1000)
#define MAX_DFUNCS (1000)
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

#define MAX_NESTS (32)
static size_t num_symbols[MAX_NESTS] = {};
static size_t current_nest           = 0;


//arrays symbol table
#define MAX_NUM_ARRAYS (256)

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
get_symbol_index(const NodeType type, const char* symbol, const char* tspecifier)
{

  int counter = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if (symbol_table[i].type & type && (!tspecifier || !strcmp(symbol_table[i].tspecifier,tspecifier)))
    {
	    if(!strcmp(symbol_table[i].identifier,symbol))
		    return counter;
	    counter++;
    }
  }
  return -1;
}

static const Symbol*
get_symbol(const NodeType type, const int index, const char* tspecifier)
{

  int counter = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if (symbol_table[i].type & type && (!tspecifier || !strcmp(symbol_table[i].tspecifier,tspecifier)))
    {
	    if(counter == index)
		    return &symbol_table[i];
	    counter++;
    }
  }
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
  const bool is_field_without_type_qualifiers = tspecifier && !strcmp(tspecifier,"Field") && symbol_table[num_symbols[current_nest]].tqualifiers.size == 0;
  const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
  if(!is_field_without_type_qualifiers || !has_optimization_info)
  	return num_symbols[current_nest]-1;


   const int field_index = get_symbol_index(NODE_VARIABLE_ID, id, "Field");
   bool is_auxiliary = true;
   bool is_communicated = false;
   for(size_t k = 0; k < num_kernels; ++k)
   {
	   is_auxiliary &= (!written_fields[field_index + num_fields*k] || !field_has_stencil_op[field_index + num_fields*k]);
	   is_communicated &= !field_has_stencil_op[field_index + num_fields*k];
   }
   if(is_auxiliary)
   {
	   push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, "auxiliary");
	   if(is_communicated)
	   	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, "communicated");
   }

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
void
get_array_var_length(const char* var, const ASTNode* root, char* dst)
{
	    ASTNode* var_identifier = get_node_by_buffer_and_type(var,NODE_VARIABLE_ID,root);
	    const ASTNode* decl = get_parent_node(NODE_DECLARATION,var_identifier);
	    assert(decl);
	    combine_all(decl->rhs->lhs->rhs,dst);
}
void
gen_d_offsets(FILE* fp, const char* datatype_scalar, const bool declarations, const ASTNode* root)
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

	    char array_length_str[4098];
	    get_array_var_length(symbol_table[i].identifier,root,array_length_str);
            sprintf(running_offset,"%s+%s",running_offset,array_length_str);
    }
  }
   if(declarations)
        fprintf(fp,"\n#ifndef D_%s_ARRAYS_LEN\n#define D_%s_ARRAYS_LEN (%s)\n#endif\n", strupr(define_name), strupr(define_name),running_offset);
   else
        fprintf(fp, "};");
}

void
gen_array_lengths(FILE* fp, const char* datatype_scalar, const ASTNode* root)
{
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  fprintf(fp, "static const int %s_array_lengths[] __attribute__((unused)) = {", convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
	char array_length_str[4098];
	get_array_var_length(symbol_table[i].identifier,root,array_length_str);

  	if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
      		fprintf(fp, "%s,", array_length_str);
	else if(!str_vec_contains(symbol_table[i].tqualifiers,"const"))
      		fprintf(fp, "(int)%s,", array_length_str);
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
        !strcmp(symbol_table[i].tspecifier, datatype) && !str_vec_contains(symbol_table[i].tqualifiers,"const"))
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
  	fprintf(fp, "NUM_%s_OUTPUT_ARRAYS} %sArrayOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));
  else
  	fprintf(fp, "NUM_%s_OUTPUTS} %sOutputParam;",strupr(convert_to_define_name(datatype)),convert_to_enum_name(datatype));
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

bool
check_symbol(const NodeType type, const char* name, const char* tspecifier, const char* tqualifier)
{
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (symbol_table[i].type & type && !strcmp(symbol_table[i].identifier,name) 
        && (!tqualifier || str_vec_contains(symbol_table[i].tqualifiers,tqualifier)) && (!tspecifier || !strcmp(symbol_table[i].tspecifier,tspecifier)))
	    return true;
  return false;
}


bool
check_for_vtxbuf(const ASTNode* node)
{
	//global vtxbuf
	if(check_symbol(NODE_VARIABLE_ID, node->buffer, "VertexBufferHandle", NULL) || check_symbol(NODE_VARIABLE_ID, node->buffer, "VertexBufferHandle*", NULL))
		return true;
	const ASTNode* func = get_parent_node(NODE_FUNCTION,node);
	if(!func)
		return false;
	const ASTNode* param_list = func->rhs->lhs;
	if(!param_list)
		return false;
	char* kernel_search_buffer = strdup(node->buffer);;
	//remove internal input string in case it is a kernel input when doing mem accesses
	remove_substring(kernel_search_buffer,"AC_INTERNAL_INPUT");
	const ASTNode* search = get_node_by_buffer(kernel_search_buffer,param_list);
	free(kernel_search_buffer);
	if(!search)
		return false;
	//make sure this version is not in the param list since otherwise breaks stuff
	const ASTNode* param_search = get_node_by_id(node->id,param_list);
	if(param_search)
		return false;
	const ASTNode* tspec = get_node(NODE_TSPEC, search->parent->lhs);
	if(!tspec)
		return false;
	const bool is_vtxbuf = !strcmp(tspec->lhs->buffer,"VertexBufferHandle");
	if(!is_vtxbuf)
		return false;
        if(!(func->type & NODE_KFUNCTION))
                return false;
	return true;
}
ASTNode* 
gen_3d_array_access(ASTNode* node)
{
	char* x_index = malloc(sizeof(char)*1000);
	char* y_index = malloc(sizeof(char)*1000);
	char* z_index = malloc(sizeof(char)*1000);
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
	char* res = malloc(sizeof(char)*4000);
	res[0] = '\0';
	base->prefix= strdup("vba.in[");
	ASTNode* rhs = astnode_create(NODE_UNKNOWN, NULL, NULL);
	sprintf(res,"[IDX(%s,%s,%s)]",x_index,y_index,z_index);
	free(x_index);
	free(y_index);
	free(z_index);
	rhs->buffer = strdup(res);

	sprintf(res,"%s",node->buffer);
        base->buffer = strdup(res);
        base->rhs = rhs;
	rhs->prefix= strdup("]");

        free(base->buffer);
        sprintf(res,"%s",node->buffer);
        base->buffer = strdup(res);
	free(res);
	return base;
}

ASTNode* 
gen_4d_array_access(ASTNode* node)
{
	char* array_index = malloc(sizeof(char)*1000);
	char* x_index = malloc(sizeof(char)*1000);
	char* y_index = malloc(sizeof(char)*1000);
	char* z_index = malloc(sizeof(char)*1000);
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
	combine_all(start->rhs,array_index);
	combine_all(start->parent->rhs,x_index);
	combine_all(start->parent->rhs,y_index);
	combine_all(start->parent->parent->rhs,z_index);
	ASTNode* base= start->parent->parent->parent;
	base->lhs = NULL;
	base->rhs = NULL;
	base->infix= NULL;
	base->postfix= NULL;
	char* res = malloc(sizeof(char)*4000);
	res[0] = '\0';
	base->prefix= strdup("vba.in[");
	ASTNode* rhs = astnode_create(NODE_UNKNOWN, NULL, NULL);
	sprintf(res,"[IDX(%s,%s,%s)]",x_index,y_index,z_index);
	free(x_index);
	free(y_index);
	free(z_index);
	rhs->buffer = strdup(res);

	sprintf(res,"%s",node->buffer);
        base->buffer = strdup(res);
        base->rhs = rhs;
	rhs->prefix= strdup("]");

        free(base->buffer);
        sprintf(res,"%s[%s]",node->buffer,array_index);
        base->buffer = strdup(res);
	free(res);
	free(array_index);
	return base;
}


void
gen_array_reads(ASTNode* node, bool gen_mem_accesses, const char* datatype_scalar)
{
  if(node->lhs)
    gen_array_reads(node->lhs,gen_mem_accesses,datatype_scalar);
  if(node->rhs)
    gen_array_reads(node->rhs,gen_mem_accesses,datatype_scalar);
  if(!node->buffer)
	  return;
  char* datatype = malloc(sizeof(char)*(1000));
  sprintf(datatype,"%s*",datatype_scalar);
  const int l_current_nest = 0;
  for (size_t i = 0; i < num_symbols[l_current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
    (!strcmp(symbol_table[i].tspecifier,datatype)) && !strcmp(node->buffer,symbol_table[i].identifier) && node->parent->parent->parent->rhs && !str_vec_contains(symbol_table[i].tqualifiers,"const"))
    {
  	if(gen_mem_accesses)
  	{
  		char* big_array_name = malloc(sizeof(char)*1000);
  		sprintf(big_array_name,"big_%s_array",convert_to_define_name(datatype_scalar));
  		node->buffer = strdup(big_array_name);
  		node->type |= NODE_INPUT;
  		free(big_array_name);
  		return;
  	}
  	char* new_name = malloc(sizeof(char)*(4096-19));
  	new_name[0] = '\0';
  	char* arrays_name = malloc(sizeof(char)*(4096/2));
  	sprintf(arrays_name,"%s_arrays",convert_to_define_name(datatype_scalar));
  	if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
  	{
  		char* index_str = malloc(sizeof(char)*(4096/2));
  		combine_all(node->parent->parent->parent->rhs,index_str);
  		char* res = malloc(sizeof(char)*(4096));
  		sprintf(res,"d_%s[%s_offset+(%s)]\n",arrays_name,node->buffer,index_str);
  		ASTNode* base = node->parent->parent->parent->parent;
  		base->buffer = strdup(res);
  		base->lhs=NULL;
  		base->rhs=NULL;
  		base->prefix=NULL;
  		base->postfix=NULL;
		free(res);
		free(index_str);
  		return;
  	}
  	sprintf(new_name,"__ldg(&vba.%s[(int)%s]",arrays_name,node->buffer);
  	node->parent->parent->parent->postfix = strdup("])");
  	node->buffer = strdup(new_name);
  	free(new_name);
  	free(arrays_name);
    }
  }
  free(datatype);
}

void
read_user_enums_recursive(const ASTNode* node,string_vec user_enums, string_vec* user_enum_options)
{
	if(node->type == NODE_ENUM_DEF)
	{
		char* tmp = malloc(sizeof(char)*4096);
		combine_all(node->rhs,tmp);
		char* enum_def = malloc(sizeof(char)*4096);
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
		free(tmp);
		free(enum_def);
	}
	if(node->lhs)
		read_user_enums_recursive(node->lhs,user_enums,user_enum_options);
	if(node->rhs)
		read_user_enums_recursive(node->rhs,user_enums,user_enum_options);
}
typedef struct
{
	string_vec names;
	string_vec options[100];
} user_enums_info;

user_enums_info
read_user_enums(const ASTNode* node)
{
        static string_vec user_enum_options[100];
	string_vec user_enums;
        init_str_vec(&user_enums);
	for(int i = 0; i < 100; ++i)
	  init_str_vec(&user_enum_options[i]);
	read_user_enums_recursive(node,user_enums,user_enum_options);
	user_enums_info res;
	res.names = user_enums;
	for(int i = 0; i < 100; ++i)
		res.options[i]  = user_enum_options[i];
	return res;
}
typedef struct
{
	string_vec user_structs;
	string_vec* user_struct_field_names;
	string_vec* user_struct_field_types;
} structs_info;
static inline void
process_declaration(const ASTNode* field,int struct_index,structs_info* params)
{
	char type[4096];
	char name[4096];
	type[0] = '\0';
	combine_buffers(get_node(NODE_TSPEC, field),type);
	combine_buffers(field->rhs,name);

	push(&(params->user_struct_field_types[struct_index]), type);
	push(&(params->user_struct_field_names[struct_index]), name);
}
void
read_user_structs_recursive(const ASTNode* node, structs_info* params)
{
	if(node->lhs)
		read_user_structs_recursive(node->lhs,params);
	if(node->rhs)
		read_user_structs_recursive(node->rhs,params);
	if(node->type != NODE_STRUCT_DEF)
		return;
	const int struct_index = push(&(params->user_structs),node->lhs->buffer);
	ASTNode* fields_head = node->rhs;
	while(fields_head->rhs)
	{
		process_declaration(fields_head->rhs,struct_index,params);
		fields_head = fields_head->lhs;
	}
	process_declaration(fields_head->lhs,struct_index,params);
}
void
remove_user_structs(ASTNode* node)
{
	if(node->lhs)
		remove_user_structs(node->lhs);
	if(node->rhs)
		remove_user_structs(node->rhs);
	if(node->type != NODE_STRUCT_DEF)
		return;
	node->lhs = NULL;
	node->rhs = NULL;
}
structs_info
read_user_structs(const ASTNode* root)
{
	string_vec* user_struct_field_names = malloc(sizeof(string_vec)*100);
	string_vec* user_struct_field_types = malloc(sizeof(string_vec)*100);
	string_vec user_structs;
	init_str_vec(&user_structs);
	for(int i = 0; i< 100; ++i)
	{
		init_str_vec(&user_struct_field_names[i]);
		init_str_vec(&user_struct_field_types[i]);
	}
	structs_info res = {user_structs,user_struct_field_names, user_struct_field_types};
	read_user_structs_recursive(root, &res);
	return res;
}

void
process_param_codegen(ASTNode* kernel_root, const ASTNode* param, char* structs_info, string_vec* added_params_to_stencil_accesses, const bool gen_mem_accesses)
{
				char* param_type = malloc(4096*sizeof(char));
                                combine_buffers(param->lhs, param_type);
				char* param_str = malloc(4096*sizeof(char));
				param_str[0] = '\0';
                              	sprintf(param_str,"%s %s;",param_type, param->rhs->buffer);
				add_node_type(NODE_INPUT, kernel_root,param->rhs->buffer);
				strprepend(structs_info,param_str);
				if(str_vec_contains(*added_params_to_stencil_accesses,param->rhs->buffer))
					return;
				push(added_params_to_stencil_accesses,strdup(param->rhs->buffer));
				char* default_param = malloc(4096*sizeof(char));
			        sprintf(default_param,"{}");
				char* tmp = malloc(4096*2*sizeof(char));
				sprintf(tmp," %s %sAC_INTERNAL_INPUT = %s;",param_type,param->rhs->buffer,default_param);
				if(gen_mem_accesses)
					file_prepend("user_kernels.h.raw",tmp);
				free(param_type);
				free(tmp);
				free(param_str);
				free(default_param);

}

void
gen_kernel_structs_recursive(const ASTNode* node, string_vec* added_params_to_stencil_accesses, char* user_kernel_params_struct_str,const bool gen_mem_accesses)
{
	if(node->lhs)
		gen_kernel_structs_recursive(node->lhs,added_params_to_stencil_accesses,user_kernel_params_struct_str,gen_mem_accesses);
	if(node->rhs)
		gen_kernel_structs_recursive(node->rhs,added_params_to_stencil_accesses,user_kernel_params_struct_str,gen_mem_accesses);
	if(!(node->type & NODE_KFUNCTION))
		return;
	if(!node->rhs->lhs)
		return;

        ASTNode* param_list_head = node->rhs->lhs;
        ASTNode* compound_statement = node->rhs->rhs;
        param_list_head->type |= NODE_NO_OUT;
	char* structs_info = malloc(sizeof(char)*10000);
	structs_info[0] = '\0';
        while(param_list_head->rhs)
        {
	  process_param_codegen(compound_statement,param_list_head->rhs,structs_info,added_params_to_stencil_accesses,gen_mem_accesses);
          param_list_head = param_list_head->lhs;
        }

        ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, node->lhs);
	process_param_codegen(compound_statement,param_list_head->lhs,structs_info,added_params_to_stencil_accesses,gen_mem_accesses);
	char* kernel_params_struct = malloc(10000*sizeof(char));
	sprintf(kernel_params_struct,"typedef struct %sInputParams {%s} %sInputParams;\n",fn_identifier->buffer,structs_info,fn_identifier->buffer);

	strcat(kernel_params_struct,"\n");
	file_prepend("user_structs.h",kernel_params_struct);
	free(structs_info);

	char* tmp = malloc(4096*sizeof(char));
	sprintf(tmp,"%sInputParams %s;\n", fn_identifier->buffer,fn_identifier->buffer);
	strcat(user_kernel_params_struct_str,tmp);
}
void
gen_kernel_structs(const ASTNode* root, const bool gen_mem_accesses)
{
	char user_kernel_params_struct_str[10000];
	user_kernel_params_struct_str[0] = '\0';
	sprintf(user_kernel_params_struct_str,"typedef union acKernelInputParams {\n");
	string_vec added_params_to_stencil_accesses;
    	init_str_vec(&added_params_to_stencil_accesses);
	gen_kernel_structs_recursive(root,&added_params_to_stencil_accesses,user_kernel_params_struct_str,gen_mem_accesses);
    	free_str_vec(&added_params_to_stencil_accesses);
	strcat(user_kernel_params_struct_str,"} acKernelInputParams;\n");

	FILE* fp_structs = fopen("user_structs.h","a");
	fprintf(fp_structs,"\n%s\n",user_kernel_params_struct_str);
	fclose(fp_structs);
}

void
gen_user_structs(const ASTNode* root)
{
	structs_info info = read_user_structs(root);
	for(size_t i = 0; i < info.user_structs.size; ++i)
	{
		char* struct_def = malloc(sizeof(char)*5000);
		sprintf(struct_def,"typedef struct %s {",info.user_structs.data[i]);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
		{
			const char* type = info.user_struct_field_types[i].data[j];
			const char* name = info.user_struct_field_names[i].data[j];
			strcat(struct_def,type);
			strcat(struct_def," ");
			strcat(struct_def,name);
			strcat(struct_def,";");
		}
		strcat(struct_def, "} ");
		strcat(struct_def, info.user_structs.data[i]);
		strcat(struct_def, ";\n");
        	file_prepend("user_structs.h",struct_def);
        	free(struct_def);
	}
}
void
get_kernel_param_types_and_names(const ASTNode* node, const char* kernel_name, string_vec* types_dst, string_vec* names_dst)
{
	if(node->lhs)
		get_kernel_param_types_and_names(node->lhs,kernel_name,types_dst,names_dst);
	if(node->rhs)
		get_kernel_param_types_and_names(node->rhs,kernel_name,types_dst,names_dst);
	if(!(node->type & NODE_KFUNCTION))
		return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
	if(!fn_identifier)
		return;
	if(strcmp(fn_identifier->buffer, kernel_name))
		return;
        ASTNode* param_list_head = node->rhs->lhs;
	char* param_type = malloc(4096*sizeof(char));
        while(param_list_head->rhs)
        {
	  const ASTNode* param = param_list_head->rhs;
          combine_buffers(param->lhs, param_type);
	  push(types_dst,param_type);
	  push(names_dst,param->rhs->buffer);
          param_list_head = param_list_head->lhs;
        }
	const ASTNode* param = param_list_head->lhs;
        combine_buffers(param->lhs, param_type);
	push(types_dst,param_type);
	push(names_dst,param->rhs->buffer);
	free(param_type);
}
string_vec
get_func_call_params(const ASTNode* func_call)
{
		string_vec params;
		init_str_vec(&params);
		ASTNode* param_list_head = func_call->rhs;
		while(param_list_head->rhs)
		{
			char* param = NULL;
			if(get_node_by_token(IDENTIFIER,param_list_head->rhs))
				 param = get_node_by_token(IDENTIFIER,param_list_head->rhs)->buffer;
			if(get_node_by_token(NUMBER,param_list_head->rhs))
				 param = get_node_by_token(NUMBER,param_list_head->rhs)->buffer;
			if(get_node_by_token(REALNUMBER,param_list_head->rhs))
				 param = get_node_by_token(REALNUMBER,param_list_head->rhs)->buffer;
			assert(param);
			push(&params,param);
			param_list_head = param_list_head->lhs;
		}
		char* param= NULL;
		if(get_node_by_token(IDENTIFIER,param_list_head->lhs))
			 param = get_node_by_token(IDENTIFIER,param_list_head->lhs)->buffer;
		if(get_node_by_token(NUMBER,param_list_head->lhs))
			 param = get_node_by_token(NUMBER,param_list_head->lhs)->buffer;
		if(get_node_by_token(REALNUMBER,param_list_head->lhs))
			 param = get_node_by_token(REALNUMBER,param_list_head->lhs)->buffer;
		assert(param);
		push(&params,param);
		return params;
}

void gen_loader(const ASTNode* kernel_call, const ASTNode* root, const char* prefix, string_vec* input_symbols, string_vec* input_types)
{
		char tmp[4000];
		const char* func_name = get_node_by_token(IDENTIFIER,kernel_call)->buffer;
		if(!strcmp(func_name,"periodic"))
			return;
		ASTNode* param_list_head = kernel_call->rhs;
		if(!param_list_head)
			return;
		string_vec params = get_func_call_params(kernel_call);
		bool is_boundcond = false;
		for(size_t i = 0; i< params.size; ++i)
			is_boundcond |= (strstr(params.data[i],"BOUNDARY_") != NULL);
		string_vec param_types;
		init_str_vec(&param_types);

		string_vec param_list_names;
		init_str_vec(&param_list_names);

		get_kernel_param_types_and_names(root,func_name,&param_types,&param_list_names);

		char* loader_str = malloc(sizeof(char)*4000);
		sprintf(loader_str,"auto %s_%s_loader = [](ParamLoadingInfo p){\n",prefix, func_name);
		const int params_offset = is_boundcond ? 2 : 0;
		if(!is_boundcond)
		{
			if(param_types.size != params.size)
			{
				fprintf(stderr,"Number of inputs for %s in ComputeSteps does not match the number of input params\n", func_name);
				exit(EXIT_FAILURE);
			}
		}
		for(size_t i = 0; i < param_types.size-params_offset; ++i)
		{
			if(is_number(params.data[i]) || is_real(params.data[i]))
				sprintf(tmp, "p.params -> %s.%s = %s;\n", func_name, param_list_names.data[i], params.data[i]);
			else if(!strcmp(param_types.data[i],"AcReal"))
			{
				sprintf(tmp, "p.params -> %s.%s = acDeviceGetRealInput(acGridGetDevice(),%s);\n", func_name,param_list_names.data[i], params.data[i]);
			}
			else if(!strcmp(param_types.data[i],"int"))
				sprintf(tmp, "p.params -> %s.%s = acDeviceGetIntInput(acGridGetDevice(),%s); \n", func_name,param_list_names.data[i], params.data[i]);
			strcat(loader_str,tmp);
			if(!str_vec_contains(*input_symbols,params.data[i]))
			{
				if(!is_number(params.data[i]) && !is_real(params.data[i]))
				{
					push(input_symbols,params.data[i]);
					push(input_types,param_types.data[i]);
				}
			}
		}
		//add predefined input params for boundcond functions
		if(is_boundcond)
		{
			sprintf(tmp, "p.params -> %s.boundary_normal= p.boundary_normal;\n",func_name);
			strcat(loader_str,tmp);
			sprintf(tmp, "p.params -> %s.vtxbuf = p.vtxbuf;\n",func_name);
			strcat(loader_str,tmp);
		}
		strcat(loader_str,"};\n");
		file_prepend("user_loaders.h",loader_str);

		free_str_vec(&param_types);
		free_str_vec(&param_list_names);
		free_str_vec(&params);
		free(loader_str);
}
	 
void
gen_taskgraph_kernel_entry(const ASTNode* kernel_call, const ASTNode* root, char* global_res, string_vec* input_symbols, string_vec* input_types, const char* taskgraph_name)
{
	assert(kernel_call);
	char* res = malloc(sizeof(char)*4000);
	const char* func_name = get_node_by_token(IDENTIFIER,kernel_call)->buffer;
	char* fields_in_str  = malloc(sizeof(char)*4000);
	char* fields_out_str = malloc(sizeof(char)*4000);
	char* communicated_fields_before = malloc(sizeof(char)*4000);
	char* communicated_fields_after = malloc(sizeof(char)*4000);
	sprintf(fields_in_str, "%s", "{");
	sprintf(fields_out_str, "%s", "{");
	sprintf(communicated_fields_before, "%s", "{");
	sprintf(communicated_fields_after, "%s", "{");
	char* tmp = malloc(sizeof(char)*4000);
	const int kernel_index = get_symbol_index(NODE_KFUNCTION_ID,func_name,NULL);
	char* all_fields = malloc(sizeof(char)*4000);
	all_fields[0] = '\0';
	for(size_t field = 0; field < num_fields; ++field)
	{
		const bool field_in  = (read_fields[field + num_fields*kernel_index] || field_has_stencil_op[field + num_fields*kernel_index]);
		const bool field_out = (written_fields[field + num_fields*kernel_index]);
		const char* field_str = get_symbol(NODE_VARIABLE_ID,field,"Field")->identifier;
		strcat(all_fields,",");
		strcat(all_fields,field_str);
		sprintf(tmp,"%s,",field_str);
		if(field_in)
			strcat(fields_in_str,tmp);
		if(field_out)
			strcat(fields_out_str,tmp);
	}
	strcat(fields_in_str,  "}");
	strcat(fields_out_str, "}");
	if(kernel_call->rhs)
		sprintf(res,"acCompute(KERNEL_%s,%s,%s,%s_%s_loader),\n",func_name,fields_in_str,fields_out_str,taskgraph_name,func_name);
	else
		sprintf(res,"acCompute(KERNEL_%s,%s,%s),\n",func_name,fields_in_str,fields_out_str);
	gen_loader(kernel_call,root,taskgraph_name,input_symbols,input_types);
	free(all_fields);
	free(fields_in_str);
	free(fields_out_str);
	free(tmp);
	strcat(global_res,res);
	free(res);
}
//void
//gen_taskgraph_entry(const ASTNode* function_call, const ASTNode* root, char* global_res)
//{
//	static bool* field_communicated = NULL;
//  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
//	if(!field_communicated && has_optimization_info)
//	{
//		field_communicated = (bool*)malloc(sizeof(bool)*num_fields);
//		memset(field_communicated,0,sizeof(bool)*num_fields);
//	}
//	char* res = malloc(sizeof(char)*4000);
//	const char* func_name = get_node_by_token(IDENTIFIER,function_call)->buffer;
//	if(!strcmp(func_name,"periodic_boundconds"))
//	{
//	}
//	else if(!strcmp(func_name,"communicate"))
//	{
//		if(function_call->rhs)
//		{
//			char* params = malloc(4000*sizeof(char));
//			combine_all(function_call->rhs,params);
//			sprintf(res,"acHaloExchange({%s}),\n",params);
//		}
//		else
//			sprintf(res, "acHaloExchange(all_fields),\n");
//	}
//	strcat(global_res,res);
//
//}
int_vec
get_taskgraph_kernel_calls(const ASTNode* function_call_list_head, int n)
{

	int_vec calls;
	init_int_vec(&calls);
	ASTNode* function_call = function_call_list_head->lhs;
	char* func_name = get_node_by_token(IDENTIFIER,function_call)->buffer;
	if(check_symbol(NODE_KFUNCTION_ID,func_name,NULL,NULL))
		push_int(&calls,get_symbol_index(NODE_KFUNCTION_ID,func_name,NULL));
	while(--n)
	{
		function_call_list_head = function_call_list_head->parent;
		function_call = function_call_list_head->rhs;
		func_name = get_node_by_token(IDENTIFIER,function_call)->buffer;
		if(check_symbol(NODE_KFUNCTION_ID,func_name,NULL,NULL))
			push_int(&calls,get_symbol_index(NODE_KFUNCTION_ID,func_name,NULL));
	}
	return calls;
}
void
compute_next_level_set(bool* src, const int_vec kernel_calls, bool* field_written_to, int* call_level_set)
{
	memset(field_written_to,0,sizeof(bool)*num_fields);
	for(size_t i = 0; i < kernel_calls.size; ++i)
	{
		const int kernel_index = kernel_calls.data[i];
		bool can_compute = true;
		for(size_t j = 0; j < num_fields; ++j)
		{
			can_compute &= !((read_fields[j + num_fields*kernel_index] || field_has_stencil_op[j + num_fields*kernel_index]) && field_written_to[j]);
			if(call_level_set[i] == -1)
				field_written_to[j] |= written_fields[j + num_fields*kernel_index];
		}
		if(call_level_set[i] == -1 &&  can_compute)
			src[i] = true;
	}
}
#define BOUNDARY_X_BOT (1 << 0)
#define BOUNDARY_X_TOP (1 << 1)
#define BOUNDARY_Y_BOT (1 << 2)
#define BOUNDARY_Y_TOP (1 << 3)
#define BOUNDARY_Z_BOT (1 << 4)
#define BOUNDARY_Z_TOP (1 << 5)
#define BOUNDARY_X (BOUNDARY_X_BOT | BOUNDARY_X_TOP)
#define BOUNDARY_Y (BOUNDARY_Y_BOT | BOUNDARY_Y_TOP)
#define BOUNDARY_Z (BOUNDARY_Z_BOT | BOUNDARY_Z_TOP)

int
get_boundary_int(const char* boundary_in)
{
	int res = 0;
	if(!strstr(boundary_in,"BOUNDARY_"))
	{
		fprintf(stderr,"incorrect boundary specification: %s\n",boundary_in);
		exit(EXIT_FAILURE);
	}
	char* boundary = remove_substring(strdup(boundary_in),"BOUNDARY");
	if(strstr(boundary,"BOT"))
	{
		if(strstr(boundary,"X"))
			res |= BOUNDARY_X_BOT;
		if(strstr(boundary,"Y"))
			res |= BOUNDARY_Y_BOT;
		if(strstr(boundary,"Z"))
			res |= BOUNDARY_Z_BOT;
	}
	else if(strstr(boundary,"TOP"))
	{
		if(strstr(boundary,"X"))
			res |= BOUNDARY_X_TOP;
		if(strstr(boundary,"Y"))
			res |= BOUNDARY_Y_TOP;
		if(strstr(boundary,"Z"))
			res |= BOUNDARY_Z_TOP;
	}
	else
	{
		if(strstr(boundary,"X"))
			res |= BOUNDARY_X;
		if(strstr(boundary,"Y"))
			res |= BOUNDARY_Y;
		if(strstr(boundary,"Z"))
			res |= BOUNDARY_Z;
	}
	free(boundary);
	return res;

}
void
process_boundcond(const ASTNode* func_call, char** res, const ASTNode* root, const char* boundconds_name, string_vec* input_symbols,string_vec* input_types)
{
	char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);
	const int boundaries[] = {BOUNDARY_X_BOT, BOUNDARY_Y_BOT,BOUNDARY_Z_BOT,BOUNDARY_X_TOP,BOUNDARY_Y_TOP,BOUNDARY_Z_TOP};
	const int num_boundaries = 6;
	bool* fields_included = (bool*)malloc(sizeof(bool)*num_fields);
	memset(fields_included,0,sizeof(bool)*num_fields);


	ASTNode* param_list_head = func_call->rhs;
	string_vec params = get_func_call_params(func_call);
	for(size_t field = 0; field < num_fields; ++field)
		fields_included[field] = str_vec_contains(params,get_symbol(NODE_VARIABLE_ID,field,"Field")->identifier);
	free_str_vec(&params);
	//if none are included then by default all are included
	bool none_included = true;
	for(size_t field = 0; field < num_fields; ++field)
		none_included &= !fields_included[field];
	for(size_t field = 0; field < num_fields; ++field)
		fields_included[field] |= none_included;

	for(size_t field = 0; field < num_fields; ++field)
	     for(int bc = 0;  bc < num_boundaries; ++bc) 
	     	     res[field + num_fields*bc] = (boundary_int & boundaries[bc] && fields_included[field]) ? func_name : res[field + num_fields*bc];
	free(fields_included);
	if(!strcmp(func_name,"periodic"))
		return;
	char* prefix = malloc(sizeof(char)*4000);
	for(int bc = 0;  bc < num_boundaries; ++bc) 
	{
		if(boundary_int & boundaries[bc])
		{
			sprintf(prefix,"%s_",boundconds_name);
			if(bc == 0)
				strcat(prefix,"X_BOT");
			if(bc == 1)
				strcat(prefix,"Y_BOT");
			if(bc == 2)
				strcat(prefix,"Z_BOT");
			if(bc == 3)
				strcat(prefix,"X_TOP");
			if(bc == 4)
				strcat(prefix,"Y_TOP");
			if(bc == 5)
				strcat(prefix,"Z_TOP");
			gen_loader(func_call,root,prefix,input_symbols,input_types);
		}
	}
	free(prefix);
}
void
get_field_boundconds_recursive(const ASTNode* node, const ASTNode* root, char** res, const char* boundconds_name, string_vec* input_symbols, string_vec* input_types)
{
	if(node->lhs)
		get_field_boundconds_recursive(node->lhs,root,res,boundconds_name,input_symbols,input_types);
	if(node->rhs)
		get_field_boundconds_recursive(node->rhs,root,res,boundconds_name,input_symbols,input_types);
	if(node->type != NODE_BOUNDCONDS_DEF)
		return;
	if(strcmp(node->lhs->buffer,boundconds_name))
		return;
	const char* name = node->lhs->buffer;
	const ASTNode* function_call_list_head = node->rhs;
	int n_entries = 1;
	while(function_call_list_head->rhs)
	{
		++n_entries;
		function_call_list_head = function_call_list_head->lhs;
	}
	process_boundcond(function_call_list_head->lhs,res,root,boundconds_name,input_symbols,input_types);
	while(--n_entries)
	{
		function_call_list_head = function_call_list_head->parent;
		process_boundcond(function_call_list_head->rhs,res,root,boundconds_name,input_symbols,input_types);
	}
}
char**
get_field_boundconds(const ASTNode* root, const char* boundconds_name, string_vec* input_symbols, string_vec* input_types)
{
	char** res = NULL;
  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	if(!has_optimization_info)
		return res;
	const int num_boundaries = 6;
	res = malloc(sizeof(char*)*num_fields*num_boundaries);
	memset(res,0,sizeof(char*)*num_fields*num_boundaries);
	get_field_boundconds_recursive(root,root,res,boundconds_name,input_symbols,input_types);
	return res;
}
void
gen_user_taskgraphs_recursive(const ASTNode* node, const ASTNode* root, string_vec* input_symbols, string_vec* input_types)
{
  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	if(!has_optimization_info)
		return;
	if(node->lhs)
		gen_user_taskgraphs_recursive(node->lhs,root,input_symbols,input_types);
	if(node->rhs)
		gen_user_taskgraphs_recursive(node->rhs,root,input_symbols,input_types);
	if(node->type != NODE_TASKGRAPH_DEF)
		return;
	const char* boundconds_name = node->lhs->rhs->buffer;
	char** field_boundconds = get_field_boundconds(root,boundconds_name,input_symbols,input_types);
	const int num_boundaries = 6;
	bool* field_boundconds_processed = (bool*)malloc(num_fields*num_boundaries);

	//for(size_t field = 0; field < num_fields; ++field)
	//{
	//	printf("%lu|x_bot: %s\n",field,field_boundconds[field + num_fields*0]);
	//	printf("%lu|y_bot: %s\n",field,field_boundconds[field + num_fields*1]);
	//	printf("%lu|z_bot: %s\n",field,field_boundconds[field + num_fields*2]);
	//	printf("%lu|x_top: %s\n",field,field_boundconds[field + num_fields*3]);
	//	printf("%lu|y_top: %s\n",field,field_boundconds[field + num_fields*4]);
	//	printf("%lu|z_top: %s\n",field,field_boundconds[field + num_fields*5]);
	//}
	
	const char* name = node->lhs->lhs->buffer;
	char* res = malloc(sizeof(char)*10000);
	sprintf(res, "AcTaskGraph* %s = acGridBuildTaskGraph({\n",name);
	const ASTNode* function_call_list_head = node->rhs;
	//have to traverse in reverse order to generate the right order in taskgraph
	int n_entries = 1;
	while(function_call_list_head->rhs)
	{
		function_call_list_head = function_call_list_head -> lhs;
		++n_entries;
	}

	int_vec kernel_calls = get_taskgraph_kernel_calls(function_call_list_head,n_entries);

	int_vec kernel_calls_in_level_order;
	init_int_vec(&kernel_calls_in_level_order);
	bool* field_halo_in_sync = malloc(sizeof(bool)*num_fields);
	bool* field_out_from_last_level_set = malloc(sizeof(bool)*num_fields);
	bool* field_out_from_level_set = malloc(sizeof(bool)*num_fields);
	bool* field_stencil_ops_at_next_level_set = malloc(sizeof(bool)*num_fields);
	bool* current_level_set = malloc(sizeof(bool)*num_kernels);
	bool* next_level_set    = malloc(sizeof(bool)*num_kernels);
	bool* field_need_to_communicate = malloc(sizeof(bool)*num_fields);
	memset(field_halo_in_sync,0,sizeof(bool)*num_fields);
	int n_level_sets = 0;
	int* call_level_set  = malloc(sizeof(int)*kernel_calls.size);
	const int MAX_TASKS = 100;
	bool* field_needs_to_be_communicated_before_level_set = malloc(sizeof(int)*MAX_TASKS*num_fields);

	memset(call_level_set,-1,sizeof(int)*kernel_calls.size);
	memset(field_out_from_level_set,0,sizeof(bool)*num_fields);
	memset(field_out_from_last_level_set,0,sizeof(bool)*num_fields);
	
	bool all_processed = false;
	while(!all_processed)
	{
		memset(field_stencil_ops_at_next_level_set,0,sizeof(bool)*num_fields);
		memset(field_need_to_communicate,0,sizeof(bool)*num_fields);
		memset(next_level_set,0,sizeof(bool)*num_kernels);
		compute_next_level_set(next_level_set, kernel_calls,field_out_from_level_set,call_level_set);
		for(size_t i = 0; i < kernel_calls.size; ++i)
		{
			if(next_level_set[i])
			{
				call_level_set[i] = n_level_sets;
				const int k = kernel_calls.data[i];
				for(size_t j = 0; j < num_fields; ++j)
					field_stencil_ops_at_next_level_set[j] |= field_has_stencil_op[j + num_fields*k];
			}
		}
		for(size_t j = 0; j < num_fields; ++j)
		    field_halo_in_sync[j] &= !field_out_from_last_level_set[j];
		for(size_t j = 0; j < num_fields; ++j)
		    field_need_to_communicate[j] |= (!field_halo_in_sync[j] && field_stencil_ops_at_next_level_set[j]);
		for(size_t j = 0; j < num_fields; ++j)
		{
			field_halo_in_sync[j] |= field_need_to_communicate[j];
			field_needs_to_be_communicated_before_level_set[j + num_fields*n_level_sets] = field_need_to_communicate[j];

		}
		bool* swap_tmp;
		swap_tmp = field_out_from_level_set;
		field_out_from_level_set = field_out_from_last_level_set;
		field_out_from_last_level_set = swap_tmp;
		++n_level_sets;
		all_processed = true;

		for(size_t k = 0; k < kernel_calls.size; ++k)
			all_processed &= (call_level_set[k] != -1);

	}
	
	free(field_halo_in_sync);
	free(field_out_from_level_set);
	free(field_stencil_ops_at_next_level_set);
	free(next_level_set);
	free(field_need_to_communicate);
	char all_fields[4000];
	all_fields[0] = '\0';
	for(size_t field = 0; field < num_fields; ++field)
	{
		const char* field_str = get_symbol(NODE_VARIABLE_ID,field,"Field")->identifier;
		strcat(all_fields,field_str);
		strcat(all_fields,",");
	}
	for(int level_set = 0; level_set < n_level_sets; ++level_set)
	{
		memset(field_boundconds_processed,0,num_fields*num_boundaries*sizeof(bool));
		bool need_to_communicate = false;
		char communicated_fields_str[4000];
		sprintf(communicated_fields_str,"{");
		for(size_t field = 0; field < num_fields; ++field)
		{
			need_to_communicate |= field_needs_to_be_communicated_before_level_set[field + num_fields*level_set];
			if(field_needs_to_be_communicated_before_level_set[field + num_fields*level_set])
			{
				const char* field_str = get_symbol(NODE_VARIABLE_ID,field,"Field")->identifier;
				strcat(communicated_fields_str,field_str);
				strcat(communicated_fields_str,",");
			}
		}
		strcat(communicated_fields_str,"}");
		if(need_to_communicate)
		{
			strcat(res,"acHaloExchange(");
			strcat(res,communicated_fields_str);
			strcat(res,"),\n");

			const char* x_boundcond = field_boundconds[0 + num_fields*0];
			const char* y_boundcond = field_boundconds[0 + num_fields*1];
			const char* z_boundcond = field_boundconds[0 + num_fields*2];

			if(!strcmp(x_boundcond,"periodic") || !strcmp(y_boundcond,"periodic") || !strcmp(z_boundcond,"periodic"))
			{
				if(!strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_XYZ,BOUNDCOND_PERIODIC,");
				else if(!strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_XY,BOUNDCOND_PERIODIC,");
				else if(!strcmp(x_boundcond,"periodic") && strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_XZ,BOUNDCOND_PERIODIC,");
				else if(!strcmp(x_boundcond,"periodic") && strcmp(y_boundcond,"periodic") && strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_X,BOUNDCOND_PERIODIC,");
				else if(strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_YZ,BOUNDCOND_PERIODIC,");
				else if(strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_Y,BOUNDCOND_PERIODIC,");
				else if(strcmp(x_boundcond,"periodic") && strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"acBoundaryCondition(BOUNDARY_Z,BOUNDCOND_PERIODIC,");
				strcat(res,communicated_fields_str);
				strcat(res,"),\n");
			}

			for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
				for(size_t field = 0; field < num_fields; ++field)
					field_boundconds_processed[field + num_fields*boundcond]  = !strcmp(field_boundconds[field + num_fields*boundcond],"periodic")  || !field_needs_to_be_communicated_before_level_set[field + num_fields*level_set];

			bool all_are_processed = false;
			while(!all_are_processed)
			{
				for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
				{
					char* processed_boundcond = NULL;

					for(size_t field = 0; field < num_fields; ++field)
						processed_boundcond = !field_boundconds_processed[field + num_fields*boundcond] ? field_boundconds[field + num_fields*boundcond] : processed_boundcond;
					if(!processed_boundcond) continue;
					char* boundary_str;
					if(boundcond == 0)
						boundary_str = "X_BOT";
					else if(boundcond == 1)
						boundary_str = "Y_BOT";
					else if(boundcond == 2)
						boundary_str = "Z_BOT";
					else if(boundcond == 3)
						boundary_str = "X_TOP";
					else if(boundcond == 4)
						boundary_str = "Y_TOP";
					else if(boundcond == 5)
						boundary_str = "Z_TOP";
					strcat(res,"acBoundaryCondition(BOUNDARY_");
					strcat(res,boundary_str);
					strcat(res,",KERNEL_");
					strcat(res,processed_boundcond);
					strcat(res,",");
					strcat(res,"{");
					for(size_t field = 0; field < num_fields; ++field)
					{
						const char* field_str = get_symbol(NODE_VARIABLE_ID,field,"Field")->identifier;
						const char* boundcond_str = field_boundconds[field + num_fields*boundcond];
						if(strcmp(boundcond_str,processed_boundcond)) continue;
						if(field_boundconds_processed[field + num_fields*boundcond]) continue;
						field_boundconds_processed[field + num_fields*boundcond] |= true;
						strcat(res,field_str);
						strcat(res,",");
					}
					strcat(res,"},");
					strcat(res,boundconds_name);
					strcat(res,"_");
					strcat(res,boundary_str);
					strcat(res,"_");
					strcat(res,processed_boundcond);
					strcat(res,"_loader");
					strcat(res,"),\n");
				}
				all_are_processed = true;
				for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
					for(size_t field = 0; field < num_fields; ++field)
						all_are_processed &= field_boundconds_processed[field + num_fields*boundcond];
			}
		}
		for(size_t call = 0; call < kernel_calls.size; ++call) 
		{
			if(call_level_set[call] == level_set)
			{
				int call_index = call;
				if(call_index == 0)
					gen_taskgraph_kernel_entry(function_call_list_head->lhs,root,res,input_symbols,input_types,name);
				else
				{
					const ASTNode* new_head = function_call_list_head;
					while(call_index--)
						new_head = new_head->parent;
					gen_taskgraph_kernel_entry(new_head->rhs,root,res,input_symbols,input_types,name);
				}


			}
		}
	}
	strcat(res,"});\n");
	file_prepend("user_taskgraphs.h", res);
	free(res);
	free_int_vec(&kernel_calls);
	free_int_vec(&kernel_calls_in_level_order);
}


void
gen_input_enums(FILE* fp, string_vec input_symbols, string_vec input_types, const char* datatype)
{
  const char* datatype_scalar = datatype;
  fprintf(fp,"typedef enum {");
  for (size_t i = 0; i < input_types.size; ++i)
  {
	  if(strcmp(datatype,input_types.data[i]))
		  continue;
	  fprintf(fp,"%s,",input_symbols.data[i]);
  }
  fprintf(fp, "NUM_%s_INPUT_PARAMS} %sInputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));
	
}
void
gen_user_taskgraphs(FILE* fp, const ASTNode* root)
{
	string_vec input_symbols;
	string_vec input_types;
	init_str_vec(&input_symbols);
	init_str_vec(&input_types);
	gen_user_taskgraphs_recursive(root,root,&input_symbols,&input_types);
	gen_input_enums(fp,input_symbols,input_types,"AcReal");
	gen_input_enums(fp,input_symbols,input_types,"int");
	free_str_vec(&input_symbols);
	free_str_vec(&input_types);
}

ASTNode*
get_node_with_buffer(const ASTNode* node, NodeType type, const char* buffer)
{
	if(node->type & type && !strcmp(node->lhs->buffer,buffer))
		return (ASTNode*) node;
	ASTNode* res = NULL;
	res = (res) ? res 
		    : (node->lhs) ? get_node_with_buffer(node->lhs,type,buffer) : NULL;
	res = (res) ? res 
		    : (node->rhs) ? get_node_with_buffer(node->rhs,type,buffer) : NULL;
	return res;
}

typedef struct
{
	int* nums;
	string_vec* vals;
} combinations;

typedef struct
{
	string_vec* names;
	string_vec* options;
} combinatorial_params;
typedef struct
{
	char* type;
	char* name;
} variable;

void
add_param_combinations(const variable var, const int kernel_index,const char* prefix, user_enums_info user_enums, combinatorial_params combinatorials, structs_info struct_info)
{
	char full_name[4096];
	sprintf(full_name,"%s%s",prefix,var.name);
	if(str_vec_contains(struct_info.user_structs,var.type))
	{
	  const int struct_index = str_vec_get_index(struct_info.user_structs,var.type);
	  string_vec struct_field_types = struct_info.user_struct_field_types[struct_index];
	  string_vec struct_field_names = struct_info.user_struct_field_names[struct_index];
	  for(size_t i=0; i<struct_field_types.size; ++i)
	  {
		  char new_prefix[10000];
		  sprintf(new_prefix, "%s%s.",prefix,var.name);
		  add_param_combinations((variable){struct_field_types.data[i],struct_field_names.data[i]},kernel_index,new_prefix,user_enums,combinatorials,struct_info);
	  }
	}
	if(str_vec_contains(user_enums.names,var.type))
	{
		const int param_index = push(&combinatorials.names[kernel_index],full_name);

		const int enum_index = str_vec_get_index(user_enums.names,var.type);
		string_vec options  = user_enums.options[enum_index];
		for(size_t i = 0; i < options.size; ++i)
		{
			push(&combinatorials.options[kernel_index+100*param_index],options.data[i]);
		}
	}
	if(!strcmp("bool",var.type))
	{
		const int param_index = push(&combinatorials.names[kernel_index],full_name);
		push(&combinatorials.options[kernel_index+100*param_index],"false");
		push(&combinatorials.options[kernel_index+100*param_index],"true");
	}
}



void
gen_all_possibilities(string_vec res, int kernel_index, size_t my_index,combinations combinations,combinatorial_params combinatorials)
{
	if(my_index == combinatorials.names[kernel_index].size-1)
	{
		for(size_t i = 0; i<combinatorials.options[kernel_index+100*my_index].size; ++i)
		{

			combinations.vals[kernel_index + MAX_KERNELS*combinations.nums[kernel_index]] = str_vec_copy(res);
			push(&combinations.vals[kernel_index + MAX_KERNELS*combinations.nums[kernel_index]], combinatorials.options[kernel_index+100*my_index].data[i]);
			++combinations.nums[kernel_index];
			
		}
		return;
	}
	else
	{
		for(size_t i = 0; i<combinatorials.options[kernel_index+100*my_index].size; ++i)
		{
			string_vec copy = str_vec_copy(res);
			push(&copy, combinatorials.options[kernel_index+100*my_index].data[i]);
			gen_all_possibilities(copy,kernel_index,my_index+1,combinations,combinatorials);
		}
	}
}
void 
gen_combinations(int kernel_index,combinations combinations ,combinatorial_params combinatorials)
{
	string_vec base;
	init_str_vec(&base);
	if(combinatorials.names[kernel_index].size > 0)
		gen_all_possibilities(base, kernel_index,0,combinations,combinatorials);
	free_str_vec(&base);
}
void
gen_kernel_num_of_combinations_recursive(const ASTNode* node, combinations combinations, user_enums_info user_enums, string_vec* user_kernels_with_input_params,combinatorial_params combinatorials, structs_info struct_info)
{
	if(node->lhs)
	{
		gen_kernel_num_of_combinations_recursive(node->lhs,combinations,user_enums,user_kernels_with_input_params,combinatorials,struct_info);
	}
	if(node->rhs)
		gen_kernel_num_of_combinations_recursive(node->rhs,combinations,user_enums,user_kernels_with_input_params,combinatorials,struct_info);
	if(node->type & NODE_KFUNCTION && node->rhs->lhs)
	{
	   const char* kernel_name = get_node(NODE_KFUNCTION_ID, node)->buffer;
	   const int kernel_index = push(user_kernels_with_input_params,kernel_name);
	   ASTNode* param_list_head = node->rhs->lhs;
	   char* type = malloc(sizeof(char)*4096);
	   char* name = malloc(sizeof(char)*4096);
	   while(param_list_head->rhs)
	   {

	        const ASTNode* type_node = get_node(NODE_TSPEC,param_list_head->rhs);
	   	combine_buffers(type_node,type);
	   	combine_buffers(param_list_head->rhs->rhs,name);
	        add_param_combinations((variable){type,name},kernel_index,"",user_enums,combinatorials,struct_info);
	        param_list_head = param_list_head->lhs;
	   }
	   const ASTNode* type_node = get_node(NODE_TSPEC,param_list_head->lhs);
	   combine_buffers(type_node,type);
	   combine_buffers(param_list_head->lhs->rhs,name);
	   add_param_combinations((variable){type,name},kernel_index,"",user_enums,combinatorials,struct_info);
	   gen_combinations(kernel_index,combinations,combinatorials);
	   free(type);
	   free(name);
	}
}
void
gen_kernel_num_of_combinations(const ASTNode* root, combinations combinations, string_vec* user_kernels_with_input_params,string_vec* user_kernel_combinatorial_params)
{
  	user_enums_info user_enums = read_user_enums(root);

	string_vec user_kernel_combinatorial_params_options[100*100];
	for(int i = 0; i < 100; ++i)
	  for(int j=0;j<100;++j)
	  	  init_str_vec(&user_kernel_combinatorial_params_options[i+100*j]);
	structs_info struct_info = read_user_structs(root);

	gen_kernel_num_of_combinations_recursive(root,combinations,user_enums,user_kernels_with_input_params,(combinatorial_params){user_kernel_combinatorial_params,user_kernel_combinatorial_params_options},struct_info);

  	free_str_vec(&user_enums.names);
	for(int i = 0; i < 100; ++i)
	{
	  for(int j=0;j<100;++j)
	  	  free_str_vec(&user_kernel_combinatorial_params_options[i+100*j]);
  	  free_str_vec(&user_enums.options[i]);
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

typedef struct
{
	string_vec outputs;
	string_vec conditions;
	op_vec ops;
} reduce_info;


void
get_reduce_info(const ASTNode* node, reduce_info* src, reduce_info* dfuncs_info)
{
	if(node->lhs) 
		get_reduce_info(node->lhs,src,dfuncs_info);
	if(node->rhs) 
		get_reduce_info(node->rhs,src,dfuncs_info);
	if(!(node->type & NODE_FUNCTION_CALL))
		return;
	char* func_name = malloc(sizeof(char)*5000);
	combine_buffers(node->lhs,func_name);
	if(!strcmp(func_name,"reduce_sum"))
		push_op(&src->ops,REDUCE_SUM);
	if(!strcmp(func_name,"reduce_min"))
		push_op(&src->ops,REDUCE_MIN);
	if(!strcmp(func_name,"reduce_max"))
		push_op(&src->ops,REDUCE_MAX);

	if (!strcmp(func_name,"reduce_sum") || !strcmp(func_name,"reduce_min") || !strcmp(func_name,"reduce_max"))
	{
	  char* condition = malloc(5000*sizeof(char));
	  combine_buffers(node->rhs->lhs->lhs,condition);
	  push(&src->conditions,condition);
	  free(condition);

	  char* output = malloc(5000*sizeof(char));
	  combine_buffers(node->rhs->rhs,output);
	  push(&src->outputs,output);
	  free(output);
	}

	const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,NULL);
	free(func_name);
	if(dfunc_index < 0)
		return;
	for(size_t i = 0; i < dfuncs_info[dfunc_index].ops.size; ++i)
	{
		push_op(&src->ops,dfuncs_info[dfunc_index].ops.data[i]);
		push(&src->conditions,dfuncs_info[dfunc_index].conditions.data[i]);
		push(&src->outputs,dfuncs_info[dfunc_index].outputs.data[i]);
	}
}
void
get_dfuncs_reduce_info(const ASTNode* node,reduce_info* src)
{
	if(node->lhs) 
		get_dfuncs_reduce_info(node->lhs,src);
	if(node->rhs) 
		get_dfuncs_reduce_info(node->rhs,src);
	if(!(node->type & NODE_DFUNCTION))
		return;
	char* func_name = malloc(sizeof(char)*5000);
        combine_buffers(node->lhs,func_name);
        const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,NULL);
	get_reduce_info(node,&src[dfunc_index],src);
	free(func_name);
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
gen_kernel_postfixes_recursive(ASTNode* node, const bool gen_mem_accesses, reduce_info* dfuncs_info,reduce_info* kernel_reduce_info)
{
	if(node->lhs)
		gen_kernel_postfixes_recursive(node->lhs,gen_mem_accesses,dfuncs_info,kernel_reduce_info);
	if(node->rhs)
		gen_kernel_postfixes_recursive(node->rhs,gen_mem_accesses,dfuncs_info,kernel_reduce_info);
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
	reduce_info reduce_info;
	init_op_vec(&reduce_info.ops);
	init_str_vec(&reduce_info.conditions);
	init_str_vec(&reduce_info.outputs);

	get_reduce_info(node,&reduce_info,dfuncs_info);
	if(reduce_info.ops.size == 0)
	{
	  strcat(new_postfix,"}");
	  compound_statement->postfix = strdup(new_postfix);
	  return;
	}
	const int kernel_index = get_kernel_index(get_node(NODE_KFUNCTION_ID,node)->buffer);
	const ASTNode* fn_identifier = get_node(NODE_KFUNCTION_ID,node);
	assert(reduce_info.ops.size  == reduce_info.outputs.size && reduce_info.ops.size == reduce_info.conditions.size);

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

	for(size_t i = 0; i < reduce_info.ops.size; ++i)
	{
		ReduceOp reduce_op = reduce_info.ops.data[i];
		char* condition = reduce_info.conditions.data[i];
		char* output = reduce_info.outputs.data[i];
		push_op(&kernel_reduce_info[kernel_index].ops,  reduce_op);
		push(&kernel_reduce_info[kernel_index].outputs,  output);
	 	//HACK!
	 	if(!strstr(condition,fn_identifier->buffer))
	 	{
	 	        char* ptr = strdup(strtok(condition, "=="));
	 	        char* ptr2 = strtok(NULL, "==");
			char* new_condition = malloc(sizeof(char)*4096);
			remove_suffix(ptr,"___AC_INTERNAL");
	 	        sprintf(new_condition, "vba.kernel_input_params.%s.%s == %s",fn_identifier->buffer,ptr,ptr2);
	 	        condition = strdup(new_condition);
			free(new_condition);
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
	free_op_vec(&reduce_info.ops);
	free_str_vec(&reduce_info.conditions);
	free_str_vec(&reduce_info.outputs);
	free(new_postfix);
	free(tmp);
	free(res_name);
	free(output_str);
}
void
gen_kernel_postfixes(ASTNode* root, const bool gen_mem_accesses,reduce_info* kernel_reduce_info)
{
	reduce_info dfuncs_info[MAX_DFUNCS];
	for(int i = 0; i < MAX_DFUNCS; ++i)
	{
	  init_op_vec( &dfuncs_info[i].ops);
	  init_str_vec(&dfuncs_info[i].conditions);
	  init_str_vec(&dfuncs_info[i].outputs);
	}
	get_dfuncs_reduce_info(root,dfuncs_info);
	gen_kernel_postfixes_recursive(root,gen_mem_accesses,dfuncs_info,kernel_reduce_info);
	for(int i = 0; i < MAX_DFUNCS; ++i)
	{
          free_op_vec( &dfuncs_info[i].ops);
	  free_str_vec(&dfuncs_info[i].conditions);
	  free_str_vec(&dfuncs_info[i].outputs);
	}
}
void
gen_kernel_reduce_outputs(reduce_info* kernel_reduce_info)
{
  size_t kernel_iterator = 0;
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
      	if(kernel_reduce_info[kernel_iterator].outputs.size < (size_t) j+1)
	        fprintf(fp,"%d,",-1);
      	else
	      	fprintf(fp,"(int)%s,",kernel_reduce_info[kernel_iterator].outputs.data[j]);
      }
      fprintf(fp,"%s","},");
      ++kernel_iterator;
    }
  fprintf(fp,"%s","};\n");

  fprintf(fp,"%s","typedef enum KernelReduceOp\n{\n\tNO_REDUCE,\n\tREDUCE_MIN,\n\tREDUCE_MAX,\n\tREDUCE_SUM,\n} KernelReduceOp;\n");
  fprintf(fp,"%s","static const KernelReduceOp kernel_reduce_ops[NUM_KERNELS][NUM_REAL_OUTPUTS] = { ");
  int iterator = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
    {
      fprintf(fp,"%s","{");
      for(int j = 0; j < num_real_reduce_output; ++j)
      {

      	if(kernel_reduce_info[iterator].ops.size < (size_t) j+1)
        	fprintf(fp,"%s,","NO_REDUCE");
	else
	{
      		switch(kernel_reduce_info[iterator].ops.data[j])
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
}
void
gen_kernel_postfixes_and_reduce_outputs(ASTNode* root, const bool gen_mem_accesses)
{
  reduce_info kernel_reduce_info[MAX_KERNELS];
  for(int i = 0; i < 100; ++i)
  {
	  init_str_vec(&kernel_reduce_info[i].outputs);
	  init_op_vec(&kernel_reduce_info[i].ops);
  }
  gen_kernel_postfixes(root,gen_mem_accesses,kernel_reduce_info);

  if(!gen_mem_accesses) gen_kernel_reduce_outputs(kernel_reduce_info);
  for(int i = 0; i < 100; ++i)
  {
	  free_str_vec(&kernel_reduce_info[i].outputs);
	  free_op_vec(&kernel_reduce_info[i].ops);
  }
}
void
gen_kernel_ifs(ASTNode* node, const combinations combinations, string_vec user_kernels_with_input_params,string_vec* user_kernel_combinatorial_params)
{
	if(node->lhs)
		gen_kernel_ifs(node->lhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		gen_kernel_ifs(node->rhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(!(node->type & NODE_KFUNCTION))
		return;
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,get_node(NODE_KFUNCTION_ID,node)->buffer);
	if(kernel_index == -1)
		return;
	string_vec combination_params = user_kernel_combinatorial_params[kernel_index];
	if(combination_params.size == 0)
		return;
	FILE* fp = fopen("user_kernels_ifs.h","a");
	FILE* fp_defs = fopen("user_defines.h","a");
	ASTNode* old_parent = node->parent;
	for(int i = 0; i < combinations.nums[kernel_index]; ++i)
	{
		string_vec combination_vals = combinations.vals[kernel_index + MAX_KERNELS*i];
		char* res = malloc(sizeof(char)*4096);
		char* tmp = malloc(sizeof(char)*4096);
		sprintf(res,"if(kernel_enum == KERNEL_%s ",get_node(NODE_KFUNCTION_ID,node)->buffer);
		for(size_t j = 0; j < combination_vals.size; ++j)
		{
			sprintf(tmp, " && vba.kernel_input_params.%s.%s ==  %s ",get_node(NODE_KFUNCTION_ID,node)->buffer,combination_params.data[j],combination_vals.data[j]);
			strcat(res,tmp);
		}
		strcat(res,")\n{\n\t");
		sprintf(tmp,"return %s_optimized_%d;\n}\n",get_node(NODE_KFUNCTION_ID,node)->buffer,i);
		strcat(res,tmp);
		//sprintf(res,"%sreturn %s_optimized_%d;\n}\n",tmp);
		fprintf(fp_defs,"%s_optimized_%d,",get_node(NODE_KFUNCTION_ID,node)->buffer,i);
		fprintf(fp,"%s",res);
		bool is_left = (old_parent->lhs == node);
		ASTNode* new_parent = astnode_create(NODE_UNKNOWN,node,astnode_dup(node,old_parent));
		char* new_name = malloc(sizeof(char)*4096);
		sprintf(new_name,"%s_optimized_%d",get_node(NODE_KFUNCTION_ID,node)->buffer,i);
		((ASTNode*) get_node(NODE_KFUNCTION_ID,new_parent->rhs))->buffer= strdup(new_name);
		new_parent->rhs->parent = new_parent;
		node->parent = new_parent;
		if(is_left)
			old_parent ->lhs = new_parent;
		else
			old_parent ->rhs = new_parent;
		old_parent = new_parent;
		free(tmp);
		free(res);
		free(new_name);
	}
	printf("NUM of combinations: %d\n",combinations.nums[kernel_index]);
	fclose(fp);
	fprintf(fp_defs,"}\n");
	fclose(fp_defs);
}
void
gen_kernel_input_params(ASTNode* node, bool gen_mem_accesses, const string_vec* vals, string_vec user_kernels_with_input_params, string_vec* user_kernel_combinatorial_params)
{
	if(node->lhs)
		gen_kernel_input_params(node->lhs,gen_mem_accesses,vals,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		gen_kernel_input_params(node->rhs,gen_mem_accesses,vals,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(!(node->type & NODE_INPUT && node->buffer))
		return;
	if(gen_mem_accesses)
	{
		if(!strstr(node->buffer,"AC_INTERNAL_INPUT"))
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

	char* res = malloc(sizeof(char)*4096);
	if(combinations_index == -1)
	{
	  	sprintf(res,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
	  	node->buffer = strdup(res);
		free(res);
		return;
	}
	const string_vec combinations = vals[kernel_index + MAX_KERNELS*combinations_index];
	char* full_name = malloc(sizeof(char)*4096);
	if(node->parent->parent->parent->rhs)
	{
		char* member_str = malloc(sizeof(char)*4096);
		member_str[0] = '\0';
		combine_buffers(node->parent->parent->parent->rhs,member_str);
		sprintf(full_name,"%s.%s",node->buffer,member_str);
		free(member_str);
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
	free(full_name);
	free(res);
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
      char** tqualifiers = malloc(sizeof(char*)*MAX_ID_LEN);
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
	  {
		if(strcmp(tqualifiers[i],"boundary_condition"))
          		fprintf(stream, "%s ", tqualifiers[i]);
	  }

        if (tspec){
          fprintf(stream, "%s ", tspec);
        }
        else if (!get_parent_node(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID) &&
                 !(node->type & NODE_INPUT) &&
                 !(node->type & NODE_KFUNCTION_ID) &&
                 !strstr(node->buffer, "__ldg") &&
		 !(node->type & NODE_NO_AUTO)
		 )
	{
          fprintf(stream, "auto ");
	  const ASTNode* func_call_node = get_parent_node(NODE_FUNCTION_CALL,node);
	  if(func_call_node)
	  {
		if(get_node_by_token(IDENTIFIER,func_call_node->lhs)->id == node->id)
		{
			fprintf(stderr,"Undeclared function used: %s\n",node->buffer);
			exit(EXIT_FAILURE);
		}

	  }
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
	}
      }
      if (!(node->type & NODE_MEMBER_ID))
        add_symbol(node->type, tqualifiers, n_tqualifiers, tspec, node->buffer);

      free(tqualifiers);
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
gen_3d_array_accesses_recursive(ASTNode* node)
{
	if(node->lhs)
		gen_3d_array_accesses_recursive(node->lhs);
	if(node->rhs)
		gen_3d_array_accesses_recursive(node->rhs);
	if(!(node->token == IDENTIFIER))
		return;
	if(!node->buffer)
		return;
	if(!node->parent)
		return;
	if(!node->parent->parent)
		return;
	if(!node->parent->parent->parent)
		return;
	//discard global const declarations
	if(node->parent->parent->parent->type & NODE_ASSIGN_LIST)
		return;
	if(!check_for_vtxbuf(node) && !check_symbol(NODE_VARIABLE_ID,node->buffer,"Field",NULL))
		return;

	char* tmp = malloc(10000*sizeof(char));
	combine_all(node->parent->parent->parent->parent,tmp);
	const char* search_str = "][";
	const char* substr = strstr(tmp,search_str);
	free(tmp);
	if(!substr)
		return;
	ASTNode* base;
	if(check_symbol(NODE_VARIABLE_ID, node->buffer, "VertexBufferHandle*", NULL))
		base = gen_4d_array_access(node);
	else
		base = gen_3d_array_access(node);
	if(!check_symbol(NODE_ANY,node->buffer,NULL,NULL))
        	base->type |= NODE_INPUT;
}
void
gen_3d_array_accesses(ASTNode* root)
{
  symboltable_reset();
  traverse(root, 0, NULL);
  //for(size_t i = 0; i < num_symbols[0]; ++i)
  //{
  //        if(str_vec_contains(symbol_table[i].tqualifiers,"const"))
  //        {
  //      	  printf("specifier: %s\n",symbol_table[i].tspecifier);
  //      	  printf("name: %s\n",symbol_table[i].identifier);
  //        }
  //}
  gen_3d_array_accesses_recursive((ASTNode*)root);
  symboltable_reset();
}

void
gen_dconsts(const ASTNode* root, FILE* stream)
{
  symboltable_reset();
  traverse(root, NODE_FUNCTION | NODE_VARIABLE | NODE_STENCIL | NODE_HOSTDEFINE | NODE_NO_OUT,
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
	char* assignment_val = malloc(sizeof(char)*4098);
	while(def_list_head -> rhs)
	{
		const ASTNode* def = def_list_head->rhs;
		const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
		if(!name) return;
        	const ASTNode* assignment = def->rhs;
		if(!assignment) return;
		combine_all(assignment,assignment_val);
		const char* datatype = tspec->lhs->buffer;
		char* datatype_scalar = remove_substring(strdup(datatype),"*");
		remove_substring(datatype_scalar,"*");
		if(strstr(assignment_val,","))
		{
			fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s[] = {%s};\n#endif\n",datatype_scalar, name, assignment_val);
		}
		else
		{
			fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s = %s;\n#endif\n",datatype_scalar, name, assignment_val);
		}
		def_list_head = def_list_head -> lhs;
		free(datatype_scalar);
	}
	const ASTNode* def = def_list_head->lhs;
	const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
	if(!name) return;
        const ASTNode* assignment = def->rhs;
	if(!assignment) return;
	combine_all(assignment,assignment_val);
	const char* datatype = tspec->lhs->buffer;
	char* datatype_scalar = remove_substring(strdup(datatype),"*");
	if(strstr(assignment_val,","))
	{
		fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s[] = {%s};\n#endif\n",datatype_scalar, name, assignment_val);
	}
	else
	{
		fprintf(fp, "\n#ifdef __cplusplus\nconst %s %s = %s;\n#endif\n",datatype_scalar, name, assignment_val);
	}
	free(assignment_val);
	free(datatype_scalar);
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
    char* cmdoptions = malloc(sizeof(char)*4096);
    cmdoptions[0] = '\0';
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
    char* buf = malloc(sizeof(char)*4096);
    while (fgets(buf, sizeof(buf), proc))
      strcat(sdefinitions, buf);

    pclose(proc);

    strcat(prefix, sdefinitions);
    free(sdefinitions);

    strcat(prefix, dfunctions);

    astnode_set_prefix(prefix, compound_statement);
    free(prefix);
    free(cmdoptions);
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
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, fp);

  symboltable_reset();
  traverse(root, 0, NULL);
  num_fields = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"Field"))
      ++num_fields;


  num_kernels = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    num_kernels += ((symbol_table[i].type & NODE_KFUNCTION_ID) != 0);

  
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
  string_vec field_names;
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
  //free_str_vec(&field_names);






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

  gen_user_taskgraphs(fp,root);

  const char* array_datatypes[] = {"int","AcReal"};
  for (size_t i = 0; i < sizeof(array_datatypes)/sizeof(array_datatypes[0]); ++i) {
  	gen_array_lengths(fp,array_datatypes[i],root);
  	gen_array_is_dconst(fp,array_datatypes[i]);
  	gen_d_offsets(fp,array_datatypes[i],false,root);
  	gen_d_offsets(fp,array_datatypes[i],true,root);
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

  fp = fopen("user_constants.h","w");
  gen_const_variables(root,fp);
  fclose(fp);
  symboltable_reset();
}


static void
gen_user_kernels(const ASTNode* root, const char* out, const bool gen_mem_accesses)
{
  symboltable_reset();
  traverse(root, NODE_NO_OUT, NULL);

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
void
replace_dynamic_coeffs_stencilpoint(ASTNode* node)
{
  if(node->lhs)
    replace_dynamic_coeffs_stencilpoint(node->lhs);
  if(node->buffer)
  {
    if(check_symbol(NODE_DCONST_ID, node->buffer, "AcReal", NULL) || check_symbol(NODE_DCONST_ID, node->buffer, "int", NULL))
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
append_to_identifiers(const char* str_to_append, ASTNode* node, const char* str_to_check)
{
	if(node->lhs)
		append_to_identifiers(str_to_append,node->lhs,str_to_check);
	if(node->rhs)
		append_to_identifiers(str_to_append,node->rhs,str_to_check);
	if(!node->buffer)
		return;
	if(node->token != IDENTIFIER)
		return;
	if(node->type & NODE_FUNCTION_CALL)
		return;
	if(check_symbol(NODE_STENCIL_ID, node->buffer, NULL, NULL))
		return;
	if(check_symbol(NODE_FUNCTION_ID, node->buffer, NULL, NULL))
		return;
	if(strcmp(node->buffer,str_to_check))
		return;
	if(strstr(node->buffer,"AC_INTERNAL"))
		return;
	char* new_name = malloc(sizeof(char)*4000);
	sprintf(new_name,"%s___AC_INTERNAL_%s",node->buffer,str_to_append);
	free(node->buffer);
	node->buffer = strdup(new_name);
	free(new_name);
}
void
rename_local_vars(const char* str_to_append, ASTNode* node, ASTNode* root)
{
	if(node->lhs)
		rename_local_vars(str_to_append,node->lhs,root);
	if(node->rhs)
		rename_local_vars(str_to_append,node->rhs,root);
	if(!(node->type & NODE_DECLARATION))
		return;
	const char* name = strdup(get_node_by_token(IDENTIFIER,node)->buffer);
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	append_to_identifiers(str_to_append,root,name);
}
void
gen_dfunc_internal_names(ASTNode* node)
{

	if(node->lhs)
		gen_dfunc_internal_names(node->lhs);
	if(node->rhs)
		gen_dfunc_internal_names(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
	//to exclude input params
	//rename_local_vars(fn_identifier->buffer,node->rhs->rhs,node->rhs->rhs);
	rename_local_vars(fn_identifier->buffer,node->rhs,node->rhs);
}
void
rename_all(const char* to_rename, const char* new_name, ASTNode* node)
{
	if(node->lhs)
		rename_all(to_rename, new_name,node->lhs);
	if(node->rhs)
		rename_all(to_rename, new_name,node->rhs);
	char** src;
	if(node->buffer && !strcmp(to_rename,node->buffer))
		src = &node->buffer;
	else if(node->infix && !strcmp(to_rename,node->infix))
		src = &node->infix;
	else if(node->postfix && !strcmp(to_rename,node->postfix))
		src = &node->postfix;
	else if(node->prefix&& !strcmp(to_rename,node->prefix))
		src = &node->prefix;
	else
		return;
	free(*src);
	*src = strdup(new_name);
}
void
remove_nodes(const NodeType type, ASTNode* node)
{
	if(node->lhs)
		remove_nodes(type,node->lhs);
	if(node->rhs)
		remove_nodes(type,node->rhs);
	if(!(node->type & type))
		return;
	node->lhs = NULL;
	node->rhs = NULL;
}
void
gen_dfunc_macros(ASTNode* node)
{
	if(node->lhs)
		gen_dfunc_macros(node->lhs);
	if(node->rhs)
		gen_dfunc_macros(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	FILE* fp = fopen("user_dfuncs.h","a");
	if(node->rhs->lhs)
	{
		add_node_type(NODE_NO_AUTO, node->rhs->lhs, NULL);
		rename_all("const ","",node->rhs->lhs);
		rename_all("&","",node->rhs->lhs);
		remove_nodes(NODE_TSPEC,node->rhs->lhs);
		remove_nodes(NODE_TQUAL,node->rhs->lhs);
	}
	rename_all("return","",node->rhs->rhs);
	free(node->prefix);
	free(node->postfix);
	free(node->infix);
	node->prefix = strdup("#define ");
	node->infix = strdup("");
	node->postfix= strdup(")\n");
	node->rhs->infix = strdup(")(");
	traverse(node,NODE_NO_OUT,fp);
	fclose(fp);
}
void
remove_auto_from_func_calls(const char* func_name, ASTNode* node)
{
	if(node->lhs)
		remove_auto_from_func_calls(func_name,node->lhs);
	if(node->rhs)
		remove_auto_from_func_calls(func_name,node->rhs);
	if(!(node->type & NODE_FUNCTION_CALL))
		return;
	ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node->lhs);
	if(!strcmp(func_name,fn_identifier->buffer))
		fn_identifier->type |= NODE_NO_AUTO;
}
void
remove_inlined_dfunc_nodes(ASTNode* node, ASTNode* root)
{
	if(node->lhs)
		remove_inlined_dfunc_nodes(node->lhs,root);
	if(node->rhs)
		remove_inlined_dfunc_nodes(node->rhs,root);
	if(!(node->type & NODE_DFUNCTION))
		return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
	if(!check_symbol(NODE_DFUNCTION_ID, fn_identifier->buffer, NULL, "inline"))
		return;
	node->lhs = NULL;
	node->rhs = NULL;
	node->prefix = NULL;
	node->infix= NULL;
	node->postfix= NULL;
	remove_auto_from_func_calls(fn_identifier->buffer,root);
}
void
remove_constexpr(ASTNode* node)
{
	if(node->lhs)
		remove_constexpr(node->lhs);
	if(node->rhs)
		remove_constexpr(node->rhs);
	if(node->buffer)
		remove_substring(node->buffer,"constexpr");
}
void
transform_arrays_to_std_arrays(ASTNode* node)
{
	if(node->lhs)
		transform_arrays_to_std_arrays(node->lhs);
	if(node->rhs)
		transform_arrays_to_std_arrays(node->rhs);
	if(!(node->type & NODE_DECLARATION))
		return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node->lhs);
	if(!tspec)
		return;
	if(!get_parent_node(NODE_FUNCTION,node))
		return;
	if(!node->rhs)
		return;
	if(!node->rhs->lhs)
		return;
	if(!node->rhs->lhs->rhs)
		return;
	char* size = malloc(sizeof(char)*4098);
	const char* type = tspec->lhs->buffer;
	combine_all(node->rhs->lhs->rhs,size);
	node->rhs->lhs->infix = NULL;
	node->rhs->lhs->postfix= NULL;
	node->rhs->lhs->rhs = NULL;
	char* new_res = malloc(sizeof(char)*4098);
	sprintf(new_res,"std::array<%s,%s>",type,size);
	tspec->lhs->buffer = strdup(new_res);
	free(size);
	free(new_res);
	//remove unneeded braces if assignment
	if(node->parent->type & NODE_ASSIGNMENT && node->parent->rhs)
		node->parent->rhs->prefix = NULL;
}
void
gen_kernel_combinatorial_optimizations_and_input(ASTNode* root, const bool gen_mem_accesses, const bool optimize_conditionals)
{
  string_vec user_kernel_combinatorial_params[100];
  string_vec user_kernels_with_input_params;


  int nums[100] = {0};
  string_vec* vals = malloc(sizeof(string_vec)*MAX_KERNELS*MAX_COMBINATIONS);
  combinations param_in = {nums, vals};

  init_str_vec(&user_kernels_with_input_params);
  for(int i = 0; i < 100; ++i)
    init_str_vec(&user_kernel_combinatorial_params[i]);
  gen_kernel_num_of_combinations(root,param_in,&user_kernels_with_input_params,user_kernel_combinatorial_params);
  if(optimize_conditionals)
  	gen_kernel_ifs(root,param_in,user_kernels_with_input_params,user_kernel_combinatorial_params);
  gen_kernel_input_params(root,gen_mem_accesses,param_in.vals,user_kernels_with_input_params,user_kernel_combinatorial_params);



  free_str_vec(&user_kernels_with_input_params);
  free(param_in.vals);
  for(int i = 0; i < 100; ++i)
	  free_str_vec(&user_kernel_combinatorial_params[i]);
}
void
generate(const ASTNode* root_in, FILE* stream, const bool gen_mem_accesses, const bool optimize_conditionals)
{ 
  ASTNode* root = astnode_dup(root_in,NULL);
  assert(root);

  gen_kernel_structs(root,gen_mem_accesses);
  gen_user_structs(root);
  gen_user_defines(root, "user_defines.h");
  gen_user_kernels(root, "user_declarations.h", gen_mem_accesses);

  gen_dfunc_internal_names(root);
  gen_3d_array_accesses(root);
  gen_kernel_combinatorial_optimizations_and_input(root,gen_mem_accesses,optimize_conditionals);


  // Fill the symbol table
  traverse(root, NODE_NO_OUT, NULL);


  gen_kernel_postfixes_and_reduce_outputs(root,gen_mem_accesses);

  // print_symbol_table();

  // Generate user_kernels.h
  fprintf(stream, "#pragma once\n");

  size_t num_stencils = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      ++num_stencils;




  // Device constants
  // gen_dconsts(root, stream);
  const char* array_datatypes[] = {"int","AcReal"};
  for (size_t i = 0; i < sizeof(array_datatypes)/sizeof(array_datatypes[0]); ++i)
  	gen_array_reads(root,gen_mem_accesses,array_datatypes[i]);

  // Stencils

  // Stencil generator
  FILE* stencilgen = fopen(STENCILGEN_HEADER, "w");
  assert(stencilgen);

  // Stencil ops
  symboltable_reset();
  traverse(root, NODE_NO_OUT, NULL);
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
               NODE_HOSTDEFINE | NODE_NO_OUT,
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
               NODE_HOSTDEFINE | NODE_NO_OUT,
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
  //gen_dfunc_internal_names(root);
  transform_arrays_to_std_arrays(root);
  gen_dfunc_macros(astnode_dup(root,NULL));
  remove_inlined_dfunc_nodes(root,root);
  if(gen_mem_accesses) remove_constexpr(root);
  symboltable_reset();
  char* dfunctions;
  size_t sizeloc;
  FILE* dfunc_fp = open_memstream(&dfunctions, &sizeloc);
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_KFUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           dfunc_fp);
  fflush(dfunc_fp);

  // Kernels
  symboltable_reset();
  gen_kernels(root, dfunctions, gen_mem_accesses);
  fclose(dfunc_fp); // Frees dfunctions also

  symboltable_reset();
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           stream);

  // print_symbol_table();
  free(written_fields);
  free(read_fields);
  free(field_has_stencil_op);
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

  FILE* fp = fopen("user_written_fields.bin", "rb");
  written_fields = (int*)malloc(num_kernels*num_fields*sizeof(int));
  fread(written_fields, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);

  fp = fopen("user_read_fields.bin", "rb");
  read_fields = (int*)malloc(num_kernels*num_fields*sizeof(int));
  fread(read_fields, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);

  fp = fopen("user_field_has_stencil_op.bin", "rb");
  field_has_stencil_op = (int*)malloc(num_kernels*num_fields*sizeof(int));
  fread(field_has_stencil_op, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);


//  const size_t k = 8;
//  for(size_t i = 0; i < num_fields; ++i)
//	  printf("written to: %d\n",written_fields[i + num_fields*k]);
//  for(size_t i = 0; i < num_fields; ++i)
//	  printf("read from: %d\n",read_fields[i + num_fields*k]);
//  for(size_t i = 0; i < num_fields; ++i)
//	  printf("has stencil op: %d\n",field_has_stencil_op[i + num_fields*k]);
}
