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
static inline char*
to_upper_case(const char* src)
{
	char* res = strdup(src);
	res[0] = (res[0] == '\0') ? res[0] : toupper(res[0]);
	return res;
}

static int* written_fields = NULL;
static int* read_fields   = NULL;
static int* field_has_stencil_op = NULL;
static size_t num_fields = 0;
static size_t num_kernels = 0;
static size_t num_dfuncs = 0;


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
get_symbol_by_index(const NodeType type, const int index, const char* tspecifier)
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
static const Symbol*
get_symbol(const NodeType type, const char* symbol, const char* tspecifier)
{
	return get_symbol_by_index(type,get_symbol_index(type,symbol,tspecifier), tspecifier);
}

static int 
add_symbol(const NodeType type, char* const* tqualifiers, const size_t n_tqualifiers, const char* tspecifier,
           const char* id)
{
  assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);

  symbol_table[num_symbols[current_nest]].type          = type;
  symbol_table[num_symbols[current_nest]].tspecifier[0] = '\0';

  
  init_str_vec(&symbol_table[num_symbols[current_nest]].tqualifiers);
  for(size_t i = 0; i < n_tqualifiers; ++i)
      	push(&symbol_table[num_symbols[current_nest]].tqualifiers,tqualifiers[i]);
  if(
        tspecifier && 
	(!strcmp(tspecifier,"AcReal*")  || 
	 !strcmp(tspecifier,"int*")     || 
	 !strcmp(tspecifier,"bool*")    || 
	 !strcmp(tspecifier,"int3*")    || 
	 !strcmp(tspecifier,"AcReal3*") || 
	 !strcmp(tspecifier,"int")      || 
	 !strcmp(tspecifier,"AcReal")   || 
	 !strcmp(tspecifier,"bool")     || 
	 !strcmp(tspecifier,"AcReal3")  || 
	 !strcmp(tspecifier,"int3")) && 
	n_tqualifiers==0 && current_nest == 0)
    push(&symbol_table[num_symbols[current_nest]].tqualifiers,"dconst");

  if (tspecifier)
    strcpy(symbol_table[num_symbols[current_nest]].tspecifier, tspecifier);

  strcpy(symbol_table[num_symbols[current_nest]].identifier, id);

  ++num_symbols[current_nest];
  const bool is_field_without_type_qualifiers = tspecifier && !strcmp(tspecifier,"Field") && symbol_table[num_symbols[current_nest]].tqualifiers.size == 0;
  const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
  if(!is_field_without_type_qualifiers)
  	return num_symbols[current_nest]-1;
	  
  if(!has_optimization_info)
  {
  	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, "communicated");
  	return num_symbols[current_nest]-1;
  } 



   const int field_index = get_symbol_index(NODE_VARIABLE_ID, id, "Field");
   bool is_auxiliary = true;
   bool is_communicated = false;
   for(size_t k = 0; k < num_kernels; ++k)
   {
	   is_auxiliary &= (!written_fields[field_index + num_fields*k] || !field_has_stencil_op[field_index + num_fields*k]);
	   is_communicated |= field_has_stencil_op[field_index + num_fields*k];
   }
   if(is_auxiliary)
   {
	   push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, "auxiliary");
	   if(is_communicated)
	   	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, "communicated");
   }
   else
   {
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
								  
  add_symbol(NODE_VARIABLE_ID, NULL, 0, NULL, "AC_INTERNAL_big_real_array");
  add_symbol(NODE_VARIABLE_ID, NULL, 0, NULL, "AC_INTERNAL_big_int_array");

  // add_symbol(NODE_UNKNOWN, NULL, NULL, "true");
  // add_symbol(NODE_UNKNOWN, NULL, NULL, "false");

  char* field3_tq[1] = {"Field3"};
  char* real_tq[1]   = {"AcReal"};
  char* real3_tq[1]  = {"AcReal3"};
  char* const_tq[1]  = {"const"};
  add_symbol(NODE_FUNCTION_ID, real_tq, 1, NULL, "previous_base");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "write_base");  // TODO RECHECK
							 //
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "reduce_sum");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "reduce_min");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "reduce_max");  // TODO RECHECK
  //In develop
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "read_w");
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write_w");
  add_symbol(NODE_FUNCTION_ID, field3_tq, 1, NULL, "MakeField3"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, real_tq,  1, NULL, "AC_dot");    // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, real3_tq, 1, NULL, "cross");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, real_tq,  1, NULL, "len");    // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "uint64_t");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "UINT64_MAX"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "rand_uniform");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "multm2_sym");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "diagonal");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "sum");   // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "AC_REAL_PI");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "NUM_FIELDS");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "NUM_VTXBUF_HANDLES");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "NUM_ALL_FIELDS");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "FIELD_IN");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "FIELD_OUT");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, "IDX");

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, "bool", "true");
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, "bool", "false");

  // Astaroth 2.0 backwards compatibility START
  // (should be actually built-in externs in acc-runtime/api/acc-runtime.h)
  char* tqualifiers[1] = {"dconst"};
  char* const_qualifier[1] = {"const"};

  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1,"int", "AC_xy_plate_bufsize");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1,"int", "AC_xz_plate_bufsize");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1,"int", "AC_yz_plate_bufsize");


  //For special reductions
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", "AC_center_x");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", "AC_center_y");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", "AC_center_z");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", "AC_sum_radius");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", "AC_window_radius");

  // (BC types do not belong here, BCs not handled with the DSL)
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_bc_type_bot_x");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_bc_type_bot_y");
#if TWO_D == 0
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_bc_type_bot_z");
#endif

  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_bc_type_top_x");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_bc_type_top_y");
#if TWO_D == 0
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_bc_type_top_z");
#endif
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", "AC_init_type");
  add_symbol(NODE_VARIABLE_ID, const_qualifier, 1, "int", "STENCIL_ORDER");
  // Astaroth 2.0 backwards compatibility END
  int index = add_symbol(NODE_VARIABLE_ID, NULL, 0 , "int3", "blockDim");
  symbol_table[index].tqualifiers.size = 0;
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
             !strcmp(symbol_table[i].tspecifier,"Kernel") ? "kernel" : "device");

    if (str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
      printf("(dconst)");

    if (!strcmp(symbol_table[i].tspecifier,"Stencil"))
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
	if(strstr(name,"Ac")) return name;
	char* res = malloc(sizeof(char)* (strlen(name) + strlen("Ac") + 100));
	sprintf(res,"Ac%s",to_upper_case(name));
	return res;
}
const char*
convert_to_define_name(const char* name)
{
	char* res = strdup(name);
	if(strlen(res) > 2 && res[0]  == 'A' && res[1] == 'c')
	{
		res = &res[2];
		res[0] = tolower(res[0]);
	}
	return res;
}
void
get_array_acccesses_recursive(const ASTNode* node, string_vec* dst)
{
	if(node->lhs)
		get_array_acccesses_recursive(node->lhs,dst);
	if(node->rhs)
		get_array_acccesses_recursive(node->rhs,dst);
	if(node->type == NODE_ARRAY_ACCESS)
	{
		char* tmp = malloc(sizeof(char)*10000);
		combine_all(node->rhs,tmp);
		push(dst,tmp);
	}
}
string_vec
get_array_accesses(const ASTNode* base)
{
	    string_vec dst = VEC_INITIALIZER;
	    if(!base) return dst;
	    get_array_acccesses_recursive(base,&dst);
	    return dst;

}
string_vec 
get_array_var_dims(const char* var_in, const ASTNode* root)
{
	    char* var = strdup(var_in);
	    strip_whitespace(var);
	
	    const ASTNode* var_identifier = get_node_by_buffer_and_type(var,NODE_VARIABLE_ID,root);
	    const ASTNode* decl = get_parent_node(NODE_DECLARATION,var_identifier);

	    const ASTNode* access_start = get_node(NODE_ARRAY_ACCESS,decl);
	    return get_array_accesses(access_start);
}
void
get_array_var_length(const char* var, const ASTNode* root, char* dst)
{
	    sprintf(dst,"%s","");
	    string_vec tmp = get_array_var_dims(var,root);
	    for(size_t i = 0; i < tmp.size; ++i)
	    {
		    if(i) strcat(dst,"*");
		    strcat(dst,tmp.data[i]);
	    }
	    free_str_vec(&tmp);
}

void
gen_array_info(FILE* fp, const char* datatype_scalar, const ASTNode* root)
{

  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  const char* define_name =  convert_to_define_name(datatype_scalar);
  {
  	char running_offset[4096];
  	  sprintf(running_offset,"0");
  	  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  {
  	    if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	        !strcmp(symbol_table[i].tspecifier,datatype) && str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
  	    {
  	            fprintf(fp,"\n#ifndef %s_offset\n#define %s_offset (%s)\n#endif\n",symbol_table[i].identifier,symbol_table[i].identifier,running_offset);
  	            char array_length_str[4098];
  	            get_array_var_length(symbol_table[i].identifier,root,array_length_str);
		    strcat(running_offset,"+");
		    strcat(running_offset,array_length_str);
  	    }
  	  }
  	  fprintf(fp,"\n#ifndef D_%s_ARRAYS_LEN\n#define D_%s_ARRAYS_LEN (%s)\n#endif\n", strupr(define_name), strupr(define_name),running_offset);
  }
  char running_offset[4096];
  sprintf(running_offset,"0");
  fprintf(fp, "static const array_info %s_array_info[] __attribute__((unused)) = {", convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(
		str_vec_contains(symbol_table[i].tqualifiers,"const") ||
		str_vec_contains(symbol_table[i].tqualifiers,"run_const")
	) continue;
	char array_length_str[4098];
	get_array_var_length(symbol_table[i].identifier,root,array_length_str);
	fprintf(fp,"%s","{");

	if(str_vec_contains(symbol_table[i].tqualifiers,"gmem")) fprintf(fp, "(int)%s,", array_length_str);
	else fprintf(fp, "%s,", array_length_str);

        if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))  fprintf(fp,"true,");
        else fprintf(fp, "false,");

        if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
  	{
  	        fprintf(fp,"%s,",running_offset);
		strcat(running_offset,"+");
		strcat(running_offset,array_length_str);
  	}
	else
	{
  		fprintf(fp,"%d,",-1);
	}

	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
      	fprintf(fp, "%lu,", dims.size);

	fprintf(fp,"%s","{");
	for(size_t dim = 0; dim < 3; ++dim)
	{
		if(dim >= dims.size) fprintf(fp,"%s,","-1");
		else fprintf(fp,"%s,",dims.data[dim]);
	}
	fprintf(fp,"%s","},");

	free_str_vec(&dims);
        fprintf(fp, "\"%s\",", symbol_table[i].identifier);
	fprintf(fp,"%s","},");
    }
  }

  //runtime array lengths come after other arrays
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(!str_vec_contains(symbol_table[i].tqualifiers,"run_const")) continue;
	char array_length_str[4098];
	get_array_var_length(symbol_table[i].identifier,root,array_length_str);
	fprintf(fp,"%s","{");
      	fprintf(fp, "%s,", array_length_str);
        fprintf(fp, "false,");
  	fprintf(fp,"%d,",-1);

	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
      	fprintf(fp, "%lu,", dims.size);

	fprintf(fp,"%s","{");
	for(size_t dim = 0; dim < 3; ++dim)
	{
		if(dim >= dims.size) fprintf(fp,"%s,","-1");
		else fprintf(fp,"%s,",dims.data[dim]);
	}
	fprintf(fp,"%s","},");
	free_str_vec(&dims);

        fprintf(fp, "\"%s\",", symbol_table[i].identifier);
	fprintf(fp,"%s","},");
    }
  }
  //pad one extra to silence warnings
  fprintf(fp,"{-1,false,-1,-1,{-1,-1,-1},\"AC_EXTRA_PADDING\"}");
  fprintf(fp, "};");

}

/**
void
gen_array_dims(FILE* fp, const char* datatype_scalar, const ASTNode* root)
{
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  fprintf(fp, "static const int %s_array_num_dims[] __attribute__((unused)) = {", convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(str_vec_contains(symbol_table[i].tqualifiers,"const") || str_vec_contains(symbol_table[i].tqualifiers,"run_const")) continue;
	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
      	fprintf(fp, "%lu,", dims.size);
	free_str_vec(&dims);
    }

  //runtime array lengths come after other arrays
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(!str_vec_contains(symbol_table[i].tqualifiers,"run_const")) continue;
	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
      	fprintf(fp, "%lu,", dims.size);
	free_str_vec(&dims);
    }
  fprintf(fp, "};\n");

  fprintf(fp, "static const int3 %s_array_dims[] __attribute__((unused)) = {", convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(str_vec_contains(symbol_table[i].tqualifiers,"const") || str_vec_contains(symbol_table[i].tqualifiers,"run_const")) continue;
	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
	fprintf(fp,"%s","{");
	for(size_t dim = 0; dim < 3; ++dim)
	{
		if(dim >= dims.size) fprintf(fp,"%s,","-1");
		else fprintf(fp,"%s,",dims.data[dim]);
	}
	fprintf(fp,"%s","},");
	free_str_vec(&dims);
    }

  //runtime array lengths come after other arrays
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(!str_vec_contains(symbol_table[i].tqualifiers,"run_const")) continue;
	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
	fprintf(fp,"%s","{");
	for(size_t dim = 0; dim < 3; ++dim)
	{
		if(dim >= dims.size) fprintf(fp,"%s,","-1");
		else fprintf(fp,"%s,",dims.data[dim]);
	}
	fprintf(fp,"%s","},");
	free_str_vec(&dims);
    }
  fprintf(fp, "};\n");
}
**/

void
gen_dmesh_declarations(const char* datatype_scalar)
{
	FILE* fp = fopen("device_mesh_info_decl.h","a");
	fprintf(fp,"%s %s_params[NUM_%s_PARAMS];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
}

void
gen_array_declarations(const char* datatype_scalar, const bool gen_mem_accesses)
{
	char tmp[7000];

	sprintf(tmp,"%s* %s_arrays[NUM_%s_ARRAYS+1];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	file_append("array_decl.h",tmp);

	sprintf(tmp,"if constexpr(std::is_same<P,%sArrayParam>::value) return vba.%s_arrays[(int)param];\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	file_append("get_vba_array.h",tmp);

	sprintf(tmp,"if constexpr(std::is_same<P,%sCompParam>::value) return (%s){};\n",convert_to_enum_name(datatype_scalar),datatype_scalar);
	file_append("get_default_value.h",tmp);

	sprintf(tmp,"if constexpr(std::is_same<P,%sCompArrayParam>::value) return (%s){};\n",convert_to_enum_name(datatype_scalar),datatype_scalar);
	file_append("get_default_value.h",tmp);


	sprintf(tmp,"if constexpr(std::is_same<P,%sCompArrayParam>::value) return config.%s_arrays[(int)param];\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	file_append("get_from_comp_config.h",tmp);

	sprintf(tmp,"if constexpr(std::is_same<P,%sCompParam>::value) return config.%s_params[(int)param];\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	file_append("get_from_comp_config.h",tmp);


	FILE* fp = fopen("get_config_param.h","a");
	fprintf(fp,"if constexpr(std::is_same<P,%sParam>::value) return config.%s_params[(int)param];\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	fprintf(fp,"if constexpr(std::is_same<P,%sArrayParam>::value) return config.%s_arrays[(int)param];\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	fclose(fp);

	fp = fopen("memcpy_to_gmem_arrays.h","a");
	fprintf(fp,"if constexpr(std::is_same<P,%sArrayParam>::value) cudaMemcpyToSymbol(gmem_%s_arrays[(int)param], &ptr, sizeof(ptr), 0, cudaMemcpyHostToDevice);\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	fclose(fp);

	fp = fopen("memcpy_from_gmem_arrays.h","a");
	fprintf(fp,"if constexpr(std::is_same<P,%sArrayParam>::value) cudaMemcpyFromSymbol(&ptr,gmem_%s_arrays[(int)param],sizeof(ptr), 0, cudaMemcpyDeviceToHost);\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	fclose(fp);


	sprintf(tmp,"if constexpr(std::is_same<P,%sCompParam>::value) return %s_comp_param_names[(int)param];\n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	file_append("get_param_name.h",tmp);


	fp = fopen("get_num_params.h","a");
	fprintf(fp," (std::is_same<P,%sParam>::value)      ? NUM_%s_PARAMS : \n",convert_to_enum_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fprintf(fp," (std::is_same<P,%sArrayParam>::value) ? NUM_%s_ARRAYS : \n",convert_to_enum_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));

	fprintf(fp," (std::is_same<P,%sCompParam>::value)      ? NUM_%s_COMP_PARAMS : \n",convert_to_enum_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fprintf(fp," (std::is_same<P,%sCompArrayParam>::value) ? NUM_%s_COMP_ARRAYS : \n",convert_to_enum_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
	
	fp = fopen("get_array_info.h","a");
	fprintf(fp," if(std::is_same<P,%sArrayParam>::value) return %s_array_info[(int)array]; \n",convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar));
	fprintf(fp," if(std::is_same<P,%sCompArrayParam>::value) return %s_array_info[(int)array + NUM_%s_ARRAYS]; \n",
	convert_to_enum_name(datatype_scalar),convert_to_define_name(datatype_scalar), strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);

	fp = fopen("dconst_decl.h","a");
	fprintf(fp,"%s __device__ __forceinline__ DCONST(const %sParam& param){return d_mesh_info.%s_params[(int)param];}\n"
			,datatype_scalar, convert_to_enum_name(datatype_scalar), convert_to_define_name(datatype_scalar));
	fclose(fp);

	fp = fopen("dconst_accesses_decl.h","a");
	fprintf(fp,"%s  DCONST(const %sParam& param){%s res{}; return res; }\n"
			,datatype_scalar, convert_to_enum_name(datatype_scalar), datatype_scalar);
	fclose(fp);
	fp = fopen("get_address.h","a");
	fprintf(fp,"size_t  get_address(const %sParam& param){ return (size_t)&d_mesh_info.%s_params[(int)param];}\n"
			,convert_to_enum_name(datatype_scalar), convert_to_define_name(datatype_scalar));
	fclose(fp);
	fp = fopen("load_and_store_array.h","a");
	fprintf(fp,"cudaError_t load_array(const %s* values, const size_t bytes, const size_t offset)"
			"{ return cudaMemcpyToSymbol(d_%s_arrays,values,bytes,offset,cudaMemcpyHostToDevice); }\n"
			,datatype_scalar, convert_to_define_name(datatype_scalar));
	fprintf(fp,"cudaError_t store_array(%s* values, const size_t bytes, const size_t offset)"
			"{ return cudaMemcpyFromSymbol(values,d_%s_arrays,bytes,offset,cudaMemcpyDeviceToHost); }\n"
			,datatype_scalar, convert_to_define_name(datatype_scalar));
	fclose(fp);

	//we pad with 1 since zero sized arrays are not allowed with some CUDA compilers
	fp = fopen("dconst_arrays_decl.h","a");
	if(gen_mem_accesses)
		fprintf(fp,"%s d_%s_arrays[D_%s_ARRAYS_LEN+1] {};\n",datatype_scalar, convert_to_define_name(datatype_scalar), strupr(convert_to_define_name(datatype_scalar)));
	else
		fprintf(fp,"__device__ __constant__ %s d_%s_arrays[D_%s_ARRAYS_LEN+1];\n",datatype_scalar, convert_to_define_name(datatype_scalar), strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);

	fp = fopen("gmem_arrays_decl.h","a");
	if(gen_mem_accesses)
		fprintf(fp,"%s gmem_%s_arrays[NUM_%s_ARRAYS][1000] {};\n",datatype_scalar, convert_to_define_name(datatype_scalar), strupr(convert_to_define_name(datatype_scalar)));
	else
		fprintf(fp,"__device__ __constant__ %s* gmem_%s_arrays[NUM_%s_ARRAYS+1];\n",datatype_scalar, convert_to_define_name(datatype_scalar), strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);




	fp = fopen("load_and_store_uniform_overloads.h","a");
	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acLoadUniform(const cudaStream_t stream, const %sParam param, const %s value) { return acLoad%sUniform(stream,param,value);}\n"
		,convert_to_enum_name(datatype_scalar), datatype_scalar, to_upper_case(convert_to_define_name(datatype_scalar)));

	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acLoadUniform(const cudaStream_t stream, const %sArrayParam param, const %s* values, const size_t length) { return acLoad%sArrayUniform(stream,param,values,length);}\n"
		,convert_to_enum_name(datatype_scalar), datatype_scalar, to_upper_case(convert_to_define_name(datatype_scalar)));

	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acStoreUniform(const cudaStream_t stream, const %sParam param, %s* value) { return acStore%sUniform(stream,param,value);}\n"
		,convert_to_enum_name(datatype_scalar), datatype_scalar, to_upper_case(convert_to_define_name(datatype_scalar)));

	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acStoreUniform(const %sArrayParam param, %s* values, const size_t length) { return acStore%sArrayUniform(param,values,length);}\n"
		,convert_to_enum_name(datatype_scalar), datatype_scalar, to_upper_case(convert_to_define_name(datatype_scalar)));
	fclose(fp);

	fp = fopen("load_and_store_uniform_funcs.h","a");
	fprintf(fp, "AcResult acLoad%sUniform(const cudaStream_t, const %sParam param, const %s value) { return acLoadUniform(param,value); }\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 
	fprintf(fp, "AcResult acLoad%sArrayUniform(const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length) { return acLoadArrayUniform(param ,values, length); }\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 
	fprintf(fp, "AcResult acStore%sUniform(const cudaStream_t, const %sParam param, %s* value) { return acStoreUniform(param,value); }\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 
	fprintf(fp, "AcResult acStore%sArrayUniform(const %sArrayParam param, %s* values, const size_t length) { return acStoreArrayUniform(param ,values, length); }\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 

	fclose(fp);
	
	fp = fopen("load_and_store_uniform_header.h","a");
	fprintf(fp, "FUNC_DEFINE(AcResult, acLoad%sUniform,(const cudaStream_t, const %sParam param, const %s value));\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 

	fprintf(fp, "FUNC_DEFINE(AcResult, acLoad%sArrayUniform, (const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length));\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 

	fprintf(fp, "FUNC_DEFINE(AcResult, acStore%sUniform,(const cudaStream_t, const %sParam param, %s* value));\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 

	fprintf(fp, "FUNC_DEFINE(AcResult, acStore%sArrayUniform, (const %sArrayParam param, %s* values, const size_t length));\n",
			to_upper_case(convert_to_define_name(datatype_scalar)), convert_to_enum_name(datatype_scalar), datatype_scalar); 
	fclose(fp);


	if(!strcmp(datatype_scalar,"int")) return;
	fp = fopen("scalar_types.h","a");
	fprintf(fp,"%sParam,\n",convert_to_enum_name(datatype_scalar));
	fclose(fp);

	fp = fopen("scalar_comp_types.h","a");
	fprintf(fp,"%sCompParam,\n",convert_to_enum_name(datatype_scalar));
	fclose(fp);

	fp = fopen("array_types.h","a");
	fprintf(fp,"%sArrayParam,\n",convert_to_enum_name(datatype_scalar));
	fclose(fp);

	fp = fopen("array_comp_types.h","a");
	fprintf(fp,"%sCompArrayParam,\n",convert_to_enum_name(datatype_scalar));
	fclose(fp);
}

void
gen_comp_declarations(const char* datatype_scalar)
{
	FILE* fp = fopen("comp_decl.h","a");
	fprintf(fp,"%s %s_params[NUM_%s_COMP_PARAMS];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fprintf(fp,"const %s* %s_arrays[NUM_%s_COMP_ARRAYS];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
	fp = fopen("comp_loaded_decl.h","a");
	fprintf(fp,"bool %s_params[NUM_%s_COMP_PARAMS];\n",convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fprintf(fp,"bool  %s_arrays[NUM_%s_COMP_ARRAYS];\n",convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
}

void
gen_input_declarations(const char* datatype_scalar)
{
	FILE* fp = fopen("input_decl.h","a");
	fprintf(fp,"const %s %s_params[NUM_%s_INPUT_PARAMS+1];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
}




void
gen_enums(FILE* fp, const char* datatype_scalar)
{
  char datatype_arr[1000];
  sprintf(datatype_arr,"%s*",datatype_scalar);

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_PARAMS} %sParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && (str_vec_contains(symbol_table[i].tqualifiers,"dconst") || str_vec_contains(symbol_table[i].tqualifiers,"gmem")))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_ARRAYS} %sArrayParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && str_vec_contains(symbol_table[i].tqualifiers,"output"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_OUTPUTS} %sOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && str_vec_contains(symbol_table[i].tqualifiers,"output"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_OUTPUT_ARRAYS} %sArrayOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && str_vec_contains(symbol_table[i].tqualifiers,"run_const"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_COMP_PARAMS} %sCompParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && str_vec_contains(symbol_table[i].tqualifiers,"run_const"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_COMP_ARRAYS} %sCompArrayParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

}
void
gen_param_names(FILE* fp, const char* datatype_scalar)
{
  char datatype_arr[1000];
  sprintf(datatype_arr,"%s*",datatype_scalar);

  fprintf(fp, "static const char* %sparam_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && str_vec_contains(symbol_table[i].tqualifiers,"output"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_array_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && str_vec_contains(symbol_table[i].tqualifiers,"output"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_comp_param_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && str_vec_contains(symbol_table[i].tqualifiers,"run_const"))
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
	if(check_symbol(NODE_VARIABLE_ID, node->buffer, "Field", NULL) || check_symbol(NODE_VARIABLE_ID, node->buffer, "Field*", NULL))
		return true;
	const ASTNode* func = get_parent_node(NODE_FUNCTION,node);
	if(!func)
		return false;
	const ASTNode* param_list = func->rhs->lhs;
	if(!param_list)
		return false;
	char* kernel_search_buffer = strdup(node->buffer);
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
	const bool is_vtxbuf = !strcmp(tspec->lhs->buffer,"Field");
	if(!is_vtxbuf)
		return false;
	return true;
}
static int int_log2(int x)
{

	int res = 0;
	while (x >>= 1) ++res;
	return res;
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
	free(base->buffer);
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


        base->rhs = rhs;
	rhs->prefix= strdup("]");

	ASTNode* lhs = astnode_create(NODE_UNKNOWN, NULL, NULL);
	base->lhs = lhs;
	lhs->parent = base;
        lhs->buffer = strdup(node->buffer);
	lhs->token = IDENTIFIER;
	lhs->type |= node->type & NODE_INPUT;

	free(res);
	return base;
}


char*
get_index(const ASTNode* array_access_start, const string_vec var_dims, const bool has_dconst_dims)
{
    string_vec array_accesses = get_array_accesses(array_access_start);
    char* index = malloc(sizeof(char)*4098);
    sprintf(index,"%s","");
    for(size_t j = 0; j < array_accesses.size; ++j)
    {
    	if(j)
    	{
    		strcat(index,"+");
		if(has_dconst_dims)
    			strcat(index,"DCONST");
    		strcat(index,"(");
    		for(size_t k = 0; k < j; ++k)
    		{
    			if(k) strcat(index,"*");
    			strcat(index,var_dims.data[k]);
    		}
    		strcat(index,")*");
    	}
	strcat(index,"(");
    	strcat(index,array_accesses.data[j]);
	strcat(index,")");
    }
    free_str_vec(&array_accesses);
    return index;
}

void
gen_array_reads(const ASTNode* root, ASTNode* node, const char* datatype_scalar)
{
  if(node->lhs)
    gen_array_reads(root,node->lhs,datatype_scalar);
  if(node->rhs)
    gen_array_reads(root,node->rhs,datatype_scalar);
  if(node->type != NODE_ARRAY_ACCESS)
	  return;
  if(!node->lhs) return;
  if(get_parent_node(NODE_VARIABLE,node)) return;
  char* array_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
  char* datatype = malloc(sizeof(char)*(1000));
  sprintf(datatype,"%s*",datatype_scalar);
  const int l_current_nest = 0;
  for (size_t i = 0; i < num_symbols[l_current_nest]; ++i)
  {
    if (!(symbol_table[i].type & NODE_VARIABLE_ID &&
    (!strcmp(symbol_table[i].tspecifier,datatype)) && !strcmp(array_name,symbol_table[i].identifier) && !str_vec_contains(symbol_table[i].tqualifiers,"const")))
	    continue;
    ASTNode* array_access_start = node;
    string_vec var_dims = get_array_var_dims(array_name, root);
	
    char* index = get_index(array_access_start,var_dims,
		    str_vec_contains(symbol_table[i].tqualifiers,"gmem"));
    ASTNode* base = array_access_start;
    base->lhs=NULL;
    base->rhs=NULL;
    base->prefix=NULL;
    base->postfix=NULL;
    base->infix=NULL;
    free(base->buffer);
    base->buffer = malloc(sizeof(char)*10000);
    if(str_vec_contains(symbol_table[i].tqualifiers,"dconst"))
    	sprintf(base->buffer,"d_%s_arrays[%s_offset+(%s)]",convert_to_define_name(datatype_scalar),array_name,index);
    else if(str_vec_contains(symbol_table[i].tqualifiers,"gmem"))
    {
	
    	if(str_vec_contains(symbol_table[i].tqualifiers,"dynamic"))
    		sprintf(base->buffer,"gmem_%s_arrays[(int)%s][%s]",convert_to_define_name(datatype_scalar),array_name,index);
	else
    		sprintf(base->buffer,"__ldg(&gmem_%s_arrays[(int)%s][%s])",convert_to_define_name(datatype_scalar),array_name,index);
    }
    else
    {
	    fprintf(stderr,"Fatal error: no case for array read\n");
	    exit(EXIT_FAILURE);
    }
    free_str_vec(&var_dims);
    free(index);
  }
  free(datatype);
}

bool
is_user_enum_option(const ASTNode* node, const char* identifier)
{
	bool res = false;
	if(node->type == NODE_ENUM_DEF)
	{
		if(get_node_by_buffer_and_token(identifier,IDENTIFIER,node))
			return true;
	}
	res |= (node->lhs && is_user_enum_option(node->lhs,identifier));
	res |= (node->rhs && is_user_enum_option(node->rhs,identifier));
	return res;
}
void
read_user_enums_recursive(const ASTNode* node,string_vec* user_enums, string_vec* user_enum_options)
{
	if(node->type == NODE_ENUM_DEF)
	{
		const int enum_index = push(user_enums,node->lhs->buffer);
		ASTNode* enums_head = node->rhs;
		while(enums_head->rhs)
		{
			push(&user_enum_options[enum_index],get_node_by_token(IDENTIFIER,enums_head->rhs)->buffer);
			enums_head = enums_head->lhs;
		}
		push(&user_enum_options[enum_index],get_node_by_token(IDENTIFIER,enums_head->lhs)->buffer);
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
        string_vec user_enum_options[100] = { [0 ... 100-1] = VEC_INITIALIZER};
	string_vec user_enums = VEC_INITIALIZER;
	read_user_enums_recursive(node,&user_enums,user_enum_options);
	user_enums_info res;
	res.names = user_enums;
	memcpy(&res.options,&user_enum_options,sizeof(string_vec)*100);
	return res;
}
typedef struct
{
	string_vec user_structs;
	string_vec* user_struct_field_names;
	string_vec* user_struct_field_types;
} structs_info;
void
free_structs_info(structs_info* info)
{
	free_str_vec(&info->user_structs);
	for(int i = 0; i < 100; ++i)
	{
		free_str_vec(&info->user_struct_field_names[i]);
		free_str_vec(&info->user_struct_field_types[i]);
	}
}

structs_info 
get_structs_info()
{
	string_vec* names       = malloc(sizeof(string_vec)*100);
	string_vec* types = malloc(sizeof(string_vec)*100);
	string_vec  structs = VEC_INITIALIZER;
	for(int i = 0; i < 100; ++i)
	{
		string_vec tmp = VEC_INITIALIZER;
		names[i] = tmp;
		types[i] = tmp;
	}
	structs_info res = {.user_structs = structs, .user_struct_field_names = names, .user_struct_field_types = types};
	return res;
}
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
	const int struct_index = push(&(params->user_structs),get_node(NODE_TSPEC,node->lhs)->buffer);
	ASTNode* fields_head = node->rhs;
	const int num_of_nodes = count_num_of_nodes_in_list(node->rhs);
	int counter = num_of_nodes;
	while(--counter)
		fields_head = fields_head -> lhs;
	process_declaration(fields_head->lhs,struct_index,params);
	counter = num_of_nodes;

	while(--counter)
	{
		fields_head = fields_head->parent;
		process_declaration(fields_head->rhs,struct_index,params);
	}
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
	structs_info res = get_structs_info();
	read_user_structs_recursive(root, &res);
	return res;
}

void
process_param_codegen(ASTNode* kernel_root, const ASTNode* param, char* structs_info_str, string_vec* added_params_to_stencil_accesses, const bool gen_mem_accesses)
{
				char* param_type = malloc(4096*sizeof(char));
                                combine_buffers(param->lhs, param_type);
				char* param_str = malloc(4096*sizeof(char));
				param_str[0] = '\0';
                              	sprintf(param_str,"%s %s;",param_type, param->rhs->buffer);
				add_node_type(NODE_INPUT, kernel_root,param->rhs->buffer);
				strprepend(structs_info_str,param_str);
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
	char* structs_info_str = malloc(sizeof(char)*10000);
	structs_info_str[0] = '\0';
        while(param_list_head->rhs)
        {
	  process_param_codegen(compound_statement,param_list_head->rhs,structs_info_str,added_params_to_stencil_accesses,gen_mem_accesses);
          param_list_head = param_list_head->lhs;
        }

        ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, node->lhs);
	process_param_codegen(compound_statement,param_list_head->lhs,structs_info_str,added_params_to_stencil_accesses,gen_mem_accesses);
	char* kernel_params_struct = malloc(10000*sizeof(char));
	sprintf(kernel_params_struct,"typedef struct %sInputParams {%s} %sInputParams;\n",fn_identifier->buffer,structs_info_str,fn_identifier->buffer);

	strcat(kernel_params_struct,"\n");
	file_prepend("user_input_typedefs.h",kernel_params_struct);
	free(structs_info_str);

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
	string_vec added_params_to_stencil_accesses = VEC_INITIALIZER;
	gen_kernel_structs_recursive(root,&added_params_to_stencil_accesses,user_kernel_params_struct_str,gen_mem_accesses);
    	free_str_vec(&added_params_to_stencil_accesses);
	strcat(user_kernel_params_struct_str,"} acKernelInputParams;\n");

	FILE* fp_structs = fopen("user_input_typedefs.h","a");
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
        	file_prepend("user_typedefs.h",struct_def);
        	free(struct_def);
	}
}

typedef struct
{
	string_vec types;
	string_vec expr;
} func_params_info;

#define FUNC_PARAMS_INITIALIZER {.types = VEC_INITIALIZER, .expr = VEC_INITIALIZER}
void
free_func_params_info(func_params_info* info)
{
	free_str_vec(&info -> types);
	free_str_vec(&info -> expr);
}

void
get_function_param_types_and_names_recursive(const ASTNode* node, const char* func_name, string_vec* types_dst, string_vec* names_dst)
{
	if(node->lhs)
		get_function_param_types_and_names_recursive(node->lhs,func_name,types_dst,names_dst);
	if(node->rhs)
		get_function_param_types_and_names_recursive(node->rhs,func_name,types_dst,names_dst);
	if(!(node->type & NODE_FUNCTION))
		return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
	if(!fn_identifier)
		return;
	if(strcmp(fn_identifier->buffer, func_name))
		return;
        ASTNode* param_list_head = node->rhs->lhs;
	if(!param_list_head) return;
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
func_params_info
get_function_param_types_and_names(const ASTNode* node, const char* func_name)
{
	func_params_info res = FUNC_PARAMS_INITIALIZER;
	get_function_param_types_and_names_recursive(node,func_name,&res.types,&res.expr);
	return res;
}
char*
get_expr_type(const ASTNode* node, const ASTNode* root);

char*
get_user_struct_member_expr(const ASTNode* node, const ASTNode* root)
{
		char* res = NULL;
		const char* struct_type = get_expr_type(node->lhs,root);
		const char* field_name = get_node(NODE_MEMBER_ID,node)->buffer;
		if(!field_name) return NULL;
		structs_info info = read_user_structs(root);
		int index = -1;
		for(size_t i = 0; i < info.user_structs.size; ++i)
			if(!strcmp(info.user_structs.data[i],struct_type)) index = i;
		if(index == -1)
			return NULL;
		int field_index = -1;
		for(size_t i = 0; i < info.user_struct_field_names[index].size; ++i)
			if(!strcmp(info.user_struct_field_names[index].data[i],field_name)) field_index = i;
		if(field_index == -1)
			return NULL;
		res = strdup(info.user_struct_field_types[index].data[field_index]);
		free_structs_info(&info);
		return res;
}
char*
get_expr_type(const ASTNode* node, const ASTNode* root)
{

	if(node->expr_type) return node->expr_type;
	//if(node->lhs) get_expr_type(node->lhs,root);
	//if(node->rhs) get_expr_type(node->rhs,root);
	char* res = node->expr_type;
	if(node->type & NODE_ARRAY_ACCESS)
	{
		const char* base_type = get_expr_type(node->lhs,root);
		res = (!base_type)   ? NULL : remove_substring(strdup(base_type),"*");
	}
	else if(node->type == NODE_PRIMARY_EXPRESSION)
	{
	      const ASTNode* identifier = get_node_by_token(IDENTIFIER,node);
	      res =
	      	(get_node_by_token(REALNUMBER,node)) ? "AcReal":
	      	(get_node_by_token(DOUBLENUMBER,node)) ? "AcReal":
	      	(get_node_by_token(NUMBER,node)) ? "int" :
	      	(get_node_by_token(STRING,node)) ? "char*" :
	              (identifier && identifier->buffer && check_symbol(NODE_ANY,identifier->buffer,"int",NULL)) ? "int":
	              (identifier && identifier->buffer && check_symbol(NODE_ANY,identifier->buffer,"AcReal",NULL)) ? "AcReal":
	              (identifier && identifier->buffer && check_symbol(NODE_ANY,identifier->buffer,"Field",NULL)) ? "Field":
	              (identifier && identifier->buffer && check_symbol(NODE_ANY,identifier->buffer,"Field3",NULL)) ? "Field3":
	      	NULL;
	}
	else if(node->type == NODE_STRUCT_EXPRESSION)
	{
		const char* base_type = get_expr_type(node->lhs,root);
		const ASTNode* left = get_node(NODE_MEMBER_ID,node);

		res = 
		!base_type ? NULL :
		!strcmp(base_type,"int3")    ? "int":
		!strcmp(base_type,"AcReal3") ? "AcReal":
		!strcmp(base_type,"Field3")  ? "Field":
		get_user_struct_member_expr(node,root);

	}
	else if(node->type & NODE_FUNCTION_CALL)
	{
		const ASTNode* func_name = get_node_by_token(IDENTIFIER,node->lhs);
		//if(excluded_funcs && func_name && str_vec_contains(*excluded_funcs,func_name->buffer)) return NULL;
		res = node->expr_type;
	}
	//else if(node->type == NODE_PRIMARY_EXPRESSION && node->expr_type)
	//	res = node->expr_type;
	else if(node->type == NODE_BINARY_EXPRESSION)
	{
		char* lhs_res = get_expr_type(node->lhs,root);
		char* rhs_res = get_expr_type(node->rhs,root);
		res = 
			!lhs_res  ? NULL :
			!rhs_res  ? NULL :
			!strcmp(lhs_res,"AcReal3") || !strcmp(rhs_res,"AcReal3") ? "AcReal3" :
			!strcmp(lhs_res,"AcReal")  || !strcmp(rhs_res,"AcReal") ?  "AcReal" :
			strcmp(lhs_res,rhs_res) ? NULL :
			lhs_res;


	}
	else if(node->type == NODE_TERNARY_EXPRESSION)
	{

		char* first_expr  = get_expr_type(node->rhs->lhs,root);
		char* second_expr = get_expr_type(node->rhs->rhs,root);
		res = 
			!first_expr ? NULL :
			!second_expr ? NULL :
			strcmp(first_expr,second_expr) ? NULL :
			first_expr;
	}
	else
	{
		if(node->lhs && !res)
			res = get_expr_type(node->lhs,root);
		if(node->rhs && !res)
			res = get_expr_type(node->rhs,root);
	}
	return res;
}


func_params_info
get_func_call_params_info(const ASTNode* func_call, const ASTNode* root)
{
		func_params_info res = FUNC_PARAMS_INITIALIZER;
		ASTNode* param_list_head = func_call->rhs;
		while(param_list_head->rhs)
		{
			char* param = malloc(sizeof(char)*10000);
			combine_all(param_list_head->rhs,param);
			assert(param);
			push(&res.expr,param);
			free(param);

			push(&res.types,get_expr_type(param_list_head->rhs,root));
			param_list_head = param_list_head->lhs;
		}
		char* param = malloc(sizeof(char)*10000);
		combine_all(param_list_head->lhs,param);
		assert(param);
		push(&res.expr,param);
		free(param);
		push(&res.types,get_expr_type(param_list_head->lhs,root));
		return res;
}

void gen_loader(const ASTNode* func_call, const ASTNode* root, const char* prefix, string_vec* input_symbols, string_vec* input_types)
{
		char tmp[4000];
		const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
		const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,NULL,NULL);
		if(!strcmp(func_name,"periodic"))
			return;
		ASTNode* param_list_head = func_call->rhs;
		if(!param_list_head)
			return;
		func_params_info call_info  = get_func_call_params_info(func_call,root);
		bool is_boundcond = false;
		for(size_t i = 0; i< call_info.expr.size; ++i)
			is_boundcond |= (strstr(call_info.expr.data[i],"BOUNDARY_") != NULL);

		func_params_info params_info =  get_function_param_types_and_names(root,func_name);

		char* loader_str = malloc(sizeof(char)*4000);
		sprintf(loader_str,"auto %s_%s_loader = [](ParamLoadingInfo p){\n",prefix, func_name);
		const int params_offset = is_boundcond ? 2 : 0;
		if(!is_boundcond)
		{
			if(params_info.types.size != call_info.expr.size)
			{
				fprintf(stderr,"Number of inputs for %s in ComputeSteps does not match the number of input params\n", func_name);
				exit(EXIT_FAILURE);
			}
		}
		for(int i = 0; i < (int)params_info.types.size-params_offset; ++i)
		{
			if(is_dfunc) continue;
			if(is_number(call_info.expr.data[i]) || is_real(call_info.expr.data[i]))
				sprintf(tmp, "p.params -> %s.%s = %s;\n", func_name, params_info.expr.data[i], call_info.expr.data[i]);
			else if(!strcmp(params_info.types.data[i],"AcReal"))
			{
				sprintf(tmp, "p.params -> %s.%s = acDeviceGetRealInput(acGridGetDevice(),%s);\n", func_name,params_info.expr.data[i], call_info.expr.data[i]);
			}
			else if(!strcmp(params_info.types.data[i],"int"))
				sprintf(tmp, "p.params -> %s.%s = acDeviceGetIntInput(acGridGetDevice(),%s); \n", func_name,params_info.expr.data[i], call_info.expr.data[i]);
			strcat(loader_str,tmp);
			if(!str_vec_contains(*input_symbols,call_info.expr.data[i]))
			{
				if(!is_number(call_info.expr.data[i]) && !is_real(call_info.expr.data[i]))
				{
					push(input_symbols,call_info.expr.data[i]);
					push(input_types,params_info.types.data[i]);
				}
			}
		}
		//TP: disabled for now
		//add predefined input params for boundcond functions
		//if(is_boundcond)
		//{
		//	sprintf(tmp, "p.params -> %s.boundary_normal= p.boundary_normal;\n",func_name);
		//	strcat(loader_str,tmp);
		//	sprintf(tmp, "p.params -> %s.vtxbuf = p.vtxbuf;\n",func_name);
		//	strcat(loader_str,tmp);
		//}
		strcat(loader_str,"};\n");
		file_prepend("user_loaders.h",loader_str);

		free_func_params_info(&params_info);
		free_func_params_info(&call_info);
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
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,func_name,"Kernel");
	if(kernel_index == -1)
	{
		fprintf(stderr,"Undeclared kernel %s used in ComputeSteps %s\n",func_name,taskgraph_name);
		exit(EXIT_FAILURE);
	}
	char* all_fields = malloc(sizeof(char)*4000);
	all_fields[0] = '\0';
	for(size_t field = 0; field < num_fields; ++field)
	{
		const bool field_in  = (read_fields[field + num_fields*kernel_index] || field_has_stencil_op[field + num_fields*kernel_index]);
		const bool field_out = (written_fields[field + num_fields*kernel_index]);
		const char* field_str = get_symbol_by_index(NODE_VARIABLE_ID,field,"Field")->identifier;
		sprintf(tmp,"%s,",field_str);
		strcat(all_fields,tmp);
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

	int_vec calls = VEC_INITIALIZER;
	ASTNode* function_call = function_call_list_head->lhs;
	char* func_name = get_node_by_token(IDENTIFIER,function_call)->buffer;
	if(check_symbol(NODE_FUNCTION_ID,func_name,NULL,NULL))
		push_int(&calls,get_symbol_index(NODE_FUNCTION_ID,func_name,"Kernel"));
	while(--n)
	{
		function_call_list_head = function_call_list_head->parent;
		function_call = function_call_list_head->rhs;
		func_name = get_node_by_token(IDENTIFIER,function_call)->buffer;
		if(check_symbol(NODE_FUNCTION_ID,func_name,"Kernel",NULL))
			push_int(&calls,get_symbol_index(NODE_FUNCTION_ID,func_name,"Kernel"));
	}
	return calls;
}
void
compute_next_level_set(bool* src, const int_vec kernel_calls, bool* field_written_to, int* call_level_set)
{
	bool* field_consumed = (bool*)malloc(sizeof(bool)*num_fields);
	memset(field_written_to,0,sizeof(bool)*num_fields);
	memset(field_consumed,0,sizeof(bool)*num_fields);
	for(size_t i = 0; i < kernel_calls.size; ++i)
	{
		if(call_level_set[i] == -1)
		{
		  const int kernel_index = kernel_calls.data[i];
		  bool can_compute = true;
		  for(size_t j = 0; j < num_fields; ++j)
		  {
			const int index = j + num_fields*kernel_index;
			const bool field_accessed = read_fields[index] || field_has_stencil_op[index];
		  	can_compute &= !(field_consumed[j] && field_accessed);
		  	field_consumed[j] |= written_fields[index];
		  }
		  for(size_t j = 0; j < num_fields; ++j)
		  	field_written_to[j] |= (can_compute && written_fields[j + num_fields*kernel_index]);
		  src[i] |= can_compute;
		}
	}
	free(field_consumed);
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
bool* get_fields_included(const func_params_info call_info)
{
	bool* fields_included = (bool*)malloc(sizeof(bool)*num_fields);
	memset(fields_included,0,sizeof(bool)*num_fields);
	for(size_t field = 0; field < num_fields; ++field)
		fields_included[field] = str_vec_contains(call_info.expr,get_symbol_by_index(NODE_VARIABLE_ID,field,"Field")->identifier);
	//if none are included then by default all are included
	bool none_included = true;
	for(size_t field = 0; field < num_fields; ++field)
		none_included &= !fields_included[field];
	for(size_t field = 0; field < num_fields; ++field)
		fields_included[field] |= none_included;
	return fields_included;
}
void
process_boundcond(const ASTNode* func_call, char** res, const ASTNode* root, const char* boundconds_name, string_vec* input_symbols,string_vec* input_types)
{
	char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);
	const int boundaries[] = {BOUNDARY_X_BOT, BOUNDARY_Y_BOT,BOUNDARY_Z_BOT,BOUNDARY_X_TOP,BOUNDARY_Y_TOP,BOUNDARY_Z_TOP};
	const int num_boundaries = 6;


	func_params_info call_info = get_func_call_params_info(func_call,root);
	bool* fields_included = get_fields_included(call_info);
	const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,NULL,NULL);
	char* prefix = malloc(sizeof(char)*4000);
	char* full_name = malloc(sizeof(char)*4000);
	for(int bc = 0;  bc < num_boundaries; ++bc) 
	{
		if(boundary_int & boundaries[bc])
		{
			if(is_dfunc) sprintf(prefix,"%s_AC_KERNEL_",boundconds_name);
			else sprintf(prefix,"%s_",boundconds_name);
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
			if(is_dfunc)
				sprintf(full_name,"%s_%s",prefix,func_name);
			else
				sprintf(full_name,"%s",func_name);
			for(size_t field = 0; field < num_fields; ++field)
	     	     		res[field + num_fields*bc] = (fields_included[field]) ? strdup(full_name) : res[field + num_fields*bc];
			if(!strcmp(func_name,"periodic"))
				gen_loader(func_call,root,prefix,input_symbols,input_types);
		}
	}
	free(fields_included);
	free(prefix);
	free(full_name);
}
void
write_dfunc_bc_kernel(const ASTNode* root, const char* prefix, const char* func_name,const bool* fields_included,const func_params_info call_info,FILE* fp)
{

	func_params_info params_info = get_function_param_types_and_names(root,func_name);
	const size_t num_of_rest_params = params_info.expr.size-1;
        free_func_params_info(&params_info);
	fprintf(fp,"Kernel %s_%s()\n{\n",prefix,func_name);
	for(size_t i = 0; i < num_fields; ++i)
	{
		if(!fields_included[i]) continue;
		fprintf(fp,"\t%s(%s",func_name,get_symbol_by_index(NODE_VARIABLE_ID,i,"Field")->identifier);
		for(size_t j = 0; j < num_of_rest_params; ++j)
			fprintf(fp,",%s",call_info.expr.data[j]);
		fprintf(fp,"%s\n",")");
	}
	fprintf(fp,"%s\n","}");
}
void
gen_dfunc_bc_kernel(const ASTNode* func_call, FILE* fp, const ASTNode* root, const char* boundconds_name)
{
	char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);
	const int boundaries[] = {BOUNDARY_X_BOT, BOUNDARY_Y_BOT,BOUNDARY_Z_BOT,BOUNDARY_X_TOP,BOUNDARY_Y_TOP,BOUNDARY_Z_TOP};
	const int num_boundaries = 6;


	func_params_info call_info = get_func_call_params_info(func_call,root);
	bool* fields_included = get_fields_included(call_info);

	if(!strcmp(func_name,"periodic"))
		return;
	char* prefix = malloc(sizeof(char)*4000);
	const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,NULL,NULL);
	char* full_name = malloc(sizeof(char)*4000);
	for(int bc = 0;  bc < num_boundaries; ++bc) 
	{
		if(boundary_int & boundaries[bc])
		{
			if(is_dfunc) sprintf(prefix,"%s_AC_KERNEL_",boundconds_name);
			else sprintf(prefix,"%s_",boundconds_name);
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
			write_dfunc_bc_kernel(root,prefix,func_name,fields_included,call_info,fp);
		}
	}
	free_func_params_info(&call_info);
	free(fields_included);
	free(prefix);
	free(full_name);
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
gen_halo_exchange_and_boundconds(
		char** field_boundconds,
		char* res,
		const char* boundconds_name,
		int_vec fields,
		int_vec communicated_fields
		)
{
		const int num_boundaries = 6;
		bool* field_boundconds_processed = (bool*)malloc(num_fields*num_boundaries);
		memset(field_boundconds_processed,0,num_fields*num_boundaries*sizeof(bool));
		bool need_to_communicate = false;
		char communicated_fields_str[4000];
		sprintf(communicated_fields_str,"{");
		for(size_t i = 0; i < fields.size; ++i)
		{
			const int field = fields.data[i];
			need_to_communicate |= int_vec_contains(communicated_fields,field);
			if(int_vec_contains(communicated_fields,field))
			{
				const char* field_str = get_symbol_by_index(NODE_VARIABLE_ID,field,"Field")->identifier;
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
				strcat(res,"acBoundaryCondition(");
				if(!strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_XYZ");
				else if(!strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_XY");
				else if(!strcmp(x_boundcond,"periodic") && strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_XZ");
				else if(!strcmp(x_boundcond,"periodic") && strcmp(y_boundcond,"periodic") && strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_X");
				else if(strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_YZ");
				else if(strcmp(x_boundcond,"periodic") && !strcmp(y_boundcond,"periodic") && strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_Y");
				else if(strcmp(x_boundcond,"periodic") && strcmp(y_boundcond,"periodic") && !strcmp(z_boundcond,"periodic"))
					strcat(res,"BOUNDARY_Z");
				strcat(res,",BOUNDCOND_PERIODIC,");
				strcat(res,communicated_fields_str);
				strcat(res,"),\n");
			}

			for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
				for(size_t i = 0; i < fields.size; ++i)
					field_boundconds_processed[fields.data[i] + num_fields*boundcond]  = !strcmp(field_boundconds[fields.data[i] + num_fields*boundcond],"periodic")  
												              || !int_vec_contains(communicated_fields,fields.data[i]);

			bool all_are_processed = false;
			while(!all_are_processed)
			{
				for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
				{
					const char* processed_boundcond = NULL;

					for(size_t i = 0; i < fields.size; ++i)
						processed_boundcond = !field_boundconds_processed[fields.data[i] + num_fields*boundcond] ? field_boundconds[fields.data[i] + num_fields*boundcond] : processed_boundcond;
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
					for(size_t i = 0; i < fields.size; ++i)
					{
						const char* field_str = get_symbol_by_index(NODE_VARIABLE_ID,fields.data[i],"Field")->identifier;
						const char* boundcond_str = field_boundconds[fields.data[i] + num_fields*boundcond];
						if(strcmp(boundcond_str,processed_boundcond)) continue;
						if(field_boundconds_processed[fields.data[i] + num_fields*boundcond]) continue;
						field_boundconds_processed[fields.data[i] + num_fields*boundcond] |= true;
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
					for(size_t i = 0; i < fields.size; ++i)
						all_are_processed &= field_boundconds_processed[fields.data[i] + num_fields*boundcond];
			}
		}
		free(field_boundconds_processed);
}
ASTNode*
get_list_elem_from_leaf(const ASTNode* leaf, int index)
{

	if(index == 0)
		return leaf->lhs;
	while(index--)
		leaf = leaf->parent;
	return leaf->rhs;
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

	for(size_t field = 0; field < num_fields; ++field)
		for(int bc = 0; bc < num_boundaries; ++bc)
			if(!field_boundconds[field + num_fields*bc])
			{
				fprintf(stderr,"Fatal error: Missing boundcond for field: %lu at boundary: %d\n",field,bc);
				exit(EXIT_FAILURE);
			}

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

	int_vec kernel_calls_in_level_order = VEC_INITIALIZER;
	bool* field_halo_in_sync = malloc(sizeof(bool)*num_fields);
	bool* field_communicated_once = malloc(sizeof(bool)*num_fields);
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
		//for(size_t j = 0; j < num_fields; ++j)
			//printf("field out from level set: %d,%ld,%d\n",n_level_sets,j,field_out_from_level_set[j]);
		bool* swap_tmp;
		swap_tmp = field_out_from_level_set;
		field_out_from_level_set = field_out_from_last_level_set;
		field_out_from_last_level_set = swap_tmp;
		++n_level_sets;
		all_processed = true;

		for(size_t k = 0; k < kernel_calls.size; ++k)
			all_processed &= (call_level_set[k] != -1);
		if((size_t) n_level_sets > kernel_calls.size)
		{
			fprintf(stderr,"Bug in the compiler aborting\n");
			exit(EXIT_FAILURE);
		}

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
		const char* field_str = get_symbol_by_index(NODE_VARIABLE_ID,field,"Field")->identifier;
		strcat(all_fields,field_str);
		strcat(all_fields,",");
	}
	bool* field_written_out_before = (bool*)malloc(sizeof(bool)*num_fields);
	memset(field_written_out_before,0,sizeof(bool)*num_fields);
	for(int level_set = 0; level_set < n_level_sets; ++level_set)
	{
		int_vec fields_not_written_to = VEC_INITIALIZER;
		int_vec fields_written_to     = VEC_INITIALIZER;
		int_vec communicated_fields   = VEC_INITIALIZER;
		for(size_t i = 0; i < num_fields; ++i)
		{
			if(field_needs_to_be_communicated_before_level_set[i + num_fields*level_set])
				push_int(&communicated_fields,i);
			if(field_written_out_before[i])
				push_int(&fields_written_to,i);
			else
				push_int(&fields_not_written_to,i);
		}
		gen_halo_exchange_and_boundconds(
		  field_boundconds,
		  res,
		  boundconds_name,
		  fields_written_to,
		  communicated_fields
		);
		gen_halo_exchange_and_boundconds(
		  field_boundconds,
		  res,
		  boundconds_name,
		  fields_not_written_to,
		  communicated_fields
		);
		for(size_t call = 0; call < kernel_calls.size; ++call) 
		{
			if(call_level_set[call] == level_set)
			{
				const ASTNode* kernel_call = get_list_elem_from_leaf(function_call_list_head,call);
				gen_taskgraph_kernel_entry(
						kernel_call,
						root,res,input_symbols,input_types,name
				);
				const char* func_name = get_node_by_token(IDENTIFIER,kernel_call)->buffer;
				const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,func_name,"Kernel");
				for(size_t field = 0; field < num_fields; ++field)
					field_written_out_before[field] |= written_fields[field + num_fields*kernel_index];

			}
		}
		free_int_vec(&fields_not_written_to);
		free_int_vec(&fields_written_to);
		free_int_vec(&communicated_fields);
	}
	free(field_written_out_before);
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
gen_dfunc_bc_kernels(const ASTNode* node, const ASTNode* root, FILE* fp)
{
	if(node->lhs)
		gen_dfunc_bc_kernels(node->lhs,root,fp);
	if(node->rhs)
		gen_dfunc_bc_kernels(node->rhs,root,fp);
	if(node->type != NODE_BOUNDCONDS_DEF) return;
	const char* name = node->lhs->buffer;
	const ASTNode* function_call_list_head = node->rhs;
	int n_entries = 1;
	while(function_call_list_head->rhs)
	{
		++n_entries;
		function_call_list_head = function_call_list_head->lhs;
	}
	gen_dfunc_bc_kernel(function_call_list_head->lhs,fp,root,name);
	while(--n_entries)
	{
		function_call_list_head = function_call_list_head->parent;
		gen_dfunc_bc_kernel(function_call_list_head->rhs,fp,root,name);
	}
}
void
gen_user_taskgraphs(FILE* enums_fp, const ASTNode* root)
{
  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	const char* path = ACC_GEN_KERNELS_PATH"/boundcond_kernels.h";
	if(!has_optimization_info)
	{
		FILE* fp = fopen(path,"w");
		gen_dfunc_bc_kernels(root,root,fp);
		fclose(fp);
	}
	string_vec input_symbols = VEC_INITIALIZER;
	string_vec input_types   = VEC_INITIALIZER;
	gen_user_taskgraphs_recursive(root,root,&input_symbols,&input_types);
	gen_input_enums(enums_fp,input_symbols,input_types,"AcReal");
	gen_input_enums(enums_fp,input_symbols,input_types,"int");
	gen_input_enums(enums_fp,input_symbols,input_types,"bool");
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
} param_combinations;

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
gen_all_possibilities(string_vec res, int kernel_index, size_t my_index,param_combinations combinations,combinatorial_params combinatorials)
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
gen_combinations(int kernel_index,param_combinations combinations ,combinatorial_params combinatorials)
{
	string_vec base = VEC_INITIALIZER;
	if(combinatorials.names[kernel_index].size > 0)
		gen_all_possibilities(base, kernel_index,0,combinations,combinatorials);
	free_str_vec(&base);
}
void
gen_kernel_num_of_combinations_recursive(const ASTNode* node, param_combinations combinations, user_enums_info user_enums, string_vec* user_kernels_with_input_params,combinatorial_params combinatorials, structs_info struct_info)
{
	if(node->lhs)
	{
		gen_kernel_num_of_combinations_recursive(node->lhs,combinations,user_enums,user_kernels_with_input_params,combinatorials,struct_info);
	}
	if(node->rhs)
		gen_kernel_num_of_combinations_recursive(node->rhs,combinations,user_enums,user_kernels_with_input_params,combinatorials,struct_info);
	if(node->type & NODE_KFUNCTION && node->rhs->lhs)
	{
	   const char* kernel_name = get_node(NODE_FUNCTION_ID, node)->buffer;
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
gen_kernel_num_of_combinations(const ASTNode* root, param_combinations combinations, string_vec* user_kernels_with_input_params,string_vec* user_kernel_combinatorial_params)
{
  	user_enums_info user_enums = read_user_enums(root);

	string_vec user_kernel_combinatorial_params_options[100*100] = { [0 ... 100*100 -1] = VEC_INITIALIZER};
	structs_info struct_info = read_user_structs(root);

	gen_kernel_num_of_combinations_recursive(root,combinations,user_enums,user_kernels_with_input_params,(combinatorial_params){user_kernel_combinatorial_params,user_kernel_combinatorial_params_options},struct_info);

  	free_str_vec(&user_enums.names);
	//TP: for some reason causes double free
	//for(int i = 0; i < 100; ++i)
	//  for(int j=0;j<100;++j)
	//  	  free_str_vec(&user_kernel_combinatorial_params_options[i+100*j]);
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
free_reduce_info(reduce_info* info)
{
	free_op_vec(&info->ops);
	free_str_vec(&info->conditions);
	free_str_vec(&info->outputs);
}

#define REDUCE_INFO_INITIALIZER {.outputs = VEC_INITIALIZER, .conditions = VEC_INITIALIZER, .ops = VEC_INITIALIZER }


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
get_called_dfuncs(const ASTNode* node, int_vec* src, int_vec* dfuncs_info)
{
	if(node->lhs)
		get_called_dfuncs(node->lhs,src,dfuncs_info);
	if(node->rhs)
		get_called_dfuncs(node->rhs,src,dfuncs_info);
	if(!(node->type & NODE_FUNCTION_CALL))
		return;
	char* func_name = malloc(sizeof(char)*5000);
	combine_buffers(node->lhs,func_name);
	const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,NULL);
	free(func_name);
	if(dfunc_index < 0)
		return;
	push_int(src,dfunc_index);
	for(size_t i = 0; i < dfuncs_info[dfunc_index].size; ++i)
		push_int(src,dfuncs_info[dfunc_index].data[i]);

}
void
get_dfuncs_called_dfuncs(const ASTNode* node, int_vec* src)
{
	if(node->lhs) 
		get_dfuncs_called_dfuncs(node->lhs,src);
	if(node->rhs) 
		get_dfuncs_called_dfuncs(node->rhs,src);
	if(!(node->type & NODE_DFUNCTION))
		return;
	char* func_name = malloc(sizeof(char)*5000);
        combine_buffers(node->lhs,func_name);
        const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,NULL);
	if(dfunc_index > 0)
		get_called_dfuncs(node,&src[dfunc_index],src);
	free(func_name);
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
void
gen_kernel_postfixes_recursive(ASTNode* node, const bool gen_mem_accesses, reduce_info* dfuncs_info,reduce_info* kernel_reduce_infos)
{
	if(node->lhs)
		gen_kernel_postfixes_recursive(node->lhs,gen_mem_accesses,dfuncs_info,kernel_reduce_infos);
	if(node->rhs)
		gen_kernel_postfixes_recursive(node->rhs,gen_mem_accesses,dfuncs_info,kernel_reduce_infos);
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
	reduce_info kernel_reduce_info = REDUCE_INFO_INITIALIZER;

	get_reduce_info(node,&kernel_reduce_info,dfuncs_info);
	if(kernel_reduce_info.ops.size == 0)
	{
	  strcat(new_postfix,"}");
	  compound_statement->postfix = strdup(new_postfix);
	  return;
	}
	const ASTNode* fn_identifier = get_node(NODE_FUNCTION_ID,node);
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,fn_identifier->buffer,"Kernel");
	assert(kernel_reduce_info.ops.size  == kernel_reduce_info.outputs.size && kernel_reduce_info.ops.size == kernel_reduce_info.conditions.size);

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

	for(size_t i = 0; i < kernel_reduce_info.ops.size; ++i)
	{
		ReduceOp reduce_op = kernel_reduce_info.ops.data[i];
		char* condition = kernel_reduce_info.conditions.data[i];
		char* output = kernel_reduce_info.outputs.data[i];
		push_op(&kernel_reduce_infos[kernel_index].ops,  reduce_op);
		push(&kernel_reduce_infos[kernel_index].outputs,  output);
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
	free_reduce_info(&kernel_reduce_info);
	free(new_postfix);
	free(tmp);
	free(res_name);
	free(output_str);
}
void
gen_kernel_postfixes(ASTNode* root, const bool gen_mem_accesses,reduce_info* kernel_reduce_info)
{
	reduce_info dfuncs_info[MAX_DFUNCS] = { [0 ... MAX_DFUNCS-1] = REDUCE_INFO_INITIALIZER};
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
  //extra padding to help some compilers
  fprintf(fp,"%s","static const int kernel_reduce_outputs[NUM_KERNELS][NUM_REAL_OUTPUTS+1] = { ");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
    {
      fprintf(fp,"%s","{");
      for(int j = 0; j < num_real_reduce_output; ++j)
      {
      	if(kernel_reduce_info[kernel_iterator].outputs.size < (size_t) j+1)
	        fprintf(fp,"%d,",-1);
      	else
	      	fprintf(fp,"(int)%s,",kernel_reduce_info[kernel_iterator].outputs.data[j]);
      }
      fprintf(fp,"-1");
      fprintf(fp,"%s","},");
      ++kernel_iterator;
    }
  }
  fprintf(fp,"%s","};\n");

  fprintf(fp,"%s","typedef enum KernelReduceOp\n{\n\tNO_REDUCE,\n\tREDUCE_MIN,\n\tREDUCE_MAX,\n\tREDUCE_SUM,\n} KernelReduceOp;\n");
  //extra padding to help some compilers
  fprintf(fp,"%s","static const KernelReduceOp kernel_reduce_ops[NUM_KERNELS][NUM_REAL_OUTPUTS+1] = { ");
  int iterator = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
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
      fprintf(fp,"NO_REDUCE");
      fprintf(fp,"%s","},");
      ++iterator;
    }
  fprintf(fp,"%s","};\n");
  fclose(fp);
  fp = fopen("kernel_reduce_info.h","w");
  fprintf(fp, "\nstatic const int kernel_calls_reduce[] = {");
  for(size_t kernel = 0; kernel < num_kernels; ++kernel)
  {
      	const char* val = (kernel_reduce_info[kernel].ops.size == 0) ? "0" : "1";
	fprintf(fp,"%s,",val);
  }
  fprintf(fp, "};\n");
  fclose(fp);
}
void
gen_kernel_postfixes_and_reduce_outputs(ASTNode* root, const bool gen_mem_accesses)
{
  reduce_info kernel_reduce_info[MAX_KERNELS] = {[0 ... MAX_KERNELS-1] = REDUCE_INFO_INITIALIZER};
  gen_kernel_postfixes(root,gen_mem_accesses,kernel_reduce_info);

  gen_kernel_reduce_outputs(kernel_reduce_info);
  for(int i = 0; i < 100; ++i)
  {
	  free_str_vec(&kernel_reduce_info[i].outputs);
	  free_op_vec(&kernel_reduce_info[i].ops);
  }
}
void
gen_kernel_ifs(ASTNode* node, const param_combinations combinations, string_vec user_kernels_with_input_params,string_vec* user_kernel_combinatorial_params)
{
	if(node->lhs)
		gen_kernel_ifs(node->lhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		gen_kernel_ifs(node->rhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(!(node->type & NODE_KFUNCTION))
		return;
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,get_node(NODE_FUNCTION_ID,node)->buffer);
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
		sprintf(res,"if(kernel_enum == KERNEL_%s ",get_node(NODE_FUNCTION_ID,node)->buffer);
		for(size_t j = 0; j < combination_vals.size; ++j)
		{
			sprintf(tmp, " && vba.kernel_input_params.%s.%s ==  %s ",get_node(NODE_FUNCTION_ID,node)->buffer,combination_params.data[j],combination_vals.data[j]);
			strcat(res,tmp);
		}
		strcat(res,")\n{\n\t");
		sprintf(tmp,"return %s_optimized_%d;\n}\n",get_node(NODE_FUNCTION_ID,node)->buffer,i);
		strcat(res,tmp);
		//sprintf(res,"%sreturn %s_optimized_%d;\n}\n",tmp);
		fprintf(fp_defs,"%s_optimized_%d,",get_node(NODE_FUNCTION_ID,node)->buffer,i);
		fprintf(fp,"%s",res);
		bool is_left = (old_parent->lhs == node);
		ASTNode* new_parent = astnode_create(NODE_UNKNOWN,node,astnode_dup(node,old_parent));
		char* new_name = malloc(sizeof(char)*4096);
		sprintf(new_name,"%s_optimized_%d",get_node(NODE_FUNCTION_ID,node)->buffer,i);
		((ASTNode*) get_node(NODE_FUNCTION_ID,new_parent->rhs))->buffer= strdup(new_name);
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
safe_strcat(char** dst, const char* to_append)
{
	const size_t res_size = strlen(*dst) + strlen(to_append) + 100;
	char* res = (char*)malloc(sizeof(char)*res_size);
	sprintf(res,"%s%s",*dst,to_append);
	free(*dst);
	*dst = strdup(res);
	free(res);
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
			safe_strcat(&node->buffer,"AC_INTERNAL_INPUT");
		return;
	}
	const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
	if(!begin_scope)
		return;
	const ASTNode* fn_declaration= begin_scope->parent->parent->lhs;
	if(!fn_declaration)
		return;
	const ASTNode* fn_identifier = get_node(NODE_FUNCTION_ID,fn_declaration);
	while(!fn_identifier)
	{
		begin_scope = get_parent_node(NODE_BEGIN_SCOPE,fn_declaration);
		if(!begin_scope)
			return;
		fn_declaration= begin_scope->parent->parent->lhs;
		if(!fn_declaration)
			return;
		fn_identifier = get_node(NODE_FUNCTION_ID,fn_declaration);
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
  if (node->buffer && node->token == IDENTIFIER && !(node->type & exclude)) {
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
        const ASTNode* is_dconst = get_parent_node_exclusive(NODE_DCONST, node);
        if (is_dconst)
          fprintf(stream, "__device__ ");

        if (n_tqualifiers)
	  for(size_t i=0; i<n_tqualifiers;++i)
	  {
		if(strcmp(tqualifiers[i],"boundary_condition"))
          		fprintf(stream, "%s ", tqualifiers[i]);
	  }

        if (tspec) 
	{
	  if(strcmp(tspec,"Kernel"))
            fprintf(stream, "%s ", tspec);
        }
        else if (!get_parent_node_exclusive(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID) &&
                 !(node->type & NODE_INPUT) &&
		 !(node->no_auto)
		 )
	{
	  if(node->is_constexpr && !(node->type & NODE_FUNCTION_ID)) fprintf(stream, " constexpr ");
          fprintf(stream, "auto ");
	  const ASTNode* func_call_node = get_parent_node(NODE_FUNCTION_CALL,node);
	  if(func_call_node)
	  {
		if(get_node_by_token(IDENTIFIER,func_call_node->lhs)->id == node->id)
		{
			char tmp[10000];
		        combine_all(func_call_node,tmp);
			fprintf(stderr,"Undeclared function used: %s\n",tmp);
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
      {
        add_symbol(node->type, tqualifiers, n_tqualifiers, tspec, node->buffer);
      }

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
    if (symbol && symbol->type & NODE_VARIABLE_ID && str_vec_contains(symbol->tqualifiers,"dconst"))
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
gen_multidimensional_field_accesses_recursive(ASTNode* node)
{
	if(node->lhs)
		gen_multidimensional_field_accesses_recursive(node->lhs);
	if(node->rhs)
		gen_multidimensional_field_accesses_recursive(node->rhs);
	if(!(node->token == IDENTIFIER))
		return;
	if(!node->buffer)
		return;
	//discard global const declarations
	if(node->parent->parent->parent->type & NODE_ASSIGN_LIST)
		return;
	if(!check_for_vtxbuf(node) && !check_symbol(NODE_VARIABLE_ID,node->buffer,"Field",NULL))
		return;

	ASTNode* array_access = (ASTNode*)get_parent_node(NODE_ARRAY_ACCESS,node);
	if(!array_access || get_node_by_id(node->id,array_access->lhs)->id != node->id)	return;
	while(get_parent_node(NODE_ARRAY_ACCESS,array_access)) array_access = (ASTNode*) get_parent_node(NODE_ARRAY_ACCESS,array_access);

	string_vec array_accesses = get_array_accesses(array_access);
	array_access->rhs = NULL;
	array_access->lhs = NULL;
	array_access->buffer = NULL;
	array_access->rhs = NULL;
	array_access->infix= NULL;
	array_access->postfix= NULL;
	array_access->prefix = NULL;
	char* res = malloc(sizeof(char)*4000);
	res[0] = '\0';
	ASTNode* rhs = astnode_create(NODE_UNKNOWN, NULL, NULL);

	const char* x_index = array_accesses.data[0];
	const char* y_index = array_accesses.size > 1
			      ? array_accesses.data[1] : "0";
	const char* z_index = array_accesses.size > 2
			      ? array_accesses.data[2] : "0";
	sprintf(res,"[IDX(%s,%s,%s)]",x_index,y_index,z_index);
	rhs->buffer = strdup(res);
        array_access->rhs = rhs;
	free(res);

	ASTNode* lhs = astnode_create(NODE_UNKNOWN, NULL, NULL);
	array_access->lhs = lhs;
	lhs->prefix  = strdup("vba.in[");
	lhs->postfix = strdup("]");
	lhs->parent = array_access;
        lhs->buffer = strdup(node->buffer);
	lhs->token = IDENTIFIER;
	lhs->type |= node->type & NODE_INPUT;
}
void
gen_multidimensional_field_accesses(ASTNode* root)
{
  symboltable_reset();
  traverse(root, NODE_NO_OUT, NULL);
  gen_multidimensional_field_accesses_recursive((ASTNode*)root);
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


#define max(a,b) (a > b ? a : b)
static int
count_nest(const ASTNode* node,const NodeType type)
{
	int lhs_res =  (node->lhs) ? count_nest(node->lhs,type) : 0;
	int rhs_res =  (node->rhs) ? count_nest(node->rhs,type) : 0;
	return max(lhs_res,rhs_res) + (node->type == type);
}
void
gen_const_def(const ASTNode* def, const ASTNode* tspec, FILE* fp)
{
		char* assignment_val = malloc(sizeof(char)*10000);
		const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
		if(!name) return;
        	const ASTNode* assignment = def->rhs;
		if(!assignment) return;
		combine_all(assignment,assignment_val);
		const char* datatype = tspec->lhs->buffer;
		char* datatype_scalar = remove_substring(strdup(datatype),"*");
		remove_substring(datatype_scalar,"*");
		const ASTNode* array_initializer = get_node(NODE_ARRAY_INITIALIZER, assignment);
		const int array_dim = array_initializer ? count_nest(array_initializer,NODE_ARRAY_INITIALIZER) : 0;
		const int num_of_elems = array_initializer ? count_num_of_nodes_in_list(array_initializer->lhs) : 0;
		if(array_initializer)
		{
			const ASTNode* second_array_initializer = get_node(NODE_ARRAY_INITIALIZER, array_initializer->lhs);
			if(array_dim == 1)
			{
				fprintf(fp, "\n#ifdef __cplusplus\nconstexpr AcArray<%s,%d> %s = %s;\n#endif\n",datatype_scalar, num_of_elems, name, assignment_val);
			}
			else
			{
				const int num_of_elems_in_list = count_num_of_nodes_in_list(second_array_initializer->lhs);
				fprintf(fp, "\n#ifdef __cplusplus\nconstexpr AcArray<AcArray<%s,%d>,%d> %s = %s;\n#endif\n",datatype_scalar, num_of_elems_in_list, num_of_elems, name, assignment_val);
			}
		}
		else
		{
			fprintf(fp, "\n#ifdef __cplusplus\nconstexpr %s %s = %s;\n#endif\n",datatype_scalar, name, assignment_val);
		}
		free(datatype_scalar);
		free(assignment_val);
}
void
gen_const_variables(const ASTNode* node, FILE* fp)
{
	if(node->lhs)
		gen_const_variables(node->lhs,fp);
	if(node->rhs)
		gen_const_variables(node->rhs,fp);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	if(!has_qualifier(node,"const")) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	const ASTNode* def_list_head = node->rhs;
	while(def_list_head -> rhs)
	{
		gen_const_def(def_list_head->rhs,tspec,fp);
		def_list_head = def_list_head -> lhs;
	}
	gen_const_def(def_list_head->lhs,tspec,fp);
}

static int curr_kernel = 0;

static void
gen_kernels_recursive(const ASTNode* node, char** dfunctions,
            const bool gen_mem_accesses, int_vec* dfuncs_info)
{
  assert(node);

  if (node->type & NODE_KFUNCTION) {

    const size_t len = 64 * 1024 * 1024;
    char* prefix     = malloc(len);
    assert(prefix);
    prefix[0] = '\0';
    int_vec called_dfuncs = VEC_INITIALIZER;
    get_called_dfuncs(node, &called_dfuncs, dfuncs_info);

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

    for(size_t i = 0; i < num_dfuncs; ++i) if(int_vec_contains(called_dfuncs,i)) strcat(prefix,dfunctions[i]);

    astnode_set_prefix(prefix, compound_statement);
    free(prefix);
    free(cmdoptions);
    free_int_vec(&called_dfuncs);
  }

  if (node->lhs)
    gen_kernels_recursive(node->lhs, dfunctions, gen_mem_accesses,dfuncs_info);

  if (node->rhs)
    gen_kernels_recursive(node->rhs, dfunctions, gen_mem_accesses,dfuncs_info);
}
static void
gen_kernels(const ASTNode* node, char** dfunctions,
            const bool gen_mem_accesses)
{
  	traverse(node, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
	int_vec dfuncs_info[MAX_DFUNCS] = {[0 ... MAX_DFUNCS-1] = VEC_INITIALIZER };
	get_dfuncs_called_dfuncs(node, dfuncs_info);
	gen_kernels_recursive(node,dfunctions,gen_mem_accesses,dfuncs_info);
  	symboltable_reset();
}

// Generate User Defines
static void
gen_user_defines(const ASTNode* root, const char* out, const bool gen_mem_accesses)
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
    num_kernels += (!strcmp(symbol_table[i].tspecifier,"Kernel"));

  num_dfuncs = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    num_dfuncs += ((symbol_table[i].type & NODE_DFUNCTION_ID) != 0);

  
  // Stencils
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier,"Stencil"))

      fprintf(fp, "stencil_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_STENCILS} Stencil;");

  // Enums
  int num_of_communicated_fields=0;
  int num_of_fields=0;
  bool field_is_auxiliary[256];
  bool field_is_communicated[256];
  string_vec field_names = VEC_INITIALIZER;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if(!strcmp(symbol_table[i].tspecifier,"Field")){
      push(&field_names, symbol_table[i].identifier);
      const char* name = symbol_table[i].identifier;
      const bool is_aux = str_vec_contains(symbol_table[i].tqualifiers,"auxiliary");
      field_is_auxiliary[num_of_fields] = is_aux;
      const bool is_comm = str_vec_contains(symbol_table[i].tqualifiers,"communicated");
      num_of_communicated_fields += is_comm;
      field_is_communicated[num_of_fields] = is_comm;
      ++num_of_fields;
    }
  }
  fprintf(fp, "typedef enum {");
  //first communicated fields
  for(int i=0;i<num_of_fields;++i)
	fprintf(fp, "%s,",field_names.data[i]);

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

  fprintf(fp, "static const bool vtxbuf_is_communicated[] = {");
  for(int i=0;i<num_of_fields;++i)
    if(field_is_communicated[i])
    	fprintf(fp, "%s,", "true");
    else
    	fprintf(fp, "%s,", "false");
  fprintf(fp, "};");
  FILE* fp_vtxbuf_is_comm_func = fopen("vtxbuf_is_communicated_func.h","a");
  fprintf(fp_vtxbuf_is_comm_func ,"static __device__ constexpr __forceinline__ bool is_communicated(Field field) {\n"
             "switch(field)"
	     "{");
  for(int i=0;i<num_of_fields;++i)
  {
    const char* ret_val = (field_is_communicated[i]) ? "true" : "false";
    fprintf(fp_vtxbuf_is_comm_func,"case(%s): return %s;\n", field_names.data[i], ret_val);
  }
  fprintf(fp_vtxbuf_is_comm_func,"default: return false;\n");
  fprintf(fp_vtxbuf_is_comm_func, "}\n}\n");

  fclose(fp_vtxbuf_is_comm_func);

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
    if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
      fprintf(fp, "KERNEL_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_KERNELS} AcKernel;");
  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  // START---------------------------


  // Enum strings (convenience)
  fprintf(fp, "static const char* stencil_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier,"Stencil"))
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
    if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  structs_info s_info = read_user_structs(root);
  for (size_t i = 0; i < s_info.user_structs.size; ++i)
  {
	  char res[7000];
	  sprintf(res,"char* to_str(const %s value)\n"
		       "{\n"
		       "char* res = (char*)malloc(sizeof(char)*7000);\n"
		       "char* tmp;\n"
		       "res[0] = '{';\n"
		       ,s_info.user_structs.data[i]);

	  for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
	  {
	  	char tmp[7000];
		sprintf(tmp,"tmp = to_str(value.%s); strcat(res,tmp);\n",s_info.user_struct_field_names[i].data[j]);
		if(j < s_info.user_struct_field_names[i].size -1) strcat(tmp,"strcat(res,\",\");\n");
		strcat(tmp,"free(tmp);\n");
		strcat(res,tmp);
	  }
	  strcat(res,"strcat(res,\"}\");\n");
	  strcat(res,"return res;\n");
	  strcat(res,"}\n");
	  file_append("to_str_funcs.h",res);

	  sprintf(res,"template <>\n const char*\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], s_info.user_structs.data[i]);
	  file_append("to_str_funcs.h",res);
  }


  string_vec datatypes = s_info.user_structs;

  const char* builtin_datatypes[] = {"int","AcReal","int3","bool"};
  for (size_t i = 0; i < sizeof(builtin_datatypes)/sizeof(builtin_datatypes[0]); ++i)
	  push(&datatypes,builtin_datatypes[i]);

  user_enums_info enum_info =read_user_enums(root);
  for (size_t i = 0; i < enum_info.names.size; ++i)
  {
	  char res[7000];
	  char tmp[7000];
	  sprintf(res,"%s {\n","typedef enum");
	  for(size_t j = 0; j < enum_info.options[i].size; ++j)
	  {
		  strcat(res,enum_info.options[i].data[j]);
		  if(j < enum_info.options[i].size - 1)  strcat(res,",\n");
	  }
	  strcat(res,"} ");
	  strcat(res,enum_info.names.data[i]);
	  strcat(res,";\n");
  	  file_prepend("user_typedefs.h",res);

	  sprintf(res,"char* to_str(const %s value)\n"
		       "{\n"
		       "switch(value)\n"
		       "{\n"
		       ,enum_info.names.data[i]);

	  for(size_t j = 0; j < enum_info.options[i].size; ++j)
	  {
		  sprintf(tmp,"case %s: return strdup(\"%s\");\n",enum_info.options[i].data[j],enum_info.options[i].data[j]);
		  strcat(res,tmp);
	  }
	  strcat(res,"}\n}\n");
	  file_prepend("to_str_funcs.h",res);

	  sprintf(res,"template <>\n const char*\n get_datatype<%s>() {return \"%s\";};\n",enum_info.names.data[i], enum_info.names.data[i]);
	  file_prepend("to_str_funcs.h",res);

	  push(&datatypes,enum_info.names.data[i]);
  }


  for (size_t i = 0; i < datatypes.size; ++i)
  {
	  const char* datatype = datatypes.data[i];
	  gen_param_names(fp,datatype);
	  gen_enums(fp,datatype);

	  gen_dmesh_declarations(datatype);
	  gen_array_declarations(datatype,gen_mem_accesses);
	  gen_comp_declarations(datatype);
	  //gen_input_declarations(datatype);

  }

  fprintf(fp,"\n #ifdef __cplusplus\n");
  fprintf(fp,"typedef struct { int length; bool is_dconst; int d_offset; int num_dims; int3 dims; const char* name;} array_info;\n");
  for (size_t i = 0; i < datatypes.size; ++i)
  	  gen_array_info(fp,datatypes.data[i],root);
  fprintf(fp,"\n #endif\n");

  gen_user_taskgraphs(fp,root);


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
  if(!TWO_D)
  	fprintf(fp, "#define STENCIL_DEPTH (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define STENCIL_HEIGHT (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define STENCIL_WIDTH (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define NGHOST (STENCIL_ORDER/2)\n");
//#if AC_RUNTIME_COMPILATION
//  fprintf(fp, "#define AC_RUNTIME_COMPILATION (1)\n");
//#else
//  fprintf(fp, "#define AC_RUNTIME_COMPILATION (0)\n");
//#endif
  char cwd[10004];
  cwd[0] = '\0';
  char* err = getcwd(cwd, sizeof(cwd));
  assert(err != NULL);
  char autotune_path[10004];
  sprintf(autotune_path,"%s/autotune.csv",cwd);
  fprintf(fp,"__attribute__((unused)) static const char* autotune_csv_path= \"%s\";\n",autotune_path);

  char runtime_path[10004];
  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build/src/core/libastaroth_core.so");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_path = \"%s\";\n",runtime_path);

  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build/src/utils/libastaroth_utils.so");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_utils_path = \"%s\";\n",runtime_path);

  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_build_path = \"%s\";\n",runtime_path);

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
    if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
      fprintf(fp, "%s,", symbol_table[i].identifier); // Host layer handle
  fprintf(fp, "};");

  const char* default_param_list=  "(const int3 start, const int3 end, VertexBufferArray vba";
  FILE* fp_dec = fopen("user_kernel_declarations.h","a");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
      fprintf(fp_dec, "void __global__ %s %s);\n", symbol_table[i].identifier, default_param_list);
  fclose(fp_dec);

  if(gen_mem_accesses)
  { 
    FILE* fp_cpu = fopen("user_cpu_kernels.h","a");
    fprintf(fp_cpu, "static const Kernel cpu_kernels[] = {");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i)
      if (!strcmp(symbol_table[i].tspecifier,"Kernel"))
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
    if(check_symbol(NODE_VARIABLE_ID, node->buffer, "AcReal", "dconst") || check_symbol(NODE_VARIABLE_ID, node->buffer, "int", "dconst"))
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
  if(node->type == NODE_STENCIL)
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
	if(check_symbol(NODE_ANY, node->buffer, "Stencil", NULL))
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
	//const ASTNode* tspec = get_node(NODE_TSPEC,node);
	append_to_identifiers(str_to_append,root,name);
}
void
gen_dfunc_internal_names_recursive(ASTNode* node)
{

	if(node->lhs)
		gen_dfunc_internal_names_recursive(node->lhs);
	if(node->rhs)
		gen_dfunc_internal_names_recursive(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
	//to exclude input params
	//rename_local_vars(fn_identifier->buffer,node->rhs->rhs,node->rhs->rhs);
	rename_local_vars(fn_identifier->buffer,node->rhs,node->rhs);
}
void
gen_dfunc_internal_names(ASTNode* root)
{
  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
  gen_dfunc_internal_names_recursive(root);
  symboltable_reset();
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
gen_dfunc_macros_recursive(ASTNode* node)
{
	if(node->type & NODE_DEF)
		return;
	if(node->lhs)
		gen_dfunc_macros_recursive(node->lhs);
	if(node->rhs)
		gen_dfunc_macros_recursive(node->rhs);
	if(!(node->type & NODE_DFUNCTION))
		return;
	FILE* fp = fopen("user_dfuncs.h","a");
	if(node->rhs->lhs)
	{
		add_no_auto(node->rhs->lhs,NULL);
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
gen_dfunc_macros(ASTNode* node)
{
  symboltable_reset();
  traverse(node,NODE_NO_OUT,NULL);
  gen_dfunc_macros_recursive(node);
  symboltable_reset();

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
		fn_identifier -> no_auto = true;
}
void
remove_inlined_dfunc_nodes_recursive(ASTNode* node, ASTNode* root)
{
	if(node->lhs)
		remove_inlined_dfunc_nodes_recursive(node->lhs,root);
	if(node->rhs)
		remove_inlined_dfunc_nodes_recursive(node->rhs,root);
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
remove_inlined_dfunc_nodes(ASTNode* node, ASTNode* root)
{
  symboltable_reset();
  traverse(node,NODE_NO_OUT,NULL);
  remove_inlined_dfunc_nodes_recursive(node,root);
  symboltable_reset();

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
	if(!get_parent_node(NODE_FUNCTION,node))
		return;
	if(!node->rhs)
		return;
	if(!node->rhs->lhs)
		return;
	if(!node->rhs->lhs->rhs)
		return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node->lhs);
	if(!tspec)
		return;
	char* size = malloc(sizeof(char)*4098);
	combine_all(node->rhs->lhs->rhs,size);
	node->rhs->lhs->infix = NULL;
	node->rhs->lhs->postfix= NULL;
	node->rhs->lhs->rhs = NULL;
	char* new_res = malloc(sizeof(char)*4098);
	sprintf(new_res,"AcArray<%s,%s>",tspec->lhs->buffer,size);
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
  string_vec user_kernel_combinatorial_params[100] = {[0 ... 100 - 1] = VEC_INITIALIZER};
  string_vec user_kernels_with_input_params = VEC_INITIALIZER;


  int nums[100] = {0};
  string_vec* vals = malloc(sizeof(string_vec)*MAX_KERNELS*MAX_COMBINATIONS);
  param_combinations param_in = {nums, vals};

  gen_kernel_num_of_combinations(root,param_in,&user_kernels_with_input_params,user_kernel_combinatorial_params);
  if(optimize_conditionals)
  	gen_kernel_ifs(root,param_in,user_kernels_with_input_params,user_kernel_combinatorial_params);
  gen_kernel_input_params(root,gen_mem_accesses,param_in.vals,user_kernels_with_input_params,user_kernel_combinatorial_params);



  free_str_vec(&user_kernels_with_input_params);
  free(param_in.vals);
  for(int i = 0; i < 100; ++i)
	  free_str_vec(&user_kernel_combinatorial_params[i]);
}
bool
all_identifiers_are_constexpr(const ASTNode* node)
{
	bool res = true;
	if(node->lhs)
		res &= all_identifiers_are_constexpr(node->lhs);
	if(node->rhs)
		res &= all_identifiers_are_constexpr(node->rhs);
	if(node->token != IDENTIFIER)
		return res;
	if(!node->buffer)
		return res;
	res &= node->is_constexpr;
	return res;
}
bool
all_primary_expressions_and_func_calls_have_type(const ASTNode* node)
{
	bool res = true;
	if(node->lhs)
		res &= all_primary_expressions_and_func_calls_have_type(node->lhs);
	if(node->rhs)
		res &= all_primary_expressions_and_func_calls_have_type(node->rhs);
	if(node->type != NODE_PRIMARY_EXPRESSION && !(node->type & NODE_FUNCTION_CALL))
		return res;
	res &= node->expr_type != NULL;
	return res;
}
void
get_primary_expression_and_func_call_types_recursive(const ASTNode* node, string_vec* res)
{
	if(node->lhs)
		get_primary_expression_and_func_call_types_recursive(node->lhs,res);
	if(node->rhs)
		get_primary_expression_and_func_call_types_recursive(node->rhs,res);
	if(!(node->type == NODE_PRIMARY_EXPRESSION))
		return;
	if(node->expr_type) push(res,node->expr_type);
}
string_vec
get_primary_expression_and_func_call_types(const ASTNode* node)
{
	string_vec res = VEC_INITIALIZER; 
	get_primary_expression_and_func_call_types_recursive(node,&res);
	return res;
}

void
set_identifier_constexpr(ASTNode* node, const char* identifier_name, const bool is_constexpr)
{
	if(!node)
		return;
	if(node->lhs)
		set_identifier_constexpr(node->lhs,identifier_name,is_constexpr);
	if(node->rhs)
		set_identifier_constexpr(node->rhs,identifier_name,is_constexpr);
	if(node->token != IDENTIFIER)
		return;
	if(!node->buffer)
		return;
	if(strcmp(node->buffer,identifier_name))
		return;
	node->is_constexpr = is_constexpr;
}
void
set_primary_expression_types(ASTNode* node, const char* type, const char* identifier)
{
	if(!node)
		return;
	if(node->lhs)
		set_primary_expression_types(node->lhs,type,identifier);
	if(node->rhs)
		set_primary_expression_types(node->rhs,type,identifier);
	if(node->type != NODE_PRIMARY_EXPRESSION)
		return;
	const ASTNode* identifier_node = get_node_by_token(IDENTIFIER,node);
	if(!identifier_node || strcmp(identifier_node->buffer,identifier)) return;
	node->expr_type = strdup(type);
}
void
count_num_of_assignments_to_lhs(const ASTNode* node, const char* lhs, int* res)
{
	if(node->lhs)
		count_num_of_assignments_to_lhs(node->lhs,lhs,res);
	if(node->rhs)
		count_num_of_assignments_to_lhs(node->rhs,lhs,res);
	if(node->type == NODE_ASSIGNMENT && node->rhs)
	  *res += !strcmp(get_node_by_token(IDENTIFIER,node->lhs)->buffer,lhs);

}
bool
gen_constexpr_info_base(ASTNode* node, ASTNode* func_base, const bool gen_mem_accesses)
{
	bool res = false;
	if(node->type & NODE_ASSIGN_LIST)
		return res;
	if(node->type & NODE_FUNCTION)
		func_base = node;
	if(node->lhs)
		res |= gen_constexpr_info_base(node->lhs,func_base,gen_mem_accesses);
	if(node->rhs)
		res |= gen_constexpr_info_base(node->rhs,func_base,gen_mem_accesses);
	if(node->token == IDENTIFIER && node->buffer && !node->is_constexpr)
	{
		node->is_constexpr |= check_symbol(NODE_ANY,node->buffer,NULL,"const");
		//if array access that means we are accessing the vtxbuffer which obviously is not constexpr
 		if(!get_parent_node(NODE_ARRAY_ACCESS,node))
			node->is_constexpr |= check_symbol(NODE_VARIABLE_ID,node->buffer,"Field",NULL);
		node->is_constexpr |= check_symbol(NODE_DFUNCTION_ID,node->buffer,NULL,"constexpr");
		res |= node->is_constexpr;
	}
	if(node->type & NODE_IF && all_identifiers_are_constexpr(node->lhs) && !node->is_constexpr && !gen_mem_accesses)
	{
		node->is_constexpr = true;
		node->prefix= strdup(" constexpr (");
	}
	//TP: below sets the constexpr value of lhs the same as rhs for: lhs = rhs
	//TP: we restrict to the case that lhs is assigned only once in the function since full generality becomes too hard (would require something like static single-assignment form), and is not needed if the code is written intelligently
	if(node->type &  NODE_ASSIGNMENT && node->rhs && func_base)
	{
	  bool is_constexpr = all_identifiers_are_constexpr(node->rhs);
	  ASTNode* lhs_identifier = get_node_by_token(IDENTIFIER,node->lhs);
	  int num_of_assignments_to_lhs = 0;
	  count_num_of_assignments_to_lhs(func_base,lhs_identifier->buffer,&num_of_assignments_to_lhs);
	  if(num_of_assignments_to_lhs != 1)
		  return res;
	  if(lhs_identifier->is_constexpr == is_constexpr)
		  return res;
	  res |= is_constexpr;
	  lhs_identifier->is_constexpr = is_constexpr;
	  set_identifier_constexpr(func_base,lhs_identifier->buffer,is_constexpr);
	  if(is_constexpr && get_node(NODE_TSPEC,node->lhs))
	  {
		  ASTNode* tspec = (ASTNode*) get_node(NODE_TSPEC,node->lhs);
		  char* new_type = malloc(sizeof(char)* (strlen(tspec->lhs->buffer) + 100));
		  sprintf(new_type," constexpr %s",tspec->lhs->buffer);
		  free(tspec->lhs->buffer);
		  tspec->lhs->buffer = new_type;
	  }
	}
	if(node->type & NODE_RETURN && !node->is_constexpr)
	{
		if(all_identifiers_are_constexpr(node->rhs))
		{
			const ASTNode* dfunc_start = get_parent_node(NODE_DFUNCTION,node);
			const char* func_name = get_node(NODE_DFUNCTION_ID,dfunc_start)->buffer;
			Symbol* func_sym = (Symbol*)get_symbol(NODE_DFUNCTION_ID,func_name,NULL);
			if(func_sym)
			{
				push(&func_sym->tqualifiers,"constexpr");
				node->is_constexpr = true;
				res |= node->is_constexpr;
			}
		}
	}
	return res;
}

void
gen_constexpr_info(ASTNode* root, const bool gen_mem_accesses)
{
        traverse(root, NODE_NO_OUT, NULL);
	bool has_changed = true;
	while(has_changed)
	{
		has_changed = gen_constexpr_info_base(root,NULL, gen_mem_accesses);
	}
}

void
get_dfunc_identifiers_recursive(const ASTNode* node, string_vec* res)
{
	if(node->lhs)
		get_dfunc_identifiers_recursive(node->lhs,res);
	if(node->rhs)
		get_dfunc_identifiers_recursive(node->rhs,res);
	if(node->type & NODE_DFUNCTION_ID)
		push(res,node->buffer);
}
string_vec
get_dfunc_identifiers(const ASTNode* node)
{
	string_vec res = VEC_INITIALIZER;
	get_dfunc_identifiers_recursive(node,&res);
	return res;
}
string_vec
get_duplicate_dfuncs(const ASTNode* node)
{
  traverse(node, NODE_NO_OUT, NULL);
  string_vec dfuncs = get_dfunc_identifiers(node);
  int n_entries[dfuncs.size];
  memset(n_entries,0,sizeof(int)*dfuncs.size);
  for(size_t i = 0; i < dfuncs.size; ++i)
	  n_entries[get_symbol_index(NODE_DFUNCTION_ID,dfuncs.data[i],NULL)]++;
  string_vec res = VEC_INITIALIZER;
  for(size_t i = 0; i < dfuncs.size; ++i)
  {
	  const int index = get_symbol_index(NODE_DFUNCTION_ID,dfuncs.data[i],NULL);
	  if(index == -1) continue;
	  if(n_entries[index] > 1) push(&res,get_symbol_by_index(NODE_DFUNCTION_ID,index,NULL)->identifier);
  }
  free_str_vec(&dfuncs);
  return res;
}
bool
gen_type_info_base(ASTNode* node, const ASTNode* root, ASTNode* func_base)
{
	bool res = false;
	if(node->type & NODE_ASSIGN_LIST)
		return res;
	if(node->type & NODE_FUNCTION)
	{
		func_base = node;
		if(node->rhs->lhs)
		{
			const ASTNode* param_list_head= node->rhs->lhs;
			while(param_list_head->rhs)
			{
				const ASTNode* param = param_list_head->rhs;
				const ASTNode* identifier = get_node_by_token(IDENTIFIER,param->rhs);
				const ASTNode* type = get_node(NODE_TSPEC,param->lhs);
				if(type)
					set_primary_expression_types(func_base, type->lhs->buffer, identifier->buffer);
				param_list_head = param_list_head->lhs;
			}
			const ASTNode* param = param_list_head->lhs;
			const ASTNode* identifier = get_node_by_token(IDENTIFIER,param->rhs);
			const ASTNode* type = get_node(NODE_TSPEC,param->lhs);
			if(type)
				set_primary_expression_types(func_base, type->lhs->buffer, identifier->buffer);
		}
	}
	if(node->lhs)
		res |= gen_type_info_base(node->lhs,root,func_base);
	if(node->rhs)
		res |= gen_type_info_base(node->rhs,root,func_base);
	if(node->type & NODE_RETURN)
	{
		char* expr_type = get_expr_type(node->rhs,root);
		const ASTNode* dfunc_start = get_parent_node(NODE_DFUNCTION,node);
		const char* func_name = get_node(NODE_DFUNCTION_ID,dfunc_start)->buffer;
		Symbol* func_sym = (Symbol*)get_symbol(NODE_DFUNCTION_ID,func_name,NULL);
		if(func_sym && expr_type && !str_vec_contains(func_sym->tqualifiers,expr_type))
			push(&func_sym->tqualifiers,expr_type);
	}
	if(node->expr_type) return res;
	if(node->type == NODE_PRIMARY_EXPRESSION)
	{
		get_expr_type(node,root);
	}
	else if(node->type & NODE_EXPRESSION && all_primary_expressions_and_func_calls_have_type(node))
	{
		char* expr_type = get_expr_type(node,root);
		if(expr_type) node->expr_type = strdup(expr_type);
	}
	else if(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node))
	{
		node->expr_type = get_node(NODE_TSPEC,node)->lhs->buffer;
	}
	else if(node->type & NODE_ASSIGNMENT && node->rhs && func_base)
	{

		const char* identifier = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	       	if(get_expr_type(node->rhs,root))
		{
			node->expr_type = get_expr_type(node->rhs,root);
			set_primary_expression_types(func_base, node->expr_type, identifier);
			return true;
		}

	}
	else if(node->type & NODE_FUNCTION_CALL)
	{
		Symbol* sym = (Symbol*)get_symbol(NODE_DFUNCTION_ID | NODE_FUNCTION_ID ,get_node_by_token(IDENTIFIER,node->lhs)->buffer,NULL);
		const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
		if(node->rhs && count_num_of_nodes_in_list(node->rhs) == 1 && sym)
		{
			string_vec duplicate_dfuncs = get_duplicate_dfuncs(root);
			if(!str_vec_contains(duplicate_dfuncs,func_name))
			{
				const char* expr_type = get_expr_type(node->rhs,root);
				if(expr_type)
				{
					if(!strcmp("contract",func_name))
						printf("HI: %s,%s\n",func_name,expr_type);
					push(&sym->tqualifiers,expr_type);
				}
			}
			free_str_vec(&duplicate_dfuncs);
		}
		if(sym && sym->tqualifiers.size > 0)
		{
  			const char* builtin_datatypes[] = {"int","AcReal","int3","AcReal3","Field","Field3"};
  			for (size_t i = 0; i < sizeof(builtin_datatypes)/sizeof(builtin_datatypes[0]); ++i)
				if(str_vec_contains(sym -> tqualifiers,builtin_datatypes[i])) node->expr_type = strdup(builtin_datatypes[i]);
		}
		const Symbol* stencil = get_symbol(NODE_VARIABLE_ID,func_name,"Stencil");
		if(stencil)
			node->expr_type = strdup("AcReal");
		
	}
	res |=  node -> expr_type != NULL;
	return res;
}
void
gen_type_info(ASTNode* root)
{
	bool has_changed = true;
        traverse(root,NODE_NO_OUT, NULL);
	int iter = 0;
	while(has_changed)
	{
		printf("ITEER: %d\n",iter++);
		has_changed = gen_type_info_base(root,root,NULL);
	}
}
void
transform_runtime_vars_recursive(ASTNode* node)
{
	if(node->lhs)
		transform_runtime_vars_recursive(node->lhs);
	if(node->rhs)
		transform_runtime_vars_recursive(node->rhs);
	if(!node->buffer) return;
	if(!get_parent_node(NODE_FUNCTION,node)) return;
	const Symbol* var = get_symbol(NODE_VARIABLE_ID,node->buffer,NULL);
	if(!var) return;
	if(!str_vec_contains(var->tqualifiers,"run_const")) return;
	char* new_buffer = (!strcmp(var->tspecifier,"AcReal")) ? "0.0" :
			   (!strcmp(var->tspecifier,"AcReal*")) ? "AC_INTERNAL_big_real_array": 
			   (!strcmp(var->tspecifier,"int")) ? "0": 
			   (!strcmp(var->tspecifier,"bool")) ? "true": 
			   (!strcmp(var->tspecifier,"int*")) ? "AC_INTERNAL_big_int_array": 
			   (!strcmp(var->tspecifier,"AcReal3")) ? "AC_INTERNAL_global_real_vec": 
			   (!strcmp(var->tspecifier,"int3")) ? "AC_INTERNAL_global_int_vec": 
			   NULL;
	if(!new_buffer)
	{
		fprintf(stderr,"Fatal error: missing default type for: %s\n",var->tspecifier);
		exit(EXIT_FAILURE);
	}
	free(node->buffer);
	node->buffer = strdup(new_buffer);
	node->no_auto = true;
	node->is_constexpr = true;
}
void
transform_runtime_vars(ASTNode* root)
{
  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
  transform_runtime_vars_recursive(root);
  symboltable_reset();
}
const ASTNode*
find_dfunc_start(const ASTNode* node, const char* dfunc_name)
{
	const ASTNode* res = NULL;
	if(node->type & NODE_DFUNCTION && get_node(NODE_DFUNCTION_ID,node) && get_node(NODE_DFUNCTION_ID,node)->buffer && !strcmp(get_node(NODE_DFUNCTION_ID,node)->buffer, dfunc_name)) res = node;
	if(res == NULL && node->lhs && find_dfunc_start(node->lhs,dfunc_name)) res = find_dfunc_start(node->lhs,dfunc_name);
	if(res == NULL && node->rhs && find_dfunc_start(node->rhs,dfunc_name)) res = find_dfunc_start(node->rhs,dfunc_name);
	return res;
		
}
void
mangle_dfunc_name(ASTNode* node, const char* dfunc_name, string_vec* dst, const int dfunc_index, int* counter)
{
	if(node->lhs)
		mangle_dfunc_name(node->lhs,dfunc_name,dst,dfunc_index,counter);
	if(node->rhs)
		mangle_dfunc_name(node->rhs,dfunc_name,dst,dfunc_index,counter);
	if(!(node->type & NODE_DFUNCTION))
		return;
	if(strcmp(get_node_by_token(IDENTIFIER,node->lhs)->buffer, dfunc_name))
		return;


	func_params_info params_info = get_function_param_types_and_names(node,dfunc_name);
	char* tmp = malloc(sizeof(char)*10000);
	sprintf(tmp,"%s_AC_MANGLED_NAME_",dfunc_name);
	for(size_t i = 0; i < params_info.types.size; ++i)
	{
		push(&dst[*counter + MAX_DFUNCS*dfunc_index], params_info.types.data[i]);
		strcat(tmp,"_");
		strcat(tmp,params_info.types.data[i]);
	}
	free(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	get_node_by_token(IDENTIFIER,node->lhs)->buffer = strdup(tmp);
	
	free_func_params_info(&params_info);
	free(tmp);
	++(*counter);
}
bool
resolve_overloaded_calls(ASTNode* node, const ASTNode* root, const char* dfunc_name, string_vec* dfunc_possible_types,const int dfunc_index)
{
	bool res = false;
	if(node->lhs)
		res |= resolve_overloaded_calls(node->lhs,root,dfunc_name,dfunc_possible_types,dfunc_index);
	if(node->rhs)
		res |= resolve_overloaded_calls(node->rhs,root,dfunc_name,dfunc_possible_types,dfunc_index);
	if(!(node->type & NODE_FUNCTION_CALL))
		return res;
	if(!get_node_by_token(IDENTIFIER,node->lhs))
		return res;
	if(strcmp(get_node_by_token(IDENTIFIER,node->lhs)->buffer, dfunc_name))
		return res;
	func_params_info call_info = get_func_call_params_info(node,root);
	char* tmp = malloc(sizeof(char)*10000);
	sprintf(tmp,"%s_AC_MANGLED_NAME_",dfunc_name);
	int correct_types = -1;
	int start = MAX_DFUNCS*dfunc_index-1;
	int_vec possible_indexes = VEC_INITIALIZER;
	while(dfunc_possible_types[++start].size > 0)
	{
		if(call_info.types.size != dfunc_possible_types[start].size) continue;
		bool possible = true;
		for(size_t i = 0; i < call_info.types.size; ++i)
			possible &= !(call_info.types.data[i] && strcmp(call_info.types.data[i],dfunc_possible_types[start].data[i]) != 0);
		if(possible)
			push_int(&possible_indexes,start);
	}
	bool able_to_resolve = possible_indexes.size == 1;
	if(!able_to_resolve) { 
		combine_all(node->rhs,tmp); 
		printf("Not able to resolve: %s\n",tmp); 
		printf("HMM: %s\n",call_info.types.data[0]);
		return res;
	}
	string_vec types = dfunc_possible_types[possible_indexes.data[0]];
	for(size_t i = 0; i < types.size; ++i)
	{
		strcat(tmp,"_");
		strcat(tmp,types.data[i]);
	}
	free(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	get_node_by_token(IDENTIFIER,node->lhs)->buffer = strdup(tmp);

	free(tmp);
	free_str_vec(&call_info.expr);
	free_str_vec(&call_info.types);
	free_int_vec(&possible_indexes);
	return true;
}

void
gen_overloads(ASTNode* root)
{
  bool overloaded_something = true;
  string_vec duplicate_dfuncs = get_duplicate_dfuncs(root);
  string_vec dfunc_possible_types[MAX_DFUNCS * duplicate_dfuncs.size];
  memset(dfunc_possible_types,0,sizeof(string_vec)*MAX_DFUNCS*duplicate_dfuncs.size);
  for(size_t i = 0; i < duplicate_dfuncs.size; ++i)
  {
	int counter = 0;
  	mangle_dfunc_name(root,duplicate_dfuncs.data[i], dfunc_possible_types, i, &counter);
  }
  int iter = 0;
  while(overloaded_something)
  {
	printf("ITER: %d\n",iter++);
	overloaded_something = false;
  	symboltable_reset();
  	traverse(root, NODE_NO_OUT, NULL);
  	gen_type_info(root);
  	for(size_t i = 0; i < duplicate_dfuncs.size; ++i)
  	        overloaded_something |= resolve_overloaded_calls(root,root,duplicate_dfuncs.data[i],dfunc_possible_types,i);
  	symboltable_reset();
  }
  free_str_vec(&duplicate_dfuncs);
  for(size_t i = 0; i < MAX_DFUNCS*duplicate_dfuncs.size; ++i)
	  free_str_vec(&dfunc_possible_types[i]);
}
void
make_enum_options_non_auto(ASTNode* node, const ASTNode* root)
{
	if(node->lhs)
		make_enum_options_non_auto(node->lhs,root);
	if(node->rhs)
		make_enum_options_non_auto(node->rhs,root);
	if(!(node->token == IDENTIFIER) || !node->buffer) return;
	node->no_auto |= is_user_enum_option(root,node->buffer);
}
void
generate(const ASTNode* root_in, FILE* stream, const bool gen_mem_accesses, const bool optimize_conditionals)
{ 
  ASTNode* root = astnode_dup(root_in,NULL);
  assert(root);
  transform_runtime_vars(root);
  gen_overloads(root);
  gen_constexpr_info(root,gen_mem_accesses);
  make_enum_options_non_auto(root,root);

  gen_kernel_structs(root,gen_mem_accesses);
  gen_user_structs(root);
  gen_user_defines(root, "user_defines.h",gen_mem_accesses);
  gen_multidimensional_field_accesses(root);
  gen_user_kernels(root, "user_declarations.h", gen_mem_accesses);

  gen_kernel_combinatorial_optimizations_and_input(root,gen_mem_accesses,optimize_conditionals);


  // Fill the symbol table
  traverse(root, NODE_NO_OUT, NULL);


  gen_kernel_postfixes_and_reduce_outputs(root,gen_mem_accesses);

  // print_symbol_table();

  // Generate user_kernels.h
  fprintf(stream, "#pragma once\n");

  size_t num_stencils = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier,"Stencil"))
      ++num_stencils;




  // Device constants
  // gen_dconsts(root, stream);
  const char* array_datatypes[] = {"int","AcReal","bool","int3","AcReal3"};
  for (size_t i = 0; i < sizeof(array_datatypes)/sizeof(array_datatypes[0]); ++i)
  	gen_array_reads(root,root,array_datatypes[i]);

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
      if (!strcmp(symbol.tspecifier,"Stencil")) {
	      if(symbol.tqualifiers.size)
	      {
		if(symbol.tqualifiers.size > 1)
		{
			fprintf(stderr,"Stencils are supported only with a single type qualifier\n");
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
  //Fill symbol table
  traverse(root, 0, NULL);
  traverse(root,
           NODE_VARIABLE_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           stencil_coeffs_fp);
  fflush(stencil_coeffs_fp);

  replace_dynamic_coeffs(root);
  symboltable_reset();
  if(!TWO_D)
  	fprintf(stencilgen, "static char* "
                      "dynamic_coeffs[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = { %s };\n", stencil_coeffs);
  else
  	fprintf(stencilgen, "static char* "
                      "dynamic_coeffs[NUM_STENCILS][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = { %s };\n", stencil_coeffs);
  if(!TWO_D)
  	fprintf(stencilgen, "static char* "
                      "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = {");
  else
  	fprintf(stencilgen, "static char* "
                      "stencils[NUM_STENCILS][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = {");
  traverse(root,
           NODE_VARIABLE_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
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
           "-DTWO_D=%d "
           "-Wfloat-conversion -Wshadow -I. %s -lm "
           "-o %s",
           IMPLEMENTATION, MAX_THREADS_PER_BLOCK, TWO_D,STENCILGEN_SRC,
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
  transform_arrays_to_std_arrays(root);

  //used to enable inlined funcs
  //gen_dfunc_internal_names(root);
  //gen_dfunc_macros(astnode_dup(root,NULL));
  //remove_inlined_dfunc_nodes(root,root);

  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
  char** dfunc_strs = malloc(sizeof(char*)*num_dfuncs);
  FILE* dfunc_fps[num_dfuncs];
  const ASTNode* dfunc_heads[num_dfuncs];
  size_t sizeloc;
  for(size_t i = 0; i < num_dfuncs; ++i)
  {

	const Symbol* dfunc_symbol = get_symbol_by_index(NODE_DFUNCTION_ID,i,NULL);
	dfunc_heads[i] = (dfunc_symbol) ? find_dfunc_start(root,dfunc_symbol->identifier) : NULL;
  }
  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_DFUNCTION_ID | NODE_STENCIL | NODE_NO_OUT, NULL);
  for(size_t i = 0; i < num_dfuncs; ++i)
  {
  	dfunc_fps[i] =  open_memstream(&dfunc_strs[i], &sizeloc);
	if(dfunc_heads[i])
  		traverse(dfunc_heads[i],
  	           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_KFUNCTION |
  	               NODE_HOSTDEFINE | NODE_NO_OUT,
  	           dfunc_fps[i]);
	else
		fprintf(dfunc_fps[i],"%s","");
  	fflush(dfunc_fps[i]);
  }
	



  // Kernels
  symboltable_reset();
  gen_kernels(root, dfunc_strs, gen_mem_accesses);
  for(size_t i = 0; i < num_dfuncs; ++i)
	  fclose(dfunc_fps[i]);
  free(dfunc_strs);



  

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
  system("cp user_kernels.h user_kernels.h.backup");
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
  if(TWO_D)
  	strcat(cmd, "-DTWO_D=1 ");
  else
  	strcat(cmd, "-DTWO_D=0 ");
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


//  const size_t k = 1;
//  for(size_t i = 0; i < num_fields; ++i)
//	  printf("written to: %d\n",written_fields[i + num_fields*k]);
//  for(size_t i = 0; i < num_fields; ++i)
//	  printf("read from: %d\n",read_fields[i + num_fields*k]);
//  for(size_t i = 0; i < num_fields; ++i)
//	  printf("has stencil op: %d\n",field_has_stencil_op[i + num_fields*k]);
}


