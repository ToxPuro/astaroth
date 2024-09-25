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

void
gen_dlsym(FILE* fp, const char* func_name)
{
	fprintf(fp,"*(void**)(&%s) = dlsym(handle,\"%s\");\n",func_name,func_name);
	fprintf(fp,"if(!%s) fprintf(stderr,\"Astaroth error was not able to load %s\\n\");\n",func_name,func_name);
}


void
get_executed_conditionals(void);

#include "ast.h"
#include "tab.h"
#include <string.h>
#include <ctype.h>

#define TRAVERSE_PREAMBLE(FUNC_NAME) \
	if(node->lhs) \
		FUNC_NAME(node->lhs); \
	if(node->rhs) \
		FUNC_NAME(node->rhs); 

static node_vec    dfunc_nodes      = VEC_INITIALIZER;
static string_vec  dfunc_names      = VEC_INITIALIZER;
typedef struct
{
	string_vec names;
	int_vec    counts;
} overloaded_dfuncs;
static overloaded_dfuncs duplicate_dfuncs = {
						.names = VEC_INITIALIZER,
						.counts = VEC_INITIALIZER
					   };

//node_vec
//get_nodes_in_list_in_reverse_order(const ASTNode* head)
//{
//	node_vec res = VEC_INITIALIZER;
//	while(head -> rhs)
//	{
//		push_node(&res, head->rhs);
//		head = head ->lhs;
//	}
//	push_node(&res, head->lhs);
//	return res;
//}

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

static int_vec executed_conditionals = VEC_INITIALIZER;


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
  //We keep this as int_vec since makes comparisons so much faster
  int_vec tqualifiers;
  char tspecifier[MAX_ID_LEN];
  int tspecifier_token;
  char identifier[MAX_ID_LEN];
  } Symbol;


static string_vec tspecifier_mappings = VEC_INITIALIZER;


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

static Symbol*
symboltable_lookup_surrounding_scope(const char* identifier)
{
  if (!identifier)
    return NULL;
  int scope = current_nest;
  while(scope--)
  {
  	for (size_t i = 0; i < num_symbols[scope]; ++i)
  	  if (!strcmp(identifier, symbol_table[i].identifier))
  	    return &symbol_table[i];
  }
  return NULL;
}


static int
get_symbol_index(const NodeType type, const char* symbol, const int tspecifier)
{

  int counter = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if ((!tspecifier || tspecifier == symbol_table[i].tspecifier_token ) && symbol_table[i].type & type)
    {
	    if(!strcmp(symbol_table[i].identifier,symbol))
		    return counter;
	    counter++;
    }
  }
  return -1;
}
static const Symbol*
get_symbol_by_index(const NodeType type, const int index, const int tspecifier)
{
  int counter = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if ((!tspecifier || symbol_table[i].tspecifier_token == tspecifier) && symbol_table[i].type & type)
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
  for (size_t i = 0; i < num_symbols[0]; ++i)
	  if(symbol_table[i].type & type && !strcmp(symbol,symbol_table[i].identifier) && (!tspecifier || !strcmp(symbol_table[i].tspecifier,tspecifier)))
			  return &symbol_table[i];
  return NULL;
}

static const Symbol*
get_symbol_token(const NodeType type, const char* symbol, const int tspecifier)
{
  for (size_t i = 0; i < num_symbols[0]; ++i)
	  if((!tspecifier || tspecifier == symbol_table[i].tspecifier_token) && symbol_table[i].type & type && !strcmp(symbol,symbol_table[i].identifier))
			  return &symbol_table[i];
  return NULL;
}
#define REAL_SPECIFIER  (1 << 0)
#define INT_SPECIFIER   (1 << 1)
#define BOOL_SPECIFIER  (1 << 2)
#define REAL3_SPECIFIER (1 << 3)
#define REAL4_SPECIFIER (1 << 4)

static int 
add_symbol(const NodeType type, const int* tqualifiers, const size_t n_tqualifiers, const char* tspecifier,
           const int tspecifier_token, const char* id)
{
  if(is_number(id))
  {
	  printf("WRONG: %s\n",id);
	  exit(EXIT_FAILURE);
  }
  assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);
  symbol_table[num_symbols[current_nest]].type          = type;
  symbol_table[num_symbols[current_nest]].tspecifier[0] = '\0';
  init_int_vec(&symbol_table[num_symbols[current_nest]].tqualifiers);
  for(size_t i = 0; i < n_tqualifiers; ++i)
      	push_int(&symbol_table[num_symbols[current_nest]].tqualifiers,tqualifiers[i]);

  if (tspecifier)
  {
    strcpy(symbol_table[num_symbols[current_nest]].tspecifier, tspecifier);
  }
  symbol_table[num_symbols[current_nest]].tspecifier_token  = (tspecifier) ? tspecifier_token : 0;
  strcpy(symbol_table[num_symbols[current_nest]].identifier, id);

  const bool is_field_without_type_qualifiers = tspecifier_token && tspecifier_token == FIELD && symbol_table[num_symbols[current_nest]].tqualifiers.size == 0;
  ++num_symbols[current_nest];
  const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
  if(!is_field_without_type_qualifiers)
  	return num_symbols[current_nest]-1;
	  
  if(!has_optimization_info)
  {
  	push_int(&symbol_table[num_symbols[current_nest]-1].tqualifiers, COMMUNICATED);
  	return num_symbols[current_nest]-1;
  } 



   const int field_index = get_symbol_index(NODE_VARIABLE_ID, id, FIELD);
   bool is_auxiliary = true;
   bool is_communicated = false;
   bool is_dead         = true;
   //TP: a field is dead if its existence does not have any observable effect on the DSL computation
   //For now it means that the field is not read, no stencils called on it and not written out
   for(size_t k = 0; k < num_kernels; ++k)
   {
	   const int written        = written_fields[field_index + num_fields*k];
	   const int input_accessed = (read_fields[field_index + num_fields*k] || field_has_stencil_op[field_index + num_fields*k]);
	   is_auxiliary    &=  OPTIMIZE_FIELDS && (!written_fields[field_index + num_fields*k] || !field_has_stencil_op[field_index + num_fields*k]);
	   is_communicated |=  !OPTIMIZE_FIELDS || field_has_stencil_op[field_index + num_fields*k];
	   const bool should_be_alive = (!OPTIMIZE_FIELDS) || written_fields[field_index + num_fields*k] || field_has_stencil_op[field_index + num_fields*k] || read_fields[field_index + num_fields*k];
	   is_dead      &= !should_be_alive;

   }
   if(is_communicated)
   {
   	push_int(&symbol_table[num_symbols[current_nest]-1].tqualifiers, COMMUNICATED);
   }
   if(is_auxiliary)
	push_int(&symbol_table[num_symbols[current_nest]-1].tqualifiers, AUXILIARY);
   if(is_dead)
	push_int(&symbol_table[num_symbols[current_nest]-1].tqualifiers, DEAD);



  //return the index of the lastly added symbol
  return num_symbols[current_nest]-1;
}

typedef struct
{
	string_vec names;
	string_vec options[100];
} user_enums_info;


static user_enums_info e_info;


typedef struct
{
	string_vec user_structs;
	string_vec* user_struct_field_names;
	string_vec* user_struct_field_types;
} structs_info;

static structs_info s_info;

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
const char* primitive_datatypes[] = {"int","AcReal","bool","long","long long"};

bool is_primitive_datatype(const char* type)
{
	return !strcmps(type,"int","AcReal","bool","long","long long");
}
string_vec
get_all_datatypes()
{
  string_vec datatypes = str_vec_copy(s_info.user_structs);

  for (size_t i = 0; i < sizeof(primitive_datatypes)/sizeof(primitive_datatypes[0]); ++i)
	  push(&datatypes,primitive_datatypes[i]);

  user_enums_info enum_info = e_info;
  for (size_t i = 0; i < enum_info.names.size; ++i)
	  push(&datatypes,enum_info.names.data[i]);
  return datatypes;
}

char*
convert_to_define_name(const char* name)
{
	if(!strcmp(name,"long long"))
		return strdup("long_long");
	char* res = strdup(name);
	if(strlen(res) > 2 && res[0]  == 'A' && res[1] == 'c')
	{
		res = &res[2];
		res[0] = tolower(res[0]);
	}
	return res;
}

static void
symboltable_reset(void)
{
  for(size_t i = 0; i < SYMBOL_TABLE_SIZE ; ++i)
	  free_int_vec(&symbol_table[i].tqualifiers);

  current_nest              = 0;
  num_symbols[current_nest] = 0;

  int field3_tq[1] =  {FIELD3};
  int real_tq[1]   =  {REAL};
  int real3_tq[1]  =  {REAL3};
  int int_tq[1]  =    {INT};
  int int3_tq[1]  =   {INT3};
  int const_tq[1]  =  {CONST_QL};

  // Add built-in variables (TODO consider NODE_BUILTIN)
  add_symbol(NODE_VARIABLE_ID, NULL, 0, NULL, 0, "stderr");           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "print");           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "fprintf");           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "threadIdx");       // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "blockIdx");        // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "vertexIdx");       // TODO REMOVE
  int vertex_index = add_symbol(NODE_VARIABLE_ID, NULL, 0, "int3", INT3, "globalVertexIdx"); // TODO REMOVE
  symbol_table[vertex_index].tqualifiers.size = 0;
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "globalGridN");     // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "write_base");  // TODO RECHECK
							 //
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "reduce_sum");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "reduce_min");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0, "reduce_max");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0,"int", 0, "size");  // TODO RECHECK
  //In develop
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "read_w");
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write_w");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, "Field3",FIELD3, "MakeField3"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, "int",  INT, "len");    // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"uint64_t");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"UINT64_MAX"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, "AcReal", REAL,"rand_uniform");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, "AcReal", REAL,"AcReal");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, "AcReal", REAL,"previous_base");  // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"multm2_sym");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"diagonal");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"sum");   // TODO RECHECK

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, "AcReal", REAL,"AC_REAL_PI");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"NUM_FIELDS");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"NUM_VTXBUF_HANDLES");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"NUM_ALL_FIELDS");

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"FIELD_IN");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"FIELD_OUT");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, 0,"IDX");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, "AcReal3", REAL3,"AcReal3");

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, "bool", BOOL,"true");
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, "bool", BOOL,"false");

  // Astaroth 2.0 backwards compatibility START
  // (should be actually built-in externs in acc-runtime/api/acc-runtime.h)
  int tqualifiers[1] = {DCONST_QL};
  int const_qualifier[1] = {CONST_QL};

  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1,"int", INT,"AC_xy_plate_bufsize");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1,"int", INT,"AC_xz_plate_bufsize");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1,"int", INT,"AC_yz_plate_bufsize");


  //For special reductions
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", REAL,"AC_center_x");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", REAL,"AC_center_y");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", REAL,"AC_center_z");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", REAL,"AC_sum_radius");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "AcReal", REAL,"AC_window_radius");

  // (BC types do not belong here, BCs not handled with the DSL)
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_bc_type_bot_x");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_bc_type_bot_y");
#if TWO_D == 0
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_bc_type_bot_z");
#endif

  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_bc_type_top_x");
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_bc_type_top_y");
#if TWO_D == 0
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_bc_type_top_z");
#endif
  add_symbol(NODE_VARIABLE_ID, tqualifiers, 1, "int", INT,"AC_init_type");
  add_symbol(NODE_VARIABLE_ID, const_qualifier, 1, "int", INT,"STENCIL_ORDER");
  // Astaroth 2.0 backwards compatibility END
  int index = add_symbol(NODE_VARIABLE_ID, NULL, 0 , "int3", INT3,"blockDim");
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
	    //char tqual[4096];
	    //tqual[0] = '\0';
	    //for(size_t tqi = 0; tqi < symbol_table[i].tqualifiers.size; ++tqi)
	    //        strcat(tqual,symbol_table[i].tqualifiers.data[tqi]);
      	    //printf("(tquals: %s) ", tqual);
    }
	 

    if (symbol_table[i].type & NODE_FUNCTION_ID)
      printf("(%s function)",
             symbol_table[i].tspecifier_token == KERNEL ? "kernel" : "device");

    if (int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
      printf("(dconst)");

    if (symbol_table[i].tspecifier_token == STENCIL)
      printf("(stencil)");

    printf("\n");
  }
  printf("---\n");
}



const char*
convert_to_enum_name(const char* name)
{
	static char res[4098];
	if(!strcmp(name,"long long"))
		return "AcLongLong";
	if(strstr(name,"Ac")) return name;
	sprintf(res,"Ac%s",to_upper_case(name));
	return res;
}
string_vec
get_array_accesses(const ASTNode* base)
{
	    string_vec dst = VEC_INITIALIZER;
	    if(!base) return dst;

	    node_vec nodes = VEC_INITIALIZER;
	    get_array_access_nodes(base,&nodes);
	    for(size_t i = 0; i < nodes.size; ++i)
		push(&dst,combine_all_new(nodes.data[i]));
	    free_node_vec(&nodes);
	    return dst;
}
string_vec
get_array_var_dims(const char* var, const ASTNode* root)
{
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
		strcatprintf(dst,"%s%s",(i) ? "*" : "",tmp.data[i]);
	    free_str_vec(&tmp);
}

int default_accesses[10000] = { [0 ... 10000-1] = 1};
int read_accesses[10000] = { [0 ... 10000-1] = 0};

const  int*
get_arr_accesses(const char* datatype_scalar)
{

	char* filename;
	const char* define_name =  convert_to_define_name(strdup(datatype_scalar));
	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	asprintf(&filename,"%s_arr_accesses",define_name);
  	if(!file_exists(filename) || !has_optimization_info || !OPTIMIZE_ARRAYS)
		return default_accesses;

	FILE* fp = fopen(filename,"rb");
	int size = 1;
	fread(&size, sizeof(int), 1, fp);
	fread(read_accesses, sizeof(int), size, fp);
	fclose(fp);
	return read_accesses;
}

void
gen_array_info(FILE* fp, const char* datatype_scalar, const ASTNode* root)
{

 
  const int* accesses = get_arr_accesses(datatype_scalar);
  char datatype[1000];
  sprintf(datatype,"%s*",datatype_scalar);
  const char* define_name =  convert_to_define_name(datatype_scalar);
  {
  	char running_offset[4096];
  	  sprintf(running_offset,"0");
  	  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  {
  	    if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	        !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
  	    {
  	            fprintf(fp,"\n#ifndef %s_offset\n#define %s_offset (%s)\n#endif\n",symbol_table[i].identifier,symbol_table[i].identifier,running_offset);
  	            char array_length_str[4098];
  	            get_array_var_length(symbol_table[i].identifier,root,array_length_str);
		    strcatprintf(running_offset,"+ %s",array_length_str);
  	    }
  	  }
  	  fprintf(fp,"\n#ifndef D_%s_ARRAYS_LEN\n#define D_%s_ARRAYS_LEN (%s)\n#endif\n", strupr(define_name), strupr(define_name),running_offset);
  }
  char running_offset[4096];
  sprintf(running_offset,"0");
  int counter = 0;
  fprintf(fp, "static const array_info %s_array_info[] __attribute__((unused)) = {", convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(int_vec_contains(symbol_table[i].tqualifiers,CONST_QL, RUN_CONST)) continue;
	char array_length_str[4098];
	get_array_var_length(symbol_table[i].identifier,root,array_length_str);
	fprintf(fp,"%s","{");

        if(int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))  fprintf(fp,"true,");
        else fprintf(fp, "false,");

        if(int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
  	{
  	        fprintf(fp,"%s,",running_offset);
		strcatprintf(running_offset,"+ %s",array_length_str);
  	}
	else
	{
  		fprintf(fp,"%d,",-1);
	}

	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
      	fprintf(fp, "%lu,", dims.size);

	fprintf(fp,"%s","{{");
	for(size_t dim = 0; dim < 3; ++dim)
	{
		if(dim >= dims.size) fprintf(fp,"%s,","-1");
		else fprintf(fp,"%s,",dims.data[dim]);
	}
	fprintf(fp,"%s","},{");

	bool const_dims = true;
	for(size_t dim = 0; dim < 3; ++dim)
	{
		const bool integer_dim = (dim >= dims.size || is_number_expression(dims.data[dim]));
		const_dims &= integer_dim;
		fprintf(fp,"%s,",integer_dim ? "false" : "true");
	}

	free_str_vec(&dims);
	fprintf(fp,"%s","}},");
        fprintf(fp, "\"%s\",", symbol_table[i].identifier);
        fprintf(fp, "%s,",accesses[counter] ? "true" : "false");
	fprintf(fp,"%s","},");
	if (!accesses[counter]) push_int(&symbol_table[i].tqualifiers,DEAD);
	if (const_dims) push_int(&symbol_table[i].tqualifiers,CONST_DIMS);
	counter++;
    }
  }

  //runtime array lengths come after other arrays
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, datatype))
    {
  	if(!int_vec_contains(symbol_table[i].tqualifiers,RUN_CONST)) continue;
	fprintf(fp,"%s","{");
        fprintf(fp, "false,");
  	fprintf(fp,"%d,",-1);

	string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
      	fprintf(fp, "%lu,", dims.size);

	fprintf(fp,"%s","{{");
	for(size_t dim = 0; dim < 3; ++dim)
	{
		if(dim >= dims.size) fprintf(fp,"%s,","-1");
		else fprintf(fp,"%s,",dims.data[dim]);
	}
	fprintf(fp,"%s","},{");

	for(size_t dim = 0; dim < 3; ++dim)
		fprintf(fp,"%s,",(dim >= dims.size || is_number(dims.data[dim])) ? "false" : "true");
	fprintf(fp,"%s","}},");
	free_str_vec(&dims);
        fprintf(fp, "\"%s\",", symbol_table[i].identifier);
        fprintf(fp, "true,");
	fprintf(fp,"%s","},");
    }
  }
  //pad one extra to silence warnings
  fprintf(fp,"{false,-1,-1,{{-1,-1,-1}, {false,false,false}},\"AC_EXTRA_PADDING\",true}");
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
	fprintf(fp,"%s %s_params[NUM_%s_PARAMS+1];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
}
void

gen_gmem_array_declarations(const char* datatype_scalar, const ASTNode* root)
{
	char* define_name = convert_to_define_name(datatype_scalar);
	char* uppr_name =       strupr(define_name);
	char* upper_case_name = to_upper_case(define_name);
	const char* enum_name = convert_to_enum_name(datatype_scalar);

	char datatype[4098];
	sprintf(datatype,"%s*",datatype_scalar);

	FILE* fp = fopen("memcpy_to_gmem_arrays.h","a");
	

	fprintf(fp,"void memcpy_to_gmem_array(const %sArrayParam param,%s* &ptr)\n"
        "{\n", enum_name,datatype_scalar);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEMORY_QL) && !int_vec_contains(symbol_table[i].tqualifiers,DEAD))
	  {
		  if (!int_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS))
		  {
		  	fprintf(fp,"if (param == %s) {ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(AC_INTERNAL_gmem_%s_arrays_%s,&ptr,sizeof(ptr),0,cudaMemcpyHostToDevice)); return;} \n",symbol_table[i].identifier,define_name,symbol_table[i].identifier);
	  	  }
	  }
	fprintf(fp,"fprintf(stderr,\"FATAL AC ERROR from memcpy_to_gmem_array\\n\");\n");
	fprintf(fp,"\n(void)param;(void)ptr;}\n");



	fprintf(fp,"void memcpy_to_const_dims_gmem_array(const %sArrayParam param,const %s* ptr)\n"
	"{\n", enum_name,datatype_scalar);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEMORY_QL) && !int_vec_contains(symbol_table[i].tqualifiers,DEAD))
	  {
		  if (int_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS))
		  {
		  	fprintf(fp,"if (param == %s) {ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(AC_INTERNAL_gmem_%s_arrays_%s,ptr,sizeof(ptr[0])*get_const_dims_array_length(param),0,cudaMemcpyHostToDevice)); return;}\n",symbol_table[i].identifier,define_name,symbol_table[i].identifier);
		  }
	  }
	fprintf(fp,"fprintf(stderr,\"FATAL AC ERROR from memcpy_to_const_dims_gmem_array\\n\");\n");
	fprintf(fp,"\n(void)param;(void)ptr;}\n");


	fclose(fp);


	fp = fopen("memcpy_from_gmem_arrays.h","a");
	
	fprintf(fp,"void memcpy_from_gmem_array(const %sArrayParam param, %s* &ptr)\n"
		  "{\n", enum_name,datatype_scalar);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEMORY_QL) && !int_vec_contains(symbol_table[i].tqualifiers,DEAD))
	  {
		  if (int_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS))
		  	fprintf(fp,"if (param == %s) {ERRCHK_CUDA_ALWAYS(cudaMemcpyFromSymbol(ptr,AC_INTERNAL_gmem_%s_arrays_%s,sizeof(ptr[0])*get_const_dims_array_length(param),0,cudaMemcpyDeviceToHost)); return;}\n",symbol_table[i].identifier,define_name,symbol_table[i].identifier);
		  else
		  	fprintf(fp,"if (param == %s) {ERRCHK_CUDA_ALWAYS(cudaMemcpyFromSymbol(&ptr,AC_INTERNAL_gmem_%s_arrays_%s,sizeof(ptr),0,cudaMemcpyDeviceToHost)); return;}\n",symbol_table[i].identifier,define_name,symbol_table[i].identifier);
	  }
	fprintf(fp,"fprintf(stderr,\"FATAL AC ERROR from memcpy_from_gmem_array\\n\");\n");
	fprintf(fp,"\n(void)param;(void)ptr;}\n");
	fclose(fp);

	fp = fopen("gmem_arrays_decl.h","a");
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	{
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEMORY_QL) && !int_vec_contains(symbol_table[i].tqualifiers,DEAD))
	  {
		  if (int_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS))
		  {
			string_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
			char len[10000];
			sprintf(len,"%s","1");
			for(size_t dim = 0; dim < dims.size; ++dim)
				strcatprintf(len,"*%s",dims.data[dim]);
			fprintf(fp,"DECLARE_CONST_DIMS_GMEM_ARRAY(%s,%s,%s,%s);\n",datatype_scalar, define_name, symbol_table[i].identifier,len);
		  }
		  else
			fprintf(fp,"DECLARE_GMEM_ARRAY(%s,%s,%s);\n",datatype_scalar, define_name, symbol_table[i].identifier);
	  }
	}
	fclose(fp);
}

void
gen_array_declarations(const char* datatype_scalar, const ASTNode* root)
{

	char* define_name = convert_to_define_name(datatype_scalar);
	char* uppr_name =       strupr(define_name);
	char* upper_case_name = to_upper_case(define_name);
	const char* enum_name = convert_to_enum_name(datatype_scalar);
	fprintf_filename("array_decl.h","%s* %s_arrays[NUM_%s_ARRAYS+1];\n",datatype_scalar,define_name,uppr_name);

	fprintf_filename("get_vba_array.h","if constexpr(std::is_same<P,%sArrayParam>::value) return vba.%s_arrays[(int)param];\n",enum_name,define_name);

	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sCompParam>::value) return (%s){};\n",enum_name,datatype_scalar);

	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sCompArrayParam>::value) return (%s){};\n",enum_name,datatype_scalar);

	fprintf_filename("get_from_comp_config.h","if constexpr(std::is_same<P,%sCompArrayParam>::value) return config.%s_arrays[(int)param];\n",enum_name,define_name);

	fprintf_filename("get_from_comp_config.h","if constexpr(std::is_same<P,%sCompParam>::value) return config.%s_params[(int)param];\n",enum_name,define_name);

	FILE* fp = fopen("get_config_param.h","a");
	fprintf(fp,"if constexpr(std::is_same<P,%sParam>::value) return config.%s_params[(int)param];\n",     enum_name,define_name);
	fprintf(fp,"if constexpr(std::is_same<P,%sArrayParam>::value) return config.%s_arrays[(int)param];\n",enum_name,define_name);
	fclose(fp);

	fp = fopen("get_empty_pointer.h","a");
	fprintf(fp,"if constexpr(std::is_same<P,%sArrayParam>::value) return (%s*){};\n",enum_name,datatype_scalar);

	char datatype[4098];
	sprintf(datatype,"%s*",datatype_scalar);


	fprintf_filename("get_param_name.h","if constexpr(std::is_same<P,%sCompParam>::value) return %s_comp_param_names[(int)param];\n",enum_name,define_name);


	fp = fopen("get_num_params.h","a");
	fprintf(fp," (std::is_same<P,%sParam>::value)      ? NUM_%s_PARAMS : \n",enum_name,uppr_name);
	fprintf(fp," (std::is_same<P,%sArrayParam>::value) ? NUM_%s_ARRAYS : \n",enum_name,uppr_name);

	fprintf(fp," (std::is_same<P,%sCompParam>::value)      ? NUM_%s_COMP_PARAMS : \n",enum_name,uppr_name);
	fprintf(fp," (std::is_same<P,%sCompArrayParam>::value) ? NUM_%s_COMP_ARRAYS : \n",enum_name,uppr_name);
	fclose(fp);

	fp = fopen("get_array_info.h","a");
	fprintf(fp," if(std::is_same<P,%sArrayParam>::value) return %s_array_info[(int)array]; \n",enum_name,define_name);
	fprintf(fp," if(std::is_same<P,%sCompArrayParam>::value) return %s_array_info[(int)array + NUM_%s_ARRAYS]; \n",
	enum_name,define_name, uppr_name);
	fclose(fp);

	fp = fopen("device_set_input.h","a");
	fprintf(fp, "AcResult\nacDeviceSet%sInput(Device device, const %sInputParam param, const %s val)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_INPUT_PARAMS == 0) return AC_FAILURE;\n"
		    "\tdevice->input.%s_params[param] = val;\n"
		    "\treturn AC_SUCCESS;\n"
		    "}\n"
	,upper_case_name, enum_name, datatype_scalar, uppr_name, define_name);
	fclose(fp);

	fp = fopen("device_get_output.h","a");
	fprintf(fp, "%s\nacDeviceGet%sOutput(Device device, const %sOutputParam param)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_OUTPUTS == 0) return (%s){};\n"
		    "\treturn device->output.%s_outputs[param];\n"
		    "}\n"
	,datatype_scalar,upper_case_name, enum_name, uppr_name, datatype_scalar,define_name);
	fclose(fp);

	fp = fopen("device_get_input.h","a");
	fprintf(fp, "%s\nacDeviceGet%sInput(Device device, const %sInputParam param)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_INPUT_PARAMS == 0) return (%s){};\n"
		    "\treturn device->input.%s_params[param];\n"
		    "}\n"
	,datatype_scalar,upper_case_name, enum_name, uppr_name, datatype_scalar,define_name);
	fclose(fp);

	fp = fopen("device_set_input_decls.h","a");
	fprintf(fp,"FUNC_DEFINE(AcResult, acDeviceSet%sInput,(Device device, const %sInputParam, const %s val));\n",upper_case_name,enum_name,datatype_scalar);	
	fclose(fp);

	fp = fopen("device_get_output_decls.h","a");
	fprintf(fp,"FUNC_DEFINE(%s, acDeviceGet%sOutput,(Device device, const %sOutputParam));\n",datatype_scalar,upper_case_name,enum_name);	
	fclose(fp);

	fp = fopen("device_get_input_decls.h","a");
	fprintf(fp,"FUNC_DEFINE(%s, acDeviceGet%sInput,(Device device, const %sInputParam));\n",datatype_scalar,upper_case_name,enum_name);	
	fclose(fp);

	fp = fopen("device_set_input_overloads.h","a");
	fprintf(fp,"#ifdef __cplusplus\nstatic inline AcResult acDeviceSetInput(Device device, const %sInputParam& param, const %s& val){ return acDeviceSet%sInput(device,param,val); }\n#endif\n",enum_name, datatype_scalar, upper_case_name);	
	fclose(fp);

	fp = fopen("device_get_output_overloads.h","a");
	fprintf(fp,"#ifdef __cplusplus\nstatic inline %s acDeviceGetOutput(Device device, const %sOutputParam& param){ return acDeviceGet%sOutput(device,param); }\n#endif\n",datatype_scalar,enum_name, upper_case_name);	
	fclose(fp);

	fp = fopen("device_get_input_overloads.h","a");
	fprintf(fp,"#ifdef __cplusplus\nstatic inline %s acDeviceGetInput(Device device, const %sInputParam& param){ return acDeviceGet%sInput(device,param); }\n#endif\n",datatype_scalar,enum_name, upper_case_name);	
	fclose(fp);
	
	char* func_name;
	fp = fopen("device_set_input_loads.h","a");
	asprintf(&func_name,"acDeviceSet%sInput",upper_case_name);
	gen_dlsym(fp,func_name);
	fclose(fp);

	fp = fopen("device_get_input_loads.h","a");
	asprintf(&func_name,"acDeviceGet%sInput",upper_case_name);
	gen_dlsym(fp,func_name);
	fclose(fp);

	fp = fopen("device_get_output_loads.h","a");
	asprintf(&func_name,"acDeviceGet%sOutput",upper_case_name);
	gen_dlsym(fp,func_name);
	fclose(fp);

	free(func_name);

	fp = fopen("dconst_decl.h","a");
	fprintf(fp,"%s __device__ __forceinline__ DCONST(const %sParam& param){return d_mesh_info.%s_params[(int)param];}\n"
			,datatype_scalar, enum_name, define_name);
	fclose(fp);

	fp = fopen("rconst_decl.h","a");
	fprintf(fp,"%s __device__ __forceinline__ RCONST(const %sCompParam& param){return d_mesh_info.%s_params[0];}\n"
			,datatype_scalar, enum_name, define_name);
	fclose(fp);

	fp = fopen("get_address.h","a");
	fprintf(fp,"size_t  get_address(const %sParam& param){ return (size_t)&d_mesh_info.%s_params[(int)param];}\n"
			,enum_name, define_name);
	fclose(fp);
	fp = fopen("load_dconst_arrays.h","a");
	fprintf(fp,"cudaError_t\n"
		   "load_array(const %s* values, const size_t bytes, const %sArrayParam arr)\n"
		    "{\n",
		     datatype_scalar, enum_name);

		     

  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	{
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
		  fprintf(fp,"if (arr == %s)\n return cudaMemcpyToSymbol(AC_INTERNAL_d_%s_arrays_%s,values,bytes,0,cudaMemcpyHostToDevice);\n",symbol_table[i].identifier,define_name, symbol_table[i].identifier);
  	}
	fprintf(fp,"(void)values;(void)bytes;(void)arr;\nreturn cudaSuccess;\n}\n");

	fclose(fp);
	fp = fopen("store_dconst_arrays.h","a");

		     

	fprintf(fp,"cudaError_t\n"
		  "store_array(%s* values, const size_t bytes, const %sArrayParam arr)\n"
		    "{\n",
	datatype_scalar, enum_name);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	{
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
	  {
		  fprintf(fp,"if (arr == %s)\n return cudaMemcpyFromSymbol(values,AC_INTERNAL_d_%s_arrays_%s,bytes,0,cudaMemcpyDeviceToHost);\n",symbol_table[i].identifier,define_name, symbol_table[i].identifier);
	  }
  	}
	fprintf(fp,"(void)values;(void)bytes;(void)arr;\nreturn cudaSuccess;\n}\n");

	fclose(fp);





	fp = fopen("load_and_store_uniform_overloads.h","a");
	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acLoadUniform(const cudaStream_t stream, const %sParam param, const %s value) { return acLoad%sUniform(stream,param,value);}\n"
		,enum_name, datatype_scalar, upper_case_name);

	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acLoadUniform(const cudaStream_t stream, const %sArrayParam param, const %s* values, const size_t length) { return acLoad%sArrayUniform(stream,param,values,length);}\n"
		,enum_name, datatype_scalar, upper_case_name);

	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acStoreUniform(const cudaStream_t stream, const %sParam param, %s* value) { return acStore%sUniform(stream,param,value);}\n"
		,enum_name, datatype_scalar, upper_case_name);

	fprintf(fp,"static AcResult __attribute ((unused))"
	        "acStoreUniform(const %sArrayParam param, %s* values, const size_t length) { return acStore%sArrayUniform(param,values,length);}\n"
		,enum_name, datatype_scalar, upper_case_name);
	fclose(fp);

	fp = fopen("load_and_store_uniform_funcs.h","a");
	fprintf(fp, "AcResult acLoad%sUniform(const cudaStream_t, const %sParam param, const %s value) { return acLoadUniform(param,value); }\n",
			upper_case_name, enum_name, datatype_scalar);
	fprintf(fp, "AcResult acLoad%sArrayUniform(const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length) { return acLoadArrayUniform(param ,values, length); }\n",
			upper_case_name, enum_name, datatype_scalar);
	fprintf(fp, "AcResult acStore%sUniform(const cudaStream_t, const %sParam param, %s* value) { return acStoreUniform(param,value); }\n",
			upper_case_name, enum_name, datatype_scalar);
	fprintf(fp, "AcResult acStore%sArrayUniform(const %sArrayParam param, %s* values, const size_t length) { return acStoreArrayUniform(param ,values, length); }\n",
			upper_case_name, enum_name, datatype_scalar);

	fclose(fp);

	//char* define_name = convert_to_define_name(datatype_scalar);
	//char* uppr_name = strupr(define_name);
	//char* enum_name = convert_to_enum_name(datatype_scalar);

	fp = fopen("load_and_store_uniform_header.h","a");
	fprintf(fp, "FUNC_DEFINE(AcResult, acLoad%sUniform,(const cudaStream_t, const %sParam param, const %s value));\n",
			upper_case_name, enum_name, datatype_scalar);

	fprintf(fp, "FUNC_DEFINE(AcResult, acLoad%sArrayUniform, (const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length));\n",
			upper_case_name, enum_name, datatype_scalar);

	fprintf(fp, "FUNC_DEFINE(AcResult, acStore%sUniform,(const cudaStream_t, const %sParam param, %s* value));\n",
			upper_case_name, enum_name, datatype_scalar);

	fprintf(fp, "FUNC_DEFINE(AcResult, acStore%sArrayUniform, (const %sArrayParam param, %s* values, const size_t length));\n",
			upper_case_name, enum_name, datatype_scalar);
	fclose(fp);

	//we pad with 1 since zero sized arrays are not allowed with some CUDA compilers
	fp = fopen("dconst_arrays_decl.h","a");
	//fprintf(fp,"__device__ __constant__  %s d_%s_arrays[D_%s_ARRAYS_LEN+1];\n",datatype_scalar, define_name, uppr_name);
  	{
  		  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  		  {
  		    if (symbol_table[i].type & NODE_VARIABLE_ID &&
  		        !strcmp(symbol_table[i].tspecifier,datatype) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
  		    {
  		            char array_length_str[4098];
  		            get_array_var_length(symbol_table[i].identifier,root,array_length_str);
			    fprintf(fp,"__device__ __constant__ %s AC_INTERNAL_d_%s_arrays_%s[%s];\n",datatype_scalar, define_name,symbol_table[i].identifier,array_length_str);
  		    }
  		  }
  	}
	fclose(fp);


	fp = fopen("gmem_arrays_accessed_decl.h","a");
	fprintf(fp,"int gmem_%s_arrays_accessed[NUM_%s_ARRAYS]{};\n",define_name,uppr_name);
	fclose(fp);

	fp = fopen("gmem_arrays_output_accesses.h","a");

	fprintf(fp,"{\nFILE* fp_arr_accesses = fopen(\"%s_arr_accesses\", \"wb\");",define_name);
	fprintf(fp,"int tmp = NUM_%s_ARRAYS; fwrite(&tmp,sizeof(int),1,fp_arr_accesses); fwrite(gmem_%s_arrays_accessed,sizeof(int),NUM_%s_ARRAYS,fp_arr_accesses);",uppr_name,define_name,uppr_name);
	fprintf(fp,"fclose(fp_arr_accesses);\n}\n");
	fclose(fp);


	fp = fopen("push_to_config.h","a");
	fprintf(fp,"constexpr static void acPushToConfig(AcMeshInfo& config, %sParam param, %s val)      {config.%s_params[(int)param] = val;}",enum_name,datatype_scalar, define_name);
	fprintf(fp,"constexpr static void acPushToConfig(AcMeshInfo& config, %sArrayParam param, %s* val) {config.%s_arrays[(int)param] = val;}",enum_name,datatype_scalar,define_name);
	fclose(fp);

	fp = fopen("load_comp_info.h","a");
	fprintf(fp,"static AcResult __attribute((unused)) acLoad%sCompInfo(const %sCompParam param, const %s val, AcCompInfo* info)      {\n"
			"info->is_loaded.%s_params[(int)param] = true;\n"
			"info->config.%s_params[(int)param] = val;\n"
			"return AC_SUCCESS;\n"
			"}\n",upper_case_name ,enum_name,datatype_scalar,define_name,define_name);

	fprintf(fp,"static AcResult __attribute((unused)) acLoad%sArrayCompInfo(const %sCompArrayParam param, const %s* val, AcCompInfo* info)      {\n"
			"info->is_loaded.%s_arrays[(int)param] = true;\n"
			"info->config.%s_arrays[(int)param] = val;\n"
			"return AC_SUCCESS;\n"
			"}\n",upper_case_name ,enum_name,datatype_scalar,define_name,define_name);
	fclose(fp);
	fp = fopen("load_comp_info_overloads.h","a");
	fprintf(fp,"GEN_LOAD_COMP_INFO(%sCompParam,%s,%s)\n", enum_name, datatype_scalar, upper_case_name);
	fprintf(fp,"GEN_LOAD_COMP_INFO(%sCompArrayParam,%s*,%sArray)\n", enum_name, datatype_scalar, upper_case_name);
	fclose(fp);


	fp = fopen("is_comptime_param.h","a");
	fprintf(fp,"constexpr static bool IsCompParam(%sParam& param) {(void)param; return false;}\n",enum_name);
	fprintf(fp,"constexpr static bool IsCompParam(%sArrayParam& param) {(void)param; return false;}\n",enum_name);
	fprintf(fp,"constexpr static bool IsCompParam(%sCompArrayParam& param) {(void)param; return true;}\n",enum_name);
	fprintf(fp,"constexpr static bool IsCompParam(%sCompParam& param) {(void)param; return true;}\n",enum_name);
	fclose(fp);
	if(strcmp(datatype_scalar,"int"))
	{
		fp = fopen("scalar_types.h","a");
		fprintf(fp,"%sParam,\n",enum_name);
		fclose(fp);

		fp = fopen("scalar_comp_types.h","a");
		fprintf(fp,"%sCompParam,\n",enum_name);
		fclose(fp);

		fp = fopen("array_types.h","a");
		fprintf(fp,"%sArrayParam,\n",enum_name);
		fclose(fp);

		fp = fopen("array_comp_types.h","a");
		fprintf(fp,"%sCompArrayParam,\n",enum_name);
		fclose(fp);
	}
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

	fopen("input_decl.h","a");
	fprintf(fp,"%s %s_params[NUM_%s_COMP_PARAMS+1];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);

	fp = fopen("output_decl.h","a");
	fprintf(fp,"%s %s_outputs[NUM_%s_OUTPUTS+1];\n",datatype_scalar,convert_to_define_name(datatype_scalar),strupr(convert_to_define_name(datatype_scalar)));
	fclose(fp);
}


void
gen_enums(FILE* fp, const char* datatype_scalar)
{
  char datatype_arr[1000];
  sprintf(datatype_arr,"%s*",datatype_scalar);

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_PARAMS} %sParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL,GLOBAL_MEMORY_QL))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_ARRAYS} %sArrayParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,OUTPUT))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_OUTPUTS} %sOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,INPUT))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_INPUT_PARAMS} %sInputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && int_vec_contains(symbol_table[i].tqualifiers,OUTPUT))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_OUTPUT_ARRAYS} %sArrayOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,RUN_CONST))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_COMP_PARAMS} %sCompParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && int_vec_contains(symbol_table[i].tqualifiers,RUN_CONST))
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
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,OUTPUT))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_array_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_arr) && int_vec_contains(symbol_table[i].tqualifiers,OUTPUT))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_comp_param_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (!strcmp(symbol_table[i].tspecifier, datatype_scalar) && int_vec_contains(symbol_table[i].tqualifiers,RUN_CONST))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");


}

bool
check_symbol(const NodeType type, const char* name, const int tspecifier, const int tqualifier)
{
  const Symbol* sym = get_symbol_token(type,name,tspecifier);
  return 
	  !sym ? false :
	  !tqualifier ? true :
	  int_vec_contains(sym->tqualifiers,tqualifier);
}

bool
check_symbol_index(const NodeType type, const int index, const int tspecifier, const int tqualifier)
{
  const Symbol* sym = get_symbol_by_index(type,index,tspecifier);
  return 
	  !sym ? false :
	  !tqualifier ? true :
	  int_vec_contains(sym->tqualifiers,tqualifier);
}




const char*
get_expr_type(ASTNode* node);

static int int_log2(int x)
{

	int res = 0;
	while (x >>= 1) ++res;
	return res;
}
static ASTNode*
create_identifier_node(const char* identifier)
{	
	ASTNode* identifier_node  = astnode_create(NODE_UNKNOWN, NULL, NULL);  
	if(is_number(identifier))
		identifier_node->token    = NUMBER;
	else
		identifier_node->token    = IDENTIFIER;
	identifier_node->buffer   = strdup(identifier);
	return identifier_node;
}
static ASTNode*
create_primary_expression(const char* identifier)
{
	return astnode_create(NODE_PRIMARY_EXPRESSION,create_identifier_node(identifier),NULL);
}

ASTNode*
get_index_node(const ASTNode* array_access_start, const string_vec var_dims_in)
{
    	node_vec array_accesses = VEC_INITIALIZER;
	get_array_access_nodes(array_access_start,&array_accesses);
	
    	if(array_accesses.size != var_dims_in.size)
    	{
    	        return NULL;
    	}
	string_vec var_dims = VEC_INITIALIZER;
	if(!AC_ROW_MAJOR_ORDER)
		var_dims = str_vec_copy(var_dims_in);
	else
		for(int i = var_dims_in.size - 1; i >= 0; --i)
			push(&var_dims,var_dims_in.data[i]);

	node_vec new_accesses = VEC_INITIALIZER;
	if(!AC_ROW_MAJOR_ORDER)
    		for(size_t j = 0; j < array_accesses.size; ++j)
		{
			ASTNode* prefix_node = astnode_create(NODE_UNKNOWN,NULL,NULL);
			push_node(&new_accesses,astnode_create(NODE_UNKNOWN,prefix_node,(ASTNode*)array_accesses.data[j]));
		}
	else
    		for(int j = array_accesses.size-1; j >= 0; --j)
		{
			ASTNode* prefix_node = astnode_create(NODE_UNKNOWN,NULL,NULL);
			push_node(&new_accesses,astnode_create(NODE_UNKNOWN,prefix_node,(ASTNode*)array_accesses.data[j]));
		}
    	for(size_t j = 0; j < array_accesses.size; ++j)
    	{
		ASTNode* node = (ASTNode*) new_accesses.data[j];
    		if(j)
    		{
			ASTNode* prefix_node = node->lhs;
    			asprintf(&prefix_node->buffer,"%s","");
			prefix_node->prefix = strdup("+(");
			node_vec dim_nodes = VEC_INITIALIZER;
    			for(size_t k = 0; k < j; ++k)
				push_node(&dim_nodes,create_primary_expression(var_dims.data[k]));
			ASTNode* dims_node = build_list_node(dim_nodes,"*");
			prefix_node->lhs = dims_node;
			dims_node->parent = dims_node;
			prefix_node->postfix = strdup(")*");
    		}
	    	node->rhs->prefix = strdup("(");
	    	node->rhs->postfix = strdup(")");
    	}
	ASTNode* res = build_list_node(new_accesses,"");
    	free_node_vec(&array_accesses);
    	free_node_vec(&new_accesses);
	return res;

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
  char* datatype;
  asprintf(&datatype,"%s*",datatype_scalar);
  const int l_current_nest = 0;
  for (size_t i = 0; i < num_symbols[l_current_nest]; ++i)
  {
    if (!(symbol_table[i].type & NODE_VARIABLE_ID &&
    (!strcmp(symbol_table[i].tspecifier,datatype)) && !strcmp(array_name,symbol_table[i].identifier) && !int_vec_contains(symbol_table[i].tqualifiers,CONST_QL)))
	    continue;

    if(get_parent_node(NODE_ARRAY_ACCESS,node)) return;
    //TP: replace dead arr accesses with default initializer values
    //TP: Might not be needed after we have a conditional removal but for now need to care of dead reads to keep compiling the code
    if(int_vec_contains(symbol_table[i].tqualifiers,DEAD))
    {
	    node = node->parent;
	    node->lhs = NULL;
	    node->rhs = NULL;
	    asprintf(&node->buffer,"(%s){}",datatype_scalar);
	    return;
    }
    string_vec var_dims = get_array_var_dims(array_name, root);
	
    ASTNode* elem_index         = get_index_node(node,var_dims);
    if(!elem_index)
    {
	    fprintf(stderr,FATAL_ERROR_MESSAGE"Incorrect array access: %s\n",combine_all_new(node));
	    exit(EXIT_FAILURE);
    }
    ASTNode* base = node;
    base->lhs=NULL;
    base->rhs=NULL;
    base->prefix=NULL;
    base->postfix=NULL;
    base->infix=NULL;
    char* identifier;
    if(int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
	    asprintf(&identifier,"AC_INTERNAL_d_%s_arrays_%s",convert_to_define_name(datatype_scalar), symbol_table[i].identifier);
    else
	    asprintf(&identifier,"AC_INTERNAL_gmem_%s_arrays_%s",convert_to_define_name(datatype_scalar), symbol_table[i].identifier);
    ASTNode* identifier_node = create_primary_expression(identifier);
    free(identifier);
    identifier_node->parent = base;
    base->lhs =  identifier_node;
    ASTNode* pointer_access = astnode_create(NODE_UNKNOWN,NULL,NULL);
    ASTNode* elem_access_offset = astnode_create(NODE_UNKNOWN,NULL,NULL);
    ASTNode* elem_access        = astnode_create(NODE_UNKNOWN,elem_access_offset,elem_index); 
    ASTNode* access_node        = astnode_create(NODE_UNKNOWN,pointer_access,elem_access);
    base->rhs = access_node;
    access_node ->parent = base;

    elem_access->prefix  = strdup("[");
    elem_access->postfix = strdup("]");
    if(int_vec_contains(symbol_table[i].tqualifiers,DCONST_QL))
    {
        //asprintf(&elem_access_offset->buffer,"%s_offset",array_name);
        //elem_access -> infix = strdup("+");
        //elem_index -> prefix  = strdup("(");
        //elem_index -> postfix = strdup(")");
    }
    else if(int_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEMORY_QL))
    {
	
	//asprintf(&pointer_access->buffer,"[(int)%s]",array_name);
    	if(!int_vec_contains(symbol_table[i].tqualifiers,DYNAMIC_QL))
	{
		asprintf(&base->prefix,"%s%s",
				 is_primitive_datatype(datatype_scalar) ? "__ldg(": "",
				 is_primitive_datatype(datatype_scalar) ? "&": ""
		       );
		if(is_primitive_datatype(datatype_scalar))
			base->postfix = strdup(")");

	}
    }
    else
    {
	    fprintf(stderr,"Fatal error: no case for array read\n");
	    exit(EXIT_FAILURE);
    }
    free_str_vec(&var_dims);
  }
  free(datatype);
}


bool
is_user_enum_option(const char* identifier)
{
	for(size_t i = 0; i < e_info.names.size; ++i)
		if(str_vec_contains(e_info.options[i],identifier)) return true;
	return false;

}
const char*
get_enum(const char* option)
{
	for(size_t i = 0; i < e_info.names.size; ++i)
		if(str_vec_contains(e_info.options[i],option)) return e_info.names.data[i];
	return false;
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

structs_info 
get_structs_info()
{
	string_vec* names   = malloc(sizeof(string_vec)*100);
	string_vec* types   = malloc(sizeof(string_vec)*100);
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
	const char* struct_name = get_node_by_token(IDENTIFIER,node)->buffer;
	const int struct_index = push(&(params->user_structs),struct_name);
	node_vec field_nodes = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < field_nodes.size; ++i)
		process_declaration(field_nodes.data[i], struct_index,params);
	free_node_vec(&field_nodes);
}
structs_info
read_user_structs(const ASTNode* root)
{
	structs_info res = get_structs_info();
	read_user_structs_recursive(root, &res);
	return res;
}
void
mark_as_input(ASTNode* kernel_root, const ASTNode* param)
{
	const char* param_name = param->rhs->buffer;
	add_node_type(NODE_INPUT, kernel_root,param_name);
}

void
process_param_codegen(const ASTNode* param, FILE* stream)
{
	const char* param_type = get_node(NODE_TSPEC,param->lhs)->lhs->buffer;
	const char* param_name = param->rhs->buffer;
	fprintf(stream,"%s %s;", param_type, param_name);
}
void
mark_kernel_inputs(const ASTNode* node)
{
	if(node->lhs)
		mark_kernel_inputs(node->lhs);
	if(node->rhs)
		mark_kernel_inputs(node->rhs);
	if(!(node->type & NODE_KFUNCTION))
		return;
	if(!node->rhs->lhs)
		return;
        ASTNode* param_list_head = node->rhs->lhs;
	node_vec params = get_nodes_in_list(param_list_head);
        ASTNode* compound_statement = node->rhs->rhs;
	for(size_t i = 0; i < params.size; ++i)
		mark_as_input(compound_statement,params.data[i]);
	free_node_vec(&params);
}

void
gen_kernel_structs_recursive(const ASTNode* node, FILE* stream)
{
	if(node->lhs)
		gen_kernel_structs_recursive(node->lhs,stream);
	if(node->rhs)
		gen_kernel_structs_recursive(node->rhs,stream);
	if(!(node->type & NODE_KFUNCTION))
		return;
	if(!node->rhs->lhs)
		return;

        ASTNode* param_list_head = node->rhs->lhs;
        ASTNode* compound_statement = node->rhs->rhs;
        param_list_head->type |= NODE_NO_OUT;

        const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, node->lhs);
	FILE* fp = fopen("user_input_typedefs.h","a");
	fprintf(fp,"typedef struct %sInputParams {", fn_identifier->buffer);

	node_vec params = get_nodes_in_list(param_list_head);
	for(size_t i = 0; i < params.size; ++i)
		process_param_codegen(params.data[i],fp);
	free_node_vec(&params);

	fprintf(fp,"} %sInputParams;\n",fn_identifier->buffer);
	fclose(fp);

	fprintf(stream,"%sInputParams %s;\n", fn_identifier->buffer,fn_identifier->buffer);

}
void
gen_kernel_structs(const ASTNode* root)
{
	FILE* fp_structs = fopen("user_input_typedefs.h","a");
	fprintf(fp_structs,"typedef union acKernelInputParams {\n\n");
	gen_kernel_structs_recursive(root,fp_structs);
	fprintf(fp_structs,"} acKernelInputParams;\n\n");
	fclose(fp_structs);
}

static void
create_binary_op(const structs_info info, const int i, const char* op, FILE* fp)
{
		const char* struct_name = info.user_structs.data[i];
		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator%s(const %s& a, const %s& b)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,op,struct_name,struct_name,struct_name);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\ta.%s %s b.%s,\n",info.user_struct_field_names[i].data[j],op,info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");

		fprintf(fp,"static HOST_DEVICE_INLINE void\n"
			   "operator%s=(%s& a, const %s& b)\n"
			   "{\n"
			,op,struct_name,struct_name);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\ta.%s %s= b.%s;\n",info.user_struct_field_names[i].data[j],op,info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\n}\n");
}
static void
create_unary_op(const structs_info info, const int i, const char* op, FILE* fp)
{
		const char* struct_name = info.user_structs.data[i];
		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator%s(const %s& a)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,op,struct_name,struct_name);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\t-a.%s,\n",info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");
}

void
gen_user_enums()
{
  user_enums_info enum_info = e_info;
  for (size_t i = 0; i < enum_info.names.size; ++i)
  {
	  char res[7000];
	  sprintf(res,"%s {\n","typedef enum");
	  for(size_t j = 0; j < enum_info.options[i].size; ++j)
	  {
		  const char* separator = (j < enum_info.options[i].size - 1) ? ",\n" : "";
		  strcatprintf(res,"%s%s",enum_info.options[i].data[j],separator);
	  }
	  strcatprintf(res,"} %s;\n",enum_info.names.data[i]);
  	  file_append("user_typedefs.h",res);

	  sprintf(res,"std::string to_str(const %s value)\n"
		       "{\n"
		       "switch(value)\n"
		       "{\n"
		       ,enum_info.names.data[i]);

	  for(size_t j = 0; j < enum_info.options[i].size; ++j)
		  strcatprintf(res,"case %s: return \"%s\";\n",enum_info.options[i].data[j],enum_info.options[i].data[j]);
	  strcat(res,"}return \"\";\n}\n");
	  file_append("to_str_funcs.h",res);

	  sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n",enum_info.names.data[i], enum_info.names.data[i]);
	  file_append("to_str_funcs.h",res);
  }
}
void
gen_user_structs()
{
	for(size_t i = 0; i < s_info.user_structs.size; ++i)
	{
		const char* struct_name = s_info.user_structs.data[i];
		FILE* struct_def = fopen("user_typedefs.h","a");
		//TP: we use the struct coming from HIP/CUDA
		if(strcmp(struct_name,"int3"))
		{
			fprintf(struct_def,"typedef struct %s {",struct_name);
			for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
			{
				const char* type = s_info.user_struct_field_types[i].data[j];
				const char* name = s_info.user_struct_field_names[i].data[j];
				fprintf(struct_def, "%s %s;", type, name);
			}
			fprintf(struct_def, "} %s;\n", s_info.user_structs.data[i]);
        		fclose(struct_def);
		}

		bool all_reals = true;
		bool all_scalar_types = true;
		for(size_t j = 0; j < s_info.user_struct_field_types[i].size; ++j)
		{
			all_reals        &= !strcmp( s_info.user_struct_field_types[i].data[j],"AcReal");
			all_scalar_types &= !strcmps(s_info.user_struct_field_types[i].data[j],"AcReal","int");
		}
		if(!all_scalar_types) continue;
		FILE* fp = fopen("user_typedefs.h","a");
		fprintf(fp,"#ifdef __cplusplus\n");



		if(strcmp(struct_name,"int3"))
		{
			create_binary_op(s_info,i,"-",fp);
			create_binary_op(s_info,i,"+",fp);
			create_unary_op (s_info,i,"-",fp);
			create_unary_op (s_info,i,"+",fp);
		}

		if(!strcmp(struct_name,"AcComplex")) 
		{
			fprintf(fp,"#endif\n");
			continue;
		}
		
		if(!all_reals)  fprintf(fp,"#endif\n");
		if(!all_reals)  continue;
		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator/(const %s& a, const AcReal& b)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,struct_name,struct_name);
		for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\ta.%s/b,\n",s_info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");

		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator*(const AcReal& a, const %s& b)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,struct_name,struct_name);
		for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\ta*b.%s,\n",s_info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");
		//TP: originally had the idea that scalar* struct would only be legal i.e. struct*scalar would not be
		//But with hindsight seems to be too pedantic and doesn't really give help that much
		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator*(const %s& a, const AcReal& b)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,struct_name,struct_name);
		for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\tb*a.%s,\n",s_info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");


		fprintf(fp,"#endif\n");

				   
		fclose(fp);
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
	node_vec params = get_nodes_in_list(node->rhs->lhs);
	for(size_t i = 0; i < params.size; ++i)
	{
	  	const ASTNode* param = params.data[i];
	  	push(types_dst,combine_buffers_new(param->lhs));
	  	push(names_dst,param->rhs->buffer);
	}
	free_node_vec(&params);
}

func_params_info
get_function_param_types_and_names(const ASTNode* node, const char* func_name)
{
	func_params_info res = FUNC_PARAMS_INITIALIZER;
	get_function_param_types_and_names_recursive(node,func_name,&res.types,&res.expr);
	return res;
}




func_params_info
get_func_call_params_info(const ASTNode* func_call);
void gen_loader(const ASTNode* func_call, const ASTNode* root, const char* prefix)
{
		const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
		const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,0,0);
		if(!strcmp(func_name,"periodic"))
			return;
		ASTNode* param_list_head = func_call->rhs;
		if(!param_list_head)
			return;
		func_params_info call_info  = get_func_call_params_info(func_call);
		bool is_boundcond = false;
		for(size_t i = 0; i< call_info.expr.size; ++i)
			is_boundcond |= (strstr(call_info.expr.data[i],"BOUNDARY_") != NULL);

		func_params_info params_info =  get_function_param_types_and_names(root,func_name);

		char loader_str[10000];
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

			char* input_param = strdup(call_info.expr.data[i]);
			replace_substring(&input_param,"AC_ITERATION_NUMBER","p.step_number");
			if (is_number(call_info.expr.data[i]) || is_real(call_info.expr.data[i]) || !strcmp(input_param,"p.step_number"))
				strcatprintf(loader_str, "p.params -> %s.%s = %s;\n", func_name, params_info.expr.data[i], input_param);
			else
				strcatprintf(loader_str, "p.params -> %s.%s = acDeviceGetInput(acGridGetDevice(),%s); \n", func_name,params_info.expr.data[i], input_param);
			free(input_param);
			//TP: not used anymore TODO: remove
			//if(!str_vec_contains(*input_symbols,call_info.expr.data[i]))
			//{
			//	if(!is_number(call_info.expr.data[i]) && !is_real(call_info.expr.data[i]))
			//	{
			//		push(input_symbols,call_info.expr.data[i]);
			//		push(input_types,params_info.types.data[i]);
			//	}
			//}
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
		file_append("user_loaders.h",loader_str);

		free_func_params_info(&params_info);
		free_func_params_info(&call_info);
}

static int_vec field_remappings = VEC_INITIALIZER;

const char*
get_field_name(const int field)
{
	const int correct_field_index = int_vec_get_index(field_remappings,field);
	return get_symbol_by_index(NODE_VARIABLE_ID,field,FIELD)->identifier;
}
	 
void
gen_taskgraph_kernel_entry(const ASTNode* kernel_call, const ASTNode* root, char* res, const char* taskgraph_name)
{
	assert(kernel_call);
	const char* func_name = get_node_by_token(IDENTIFIER,kernel_call)->buffer;
	char* fields_in_str  = malloc(sizeof(char)*4000);
	char* fields_out_str = malloc(sizeof(char)*4000);
	char* communicated_fields_before = malloc(sizeof(char)*4000);
	char* communicated_fields_after = malloc(sizeof(char)*4000);
	sprintf(fields_in_str, "%s", "{");
	sprintf(fields_out_str, "%s", "{");
	sprintf(communicated_fields_before, "%s", "{");
	sprintf(communicated_fields_after, "%s", "{");
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,func_name,KERNEL);
	if(kernel_index == -1)
	{
		fprintf(stderr,FATAL_ERROR_MESSAGE"Undeclared kernel %s used in ComputeSteps %s\n",func_name,taskgraph_name);
		exit(EXIT_FAILURE);
	}
	char* all_fields = malloc(sizeof(char)*4000);
	all_fields[0] = '\0';
	for(size_t field = 0; field < num_fields; ++field)
	{
		const bool field_in  = (read_fields[field + num_fields*kernel_index] || field_has_stencil_op[field + num_fields*kernel_index]);
		const bool field_out = (written_fields[field + num_fields*kernel_index]);
		const char* field_str = get_field_name(field);
		strcatprintf(all_fields,"%s,",field_str);
		if(field_in)
			strcatprintf(fields_in_str,"%s,",field_str);
		if(field_out)
			strcatprintf(fields_out_str,"%s,",field_str);
	}
	strcat(fields_in_str,  "}");
	strcat(fields_out_str, "}");
	if(kernel_call->rhs)
		strcatprintf(res,"\tacCompute(KERNEL_%s,%s,%s,%s_%s_loader),\n",func_name,fields_in_str,fields_out_str,taskgraph_name,func_name);
	else
		strcatprintf(res,"\tacCompute(KERNEL_%s,%s,%s),\n",func_name,fields_in_str,fields_out_str);
	gen_loader(kernel_call,root,taskgraph_name);
	free(all_fields);
	free(fields_in_str);
	free(fields_out_str);
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
	if(check_symbol(NODE_FUNCTION_ID,func_name,0,0))
		push_int(&calls,get_symbol_index(NODE_FUNCTION_ID,func_name,KERNEL));
	while(--n)
	{
		function_call_list_head = function_call_list_head->parent;
		function_call = function_call_list_head->rhs;
		func_name = get_node_by_token(IDENTIFIER,function_call)->buffer;
		if(check_symbol(NODE_FUNCTION_ID,func_name,KERNEL,0))
			push_int(&calls,get_symbol_index(NODE_FUNCTION_ID,func_name,KERNEL));
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
		fprintf(stderr,FATAL_ERROR_MESSAGE"incorrect boundary specification: %s\n",boundary_in);
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
typedef struct
{
	bool* in;
	bool* out;
}
bc_fields;

bc_fields 
get_fields_included(const ASTNode* func_call, const char* boundconds_name)
{
	func_params_info call_info = get_func_call_params_info(func_call);
	char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	bc_fields res;
	char* full_name = malloc((strlen(boundconds_name) + strlen(func_name) + 500)*sizeof(char));
	sprintf(full_name,"%s__%s",boundconds_name,func_name);
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID, full_name, KERNEL);
	res.in  = (bool*)malloc(sizeof(bool)*num_fields);
	res.out = (bool*)malloc(sizeof(bool)*num_fields);
	memset(res.in,0,sizeof(bool)*num_fields);
	memset(res.out,0,sizeof(bool)*num_fields);
	if(kernel_index >= 0)
		for(size_t field = 0; field < num_fields; ++field)
		{
			res.out[field] |= written_fields[field + num_fields*kernel_index];
			res.in[field]  |= read_fields[field + num_fields*kernel_index] || field_has_stencil_op[field + num_fields*kernel_index];
		}
	//if rest_fields include all 
	//at the moment not supported
	const bool all_included = str_vec_contains(call_info.expr,"REST_FIELDS") || !strcmp(func_name,"periodic");
	if (str_vec_contains(call_info.expr,"REST_FIELDS"))
	{
		fprintf(stderr,FATAL_ERROR_MESSAGE"REST_FIELDS not supported right now\n");
		exit(EXIT_FAILURE);
	}
	for(size_t field = 0; field < num_fields; ++field)
	{
		res.out[field] |= all_included;
		res.in[field]  |= all_included;
	}
	free(full_name);
	free_func_params_info(&call_info);
	return res;
}
const char*
boundary_str(const int bc)
{
       return
               bc == 0 ? "BOUNDARY_X_BOT" :
               bc == 1 ? "BOUNDARY_Y_BOT" :
               bc == 2 ? "BOUNDARY_X_TOP" :
               bc == 3 ? "BOUNDARY_Y_TOP" :
               bc == 4 ? "BOUNDARY_Z_BOT" :
               bc == 5 ? "BOUNDARY_Z_TOP" :
               NULL;
}

const int boundaries[] = {BOUNDARY_X_BOT, BOUNDARY_Y_BOT,BOUNDARY_X_TOP,BOUNDARY_Y_TOP,BOUNDARY_Z_BOT, BOUNDARY_Z_TOP};

void
process_boundcond(const ASTNode* func_call, char** res, const ASTNode* root, const char* boundconds_name)
{
	char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);
	const int num_boundaries = TWO_D ? 4 : 6;


	bc_fields fields  = get_fields_included(func_call,boundconds_name);
	const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,0,0);
	char* prefix = malloc(sizeof(char)*4000);
	char* full_name = malloc(sizeof(char)*4000);
	for(int bc = 0;  bc < num_boundaries; ++bc) 
	{
		if(boundary_int & boundaries[bc])
		{
			if(is_dfunc) sprintf(prefix,"%s_AC_KERNEL_",boundconds_name);
			else sprintf(prefix,"%s_",boundconds_name);
			if(bc == 0)  strcat(prefix,"X_BOT");
			if(bc == 1)  strcat(prefix,"Y_BOT");
			if(bc == 2)  strcat(prefix,"X_TOP");
			if(bc == 3)  strcat(prefix,"Y_TOP");
			if(bc == 4)  strcat(prefix,"Z_BOT");
			if(bc == 5)  strcat(prefix,"Z_TOP");
			if(is_dfunc) sprintf(full_name,"%s_%s",prefix,func_name);
			else         sprintf(full_name,"%s",func_name);
			for(size_t field = 0; field < num_fields; ++field)
	     	     		res[field + num_fields*bc] = (fields.out[field]) ? strdup(full_name) : res[field + num_fields*bc];
			if(!strcmp(func_name,"periodic"))
				gen_loader(func_call,root,prefix);
		}
	}
	free(fields.in);
	free(fields.out);
	free(prefix);
	free(full_name);
}
void
remove_ending_symbols(char* str, const char symbol)
{
	int len = strlen(str);
	while(str[--len] == symbol) str[len] = '\0';
}

void
remove_suffix(char *str, const char* suffix_match) {
    char *optimizedPos = strstr(str, suffix_match);
    if (optimizedPos != NULL) {
        *optimizedPos = '\0'; // Replace suffix_match with null character
    }
}
void
write_dfunc_bc_kernel(const ASTNode* root, const char* prefix, const char* func_name,const func_params_info call_info,FILE* fp)
{

	//TP: in bc call params jump over boundary
	const int call_param_offset = 1;
	char* dfunc_name = strdup(func_name);
	remove_suffix(dfunc_name,"____");
	func_params_info params_info = get_function_param_types_and_names(root,dfunc_name);
	if(call_info.expr.size-1 != params_info.expr.size)
	{
		fprintf(stderr,FATAL_ERROR_MESSAGE"Number of inputs %lu for %s in BoundConds does not match the number of input params %lu \n", call_info.expr.size-1, dfunc_name, params_info.expr.size);
		exit(EXIT_FAILURE);

	}
	const size_t num_of_rest_params = params_info.expr.size;
        free_func_params_info(&params_info);
	fprintf(fp,"boundary_condition Kernel %s_%s()\n{\n",prefix,func_name);
	fprintf(fp,"\t%s(",dfunc_name);
	for(size_t j = 0; j <num_of_rest_params; ++j)
	{
		fprintf(fp,"%s",call_info.expr.data[j+call_param_offset]);
		if(j < num_of_rest_params-1) fprintf(fp,",");
	}
	fprintf(fp,"%s\n",")");
	fprintf(fp,"%s\n","}");
	free(dfunc_name);
}
void
gen_dfunc_bc_kernel(const ASTNode* func_call, FILE* fp, const ASTNode* root, const char* boundconds_name)
{
	const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
		
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);
	const int num_boundaries = TWO_D ? 4 : 6;


	func_params_info call_info = get_func_call_params_info(func_call);

	if(!strcmp(func_name,"periodic"))
		return;
	char* prefix;
	const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,0,0);

	if(is_dfunc) asprintf(&prefix,"%s_AC_KERNEL_",boundconds_name);
	else asprintf(&prefix,"%s_",boundconds_name);
	write_dfunc_bc_kernel(root,prefix,func_name,call_info,fp);

	free_func_params_info(&call_info);
}
void
get_field_boundconds_recursive(const ASTNode* node, const ASTNode* root, char** res, const char* boundconds_name)
{
	if(node->lhs)
		get_field_boundconds_recursive(node->lhs,root,res,boundconds_name);
	if(node->rhs)
		get_field_boundconds_recursive(node->rhs,root,res,boundconds_name);
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
	process_boundcond(function_call_list_head->lhs,res,root,boundconds_name);
	while(--n_entries)
	{
		function_call_list_head = function_call_list_head->parent;
		process_boundcond(function_call_list_head->rhs,res,root,boundconds_name);
	}
}
char**
get_field_boundconds(const ASTNode* root, const char* boundconds_name)
{
	char** res = NULL;
  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	if(!has_optimization_info)
		return res;
	const int num_boundaries = TWO_D ? 4 : 6;
	res = malloc(sizeof(char*)*num_fields*num_boundaries);
	memset(res,0,sizeof(char*)*num_fields*num_boundaries);
	get_field_boundconds_recursive(root,root,res,boundconds_name);
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
		const int num_boundaries = TWO_D ? 4 : 6;
		bool* field_boundconds_processed = (bool*)malloc(num_fields*num_boundaries);
		memset(field_boundconds_processed,0,num_fields*num_boundaries*sizeof(bool));
		bool need_to_communicate = false;
		char communicated_fields_str[40000];
		sprintf(communicated_fields_str,"{");
		//TP: To check the boundary of periodicity we need some field that is communicated to check are the periodic boundaries
		int one_communicated_field = -1;
		for(size_t i = 0; i < fields.size; ++i)
		{
			const int field = fields.data[i];
			bool communicated = int_vec_contains(communicated_fields,field);
			need_to_communicate |= communicated;
			if(communicated)
			{
				one_communicated_field = field;
				const char* field_str = get_symbol_by_index(NODE_VARIABLE_ID,field,FIELD)->identifier;
				strcatprintf(communicated_fields_str,"%s,",field_str);
			}
		}
		strcat(communicated_fields_str,"}");
		if(need_to_communicate)
		{
			strcatprintf(res,"\tacHaloExchange(%s),\n",communicated_fields_str);

			const char* x_boundcond = field_boundconds[one_communicated_field + num_fields*0];
			const char* y_boundcond = field_boundconds[one_communicated_field + num_fields*1];
			const char* z_boundcond = TWO_D ? NULL : field_boundconds[one_communicated_field + num_fields*4];
			
			const bool x_periodic = !strcmp(x_boundcond,"periodic");
			const bool y_periodic = !strcmp(y_boundcond,"periodic");
			const bool z_periodic = TWO_D ? false : !strcmp(z_boundcond,"periodic");

			char* boundary;
			asprintf(&boundary,"%s%s%s",
						x_periodic ? "X" : "",
						y_periodic ? "Y" : "",
						z_periodic ? "Z" : ""
				);
			if( x_periodic|| y_periodic || z_periodic)
				strcatprintf(res,"\tacBoundaryCondition(BOUNDARY_%s,BOUNDCOND_PERIODIC,%s),\n",boundary,communicated_fields_str);

			for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
				for(size_t i = 0; i < fields.size; ++i)
					field_boundconds_processed[fields.data[i] + num_fields*boundcond]  = !int_vec_contains(communicated_fields,fields.data[i]) || 
													     !strcmp(field_boundconds[fields.data[i] + num_fields*boundcond],"periodic");

			bool all_are_processed = false;
			while(!all_are_processed)
			{
				for(int boundcond = 0; boundcond < num_boundaries; ++boundcond)
				{
					const char* processed_boundcond = NULL;

					for(size_t i = 0; i < fields.size; ++i)
					{
						if(!check_symbol_index(NODE_VARIABLE_ID, i, FIELD, COMMUNICATED)) continue;
						processed_boundcond = !field_boundconds_processed[fields.data[i] + num_fields*boundcond] ? field_boundconds[fields.data[i] + num_fields*boundcond] : processed_boundcond;
					}
					if(!processed_boundcond) continue;
					strcatprintf(res,"\tacBoundaryCondition(BOUNDARY_%s,KERNEL_%s__%s,{",boundary_str(boundcond),boundconds_name,processed_boundcond);

					for(size_t i = 0; i < fields.size; ++i)
					{
						if(!check_symbol_index(NODE_VARIABLE_ID, i, FIELD, COMMUNICATED)) continue;
						const char* field_str = get_field_name(fields.data[i]);
						const char* boundcond_str = field_boundconds[fields.data[i] + num_fields*boundcond];
						if(strcmp(boundcond_str,processed_boundcond)) continue;
						if(field_boundconds_processed[fields.data[i] + num_fields*boundcond]) continue;
						field_boundconds_processed[fields.data[i] + num_fields*boundcond] |= true;
						strcatprintf(res,"%s,",field_str);
					}
					//TP: no loaders at the moment
					strcat(res,"}),\n");
					//strcatprintf(res,"},%s_%s_%s_loader),\n",boundconds_name,boundary_str,processed_boundcond);
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
bool
do_not_rename(const ASTNode* node, const char* str_to_check)
{
	if(!node->buffer)                                            return true;
	if(node->token != IDENTIFIER)                                return true;
	if(strcmp(node->buffer,str_to_check))                        return true;
	if(strstr(node->buffer,"AC_INTERNAL"))                       return true;
	return false;
}

void
rename_variables(ASTNode* node, const char* new_name, const char* new_expr, const char* old_name)
{
	if(node->lhs)
		rename_variables(node->lhs,new_name,new_expr,old_name);
	if(node->rhs)
		rename_variables(node->rhs,new_name,new_expr,old_name);
	if(do_not_rename(node,old_name)) return;
	free(node->buffer);
	node->buffer = strdup(new_name);
	node->expr_type = new_expr;
}

void
rename_while(const NodeType type, ASTNode* head, const char* new_name, const char* new_expr, const char* old_name)
{
	while(head->type == type)
	{
		rename_variables(head->rhs,new_name,new_expr,old_name);
		head = head->parent;
	}
}

void
make_unique_bc_calls(ASTNode* node)
{
	TRAVERSE_PREAMBLE(make_unique_bc_calls);
	if(node->type != NODE_BOUNDCONDS_DEF) return;
	const ASTNode* function_call_list_head = node->rhs;
	node_vec func_calls = get_nodes_in_list(function_call_list_head);
	for(size_t i = 0; i < func_calls.size; ++i)
	{
		ASTNode* identifier = (ASTNode*) get_node_by_token(IDENTIFIER, func_calls.data[i]);
		const char* func_name = identifier->buffer;
		ASTNode* head = func_calls.data[i]->parent;
		const bool lhs_node = func_calls.data[i]->parent->rhs == NULL;
		if(!strcmp(func_name,"periodic")) continue;
		if(!head) continue;
		char* new_name = malloc(sizeof(char)*(strlen(func_name)+10));
		int index = strlen(func_name);
		while(isdigit(func_name[--index]));
		if(strlen(func_name) >=5 && func_name[index] == '_' && func_name[index-1] == '_' && func_name[index-2] == '_' && func_name[index-3] == '_')
		{
			int num = atoi(&func_name[index+1]);
			char* tmp = strdup(func_name);
			remove_suffix(tmp,"____");
			sprintf(new_name,"%s____%d",tmp,num+1);
			free(tmp);
		}
		else
		{
			sprintf(new_name,"%s____1",func_name);
		}
		rename_while(
					NODE_UNKNOWN,
					lhs_node ? head : head->parent,
					new_name,NULL,func_name
				);
		free(identifier->buffer);
		identifier->buffer = new_name;
	}
	free_node_vec(&func_calls);
}
void
gen_user_taskgraphs_recursive(const ASTNode* node, const ASTNode* root, string_vec* names)
{
  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	if(!has_optimization_info)
		return;
	if(node->lhs)
		gen_user_taskgraphs_recursive(node->lhs,root,names);
	if(node->rhs)
		gen_user_taskgraphs_recursive(node->rhs,root,names);
	if(node->type != NODE_TASKGRAPH_DEF)
		return;
	const char* boundconds_name = node->lhs->rhs->buffer;
	char** field_boundconds = get_field_boundconds(root,boundconds_name);
	const int num_boundaries = TWO_D ? 4 : 6;

	for(size_t field = 0; field < num_fields; ++field)
	{
		if(!check_symbol_index(NODE_VARIABLE_ID, field, FIELD, COMMUNICATED)) continue;
		for(int bc = 0; bc < num_boundaries; ++bc)
			if(!field_boundconds[field + num_fields*bc])
			{
				printf("HMM :%d\n",bc);
				fprintf(stderr,FATAL_ERROR_MESSAGE"Missing boundcond for field: %s at boundary: %s\n",get_field_name(field),boundary_str(bc));
				exit(EXIT_FAILURE);

			}
	}

	const char* name = node->lhs->lhs->buffer;
	push(names,name);
	char* res = malloc(sizeof(char)*10000);
	sprintf(res, "if (graph == %s)\n\treturn acGridBuildTaskGraph({\n",name);
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
			fprintf(stderr,FATAL_ERROR_MESSAGE"Bug in the compiler aborting\n");
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
		strcatprintf(all_fields,"%s,",get_field_name(field));
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
						root,res,name
				);
				const char* func_name = get_node_by_token(IDENTIFIER,kernel_call)->buffer;
				const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,func_name,KERNEL);
				for(size_t field = 0; field < num_fields; ++field)
					field_written_out_before[field] |= written_fields[field + num_fields*kernel_index];

			}
		}
		free_int_vec(&fields_not_written_to);
		free_int_vec(&fields_written_to);
		free_int_vec(&communicated_fields);
	}
	free(field_written_out_before);
	strcat(res,"\t});\n");
	file_append("user_taskgraphs.h",res);


	free_int_vec(&kernel_calls);
	free_int_vec(&kernel_calls_in_level_order);
	free(res);
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
gen_user_taskgraphs(const ASTNode* root)
{
  	const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	make_unique_bc_calls((ASTNode*) root);
	string_vec graph_names = VEC_INITIALIZER;
	if(has_optimization_info)
	{
		gen_user_taskgraphs_recursive(root,root,&graph_names);
	}
	FILE* fp = fopen("taskgraph_enums.h","w");
	fprintf(fp,"typedef enum {");
	for(size_t i = 0; i < graph_names.size; ++i)
		fprintf(fp,"%s,",graph_names.data[i]);
	fprintf(fp,"NUM_DSL_TASKGRAPHS} AcDSLTaskGraph;\n");
	fclose(fp);
	free_str_vec(&graph_names);
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
	const char* type;
	const char* name;
} variable;


void
add_param_combinations(const variable var, const int kernel_index,const char* prefix, combinatorial_params combinatorials)
{
	char full_name[4096];
	sprintf(full_name,"%s%s",prefix,var.name);
	if(str_vec_contains(s_info.user_structs,var.type))
	{
	  const int struct_index = str_vec_get_index(s_info.user_structs,var.type);
	  string_vec struct_field_types = s_info.user_struct_field_types[struct_index];
	  string_vec struct_field_names = s_info.user_struct_field_names[struct_index];
	  for(size_t i=0; i<struct_field_types.size; ++i)
	  {
		  char new_prefix[10000];
		  sprintf(new_prefix, "%s%s.",prefix,var.name);
		  add_param_combinations((variable){struct_field_types.data[i],struct_field_names.data[i]},kernel_index,new_prefix,combinatorials);
	  }
	}
	if(str_vec_contains(e_info.names,var.type))
	{
		const int param_index = push(&combinatorials.names[kernel_index],full_name);

		const int enum_index = str_vec_get_index(e_info.names,var.type);
		string_vec options  = e_info.options[enum_index];
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
add_kernel_bool_dconst_to_combinations(const ASTNode* node, const int kernel_index, combinatorial_params dst)
{
	if(node->lhs)
		add_kernel_bool_dconst_to_combinations(node->lhs,kernel_index,dst);
	if(node->rhs)
		add_kernel_bool_dconst_to_combinations(node->rhs,kernel_index,dst);
	if(node->token != IDENTIFIER) return;
  	if(!check_symbol(NODE_VARIABLE_ID, node->buffer, BOOL, DCONST_QL)) return;
	if(str_vec_contains(dst.names[kernel_index], node->buffer)) return;
	add_param_combinations((variable){"bool",node->buffer}, kernel_index,"", dst);


}
void
gen_kernel_num_of_combinations_recursive(const ASTNode* node, param_combinations combinations, string_vec* user_kernels_with_input_params,combinatorial_params combinatorials)
{
	if(node->lhs)
	{
		gen_kernel_num_of_combinations_recursive(node->lhs,combinations,user_kernels_with_input_params,combinatorials);
	}
	if(node->rhs)
		gen_kernel_num_of_combinations_recursive(node->rhs,combinations,user_kernels_with_input_params,combinatorials);
	if(node->type & NODE_KFUNCTION && node->rhs->lhs)
	{
	   const char* kernel_name = get_node(NODE_FUNCTION_ID, node)->buffer;
	   const int kernel_index = push(user_kernels_with_input_params,kernel_name);
	   ASTNode* param_list_head = node->rhs->lhs;
	   func_params_info info = get_function_param_types_and_names(node,kernel_name);
	   for(size_t i = 0; i < info.expr.size; ++i)
	   {
		   const char* type = info.types.data[i]; 
		   const char* name = info.expr.data[i];
	           add_param_combinations((variable){type,name},kernel_index,"",combinatorials);
	   }
	   free_func_params_info(&info);
	   add_kernel_bool_dconst_to_combinations(node,kernel_index,combinatorials);
	   gen_combinations(kernel_index,combinations,combinatorials);
	}
}
void
gen_kernel_num_of_combinations(const ASTNode* root, param_combinations combinations, string_vec* user_kernels_with_input_params,string_vec* user_kernel_combinatorial_params)
{

	string_vec user_kernel_combinatorial_params_options[100*100] = { [0 ... 100*100 -1] = VEC_INITIALIZER};
	gen_kernel_num_of_combinations_recursive(root,combinations,user_kernels_with_input_params,(combinatorial_params){user_kernel_combinatorial_params,user_kernel_combinatorial_params_options});
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

	const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,0);
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
	const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,0);
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
	char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
        const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,0);
	if(dfunc_index > 0)
	    get_called_dfuncs(node,&src[dfunc_index],src);
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
        const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,0);
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
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,fn_identifier->buffer,KERNEL);
	assert(kernel_reduce_info.ops.size  == kernel_reduce_info.outputs.size && kernel_reduce_info.ops.size == kernel_reduce_info.conditions.size);

	char* res_name;
#if AC_USE_HIP
	const char* shuffle_instruction = "rocprim::warp_shuffle_down(";
	const char* warp_size  = "const size_t warp_size = rocprim::warp_size();";
	const char* warp_id= "const size_t warp_id = rocprim::warp_id();\n";
#else
	const char* shuffle_instruction = "__shfl_down_sync(0xffffffff,";
	const char* warp_size  = "constexpr size_t warp_size = 32;";
	const char* warp_id= "const size_t warp_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) / warp_size;";
#endif

	for(size_t i = 0; i < kernel_reduce_info.ops.size; ++i)
	{
		ReduceOp reduce_op = kernel_reduce_info.ops.data[i];
		char* condition = kernel_reduce_info.conditions.data[i];
		char* output = kernel_reduce_info.outputs.data[i];
		push_op(&kernel_reduce_infos[kernel_index].ops,  reduce_op);
		push(&kernel_reduce_infos[kernel_index].outputs,  output);
	 	strcatprintf(new_postfix,"if(should_reduce[(int)%s]){"
						"%s"
						"%s"
						"const size_t lane_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) % warp_size;"
						"const int warps_per_block = (blockDim.x*blockDim.y*blockDim.z + warp_size -1)/warp_size;"
						"const int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;"
						"const int out_index =  vba.reduce_offset + warp_id + block_id*warps_per_block;"
						"for(int offset = warp_size/2; offset > 0; offset /= 2){ \n"
		,output,warp_size,warp_id);
		const char* array_name =
			reduce_op == REDUCE_SUM ? "reduce_sum_res" :
			reduce_op == REDUCE_MIN ? "reduce_min_res" :
			reduce_op == REDUCE_MAX ? "reduce_max_res" :
			NULL;
		if(!array_name)
		{
			printf("WRONG!\n");
			printf("%s\n",fn_identifier->buffer);
      			exit(EXIT_FAILURE);
		}
		asprintf(&res_name,"%s[(int)%s]",array_name,output);
	 	switch(reduce_op)
	 	{
		 	case(REDUCE_SUM):

				strcatprintf(new_postfix,"%s += %s%s,offset);\n",res_name,shuffle_instruction,res_name);
				break;
		 	case(REDUCE_MIN):
				strcatprintf(new_postfix,"const AcReal shuffle_tmp = %s%s,offset);",shuffle_instruction,res_name);
				strcatprintf(new_postfix,"%s = (shuffle_tmp < %s) ? shuffle_tmp : %s;\n",res_name,res_name,res_name);
				break;
		 	case(REDUCE_MAX):
				strcatprintf(new_postfix,"const AcReal shuffle_tmp = %s%s,offset);",shuffle_instruction,res_name);
				strcatprintf(new_postfix,"%s = (shuffle_tmp > %s) ? shuffle_tmp : %s;\n",res_name,res_name,res_name);
				break;
		 	case(NO_REDUCE):
				printf("WRONG!\n");
				printf("%s\n",fn_identifier->buffer);
      				exit(EXIT_FAILURE);
	 	}

	 	strcatprintf(new_postfix,
				"}\n"
				"if(lane_id == 0) {vba.reduce_scratchpads[(int)%s][0][out_index] = %s;}}\n"
		,output,res_name);
	}
	strcat(new_postfix,"}");
	compound_statement->postfix = strdup(new_postfix);
	free_reduce_info(&kernel_reduce_info);
	free(new_postfix);
	free(res_name);
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
  FILE* fp = fopen("kernel_reduce_outputs.h","w");

  int num_real_reduce_output = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (symbol_table[i].tspecifier_token == REAL && int_vec_contains(symbol_table[i].tqualifiers,OUTPUT))
	    ++num_real_reduce_output;
  //extra padding to help some compilers
  fprintf(fp,"%s","static const int kernel_reduce_outputs[NUM_KERNELS][NUM_REAL_OUTPUTS+1] = { ");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].tspecifier_token == KERNEL)
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
    if (symbol_table[i].tspecifier_token == KERNEL)
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
gen_optimized_kernel_decls(ASTNode* node, const param_combinations combinations, const string_vec user_kernels_with_input_params,string_vec* const user_kernel_combinatorial_params)
{
	if(node->lhs)
		gen_optimized_kernel_decls(node->lhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		gen_optimized_kernel_decls(node->rhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(!(node->type & NODE_KFUNCTION))
		return;
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,get_node(NODE_FUNCTION_ID,node)->buffer);
	if(kernel_index == -1)
		return;
	string_vec combination_params = user_kernel_combinatorial_params[kernel_index];
	if(combination_params.size == 0)
		return;
	const bool left_child = (node->parent->lhs->id == node->id);
	ASTNode* base = node->parent;
	ASTNode* head = astnode_create(NODE_UNKNOWN,node,NULL);
	if(left_child) 
		base->lhs = head;
	else
		base->rhs = head;
	head->parent = base;
	node_vec optimized_decls = VEC_INITIALIZER;
	for(int i = 0; i < combinations.nums[kernel_index]; ++i)
	{
		ASTNode* new_node = astnode_dup(node,NULL);
		ASTNode* function_id = (ASTNode*) get_node(NODE_FUNCTION_ID,new_node->lhs);
		asprintf(&function_id->buffer,"%s_optimized_%d",get_node(NODE_FUNCTION_ID,node)->buffer,i);
		push_node(&optimized_decls,new_node);
	}
	head->rhs = build_list_node(optimized_decls,"");
	free_node_vec(&optimized_decls);
}
void
gen_kernel_ifs(ASTNode* node, const param_combinations combinations, const string_vec user_kernels_with_input_params,string_vec* const user_kernel_combinatorial_params)
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
	for(int i = 0; i < combinations.nums[kernel_index]; ++i)
	{
		string_vec combination_vals = combinations.vals[kernel_index + MAX_KERNELS*i];
		char* res = malloc(sizeof(char)*4096);
		sprintf(res,"if(kernel_enum == KERNEL_%s ",get_node(NODE_FUNCTION_ID,node)->buffer);
		for(size_t j = 0; j < combination_vals.size; ++j)
			strcatprintf(res, " && vba.kernel_input_params.%s.%s ==  %s ",get_node(NODE_FUNCTION_ID,node)->buffer,combination_params.data[j],combination_vals.data[j]);

		strcatprintf(res,
				")\n{\n"
				"\treturn %s_optimized_%d;\n}\n"
		,get_node(NODE_FUNCTION_ID,node)->buffer,i);
		fprintf(fp_defs,"%s_optimized_%d,",get_node(NODE_FUNCTION_ID,node)->buffer,i);
		fprintf(fp,"%s",res);
		free(res);
	}
	
	printf("NUM of combinations: %d\n",combinations.nums[kernel_index]);
	fclose(fp);
	fprintf(fp_defs,"}\n");
	fclose(fp_defs);
}
void
replace_boolean_dconsts_in_optimized(ASTNode* node, const string_vec* vals, string_vec user_kernels_with_input_params, string_vec* user_kernel_combinatorial_params)
{
	if(node->lhs)
		replace_boolean_dconsts_in_optimized(node->lhs,vals,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		replace_boolean_dconsts_in_optimized(node->rhs,vals,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->token != IDENTIFIER || !node->buffer) return;
  	if(!check_symbol(NODE_VARIABLE_ID, node->buffer, BOOL, DCONST_QL)) return;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,function->lhs);
	char* kernel_name = strdup(fn_identifier->buffer);
	const int combinations_index = get_suffix_int(kernel_name,"_optimized_");
	remove_suffix(kernel_name,"_optimized_");
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,kernel_name);
	if(combinations_index == -1)
		return;
	const string_vec combinations = vals[kernel_index + MAX_KERNELS*combinations_index];
	const int param_index = str_vec_get_index(user_kernel_combinatorial_params[kernel_index],node->buffer);
	if(param_index < 0) return;
	asprintf(&node->buffer,"%s",combinations.data[param_index]);
	node->lhs = NULL;
	node->rhs = NULL;
}
void
gen_kernel_input_params(ASTNode* node, const string_vec* vals, string_vec user_kernels_with_input_params, string_vec* user_kernel_combinatorial_params)
{
	if(node->lhs)
		gen_kernel_input_params(node->lhs,vals,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		gen_kernel_input_params(node->rhs,vals,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(!(node->type & NODE_INPUT && node->buffer))
		return;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,function->lhs);
	char* kernel_name = strdup(fn_identifier->buffer);
	const int combinations_index = get_suffix_int(kernel_name,"_optimized_");
	remove_suffix(kernel_name,"_optimized_");
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,kernel_name);

	if(combinations_index == -1)
	{
		asprintf(&node->buffer,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
		return;
	}
	const string_vec combinations = vals[kernel_index + MAX_KERNELS*combinations_index];
	const int param_index = str_vec_get_index(user_kernel_combinatorial_params[kernel_index],node->buffer);
	if(param_index < 0)
	{
		asprintf(&node->buffer,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
		return;
	}
	asprintf(&node->parent->parent->parent->buffer,"%s",combinations.data[param_index]);
	node->parent->parent->parent->infix= NULL;
	node->parent->parent->parent->lhs = NULL;
	node->parent->parent->parent->rhs = NULL;
}
int
str_to_qualifier(const char* str)
{
	if(!strcmp(str,"AcReal"))     return REAL;
	if(!strcmp(str,"AcReal3"))    return REAL3;
	if(!strcmp(str,"int"))        return INT;
	if(!strcmp(str,"int3"))       return INT3;
	if(!strcmp(str,"long"))        return LONG;
	if(!strcmp(str,"long long"))       return LONG_LONG;
	return 0;
}
const char*
qualifier_to_str(const int qualifier)
{

	if(qualifier == INT)         return "int";
	else if(qualifier == REAL)   return "AcReal";
	else if(qualifier == LONG)   return "long";
	else if(qualifier == LONG_LONG)   return "long long";
	else if(qualifier == MATRIX) return "AcMatrix";
	else if(qualifier == REAL3) return "AcReal3";
	else if(qualifier == INT3) return "int3";
	else if(qualifier == BOOL) return "bool";
	else if(qualifier == CONST_QL) return "const";
	else if(qualifier == CONSTEXPR) return "constexpr";
	else if(qualifier == VTXBUFFER) return "VertexBuffer";
	else if(qualifier == SUM)        return "Sum";
	else if(qualifier == MAX)        return "Max";
	else if(qualifier == SHARED)     return "__shared__";
	return NULL;
}
static void
check_for_undeclared_use_in_range(const ASTNode* node)
{
	  const ASTNode* range_node = get_parent_node_by_token(RANGE,node);
	  if(range_node)
	  {
		fprintf(stderr,FATAL_ERROR_MESSAGE"Undeclared variable or function used on a range expression\n");
		fprintf(stderr,"Range: %s\n",combine_all_new(range_node));
		fprintf(stderr,"Var: %s\n",node->buffer);
		fprintf(stderr,"\n");
		exit(EXIT_FAILURE);
	  }
}	
static void
check_for_undeclared_function(const ASTNode* node)
{
	  const ASTNode* func_call_node = get_parent_node(NODE_FUNCTION_CALL,node);
	  if(func_call_node)
	  {
		if(get_node_by_token(IDENTIFIER,func_call_node->lhs)->id == node->id)
		{
			const char* tmp = combine_all_new(func_call_node);
			const char* func_name = get_node_by_token(IDENTIFIER,func_call_node->lhs)->buffer;
			fprintf(stderr,FATAL_ERROR_MESSAGE);
                        if(str_vec_contains(duplicate_dfuncs.names,func_name))
                                fprintf(stderr,"Unable to resolve overloaded function: %s\nIn:\t%s\n",func_name,tmp);
                        else
                                fprintf(stderr,"Undeclared function used: %s\nIn:\t%s\n",func_name,tmp);
			const ASTNode* surrounding_func = get_parent_node(NODE_FUNCTION,node);
			if(surrounding_func)
                                fprintf(stderr,"Inside %s",get_node_by_token(IDENTIFIER,surrounding_func->lhs)->buffer);
			fprintf(stderr,"\n");
			exit(EXIT_FAILURE);
		}

	  }
}
static void
check_for_undeclared_use_in_assignment(const ASTNode* node)
{
	
	  const bool used_in_assignment = is_right_child(NODE_ASSIGNMENT,node);
	  if(used_in_assignment)
	  {
		fprintf(stderr,FATAL_ERROR_MESSAGE"Undeclared variable or function used on the right hand side of an assignment\n");
		fprintf(stderr,"Assignment: %s\n",combine_all_new(get_parent_node(NODE_ASSIGNMENT,node)));
		fprintf(stderr,"Var: %s\n",node->buffer);
		fprintf(stderr,"\n");
		exit(EXIT_FAILURE);
	  }
}
static void
check_for_undeclared(const ASTNode* node)
{
	 check_for_undeclared_use_in_range(node);
	 check_for_undeclared_function(node);
	 check_for_undeclared_use_in_assignment(node);

}
static void
translate_buffer_body(FILE* stream, const ASTNode* node)
{
  if (stream && node->buffer) {
    const Symbol* symbol = symboltable_lookup(node->buffer);
    if (symbol && symbol->type & NODE_VARIABLE_ID && int_vec_contains(symbol->tqualifiers,DCONST_QL))
      fprintf(stream, "DCONST(%s)", node->buffer);
    else if (symbol && symbol->type & NODE_VARIABLE_ID && int_vec_contains(symbol->tqualifiers,RUN_CONST))
      fprintf(stream, "RCONST(%s)", node->buffer);
    else
      fprintf(stream, "%s", node->buffer);
  }
}
static void
output_qualifiers(FILE* stream, const ASTNode* node, const int* tqualifiers, const size_t n_tqualifiers)
{
        const ASTNode* is_dconst = get_parent_node_exclusive(NODE_DCONST, node);
        if (is_dconst)
          fprintf(stream, "__device__ ");

        if (n_tqualifiers)
	  for(size_t i=0; i<n_tqualifiers;++i)
	  {
		if(tqualifiers[i] != BOUNDARY_CONDITION && tqualifiers[i] != ELEMENTAL && tqualifiers[i] != UTILITY)
          		fprintf(stream, "%s ", qualifier_to_str(tqualifiers[i]));
	  }
}
typedef struct
{
	const char* id;
	int token;
} tspecifier;

void
output_specifier(FILE* stream, const tspecifier tspec, const ASTNode* node, const bool do_undeclared_check)
{
        if (tspec.id) 
	{
	  if(tspec.token != KERNEL)
            fprintf(stream, "%s ", tspec.id);
        }
        else if (!get_parent_node_exclusive(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID) &&
                 !(node->type & NODE_INPUT) &&
		 !(node->no_auto) &&
		 !(is_user_enum_option(node->buffer)) &&
		 !(strstr(node->buffer,"AC_INTERNAL_d")) &&
		 !(strstr(node->buffer,"AC_INTERNAL_gmem"))
		 )
	{
	  if(node->is_constexpr && !(node->type & NODE_FUNCTION_ID)) fprintf(stream, " constexpr ");
	  if(node->expr_type)
		  fprintf(stream, "%s ",node->expr_type);
	  else
          	fprintf(stream, "auto ");
	  if(do_undeclared_check) check_for_undeclared(node);
	}
}

tspecifier
get_tspec(const ASTNode* node)
{
      const ASTNode* decl = get_parent_node(NODE_DECLARATION, node);
      if(!decl) return (tspecifier){NULL, 0};
      const ASTNode* tspec_node = get_node(NODE_TSPEC, decl);
      const ASTNode* tqual_node = get_node(NODE_TQUAL, decl);
      if(!tspec_node || !tspec_node->lhs) return (tspecifier){NULL,0};
      return 
	     (tspecifier)
	      {
		      tspec_node->lhs->buffer,
		      tspec_node->lhs->token
	      };
}
size_t
get_qualifiers(const ASTNode* node, int* tqualifiers)
{
      const ASTNode* decl = get_parent_node(NODE_DECLARATION, node);
      if(!decl) return 0;
      const ASTNode* tqual_node = get_node(NODE_TQUAL, decl);
      if(!tqual_node || !tqual_node->lhs) return 0;
      size_t n_tqualifiers = 0;

	  const ASTNode* tqual_list_node = tqual_node->parent;
	  //backtrack to the start of the list
	  while(tqual_list_node->parent && tqual_list_node->parent->rhs && tqual_list_node->parent->rhs->type & NODE_TQUAL)
		  tqual_list_node = tqual_list_node->parent;
	  while(tqual_list_node->rhs)
	  {
		  tqualifiers[n_tqualifiers] = tqual_list_node->rhs->lhs->token;
		  ++n_tqualifiers;
		  tqual_list_node = tqual_list_node->lhs;
	  }
	  tqualifiers[n_tqualifiers] = tqual_list_node->lhs->lhs->token;
	  ++n_tqualifiers;

      return n_tqualifiers;
}
void static 
check_for_shadowing(const ASTNode* node)
{
    if(symboltable_lookup_surrounding_scope(node->buffer) && is_right_child(NODE_DECLARATION,node) && !get_parent_node(NODE_FUNCTION_CALL,node) && get_node(NODE_TSPEC,get_parent_node(NODE_DECLARATION,node)->lhs))
    {
      // Do not allow shadowing.
      //
      // Note that if we want to allow shadowing, then the symbol table must
      // be searched in reverse order
      fprintf(stderr,
              "Error! Symbol '%s' already present in symbol table. Shadowing "
              "is not allowed.\n",
              node->buffer);
      exit(EXIT_FAILURE);
      assert(0);
    }
}
void
traverse_base(const ASTNode* node, const NodeType exclude, FILE* stream, bool do_checks)
{
  if(node->type == NODE_ENUM_DEF)   return;
  if(node->type == NODE_STRUCT_DEF) return;
  if (node->type & exclude)
	  stream = NULL;
  // Do not translate tqualifiers or tspecifiers immediately
  if (node->parent &&
      (node->parent->type & NODE_TQUAL || node->parent->type & NODE_TSPEC))
    return;

  // Prefix translation
  // Prefix translation
  if (stream && node->prefix)
      fprintf(stream, "%s", node->prefix);

  // Prefix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest < MAX_NESTS);

    ++current_nest;
    num_symbols[current_nest] = num_symbols[current_nest - 1];
  }

  // Traverse LHS
  if (node->lhs)
    traverse_base(node->lhs, exclude, stream, do_checks);

  // Add symbols to symbol table
  if (node->buffer && node->token == IDENTIFIER && !(node->type & exclude)) {
    //New test for shadowing
    if (do_checks) check_for_shadowing(node);
    if (!symboltable_lookup(node->buffer)) {
      static int tqualifiers[MAX_ID_LEN];
      size_t n_tqualifiers = get_qualifiers(node,tqualifiers);
      tspecifier tspec = get_tspec(node);


      if (stream) {
	output_qualifiers(stream,node,tqualifiers,n_tqualifiers);
	output_specifier(stream,tspec,node,do_checks);
      }
      if (!(node->type & NODE_MEMBER_ID))
      {
        add_symbol(node->type, tqualifiers, n_tqualifiers, tspec.id, tspec.token, node->buffer);
      }
    }
  }

  // Infix translation
  if (stream && node->infix) 
    fprintf(stream, "%s", node->infix);
  // Translate buffer body
  translate_buffer_body(stream, node);

  // Traverse RHS
  if (node->rhs)
    traverse_base(node->rhs, exclude, stream,do_checks);

  // Postfix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest > 0);
    --current_nest;
  }

  // Postfix translation
  if (stream && node->postfix) 
    fprintf(stream, "%s", node->postfix);
}
static inline void
traverse(const ASTNode* node, const NodeType exclude, FILE* stream)
{
	traverse_base(node,exclude,stream,false);
}

func_params_info
get_func_call_params_info(const ASTNode* func_call)
{
		func_params_info res = FUNC_PARAMS_INITIALIZER;
		if(!func_call->rhs) return res;
		node_vec params = get_nodes_in_list(func_call->rhs);
		for(size_t i = 0; i < params.size; ++i)
		{

			char* param = malloc(sizeof(char)*10000);
			combine_all(params.data[i],param);
			assert(param);
			push(&res.expr,param);
			free(param);
			push(&res.types,get_expr_type((ASTNode*) params.data[i]));
	        }	
		free_node_vec(&params);
		return res;
}

string_vec
get_struct_field_types(const char* struct_name)
{
		const structs_info info = s_info;
		string_vec res;
		for(size_t i = 0; i < info.user_structs.size; ++i)
			if(!strcmp(info.user_structs.data[i],struct_name)) res = str_vec_copy(info.user_struct_field_types[i]);
		return res;
}

char*
get_user_struct_member_expr(const ASTNode* node)
{
		char* res = NULL;
		const char* struct_type = get_expr_type(node->lhs);
		const char* field_name = get_node(NODE_MEMBER_ID,node)->buffer;
		if(!field_name) return NULL;
		const structs_info info = s_info;
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
		return res;
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


overloaded_dfuncs
get_duplicate_dfuncs(const ASTNode* node)
{
  string_vec dfuncs = get_dfunc_identifiers(node);
  int n_entries[dfuncs.size];
  memset(n_entries,0,sizeof(int)*dfuncs.size);
  for(size_t i = 0; i < dfuncs.size; ++i)
	  n_entries[get_symbol_index(NODE_DFUNCTION_ID,dfuncs.data[i],0)]++;
  string_vec res = VEC_INITIALIZER;
  int_vec    count_res = VEC_INITIALIZER;
  for(size_t i = 0; i < dfuncs.size; ++i)
  {
	  const int index = get_symbol_index(NODE_DFUNCTION_ID,dfuncs.data[i],0);
	  if(index == -1) continue;
	  const char* name = get_symbol_by_index(NODE_DFUNCTION_ID,index,0)->identifier;
	  if(n_entries[index] > 1) push(&res,name);
	  if(n_entries[index] > 1) push_int(&count_res,n_entries[index]);
  }
  free_str_vec(&dfuncs);
  return (overloaded_dfuncs){res,count_res};
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
	ASTNode* identifier_node = get_node_by_token(IDENTIFIER,node);
	if(!identifier_node || strcmp(identifier_node->buffer,identifier)) return;
	node->expr_type = type;
	identifier_node->expr_type = type;
}
static int
strcmp_null_ok(const char* a, const char* b)

{
	if(a == NULL) return -1;
	if(b == NULL) return -1;
	return strcmp(a,b);
}
static bool
node_is_function_param(const ASTNode* node)
{
	return (node->type & NODE_DECLARATION) && is_left_child(NODE_FUNCTION,node);
}
static bool
node_is_binary_expr(const ASTNode* node)
{
	return node->type == NODE_BINARY_EXPRESSION;
}

static bool
node_is_unary_expr(const ASTNode* node)
{
	return node->type == NODE_EXPRESSION && node->lhs && node->lhs->token == UNARY_OP;
}

static bool
node_is_struct_access_expr(const ASTNode* node)
{
	return node->type == NODE_STRUCT_EXPRESSION && node->rhs && get_node(NODE_MEMBER_ID, node->rhs);
}

bool
test_type(ASTNode* node, const char* type);

const char*
get_primary_expr_type(const ASTNode* node)
{
	const ASTNode* identifier = get_node_by_token(IDENTIFIER,node);
  if (identifier &&  identifier->buffer && is_user_enum_option(identifier->buffer))
    return get_enum(identifier->buffer);
	const Symbol* sym = (!identifier || !identifier->buffer) ? NULL :
	        (Symbol*)get_symbol(NODE_ANY, identifier->buffer,NULL);
	return
		(get_node_by_token(REALNUMBER,node)) ? "AcReal":
		(get_node_by_token(DOUBLENUMBER,node)) ? "AcReal":
		(get_node_by_token(NUMBER,node)) ? "int" :
		(get_node_by_token(STRING,node)) ? "char*" :
	  (!sym) ? NULL :
	  strdup(sym->tspecifier);
}
static int
n_occurances(const char* str, const char test)
{
	int res = 0;
	int i = -1;
	while(str[++i] != '\0') res += str[i] == test;
	return res;
}
char*
get_array_elem_type(char* arr_type)
{
	if(n_occurances(arr_type,'<') == 1)
	{
		int start = 0;
		while(arr_type[++start] != '<');
		int end = start;
		++start;
		while(arr_type[++end] != ',');
		arr_type[end] = '\0';
		char* tmp = malloc(sizeof(char)*1000);
		strcpy(tmp, &arr_type[start]);
		return tmp;
	}
	return arr_type;
}
//char*
//get_arr_declaration_type(const ASTNode* node)
//{
//}
char*
get_node_array_access_type(const ASTNode* node)
{
	int counter = 1;
	ASTNode* array_access_base = node->lhs;
	while(array_access_base->type == NODE_ARRAY_ACCESS)
	{
		array_access_base = array_access_base->lhs;
		++counter;
	}
	const char* base_type = get_expr_type(array_access_base);
	/**
	if (!strcmp_null_ok(base_type,"AcReal"))
	{
		char tmp[1000];
		combine_all(node,tmp);
		printf("WRONG %s\n",tmp);
	}
	**/
	return (!base_type)   ? NULL : 
		counter == 2 && !strcmp(base_type,"AcMatrix") ? "AcReal" :
		!strcmp(base_type,"AcMatrix") ? "AcRealArray" :
		strstr(base_type,"*") ? remove_substring(strdup(base_type),"*") :
		strstr(base_type,"AcArray") ? get_array_elem_type(strdup(base_type)) :
		!strcmp(base_type,"Field")  ? "AcReal" :
		NULL;
}

const char*
get_struct_expr_type(const ASTNode* node)
{
	const char* base_type = get_expr_type(node->lhs);
	const ASTNode* left = get_node(NODE_MEMBER_ID,node);
	return
		!base_type ? NULL :
		!strcmp(base_type,"AcReal3") ? "AcReal":
		!strcmp(base_type,"Field3")  ? "Field":
		get_user_struct_member_expr(node);

}
const char*
get_binary_expr_type(const ASTNode* node)
{
	const char* op = get_node_by_token(BINARY_OP,node->rhs->lhs)->buffer;
	if(op && !strcmps(op,"==",">","<",">=","<="))
		return "bool";
	ASTNode* lhs_node = !strcmp_null_ok(op,"*") && node->lhs->rhs ? node->lhs->rhs
				                                            : node->lhs;
	const char* lhs_res = get_expr_type(lhs_node);
	const char* rhs_res = get_expr_type(node->rhs);
	if(!lhs_res || !rhs_res) return NULL;
	const bool lhs_real = !strcmp(lhs_res,"AcReal");
	const bool rhs_real = !strcmp(rhs_res,"AcReal");
	const bool lhs_int   = !strcmp(lhs_res,"int");
	const bool rhs_int   = !strcmp(rhs_res,"int");
	return
		op && !strcmps(op,"+","-","*","/") && (!strcmp(lhs_res,"Field") || !strcmp(rhs_res,"Field"))   ? "AcReal"  :
		op && !strcmps(op,"+","-","*","/") && (!strcmp(lhs_res,"Field3") || !strcmp(rhs_res,"Field3")) ? "AcReal3" :
                (lhs_real || rhs_real) && (lhs_int || rhs_int) ? "AcReal" :
                !strcmp_null_ok(op,"*") && !strcmp(lhs_res,"AcMatrix") &&  !strcmp(rhs_res,"AcReal3") ? "AcReal3" :
		!strcmp(lhs_res,"AcComplex") || !strcmp(rhs_res,"AcComplex")   ? "AcComplex"  :
		lhs_real && !strcmps(rhs_res,"int","long","long long")    ?  "AcReal"  :
		!strcmp_null_ok(op,"*")     && lhs_real && !rhs_int  ?  rhs_res   :
		op && !strcmps(op,"*","/")  && rhs_real && !lhs_int  ?  lhs_res   :
		!strcmp(lhs_res,rhs_res) ? lhs_res :
		NULL;

}
const char*
get_ternary_expr_type(const ASTNode* node)
{

	const char* first_expr  = get_expr_type(node->rhs->lhs);
	const char* second_expr = get_expr_type(node->rhs->rhs);
	return 
		!first_expr ? NULL :
		!second_expr ? NULL :
		strcmp(first_expr,second_expr) ? NULL :
		first_expr;
}
void
get_assignment_expr_type(ASTNode* node)
{
	ASTNode* func_base = (ASTNode*) get_parent_node(NODE_FUNCTION,node);
	const ASTNode* decl = get_node(NODE_DECLARATION,node->lhs);
	const ASTNode* tspec = get_node(NODE_TSPEC,decl);
	const char* var_name = get_node_by_token(IDENTIFIER,decl)->buffer;
	if(tspec)
	{
		
		if(test_type(node->rhs,tspec->lhs->buffer))
		{
			node->expr_type = tspec->lhs->buffer;
	 	        if(func_base && node->expr_type)
	 	       		set_primary_expression_types(func_base, node->expr_type, var_name);
		}	
	}
	else if(get_expr_type(node->rhs))
	{
		const char* rhs_type = get_expr_type(node->rhs);
		const int n_lhs = count_num_of_nodes_in_list(node->lhs->rhs);
		if(n_lhs > 1)
		{
			string_vec types = get_struct_field_types(rhs_type);
			node_vec decls = get_nodes_in_list(node->lhs->rhs);
			for(size_t i = 0; i < decls.size; ++i)
				set_primary_expression_types(func_base, types.data[i], get_node_by_token(IDENTIFIER,decls.data[i])->buffer);
			free_str_vec(&types);
			free_node_vec(&decls);
		}
		else
		{
		      node->expr_type = get_expr_type(node->rhs);
		      set_primary_expression_types(func_base, node->expr_type, var_name);
		}
	}

}
const char*
get_type_declaration_type(ASTNode* node)
{
	node->expr_type = get_node(NODE_TSPEC,node)->lhs->buffer;
	const char* var_name = get_node_by_token(IDENTIFIER,node)->buffer;
	ASTNode* func_base = (ASTNode*) get_parent_node(NODE_FUNCTION,node);
	if(func_base && node->expr_type)
		set_primary_expression_types(func_base, node->expr_type, var_name);
	return node->expr_type;
}
void
get_dfunc_nodes(const ASTNode* node, node_vec* nodes, string_vec* names)
{
	if(node->lhs)
		get_dfunc_nodes(node->lhs,nodes,names);
	if(node->rhs)
		get_dfunc_nodes(node->rhs,nodes,names);
	if(!(node->type & NODE_DFUNCTION)) return;
	push_node(nodes,node);
	push(names,get_node_by_token(IDENTIFIER,node->lhs)->buffer);
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


bool
gen_type_info_base(ASTNode* node, const ASTNode* root);
const char*
get_func_call_expr_type(ASTNode* node)
{
	if(node->lhs->type == NODE_STRUCT_EXPRESSION)
	{
	       const ASTNode* struct_expr   = node->lhs;
               const char* struct_func_name = get_node(NODE_MEMBER_ID,struct_expr->rhs)->buffer;
               const char* base_type        = get_expr_type(struct_expr->lhs);
               if(!strcmp_null_ok(base_type,"AcMatrix") && !strcmps(struct_func_name,"col","row"))
                       return "AcReal3";
	}
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	Symbol* sym = (Symbol*)get_symbol(NODE_VARIABLE_ID | NODE_FUNCTION_ID ,func_name,NULL);
	if(sym && sym->tspecifier_token == STENCIL)
		return "AcReal";
	if(sym && sym->type & NODE_FUNCTION_ID)
	{
		const ASTNode* func = NULL;
		for(size_t i = 0; i < dfunc_nodes.size; ++i)
			if(!strcmp(dfunc_names.data[i],func_name)) func = dfunc_nodes.data[i];
		if(strlen(sym->tspecifier))
			return strdup(sym->tspecifier);
		else if(func && func->expr_type)
			return strdup(func->expr_type);
		if(!node->expr_type)
		{
			if(func)
			{
				func_params_info call_info = get_func_call_params_info(node);
				bool know_all_types = true;
				for(size_t i = 0; i < call_info.types.size; ++i)
					know_all_types &= call_info.types.data[i] != NULL;
				if(know_all_types)
				{
					func_params_info info = get_function_param_types_and_names(func,func_name);
					if(!str_vec_contains(duplicate_dfuncs.names,func_name) && call_info.types.size == info.expr.size)
					{
						ASTNode* func_copy = astnode_dup(func,NULL);
						for(size_t i = 0; i < info.expr.size; ++i)
							set_primary_expression_types(func_copy, call_info.types.data[i], info.expr.data[i]);
						gen_type_info_base(func_copy, NULL);
						if(func_copy->expr_type) 
							node->expr_type = strdup(func_copy -> expr_type);
						astnode_destroy(func_copy);
					}
					free_func_params_info(&info);
				}
				free_func_params_info(&call_info);
			}
		}

	}
	return node->expr_type;
	
}
const char*
get_array_initializer_type(ASTNode* node)
{
	//TP: do not consider more than 1d arrays for the moment
	if(get_node(NODE_ARRAY_INITIALIZER,node->lhs)) return NULL;
	node_vec elems = get_nodes_in_list(node->lhs);
	const char* expr = get_expr_type((ASTNode*) elems.data[0]);
	if(!expr) return NULL;
	const char* res = sprintf_new("AcArray<%s,%lu>",expr,elems.size);
	free_node_vec(&elems);
	return res;
}
const char*
get_struct_initializer_type(ASTNode* node)
{
	if(all_primary_expressions_and_func_calls_have_type(node->lhs))
	{
		node_vec nodes = get_nodes_in_list(node->lhs);
		string_vec types = VEC_INITIALIZER;
		for(size_t i = 0; i < nodes.size; ++i)
			push(&types,get_expr_type((ASTNode*)nodes.data[i]));
		const structs_info info = s_info;
		int n_structs_having_types = 0;
		int index = -1;
		for(size_t i = 0; i < info.user_structs.size; ++i)
		{
			bool has_same_types = info.user_struct_field_types[i].size == types.size;
			if(!has_same_types) continue;
			for(size_t j = 0; j < info.user_struct_field_types[i].size; ++j)
				has_same_types &= !strcmp(info.user_struct_field_types[i].data[j],types.data[j]);
			n_structs_having_types += has_same_types;
			//if there is only a single struct having the correct types index will be the index of that struct
			if(has_same_types) index = i;
		}
		const char* res = (n_structs_having_types== 1 && index >= 0) ? strdup(info.user_structs.data[index]) : NULL;
		free_str_vec(&types);
		free_node_vec(&nodes);
		char* parent_prefix = malloc(sizeof(res) + 100);
		sprintf(parent_prefix,"(%s)",res);
		node->parent->prefix  = parent_prefix;
		return res;
	}
	return node->expr_type;
}
const char*
get_cast_expr_type(ASTNode* node)
{
	const char* res = strdup(combine_all_new(node->lhs));
	test_type(node->rhs,res);
	return res;
}
const char*
get_expr_type(ASTNode* node)
{

	if(node->expr_type) return node->expr_type;
	const char* res = node->expr_type;
	if(node->token == CAST)
		res = get_cast_expr_type(node);
	else if(node->type & NODE_ARRAY_INITIALIZER)
		res = get_array_initializer_type(node);
	else if(node->type == NODE_PRIMARY_EXPRESSION)
		res = get_primary_expr_type(node);
	else if(node->type & NODE_STRUCT_INITIALIZER)
		res = get_struct_initializer_type(node);
	else if(node->type & NODE_ARRAY_ACCESS)
		res = get_node_array_access_type(node);
	else if(node_is_binary_expr(node))
		res = get_binary_expr_type(node);
	else if(node->type == NODE_STRUCT_EXPRESSION)
		res = get_struct_expr_type(node);
	else if(node->type == NODE_TERNARY_EXPRESSION)
		res = get_ternary_expr_type(node);
	else if(node->type == NODE_FUNCTION_CALL)
		res = get_func_call_expr_type(node);
	else if(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node))
		get_type_declaration_type(node);
	else if(node->type & NODE_ASSIGNMENT && get_parent_node(NODE_FUNCTION,node) &&  !get_node(NODE_MEMBER_ID,node->lhs) && !get_node(NODE_ARRAY_ACCESS,node->lhs))
		get_assignment_expr_type(node);
	else
	{
		if(node->lhs && !res)
			res = get_expr_type(node->lhs);
		if(node->rhs && !res)
			res = get_expr_type(node->rhs);
	}
	return res;
}

bool
test_type(ASTNode* node, const char* type)
{
	if(node->type == NODE_PRIMARY_EXPRESSION)
		return !strcmp_null_ok(node->expr_type,type);
	else if(node->type & NODE_STRUCT_INITIALIZER)
	{
		const structs_info info = s_info;
		if(!str_vec_contains(info.user_structs,type)) return false;
		const string_vec types = info.user_struct_field_types[str_vec_get_index(info.user_structs,type)];
		node_vec nodes = get_nodes_in_list(node->lhs);
		if(types.size != nodes.size)
		{
			if(nodes.size == 1)
				fprintf(stderr,FATAL_ERROR_MESSAGE"Incorrect number of initializers\n%s expects %lu members but there was only one initializer\n",type,types.size);
			else
				fprintf(stderr,FATAL_ERROR_MESSAGE"Incorrect number of initializers\n%s expects %lu members but there were %lu initializers\n",type,types.size,nodes.size);
			fprintf(stderr,"%s\n\n",combine_all_new(node));
			exit(EXIT_FAILURE);
		}
		bool res = true;
		if(nodes.size != types.size) return false;
		for(size_t i = 0; i < nodes.size; ++i)
			res &= test_type((ASTNode*)nodes.data[i], types.data[i]);
		free_node_vec(&nodes);
		return res;
	}
	return node->lhs && test_type(node->lhs,type) ? true :
	       node->rhs && test_type(node->rhs,type) ? true : 
	       false;
}

void
gen_multidimensional_field_accesses_recursive(ASTNode* node, const bool gen_mem_accesses)
{
	if(node->lhs)
		gen_multidimensional_field_accesses_recursive(node->lhs,gen_mem_accesses);
	if(node->rhs)
		gen_multidimensional_field_accesses_recursive(node->rhs,gen_mem_accesses);

	if(node->token != IDENTIFIER)
		return;
	if(!node->buffer)
		return;
	if(!node->parent)
		return;
	//discard global const declarations
	if(get_parent_node(NODE_GLOBAL,node))
		return;
	const char* type = get_expr_type(node->parent);
	if(!type || strcmps(type,"Field","VertexBufferHandle"))
		return;

	ASTNode* array_access = (ASTNode*)get_parent_node(NODE_ARRAY_ACCESS,node);
	if(!array_access || !is_left_child(NODE_ARRAY_ACCESS,node))	return;
	while(get_parent_node(NODE_ARRAY_ACCESS,array_access)) array_access = (ASTNode*) get_parent_node(NODE_ARRAY_ACCESS,array_access);


	node_vec nodes = VEC_INITIALIZER;
	get_array_access_nodes(array_access,&nodes);
	if(nodes.size != 1 && nodes.size != 3)	
	{
		fprintf(stderr,"Fatal error: only 1 and 3 -dimensional reads/writes are allowed for VertexBuffers\n");
	}


	ASTNode* idx_node = astnode_create(NODE_UNKNOWN,NULL,NULL);
	ASTNode* rhs = astnode_create(NODE_UNKNOWN, idx_node, NULL);
	if(nodes.size == 3)
	{
		idx_node->prefix  = strdup("IDX(");
		idx_node->postfix = strdup(")");
	}
	ASTNode* indexes = build_list_node(nodes,",");
	idx_node->lhs = indexes;
	indexes->parent = idx_node;

	free_node_vec(&nodes);
	ASTNode* before_lhs = NULL;
	if(gen_mem_accesses && is_left_child(NODE_ASSIGNMENT,node))
	{
		before_lhs = astnode_create(NODE_UNKNOWN,astnode_dup(node,NULL),NULL);
		before_lhs -> prefix = strdup("written_fields[");
		before_lhs -> postfix = strdup("] = 1;");
	}

	array_access->rhs = NULL;
	array_access->lhs = NULL;
	array_access->buffer = NULL;
	array_access->rhs = NULL;
	array_access->infix= NULL;
	array_access->postfix= NULL;
	array_access->prefix = NULL;

        array_access->rhs = rhs;

	ASTNode* lhs = astnode_create(NODE_UNKNOWN, before_lhs, astnode_dup(node,NULL));

	array_access->lhs = lhs;

	if(gen_mem_accesses && !is_left_child(NODE_ASSIGNMENT,node))
	{
        	rhs->postfix= strdup(")");

		lhs->infix = strdup("AC_INTERNAL_read_field(");
		lhs->postfix = strdup(",");
	}
	else
	{
        	rhs->prefix = strdup("[");
        	rhs->postfix= strdup("]");

		lhs->infix = strdup("vba.in[");
		lhs->postfix = strdup("]");
	}
	lhs->parent = array_access;
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
gen_const_def(const ASTNode* def, const ASTNode* tspec, FILE* fp, FILE* fp_non_scalar_constants)
{
		(void)fp;
		const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
		if(!name) return;
        	const ASTNode* assignment = def->rhs;
		if(!assignment) return;
		const ASTNode* struct_initializer = get_node(NODE_STRUCT_INITIALIZER,assignment);
		const ASTNode* array_initializer = get_node(NODE_ARRAY_INITIALIZER, assignment);
		const char* datatype = tspec->lhs->buffer;
		char* datatype_scalar = remove_substring(strdup(datatype),"*");
		//TP: the C++ compiler is not always able to use the structs if you don't have the conversion from the initializer list
		//TP: e.g. multiplying a matrix with a scalar won't work without the conversion
		if(struct_initializer && !array_initializer)
			if(!struct_initializer->parent->postfix)
				asprintf(&struct_initializer->parent->prefix,"(%s)",datatype_scalar);
		const char* assignment_val = combine_all_new(assignment);
		remove_substring(datatype_scalar,"*");
		const int array_dim = array_initializer ? count_nest(array_initializer,NODE_ARRAY_INITIALIZER) : 0;
		const int num_of_elems = array_initializer ? count_num_of_nodes_in_list(array_initializer->lhs) : 0;
		if(array_initializer)
		{
			const ASTNode* second_array_initializer = get_node(NODE_ARRAY_INITIALIZER, array_initializer->lhs);
			if(array_dim == 1)
			{
				fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] constexpr AcArray<%s,%d> %s = %s;\n#endif\n",datatype_scalar, num_of_elems, name, assignment_val);
			}
			else if(array_dim == 2)
			{
				const int num_of_elems_in_list = count_num_of_nodes_in_list(second_array_initializer->lhs);
				fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] constexpr AcArray<AcArray<%s,%d>,%d> %s = %s;\n#endif\n",datatype_scalar, num_of_elems_in_list, num_of_elems, name, assignment_val);
			}
			else
			{
				fprintf(stderr,FATAL_ERROR_MESSAGE"todo add 3d const arrays\n");
				exit(EXIT_FAILURE);
			}
		}
		else
		{
		        //TP: define macros have greater portability then global constants, since they do not work on some CUDA compilers
			//TP: actually can not make macros since if the user e.g. writes const nx = 3 then that define would conflict with variables in hip
                        if(is_primitive_datatype(datatype_scalar))
                                fprintf(fp_non_scalar_constants, "[[maybe_unused]] constexpr %s %s = %s;\n", datatype_scalar, name, assignment_val);
                        else
			{
                               fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] constexpr %s %s = %s;\n#endif\n",datatype_scalar, name, assignment_val);
			}
		}
		free(datatype_scalar);
}
void
gen_const_variables(const ASTNode* node, FILE* fp,FILE* fp_non_scalars)
{
	if(node->lhs)
		gen_const_variables(node->lhs,fp,fp_non_scalars);
	if(node->rhs)
		gen_const_variables(node->rhs,fp,fp_non_scalars);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	if(!has_qualifier(node,"const")) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	const ASTNode* def_list_head = node->rhs;
	while(def_list_head -> rhs)
	{
		gen_const_def(def_list_head->rhs,tspec,fp,fp_non_scalars);
		def_list_head = def_list_head -> lhs;
	}
	gen_const_def(def_list_head->lhs,tspec,fp,fp_non_scalars);
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

    for(size_t i = 0; i < num_dfuncs; ++i) 
    {
    	    const Symbol* dfunc_symbol = get_symbol_by_index(NODE_DFUNCTION_ID,i,0);
	    if(int_vec_contains(dfunc_symbol->tqualifiers,INLINE)) continue;
	    if(int_vec_contains(called_dfuncs,i)) strcat(prefix,dfunctions[i]);
    }

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

string_vec
get_names(const int token)
{
	string_vec res = VEC_INITIALIZER;
	for(size_t i = 0; i < num_symbols[0]; ++i)
		if(symbol_table[i].tspecifier_token == token)
			push(&res, symbol_table[i].identifier);
	return res;
}


void
gen_names(const char* datatype, const int token, FILE* fp)
{
	string_vec names = get_names(token); 
	fprintf(fp,"static const char* %s_names[] __attribute__((unused)) = {",datatype);
	for(size_t i = 0; i < names.size; ++i)
  		fprintf(fp, "\"%s\",", names.data[i]);
	fprintf(fp,"};\n");
	free_str_vec(&names);
}
static void
gen_field_info(FILE* fp)
{
  num_fields = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier_token == FIELD)
      ++num_fields;



  // Enums
  int num_of_communicated_fields=0;
  size_t num_of_fields=0;
  bool field_is_auxiliary[256];
  bool field_is_communicated[256];
  bool field_is_dead[256];
  size_t num_of_alive_fields=0;
  string_vec field_names = VEC_INITIALIZER;
  string_vec original_names = VEC_INITIALIZER;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier_token == FIELD)
	    push(&original_names,symbol_table[i].identifier);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if(symbol_table[i].tspecifier_token == FIELD){
      const bool is_dead = int_vec_contains(symbol_table[i].tqualifiers,DEAD);
      if(is_dead) continue;
      push(&field_names, symbol_table[i].identifier);
      const char* name = symbol_table[i].identifier;
      const bool is_aux  = int_vec_contains(symbol_table[i].tqualifiers,AUXILIARY);
      const bool is_comm = int_vec_contains(symbol_table[i].tqualifiers,COMMUNICATED);
      field_is_auxiliary[num_of_fields]    = is_aux;
      field_is_communicated[num_of_fields] = is_comm;
      num_of_communicated_fields           += is_comm;
      num_of_alive_fields                  += (!is_dead);
      field_is_dead[num_of_fields]         = is_dead;
      ++num_of_fields;
    }
  }
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if(symbol_table[i].tspecifier_token == FIELD){
      const bool is_dead = int_vec_contains(symbol_table[i].tqualifiers,DEAD);
      if(!is_dead) continue;
      push(&field_names, symbol_table[i].identifier);
      const char* name = symbol_table[i].identifier;
      const bool is_aux  = int_vec_contains(symbol_table[i].tqualifiers,AUXILIARY);
      const bool is_comm = int_vec_contains(symbol_table[i].tqualifiers,COMMUNICATED);
      field_is_auxiliary[num_of_fields]    = is_aux;
      field_is_communicated[num_of_fields] = is_comm;
      num_of_communicated_fields           += is_comm;
      num_of_alive_fields                  += (!is_dead);
      field_is_dead[num_of_fields]         = is_dead;
      ++num_of_fields;
    }
  }
  const size_t num_of_dead_fields = num_of_fields - num_of_alive_fields;
  free_int_vec(&field_remappings);
  for(size_t field = 0; field < num_fields; ++field)
  {
	  const char* new_name = field_names.data[field];
	  const int old_index  = str_vec_get_index(original_names,new_name);
	  push_int(&field_remappings,old_index);
  }
  fprintf(fp, "typedef enum {");
  //TP: IMPORTANT!! if there are dead fields NUM_VTXBUF_HANDLES is equal to alive fields not all fields.  
  //TP: the compiler is allowed to move dead field declarations till the end
  //TP: this way the user can easily loop all alive fields with the old 0:NUM_VTXBUF_HANDLES and same for the Astaroth library dead fields are skiped over automatically
  for(size_t i = 0; i < num_of_fields; ++i)
	  fprintf(fp,"%s,",field_names.data[i]);

  const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
  if(has_optimization_info)
  	fprintf(fp, "NUM_FIELDS=%ld,", num_of_alive_fields);
  else
  	fprintf(fp, "NUM_FIELDS=%ld,", num_of_fields);
  fprintf(fp, "NUM_ALL_FIELDS=%ld,", num_of_fields);
  fprintf(fp, "NUM_DEAD_FIELDS=%ld,", num_of_fields-num_of_alive_fields);
  fprintf(fp, "NUM_COMMUNICATED_FIELDS=%d,", num_of_communicated_fields);
  fprintf(fp, "} Field;\n");

  fprintf(fp,"static const int field_remappings[] = {");
  {
	  for(size_t field = 0; field < field_remappings.size; ++field)
		  fprintf(fp,"%d,",field_remappings.data[field]);
  }
  fprintf(fp, "};");

  fprintf(fp, "static const bool vtxbuf_is_auxiliary[] = {");

  for(size_t i = 0; i < num_of_fields; ++i)
    if(field_is_auxiliary[i])
        fprintf(fp, "%s,", "true");
    else
        fprintf(fp, "%s,", "false");
  fprintf(fp, "};");

  fprintf(fp, "static const bool vtxbuf_is_communicated[] = {");
  for(size_t i = 0; i < num_of_fields; ++i)
    if(field_is_communicated[i])
        fprintf(fp, "%s,", "true");
    else
        fprintf(fp, "%s,", "false");

  fprintf(fp, "};");

  fprintf(fp, "static const bool vtxbuf_is_alive[] = {");

  for(size_t i = 0; i < num_of_fields; ++i)
    if(!field_is_dead[i])
        fprintf(fp, "%s,", "true");
    else
        fprintf(fp, "%s,", "false");
  fprintf(fp, "};");

  FILE* fp_vtxbuf_is_comm_func = fopen("vtxbuf_is_communicated_func.h","w");
  fprintf(fp_vtxbuf_is_comm_func ,"static __device__ constexpr __forceinline__ bool is_communicated(Field field) {\n"
             "switch(field)"
             "{");
  for(size_t i=0;i<num_of_alive_fields;++i)
  {
    const char* ret_val = (field_is_communicated[i]) ? "true" : "false";
    fprintf(fp_vtxbuf_is_comm_func,"case(%s): return %s;\n", field_names.data[i], ret_val);
  }
  fprintf(fp_vtxbuf_is_comm_func,"default: return false;\n");
  fprintf(fp_vtxbuf_is_comm_func, "}\n}\n");

  fclose(fp_vtxbuf_is_comm_func);

  fp = fopen("field_names.h","w");
  fprintf(fp,"static const char* field_names[] __attribute__((unused)) = {");
  for(size_t i=0;i<num_of_fields;++i)
	  fprintf(fp,"\"%s\",",field_names.data[i]);
  fprintf(fp,"};\n");
  fprintf(fp, "static const char** vtxbuf_names = field_names;\n");
  fclose(fp);

  fp = fopen("get_vtxbufs_funcs.h","w");
  for(size_t i = 0; i < num_of_fields; ++i)
  	fprintf(fp,"VertexBufferHandle acGet%s() {return %s;}\n", field_names.data[i], field_names.data[i]);
  fp = fopen("get_vtxbufs_declares.h","w");
  for(size_t i = 0; i < num_of_fields; ++i)
	fprintf(fp,"FUNC_DEFINE(VertexBufferHandle, acGet%s,());\n",field_names.data[i]);	
  fp = fopen("get_vtxbufs_loads.h","w");
  for(size_t i = 0; i < num_of_fields; ++i)
  {
	char* func_name;
	asprintf(&func_name,"acGet%s",field_names.data[i]);
	gen_dlsym(fp,func_name);
	free(func_name);
  }
  fclose(fp);
}
// Generate User Defines
static void
gen_user_defines(const ASTNode* root, const char* out)
{
  FILE* fp = fopen(out, "w");
  assert(fp);

  fprintf(fp, "#pragma once\n");

  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, fp);

  num_fields = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier_token == FIELD)
      ++num_fields;


  num_kernels = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    num_kernels += (symbol_table[i].tspecifier_token == KERNEL);

  num_dfuncs = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    num_dfuncs += ((symbol_table[i].type & NODE_DFUNCTION_ID) != 0);

  
  // Stencils
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier_token == STENCIL)

      fprintf(fp, "stencil_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_STENCILS} Stencil;");

  //TP: fields info is generated separately since it is different between 
  //analysis generation and normal generation
  fprintf(fp,"\n#include \"fields_info.h\"\n");
  // Enums

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
    if (symbol_table[i].tspecifier_token == KERNEL)
      fprintf(fp, "KERNEL_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_KERNELS} AcKernel;");

  fprintf(fp, "static const bool skip_kernel_in_analysis[NUM_KERNELS] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier_token == KERNEL)
    {
      if (int_vec_contains(symbol_table[i].tqualifiers,UTILITY))
	      fprintf(fp,"true,");
      else
	      fprintf(fp,"false,");
    }
  fprintf(fp, "};");
  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  // START---------------------------


  // Enum strings (convenience)
  gen_names("stencil",STENCIL,fp);
  gen_names("work_buffer",WORK_BUFFER,fp);
  gen_names("kernel",KERNEL,fp);
  //TP: field names have to be generated differently since they might get reorder because of dead fields
  //gen_names("field", FIELD,fp);
  fprintf(fp,"\n#include \"field_names.h\"\n");


  for (size_t i = 0; i < s_info.user_structs.size; ++i)
  {
	  char res[7000];
	  sprintf(res,"std::string to_str(const %s value)\n"
		       "{\n"
		       "std::string res = \"{\";"
		       "std::string tmp;\n"
		       ,s_info.user_structs.data[i]);

	  for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
	  {
		const char* middle = (j < s_info.user_struct_field_names[i].size -1) ? "res += \",\";\n" : "";
		strcatprintf(res,"res += to_str(value.%s);\n"
				"%s"
		,s_info.user_struct_field_names[i].data[j],middle);
	  }
	  strcat(res,
			  "res += \"}\";\n"
			  "return res;\n"
			  "}\n"
	  );
	  file_append("to_str_funcs.h",res);
	  const char* name = s_info.user_structs.data[i];
	  if(!strcmp(name,"AcReal2"))
	  	sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], "real2");
	  else if(!strcmp(name,"AcReal3"))
	  	sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], "real3");
	  else if(!strcmp(name,"AcReal4"))
	  	sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], "real4");
	  else if(!strcmp(name,"AcComplex"))
	  	sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], "complex");
	  else if(!strcmp(name,"AcBool3"))
	  	sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], "bool3");
	  else
	  	sprintf(res,"template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], s_info.user_structs.data[i]);
	  file_append("to_str_funcs.h",res);
  }


  string_vec datatypes = get_all_datatypes();

  for (size_t i = 0; i < datatypes.size; ++i)
  {
	  const char* datatype = datatypes.data[i];
	  gen_param_names(fp,datatype);
	  gen_enums(fp,datatype);

	  gen_dmesh_declarations(datatype);
	  gen_array_declarations(datatype,root);
	  gen_comp_declarations(datatype);
  }

  fprintf(fp,"\n#include \"array_info.h\"\n");
  fprintf(fp,"\n#include \"taskgraph_enums.h\"\n");

  free_str_vec(&datatypes);
  free_structs_info(&s_info);



  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  fprintf(fp, "\n// Redefined for backwards compatibility START\n");
  fprintf(fp, "#define NUM_VTXBUF_HANDLES (NUM_FIELDS)\n");
  fprintf(fp, "typedef Field VertexBufferHandle;\n");
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
  char cwd[9000];
  cwd[0] = '\0';
  char* err = getcwd(cwd, sizeof(cwd));
  assert(err != NULL);
  char autotune_path[10004];
  sprintf(autotune_path,"%s/autotune.csv",cwd);
  fprintf(fp,"__attribute__((unused)) static const char* autotune_csv_path= \"%s\";\n",autotune_path);

  char runtime_path[10004];
  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build/src/core/libastaroth_core.so");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_path = \"%s\";\n",runtime_path);

  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build/src/core/kernels/libkernels.so");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_runtime_path = \"%s\";\n",runtime_path);

  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build/src/utils/libastaroth_utils.so");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_utils_path = \"%s\";\n",runtime_path);

  sprintf(runtime_path,"%s",AC_BASE_PATH"/runtime_compilation/build");
  fprintf(fp,"__attribute__((unused)) static const char* runtime_astaroth_build_path = \"%s\";\n",runtime_path);

  fprintf(fp,"__attribute__((unused)) static const char* acc_compiler_path = \"%s\";\n", ACC_COMPILER_PATH);

  fclose(fp);

  //Done to refresh the autotune file when recompiling DSL code
  fp = fopen(autotune_path,"w");
  fclose(fp);

  fp = fopen("user_constants.h","w");
  FILE* fp_non_scalar_constants = fopen("user_non_scalar_constants.h","w");
  gen_const_variables(root,fp,fp_non_scalar_constants);
  fclose(fp);
  fclose(fp_non_scalar_constants);
}


static void
gen_user_kernels(const char* out)
{
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

  const char* default_param_list=  "(const int3 start, const int3 end, VertexBufferArray vba";
  FILE* fp_dec = fopen("user_kernel_declarations.h","a");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier_token == KERNEL)
      fprintf(fp_dec, "void __global__ %s %s);\n", symbol_table[i].identifier, default_param_list);

  fprintf(fp_dec, "static const Kernel kernels[] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier_token == KERNEL)
      fprintf(fp_dec, "%s,", symbol_table[i].identifier); // Host layer handle
  fprintf(fp_dec, "};");

  fclose(fp_dec);

  // Astaroth 2.0 backwards compatibility END

  fclose(fp);
}

void
replace_dynamic_coeffs_stencilpoint(ASTNode* node)
{
  TRAVERSE_PREAMBLE(replace_dynamic_coeffs_stencilpoint);
  if(!node->buffer) return;
  if(!check_symbol(NODE_VARIABLE_ID, node->buffer, REAL, DCONST_QL) && !check_symbol(NODE_VARIABLE_ID, node->buffer, INT, DCONST_QL)) return;
  //replace with zero to compile the stencil
  node->buffer = strdup("NAN");
  node->prefix=strdup("AcReal(");
  node->postfix = strdup(")");
}
void replace_dynamic_coeffs(ASTNode* node)
{
  TRAVERSE_PREAMBLE(replace_dynamic_coeffs);
  if(node->type != NODE_STENCIL) return;
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
//void
//rename_identifiers(const char* new_name, ASTNode* node, const char* str_to_check)
//{
//	if(node->lhs)
//		rename_identifiers(new_name,node->lhs,str_to_check);
//	if(node->rhs)
//		rename_identifiers(new_name,node->rhs,str_to_check);
//	if(!node->buffer)                    return;
//	if(node->token != IDENTIFIER)        return;
//	if(node->type & NODE_FUNCTION_CALL)  return;
//	if(check_symbol(NODE_ANY, node->buffer, "Stencil", NULL))    return;
//	if(check_symbol(NODE_FUNCTION_ID, node->buffer, NULL, NULL)) return;
//	if(strcmp(node->buffer,str_to_check))  return;
//	if(strstr(node->buffer,"AC_INTERNAL")) return;
//	free(node->buffer);
//	node->buffer = strdup(new_name);
//}
//void
//rename_local_vars(const char* str_to_append, ASTNode* node, ASTNode* root)
//{
//	if(node->lhs)
//		rename_local_vars(str_to_append,node->lhs,root);
//	if(node->rhs)
//		rename_local_vars(str_to_append,node->rhs,root);
//	if(!(node->type & NODE_DECLARATION)) return;
//	const ASTNode* func = get_parent_node(NODE_FUNCTION,node);
//	if(!func) return;
//	const char* func_name = get_node_by_token(IDENTIFIER,func->lhs)->buffer;
//	const char* name = get_node_by_token(IDENTIFIER,node)->buffer;
//	char* new_name = malloc(10000*sizeof(char));
//	sprintf(new_name,"%s__AC_INTERNAL_%s",name,func_name);
//	rename_identifiers(new_name,root,name);
//	free(new_name);
//}

void
rename_identifiers(ASTNode* node, const char* old_name, const char* new_name)
{
	if(node->lhs)
		rename_identifiers(node->lhs,old_name,new_name);
	if(node->rhs)
		rename_identifiers(node->rhs,old_name,new_name);
	if(node->token != IDENTIFIER) return;
	if(strcmp(node->buffer,old_name)) return;
	printf("renamed :%s\n",old_name);
	free(node->buffer);
	node->buffer = strdup(new_name);
}
void
append_to_identifiers(const char* str_to_append, ASTNode* node, const char* str_to_check)
{
	if(node->lhs)
		append_to_identifiers(str_to_append,node->lhs,str_to_check);
	if(node->rhs)
		append_to_identifiers(str_to_append,node->rhs,str_to_check);
	if(do_not_rename(node,str_to_check)) return;
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

	TRAVERSE_PREAMBLE(gen_dfunc_internal_names_recursive);
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
  gen_dfunc_internal_names_recursive(root);
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
const ASTNode*
get_dfunc(const char* name)
{
		for(size_t i = 0; i < dfunc_names.size; ++i)
			if(!strcmp(name,dfunc_names.data[i])) return dfunc_nodes.data[i];
		return NULL;
}
static ASTNode*
create_assignment(const ASTNode* lhs, const ASTNode* assign_expr, const char* op);
void
add_to_node_list(ASTNode* head, const ASTNode* new_node)
{
	while(head->rhs) head = head->lhs;
	ASTNode* last_elem = head->lhs;
	ASTNode* new_last = astnode_create(NODE_UNKNOWN,astnode_dup(new_node,NULL),NULL);
	ASTNode* node = astnode_create(NODE_UNKNOWN,new_last,last_elem);
	*head = *node;
}
//void
//inline_dfuncs_recursive(ASTNode* node)
//{
//	TRAVERSE_PREAMBLE(inline_dfuncs_recursive);
//	if(!(node->type & NODE_FUNCTION_CALL) || !node->lhs ) return;
//	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
//	const ASTNode* dfunc = get_dfunc(func_name);
//	if(!dfunc) return;
//	if(!dfunc->rhs->rhs->lhs) return;
//	if(get_node_by_token(RETURN,dfunc->rhs->rhs)) return;
//	ASTNode* new_dfunc = astnode_dup(dfunc,NULL);
//	ASTNode* dfunc_statements = new_dfunc->rhs->rhs->lhs;
//	if(!func_name || !check_symbol(NODE_DFUNCTION_ID,func_name,0,INLINE)) return;
//	ASTNode* statement = (ASTNode*) get_parent_node(NODE_STATEMENT_LIST_HEAD,node);
//	node_vec params = get_nodes_in_list(node->rhs);
//	func_params_info params_info = get_function_param_types_and_names(dfunc,func_name);
//	for(size_t i = 0; i < params.size; ++i)
//	{
//		ASTNode* copy_assignment = create_assignment(params_info.expr.data[i],params.data[i],"=");
//		add_to_node_list(dfunc_statements,copy_assignment);
//	}
//	char tmp[10000];
//	combine_all(new_dfunc->rhs->rhs,tmp);
//  	gen_multidimensional_field_accesses_recursive(new_dfunc->rhs->rhs);
//	node->parent->lhs = new_dfunc->rhs->rhs;
//	free_func_params_info(&params_info);
//	free_node_vec(&params);
//}
//void
//inline_dfuncs(ASTNode* node)
//{
//  traverse(node,NODE_NO_OUT,NULL);
//  inline_dfuncs_recursive(node);
//}
void
transform_arrays_to_std_arrays(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_arrays_to_std_arrays);
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
typedef struct
{
	string_vec* kernel_combinatorial_params;
	string_vec kernels_with_input_params;
	param_combinations params;
} combinatorial_params_info;

combinatorial_params_info
get_combinatorial_params_info(const ASTNode* root)
{
  string_vec* user_kernel_combinatorial_params = malloc(sizeof(string_vec)*100);
  memset(user_kernel_combinatorial_params,0,100*sizeof(string_vec));
  string_vec user_kernels_with_input_params = VEC_INITIALIZER;
  int* nums = malloc(sizeof(int)*100);
  memset(nums,0,sizeof(int)*100);
  string_vec* vals = malloc(sizeof(string_vec)*MAX_KERNELS*MAX_COMBINATIONS);
  param_combinations param_in = {nums, vals};
  gen_kernel_num_of_combinations(root,param_in,&user_kernels_with_input_params,user_kernel_combinatorial_params);
  return (combinatorial_params_info){user_kernel_combinatorial_params, user_kernels_with_input_params, param_in};
}
void
free_combinatorial_params_info(combinatorial_params_info* info)
{
  free_str_vec(&info->kernels_with_input_params);
  free(info->params.vals);
  for(int i = 0; i < 100; ++i)
	  free_str_vec(&info->kernel_combinatorial_params[i]);
}
void
gen_kernel_combinatorial_optimizations_and_input(ASTNode* root, const bool optimize_conditionals)
{
  combinatorial_params_info info = get_combinatorial_params_info(root);
  if(optimize_conditionals)
  {
  	gen_kernel_ifs(root,info.params,info.kernels_with_input_params,info.kernel_combinatorial_params);
	gen_optimized_kernel_decls(root,info.params,info.kernels_with_input_params,info.kernel_combinatorial_params);
  }
  free_combinatorial_params_info(&info);
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
	//size of an AcArray that needs to be known at compile time
	if(!strcmp(node->buffer,"size")) return true;
	res &= node->is_constexpr;
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
count_num_of_assignments_to_lhs(const ASTNode* node, const char* lhs, int* res)
{
	if(node->lhs)
		count_num_of_assignments_to_lhs(node->lhs,lhs,res);
	if(node->rhs)
		count_num_of_assignments_to_lhs(node->rhs,lhs,res);
	if(node->type == NODE_ASSIGNMENT && node->rhs)
	  *res += !strcmp(get_node_by_token(IDENTIFIER,node->lhs)->buffer,lhs);

}

static
bool is_return_node(const ASTNode* node)
{
	const bool res = 
		!node->lhs ? false :
		node->lhs->token == RETURN;
	return res;
}

bool
gen_constexpr_info_base(ASTNode* node)
{
	bool res = false;
	if(node->type & NODE_GLOBAL)
		return res;
	if(node->lhs)
		res |= gen_constexpr_info_base(node->lhs);
	if(node->rhs)
		res |= gen_constexpr_info_base(node->rhs);
	if(node->token == IDENTIFIER && node->buffer && !node->is_constexpr)
	{
		node->is_constexpr |= check_symbol(NODE_ANY,node->buffer,0,CONST_QL);
		//if array access that means we are accessing the vtxbuffer which obviously is not constexpr
 		if(!get_parent_node(NODE_ARRAY_ACCESS,node))
			node->is_constexpr |= check_symbol(NODE_VARIABLE_ID,node->buffer,FIELD,0);
		node->is_constexpr |= check_symbol(NODE_DFUNCTION_ID,node->buffer,FIELD,CONSTEXPR);
		res |= node->is_constexpr;
	}
	if(node->type & NODE_IF && all_identifiers_are_constexpr(node->lhs) && !node->is_constexpr)
	{
		//TP: simplification for now only consider conditionals that are not in nested scopes
		const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
		if(begin_scope->parent->parent->type & NODE_FUNCTION)
		{
			node->is_constexpr = true;
			node->prefix= strdup(" constexpr (");
			if(node->rhs->lhs->type & NODE_BEGIN_SCOPE)
				asprintf(&node->rhs->lhs->prefix,"{executed_conditionals.push_back(%d);",node->id);
		}
	}
	//TP: below sets the constexpr value of lhs the same as rhs for: lhs = rhs
	//TP: we restrict to the case that lhs is assigned only once in the function since full generality becomes too hard 
	//TP: However we get far with this approach since we turn many easy cases to SSA form which this check covers
	if(node->type &  NODE_ASSIGNMENT && node->rhs && get_parent_node(NODE_FUNCTION,node))
	{

	  ASTNode* func_base = (ASTNode*) get_parent_node(NODE_FUNCTION,node);
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
	if(is_return_node(node) && !node->is_constexpr)
	{
		if(all_identifiers_are_constexpr(node->rhs))
		{
			const ASTNode* dfunc_start = get_parent_node(NODE_DFUNCTION,node);
			const char* func_name = get_node(NODE_DFUNCTION_ID,dfunc_start)->buffer;
			Symbol* func_sym = (Symbol*)get_symbol(NODE_DFUNCTION_ID,func_name,NULL);
			if(func_sym)
			{
				push_int(&func_sym->tqualifiers,CONSTEXPR);
				node->is_constexpr = true;
				res |= node->is_constexpr;
			}
		}
	}
	return res;
}

void
gen_constexpr_info(ASTNode* root)
{
	bool has_changed = true;
	while(has_changed) has_changed = gen_constexpr_info_base(root);
}

bool
gen_type_info_base(ASTNode* node, const ASTNode* root)
{
	bool res = false;
	if(node->type & NODE_GLOBAL)
		return res;
	if(node->lhs)
		res |= gen_type_info_base(node->lhs,root);
	if(node->rhs)
		res |= gen_type_info_base(node->rhs,root);
	if(node->expr_type) return res;
	if(is_return_node(node))
	{
		const char* expr_type = get_expr_type(node->rhs);
		if(expr_type)
		{
			node->expr_type = strdup(expr_type);
			ASTNode* dfunc_start = (ASTNode*) get_parent_node(NODE_DFUNCTION,node);
			const char* func_name = get_node(NODE_DFUNCTION_ID,dfunc_start)->buffer;
			Symbol* func_sym = (Symbol*)get_symbol(NODE_DFUNCTION_ID,func_name,NULL);
			const int token = str_to_qualifier(expr_type);
			if(func_sym && !int_vec_contains(func_sym->tqualifiers,token))
				dfunc_start -> expr_type = expr_type;
		}
	}
	if(node->type & (NODE_PRIMARY_EXPRESSION | NODE_FUNCTION_CALL) ||
		(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node)) ||
		(node->type & NODE_EXPRESSION && all_primary_expressions_and_func_calls_have_type(node)) ||
		(node->type & NODE_ASSIGNMENT && node->rhs && get_parent_node(NODE_FUNCTION,node) &&  !get_node(NODE_MEMBER_ID,node->lhs))
	)
		get_expr_type(node);
	res |=  node -> expr_type != NULL;
	return res;
}
void
gen_type_info(ASTNode* root)
{
  	transform_arrays_to_std_arrays(root);
  	if(dfunc_nodes.size == 0)
  		get_dfunc_nodes(root,&dfunc_nodes,&dfunc_names);
	bool has_changed = true;
	int iter = 0;
	while(has_changed)
	{
		has_changed = gen_type_info_base(root,root);
	}
}
const ASTNode*
find_dfunc_start(const ASTNode* node, const char* dfunc_name)
{
	if(node->type & NODE_DFUNCTION && get_node(NODE_DFUNCTION_ID,node) && get_node(NODE_DFUNCTION_ID,node)->buffer && !strcmp(get_node(NODE_DFUNCTION_ID,node)->buffer, dfunc_name)) return node;
	const ASTNode* lhs_res = !node->lhs ? NULL :
		find_dfunc_start(node->lhs,dfunc_name);
	return lhs_res ? lhs_res :
	       node->rhs ? find_dfunc_start(node->rhs,dfunc_name) :
	       NULL;
}
char*
get_mangled_name(const char* dfunc_name, const string_vec types)
{
		char* tmp;
		asprintf(&tmp,"%s_AC_MANGLED_NAME_",dfunc_name);
		for(size_t i = 0; i < types.size; ++i)
			asprintf(&tmp,"%s_%s",tmp,types.data[i]);
		tmp = realloc(tmp,sizeof(char)*(strlen(tmp) + 5*types.size));
		replace_substring(&tmp,"*","ARRAY");
		return tmp;
}
void
mangle_dfunc_names(ASTNode* node, string_vec* dst, int* counters)
{
	if(node->lhs)
		mangle_dfunc_names(node->lhs,dst,counters);
	if(node->rhs)
		mangle_dfunc_names(node->rhs,dst,counters);
	if(!(node->type & NODE_DFUNCTION))
		return;
	const int dfunc_index = str_vec_get_index(duplicate_dfuncs.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	if(dfunc_index == -1)
		return;
	const char* dfunc_name = duplicate_dfuncs.names.data[dfunc_index];
	const int overload_index = counters[dfunc_index];
	func_params_info params_info = get_function_param_types_and_names(node,dfunc_name);
	counters[dfunc_index]++;
	for(size_t i = 0; i < params_info.types.size; ++i)
	{
		push(&dst[overload_index + MAX_DFUNCS*dfunc_index], params_info.types.data[i]);
	}
	free(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	get_node_by_token(IDENTIFIER,node->lhs)->buffer = get_mangled_name(dfunc_name,params_info.types);
	free_func_params_info(&params_info);
}
static bool
compatible_types(const char* a, const char* b)
{
	return !strcmp(a,b) 
	       || (!strcmp(a,"AcReal*") && strstr(b,"AcArray") && strstr(b,"AcReal")) ||
	          (!strcmp(b,"AcReal*") && strstr(a,"AcArray") && strstr(a,"AcReal")) ||
            ((!strcmp(a,"Field") && !strcmp(b,"VertexBufferHandle")) || (!strcmp(b,"Field") && !strcmp(a,"VertexBufferHandle")))
		;
}
bool
//resolve_overloaded_calls(ASTNode* node, const char* dfunc_name, string_vec* dfunc_possible_types,const int dfunc_index)
resolve_overloaded_calls(ASTNode* node, string_vec* dfunc_possible_types)
{
	bool res = false;
	if(node->lhs)
		res |= resolve_overloaded_calls(node->lhs,dfunc_possible_types);
	if(node->rhs)
		res |= resolve_overloaded_calls(node->rhs,dfunc_possible_types);
	if(!(node->type & NODE_FUNCTION_CALL))
		return res;
	if(!get_node_by_token(IDENTIFIER,node->lhs))
		return res;
	const int dfunc_index = str_vec_get_index(duplicate_dfuncs.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	if(dfunc_index == -1)
		return res;
	const char* dfunc_name = duplicate_dfuncs.names.data[dfunc_index];
	func_params_info call_info = get_func_call_params_info(node);
	if(!strcmp(dfunc_name,"dot") && call_info.types.size == 2 && !strcmp_null_ok(call_info.types.data[0],"AcRealArray") && !strcmp_null_ok(call_info.types.data[1],"AcRealArray"))
	{
		get_node_by_token(IDENTIFIER,node->lhs)->buffer = strdup("AC_dot");
		return true;
	}
	int correct_types = -1;
	int overload_index = MAX_DFUNCS*dfunc_index-1;
	int_vec possible_indexes = VEC_INITIALIZER;
  int param_offset = 0;
	while(dfunc_possible_types[++overload_index].size > 0)
	{
		bool possible = true;
    //TP: ugly hack to resolve calls in BoundConds
    if(!strcmps(call_info.expr.data[0],
            "BOUNDARY_X_TOP",
            "BOUNDARY_X_BOT",
            "BOUNDARY_Y_TOP",
            "BOUNDARY_Y_BOT",
            "BOUNDARY_Z_TOP",
            "BOUNDARY_Z_BOT",
            "BOUNDARY_X",
            "BOUNDARY_Y",
            "BOUNDARY_Z"
    ))
    {
      param_offset = 1;
    }
		if(call_info.types.size - param_offset != dfunc_possible_types[overload_index].size) continue;
		for(size_t i = param_offset; i < call_info.types.size; ++i)
		{
			const char* func_type = dfunc_possible_types[overload_index].data[i-param_offset];
			const char* call_type = call_info.types.data[i];
			//The upper one is the less strict resolver and below is the more strict resolver
			possible &= !call_type || !func_type || compatible_types(func_type,call_type);
		}
		if(possible)
			push_int(&possible_indexes,overload_index);
	}
	bool able_to_resolve = possible_indexes.size == 1;
	if(!able_to_resolve) { 
		//if(!strcmp(dfunc_name,"bc_sym_x"))
		//{
		//	char my_tmp[10000];
		//	my_tmp[0] = '\0';
		//	combine_all(node->rhs,my_tmp); 
		//	printf("Not able to resolve: %s,%d\n",my_tmp,param_offset); 
		//	printf("Not able to resolve: %s,%d\n",call_info.types.data[4],param_offset); 
		//}
		return res;
	}
	{
		free(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
		//TP: do not have to strdup since tmp is not used after
		//TP: tmp should really always point to the src buffer but if I do that
		//TP: for some reason the code slows down
		const string_vec types = dfunc_possible_types[possible_indexes.data[0]];
		get_node_by_token(IDENTIFIER,node->lhs)->buffer = get_mangled_name(dfunc_name,types);

	}

	free_str_vec(&call_info.expr);
	free_str_vec(&call_info.types);
	free_int_vec(&possible_indexes);
	return true;
}

void
gen_overloads(ASTNode* root)
{
  bool overloaded_something = true;
  string_vec dfunc_possible_types[MAX_DFUNCS * duplicate_dfuncs.names.size];
  memset(dfunc_possible_types,0,sizeof(string_vec)*MAX_DFUNCS*duplicate_dfuncs.names.size);
  int counters[duplicate_dfuncs.names.size];
  memset(counters,0,sizeof(int)*duplicate_dfuncs.names.size);
  mangle_dfunc_names(root,dfunc_possible_types,counters);
  
  symboltable_reset();
  traverse(root, NODE_NO_OUT, NULL);
  int iter = 0;
  while(overloaded_something)
  {
	overloaded_something = false;
  	gen_type_info(root);
	overloaded_something |= resolve_overloaded_calls(root,dfunc_possible_types);
  	//for(size_t i = 0; i < duplicate_dfuncs.size; ++i)
  	        //overloaded_something |= resolve_overloaded_calls(root,duplicate_dfuncs.data[i],dfunc_possible_types,i);
  }
  for(size_t i = 0; i < MAX_DFUNCS*duplicate_dfuncs.names.size; ++i)
	  free_str_vec(&dfunc_possible_types[i]);
}
static ASTNode*
create_binary_op_expr(const char* op)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,NULL,NULL);
	res->token = BINARY_OP;
	res->buffer = strdup(op);
	return res;
}
static ASTNode*
create_binary_expression(ASTNode* expression , ASTNode* unary_expression, const char* op)
{

	ASTNode* rhs = astnode_create(NODE_UNKNOWN,
					create_binary_op_expr(op),
					unary_expression);
	return 
		astnode_create(NODE_BINARY_EXPRESSION,expression,rhs);
}

static ASTNode* 
create_func_call(const char* func_name, const ASTNode* param)
{

	ASTNode* postfix_expression = astnode_create(NODE_UNKNOWN,
			     create_primary_expression(func_name),
			     NULL);
	ASTNode* func_call = astnode_create(NODE_FUNCTION_CALL,postfix_expression,astnode_dup(param,NULL));
	astnode_set_infix("(",func_call); 
	astnode_set_postfix(")",func_call); 
	return func_call;
}

static ASTNode*
create_choose_node(ASTNode* lhs_value, ASTNode* rhs_value)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,lhs_value,rhs_value);	
	astnode_set_prefix("?", res);
	astnode_set_prefix(": ",res->rhs);
	return res;
}
static ASTNode*
create_assign_op(const char* op)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,NULL,NULL);
	res->buffer = strdup(op);
	res->token = ASSIGNOP;
	return res;
}

static ASTNode*
create_assignment_body(const ASTNode* assign_expr, const char* op)
{
	ASTNode* expression_list = astnode_create(NODE_UNKNOWN,astnode_dup(assign_expr,NULL),NULL);
	return astnode_create(NODE_UNKNOWN,
				astnode_dup(create_assign_op(op),NULL),
				astnode_dup(expression_list,NULL)
				);
}
static ASTNode*
create_declaration(const char* identifier)
{
	ASTNode* empty       = astnode_create(NODE_UNKNOWN,NULL,NULL);
	ASTNode* decl_vars   = astnode_create(NODE_UNKNOWN,create_identifier_node(identifier),NULL);
	ASTNode* declaration = astnode_create(NODE_DECLARATION,empty,decl_vars);
	return declaration;
}

static ASTNode*
create_assignment(const ASTNode* lhs, const ASTNode* assign_expr, const char* op)
{

	ASTNode* res = 	
		astnode_create(NODE_ASSIGNMENT,
			      astnode_dup(lhs,NULL),
			      create_assignment_body(assign_expr,op)
			      );
	astnode_set_postfix(";",res);
	return res;
}
static ASTNode*
create_ternary_expr(ASTNode* conditional, ASTNode* lhs_value, ASTNode* rhs_value)
{

	return astnode_create(NODE_TERNARY_EXPRESSION,
			      astnode_dup(conditional,NULL),
			      create_choose_node(astnode_dup(lhs_value,NULL),astnode_dup(rhs_value,NULL))
			);
}

void
transform_field_intrinsic_func_calls_recursive(ASTNode* node, const ASTNode* root)
{
	if(node->lhs)
		transform_field_intrinsic_func_calls_recursive(node->lhs,root);
	if(node->rhs)
		transform_field_intrinsic_func_calls_recursive(node->rhs,root);
	if(node->type != NODE_FUNCTION_CALL) return;
	const ASTNode* identifier_node = get_node_by_token(IDENTIFIER,node->lhs);
	if(!identifier_node) return;
	const char* func_name = identifier_node->buffer;
	const Symbol* sym = get_symbol(NODE_FUNCTION_ID, func_name, NULL);
	if(!sym) return;
	if(!int_vec_contains(sym -> tqualifiers,REAL)) return;
	func_params_info param_info = get_func_call_params_info(node);
	if(!strcmp_null_ok(param_info.types.data[0],"Field") && param_info.expr.size == 1)
	{
		ASTNode* func_call = create_func_call("value",node->rhs);
		ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
		ASTNode* expression         = astnode_create(NODE_EXPRESSION,unary_expression,NULL);
		ASTNode* expression_list = astnode_create(NODE_UNKNOWN,expression,NULL);

		node->rhs = expression_list;
	}
	free_func_params_info(&param_info);
}

void
transform_field_unary_ops(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_field_unary_ops);
	if(!node_is_unary_expr(node)) return;
	const char* base_expr = get_expr_type(node->rhs);
	const char* unary_op = get_node_by_token(UNARY_OP,node->lhs)->buffer;
	if(strcmps(unary_op,"+","-")) return;
	if(strcmp_null_ok(base_expr,"Field") && strcmp_null_ok(base_expr,"Field3")) return;

	ASTNode*  func_call = create_func_call("value",node->rhs);
	ASTNode*  unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
	node->rhs = unary_expression;
}
void
transform_field_binary_ops(ASTNode* node)
{
	if(node->lhs)
		transform_field_binary_ops(node->lhs);
	if(node->rhs)
		transform_field_binary_ops(node->rhs);
	if(!node_is_binary_expr(node)) return;

	const char* lhs_expr = get_expr_type(node->lhs);
	const char* rhs_expr = get_expr_type(node->rhs);
	const char* op = node->rhs->lhs->buffer;
        if(!op) return;
        if(strcmps(op,"+","-","/","*")) return;

	if(!strcmp_null_ok(lhs_expr,"Field") || !strcmp_null_ok(lhs_expr,"Field3"))
	{

		ASTNode* func_call = create_func_call("value",node->lhs);
		ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
		ASTNode* expression         = astnode_create(NODE_EXPRESSION,unary_expression,NULL);
		node->lhs = expression;
	}
	if(!strcmp_null_ok(rhs_expr,"Field") || !strcmp_null_ok(rhs_expr,"Field3"))
	{

		ASTNode* func_call = create_func_call("value",node->rhs->rhs);
		ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
		node->rhs->rhs = unary_expression;
	}
}
void
gen_extra_func_definitions_recursive(const ASTNode* node, const ASTNode* root, FILE* stream)
{
	if(node->lhs)
		gen_extra_func_definitions_recursive(node->lhs,root,stream);
	if(node->rhs)
		gen_extra_func_definitions_recursive(node->rhs,root,stream);
	if(node->type != NODE_DFUNCTION) return;
	const char* dfunc_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	func_params_info info = get_function_param_types_and_names(node,dfunc_name);
	if(!has_qualifier(node,"elemental")) return;
	if(info.expr.size == 1 && !strcmp_null_ok(info.types.data[0], "AcReal") && !strstr(dfunc_name,"AC_INTERNAL_COPY"))
	{
		const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);

		fprintf(stream,"%s_AC_INTERNAL_COPY (real %s){%s}\n",dfunc_name,info.expr.data[0],func_body);
		fprintf(stream,"%s (real3 v){return real3(%s_AC_INTERNAL_COPY(v.x), %s_AC_INTERNAL_COPY(v.y), %s_AC_INTERNAL_COPY(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		fprintf(stream,"%s (Field field){return %s_AC_INTERNAL_COPY(value(field))}\n",dfunc_name,dfunc_name);
		fprintf(stream,"%s (Field3 v){return real3(%s(v.x), %s(v.y), %s(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		fprintf(stream,"%s(real[] arr){\nreal res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s_AC_INTERNAL_COPY(arr[i])\nreturn arr\n}\n",dfunc_name,dfunc_name);
	}
	else if(info.expr.size == 1 && !strcmp_null_ok(info.types.data[0], "Field") && !strstr(dfunc_name,"AC_INTERNAL_COPY"))
	{
		if(!strcmp_null_ok(node->expr_type,"AcReal"))
		{
			const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);
			fprintf(stream,"%s_AC_INTERNAL_COPY (Field %s){%s}\n",dfunc_name,info.expr.data[0],func_body);
			fprintf(stream,"%s (Field3 v){return real3(%s_AC_INTERNAL_COPY(v.x), %s_AC_INTERNAL_COPY(v.y), %s_AC_INTERNAL_COPY(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		}
		else if(!strcmp_null_ok(node->expr_type,"AcReal3"))
		{
			const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);

			fprintf(stream,"%s_AC_INTERNAL_COPY (Field %s){%s}\n",dfunc_name,info.expr.data[0],func_body);
			fprintf(stream,"%s (Field3 v){return Matrix(%s_AC_INTERNAL_COPY(v.x), %s_AC_INTERNAL_COPY(v.y), %s_AC_INTERNAL_COPY(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		}
		else
		{
			fprintf(stderr,FATAL_ERROR_MESSAGE"Missing elemental case for func: %s\n",dfunc_name);
			exit(EXIT_FAILURE);
		}
	}
	else if(info.expr.size == 2 && !strcmp_null_ok(info.types.data[0], "AcReal3") && !strcmp_null_ok(info.types.data[1], "AcReal3") && !strstr(dfunc_name,"AC_INTERNAL_COPY"))
	{
		const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);
		fprintf(stream,"%s_AC_INTERNAL_COPY (real3 %s, real3 %s){%s}\n",dfunc_name,info.expr.data[0],info.expr.data[1],func_body);
		fprintf(stream,"%s (Field3 field, real3 vec){return %s_AC_INTERNAL_COPY(value(field), vec) }\n",dfunc_name,dfunc_name);
		fprintf(stream,"%s (real3  vec,   Field3 field){return %s_AC_INTERNAL_COPY(vec, value(field)) }\n",dfunc_name,dfunc_name);
		fprintf(stream,"%s (Field3 a, Field3 b){return %s_AC_INTERNAL_COPY(value(a), value(b)) }\n",dfunc_name,dfunc_name);
	}
	free_func_params_info(&info);
}

void
transform_field_intrinsic_func_calls_and_ops(ASTNode* root)
{
  	traverse(root, NODE_NO_OUT, NULL);
	transform_field_unary_ops(root);
	transform_field_intrinsic_func_calls_recursive(root,root);
	transform_field_binary_ops(root);
}
void
gen_extra_funcs(const ASTNode* root_in, FILE* stream)
{
	push(&tspecifier_mappings,"int");
	push(&tspecifier_mappings,"AcReal");
	push(&tspecifier_mappings,"bool");
  	ASTNode* root = astnode_dup(root_in,NULL);

	symboltable_reset();
  	traverse_base(root, 0, NULL, true);
        duplicate_dfuncs = get_duplicate_dfuncs(root);

  	assert(root);
  s_info = read_user_structs(root);
	e_info = read_user_enums(root);
	gen_type_info(root);
	gen_extra_func_definitions_recursive(root,root,stream);
	free_str_vec(&duplicate_dfuncs.names);
	free_int_vec(&duplicate_dfuncs.counts);
  free_structs_info(&s_info);

}
void gen_boundcond_kernels(const ASTNode* root_in, FILE* stream)
{
    ASTNode* root = astnode_dup(root_in,NULL);
          symboltable_reset();
        traverse(root, 0, NULL);
    s_info = read_user_structs(root);
    e_info = read_user_enums(root);
    duplicate_dfuncs = get_duplicate_dfuncs(root);
    gen_overloads(root);
    make_unique_bc_calls((ASTNode*) root);
    gen_dfunc_bc_kernels(root,root,stream);
    free_structs_info(&s_info);
    //free_str_vec(&duplicate_dfuncs);
}
void
canonalize_assignments(ASTNode* node)
{
	TRAVERSE_PREAMBLE(canonalize_assignments);
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	const char* function_name = get_node_by_token(IDENTIFIER,function->lhs)->buffer;
	char* op = strdup(node->rhs->lhs->buffer);
	if(strcmps(op,"*=","-=","+=","/="))   return;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return;
	ASTNode* assign_expression = node->rhs->rhs->lhs;
	remove_substring(op,"=");
	ASTNode* binary_expression = create_binary_expression(node->lhs, assign_expression, op);
	ASTNode* assignment        = create_assignment(node->lhs, binary_expression, "="); 
	assignment->parent = node->parent;
	node->parent->lhs = assignment;
}
const ASTNode*
get_node_decl(const ASTNode* node, const char* var_name)
{
	const ASTNode* identifier = (node->type & NODE_DECLARATION && node->rhs)
				    ?  get_node_by_token(IDENTIFIER,node->rhs)
				    :  NULL;
	if(identifier && !strcmp(identifier->buffer,var_name)) return node;
	const ASTNode* lhs_res = node->lhs ? get_node_decl(node->lhs,var_name) : NULL;
	if(lhs_res) return lhs_res;
	const ASTNode* rhs_res = node->rhs ? get_node_decl(node->rhs,var_name) : NULL;
	if(rhs_res) return rhs_res;
	return NULL;
}

bool is_first_decl(const ASTNode* node, const ASTNode* begin_scope)
{
	const ASTNode* first_decl = get_node_decl(begin_scope,get_node_by_token(IDENTIFIER,node)->buffer);
	return first_decl ? first_decl->id == node->id : true;
}
bool
remove_unnecessary_assignments(ASTNode* node)
{
	if(!(node->type & NODE_ASSIGNMENT)) return false;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return false;
	const char* function_name = get_node_by_token(IDENTIFIER,function->lhs)->buffer;
	const ASTNode* if_node = get_parent_node(NODE_IF,node);
	if(!if_node) return false;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return false;
	if(strcmp(node->rhs->lhs->buffer,"="))   return false;
	if(!(if_node->rhs->lhs->type & NODE_BEGIN_SCOPE) && is_first_decl(node->lhs,function->rhs->rhs))
	{
		//remove the if node
		if(!if_node->rhs->rhs)
		{
			if_node->parent->parent->lhs = NULL;
			if_node->parent->parent->rhs = NULL;
		}
		else
		{
				if_node->parent->parent->rhs = if_node->rhs->rhs;
				if_node->rhs->rhs ->parent = if_node->parent->parent;
		}
		return true;
	}
	return false;
}

void
convert_to_ternary(ASTNode* node)
{
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const char* var_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	const char* function_name  = get_node_by_token(IDENTIFIER,function->lhs)->buffer;
	const ASTNode* if_node     = get_parent_node(NODE_IF,node);
	if(!if_node) return;
	if(!if_node->rhs->lhs) return;
	if(if_node->rhs->lhs->type & NODE_BEGIN_SCOPE) return;
	if(!get_node_by_id(node->id,if_node->rhs->lhs)) return;
	const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
	if(!begin_scope) return;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return;
	if(strcmp(node->rhs->lhs->buffer,"="))   return;
	if(get_node_by_token(ELIF,if_node)) return;
	if(if_node->parent->lhs->token == ELIF) return;

	ASTNode* assign_expression = node->rhs->rhs->lhs;
	ASTNode* conditional       = if_node->lhs;
	if(get_node_by_token(ELSE,if_node))
	{
		if(if_node->rhs->rhs->lhs->token != ELSE) return;
		const ASTNode* else_node = if_node->rhs->rhs;
		if(else_node->rhs->type & NODE_BEGIN_SCOPE) return;
		const ASTNode* second_assign = get_node(NODE_ASSIGNMENT,else_node);
		if(!second_assign) return;
		const char* second_var_name = get_node_by_token(IDENTIFIER,second_assign->lhs)->buffer;
		if(strcmp(second_var_name,var_name)) return;
		if(count_num_of_nodes_in_list(second_assign->rhs->rhs) != 1)   return;
		if(strcmp(second_assign->rhs->lhs->buffer,"="))   return;
		ASTNode* second_assign_expr = second_assign->rhs->rhs->lhs;
		//same as below except now : condition is the else condition
		//ASTNode* ternary_expr = create_ternary_expr(conditional, assign_expression ,second_assign_expr);
		ASTNode* ternary_expr = create_ternary_expr(conditional, assign_expression,second_assign_expr);
		ASTNode* assignment =   create_assignment(node->lhs,ternary_expr,"=");
		assignment->parent = if_node->parent->parent;
		if_node->parent->parent->lhs = assignment;

		return;
	}


	ASTNode* ternary_expr = create_ternary_expr(conditional, assign_expression ,create_primary_expression(var_name));
	ASTNode* assignment =   create_assignment(node->lhs,ternary_expr,"=");
	assignment->parent = if_node->parent->parent;
	if_node->parent->parent->lhs = assignment;
}
	

void
canonalize_if_assignments(ASTNode* node)
{
	//TP: we check for code like if [else] (cond){ y = x } where y is not declared in the surrounding scope
	//Then the assignment can not have an effect can the assignment can be removed
	//We also check for assignments like if (cond) {y = x} with no other cases. This can be translated to:
	//y = (cond) ? x : y;
	//This makes analyses like constexpr inference easier since it is clear which values y can be assigned to
	//And this is always safe to do for performance since x can be evaluated only if cond so the compiler has to generate IR that looks like the following:
	//if(cond)
	//  y = x
	//else
	//  y = y
	//And then of course it sees the second write as a no-op and we get back to the code we started with. 
	//Now we also convert if(cond) {y = x} else {y = z} ---> y = (cond) ? x : z;
	TRAVERSE_PREAMBLE(canonalize_if_assignments);
	const bool removed = remove_unnecessary_assignments(node);
	if(!removed)
		convert_to_ternary(node);

}
bool
is_used_in_statements(const ASTNode* head, const char* var)
{
	while(head->type == NODE_STATEMENT_LIST_HEAD)
	{
		if (get_node_by_buffer(var,head->rhs)) return true;
		head = head->parent;
	}
	return false;
}
void
remove_dead_writes(ASTNode* node)
{
	TRAVERSE_PREAMBLE(remove_dead_writes);
	const ASTNode* func = get_parent_node(NODE_FUNCTION,node);
	if(!func) return;
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const char* var_name = strdup(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
	if(begin_scope -> id != func->rhs->rhs->id) return;
	const ASTNode* head = get_parent_node(NODE_STATEMENT_LIST_HEAD,node);
	const bool final_node = is_left_child(NODE_STATEMENT_LIST_HEAD,node);
	const bool is_used_in_rest = is_used_in_statements(final_node ? head : head->parent,var_name);
	ASTNode* primary_expr = get_node_by_token(IDENTIFIER,node->lhs)->parent;
	const char* expr_type = get_expr_type(primary_expr);
	const char* primary_identifier = primary_expr->lhs->buffer;
	if(!is_used_in_rest && expr_type && 
	   //exclude written fields since they write to the vertex buffer
	   strcmp(expr_type,"Field") && 
	   //exclude writes to dynamic global arrays since they persist after the kernel
	   !check_symbol(NODE_VARIABLE_ID,primary_identifier,0,DYNAMIC_QL)
	)
	{
		node->parent->lhs = NULL;
	}
}
void
gen_ssa_in_basic_blocks(ASTNode* node)
{
	TRAVERSE_PREAMBLE(gen_ssa_in_basic_blocks);
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	ASTNode* head = (ASTNode*) get_parent_node(NODE_STATEMENT_LIST_HEAD,node);
	if(!head) return;
	const ASTNode* decl = get_node(NODE_DECLARATION,node->lhs);
	const int n_variables_declared = count_num_of_nodes_in_list(decl->rhs);
	if(n_variables_declared != 1) return;
	if(get_node(NODE_MEMBER_ID,decl->rhs)) return;
	if(get_node(NODE_ARRAY_ACCESS,decl->rhs)) return;
	const char* function_name = get_node_by_token(IDENTIFIER,function->lhs)->buffer;
	const char* var_name = strdup(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	const ASTNode* first_decl = get_node_decl(function,var_name);
	const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
	if(!get_node_by_id(first_decl->id,begin_scope)) return;
	char* new_name = malloc((strlen(var_name)+1)*sizeof(char));
	sprintf(new_name,"%s_",var_name);
	const bool final_node = is_left_child(NODE_STATEMENT_LIST_HEAD,node);
	rename_while(
					NODE_STATEMENT_LIST_HEAD,
					final_node ? head : head->parent,
					new_name,node->expr_type, var_name
				      );
	get_node_by_token(IDENTIFIER,node->lhs)->buffer = strdup(new_name);
	free(new_name);
}
void
canonalize(ASTNode* node)
{
	canonalize_assignments(node);
	//canonalize_if_assignments(node);
}
void
preprocess(ASTNode* root, const bool optimize_conditionals)
{
  free_node_vec(&dfunc_nodes);
  free_str_vec(&dfunc_names);
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  canonalize(root);

  transform_field_intrinsic_func_calls_and_ops(root);
  traverse(root, 0, NULL);
  //We use duplicate dfuncs from gen_boundcond_kernels
  //duplicate_dfuncs = get_duplicate_dfuncs(root);
  gen_overloads(root);
  mark_kernel_inputs(root);
  gen_kernel_combinatorial_optimizations_and_input(root,optimize_conditionals);
  free_structs_info(&s_info);
}

static size_t
count_stencils()
{
  size_t num_stencils = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier_token == STENCIL)
      ++num_stencils;
  return num_stencils;
}

void
stencilgen(ASTNode* root)
{
  const size_t num_stencils = count_stencils();
  FILE* stencilgen = fopen(STENCILGEN_HEADER, "w");
  assert(stencilgen);

  // Stencil ops
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
       if(symbol.tspecifier_token == STENCIL) {
	      if(symbol.tqualifiers.size)
	      {
		if(symbol.tqualifiers.size > 1)
		{
			fprintf(stderr,"Stencils are supported only with a single type qualifier\n");
			exit(EXIT_FAILURE);
		}
        	fprintf(stencilgen, "\"%s\",",qualifier_to_str(symbol.tqualifiers.data[0]));
	      }
	      else
        	fprintf(stencilgen, "\"sum\",");
      }
    }
    fprintf(stencilgen, "};");
  }

  // Stencil coefficients
  char* stencil_coeffs;
  size_t file_size;
  FILE* stencil_coeffs_fp = open_memstream(&stencil_coeffs, &file_size);
  //Fill symbol table
  traverse(root,
           NODE_VARIABLE_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           stencil_coeffs_fp);
  fflush(stencil_coeffs_fp);

  replace_dynamic_coeffs(root);
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
}

//These are the same for mem_accesses pass and normal pass
void
gen_output_files(ASTNode* root)
{
  traverse(root, 0, NULL);
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);

  file_append("user_typedefs.h","#include \"func_attributes.h\"\n");
  gen_user_enums();
  gen_user_structs();
  gen_user_defines(root, "user_defines.h");
  gen_kernel_structs(root);
  FILE* fp = fopen("user_kernel_declarations.h","w");
  fclose(fp);
  gen_user_kernels("user_declarations.h");
  fp = fopen("user_kernel_declarations.h","a");
  fclose(fp);
  stencilgen(root);

}
bool
eliminate_conditionals_base(ASTNode* node)
{
	bool res = false;
	if(node->lhs)
		res |= eliminate_conditionals_base(node->lhs);
	if(node->rhs)
		res |= eliminate_conditionals_base(node->rhs);
	if(node->type & NODE_IF && node->is_constexpr)
	{
		const bool is_executed = int_vec_contains(executed_conditionals,node->id);
		const bool is_elif = get_node_by_token(ELIF,node->parent->lhs)  != NULL;
		if(is_executed)
		{
			//TP: now we know that this constexpr conditional is taken
			//TP: this means that its condition has to be always true given its constexpr nature, thus if the previous conditionals were not taken this is always taken
			//TP: since we iterate the conditionals in order we can remove the conditionals on the right that can not be taken
			res |= node->rhs->rhs != NULL;
			node->rhs->rhs = NULL;
			//TP: if is not elif this is the base case and there is only a single redundant check left
			if(!is_elif)
			{
				node->rhs->lhs->parent = node->parent->parent;
				node->parent->parent->lhs = node->rhs->lhs;
				return true;
			}
		}
		else
		{
			const bool has_more_cases = node->rhs->rhs && node->rhs->rhs->lhs->token == ELIF;
			if(has_more_cases)
			{
				ASTNode* elif = node->rhs->rhs;
				ASTNode* elif_if_statement = elif->rhs;
				elif_if_statement->parent = node->parent;
				node->parent->rhs = elif_if_statement;
			}
			//Else is the only possibility take it
			else if(node->rhs->rhs->lhs->token == ELSE)
			{
				ASTNode* else_node = node->rhs->rhs;
				ASTNode* statement = node->parent->parent;
				statement->lhs = else_node->rhs;
				else_node->rhs->parent = statement;
			}
			return true;
		}
	}
	return res;

}
void
eliminate_conditionals(ASTNode* node)
{
	eliminate_conditionals_base(node);
	bool process = true;
	while(process)
		process = eliminate_conditionals_base(node);
}


void
clean_stream(FILE* stream)
{
	freopen(NULL,"w",stream);
}


void
gen_analysis_stencils(FILE* stream)
{
  string_vec stencil_names = get_names(STENCIL);
  for (size_t i = 0; i < stencil_names.size; ++i)
    fprintf(stream,"AcReal %s(const Field& field_in)"
           "{stencils_accessed[field_in][stencil_%s]=1;return AcReal(1.0);};\n",
           stencil_names.data[i], stencil_names.data[i]);
  free_str_vec(&stencil_names);
}

//void                     
//reorder_dead_fields_last(ASTNode* node)
//{
//	TRAVERSE_PREAMBLE(reorder_dead_fields_last);
//	if(!(node->type & NODE_DECLARATION)) return;
//	if(!(node->type & NODE_GLOBAL))      return;
//	const ASTNode* tspec = get_node(NODE_TSPEC,node->lhs);
//	if(!tspec)  return;
//	if(tspec->lhs->token != FIELD) return;
//	node_vec identifiers = get_nodes_in_list(node->rhs);
//	for(size_t field = 0; field < identifiers.size; field++)
//	{
//		const Symbol* sym = get_symbol_token(NODE_VARIABLE_ID,get_node_by_token(IDENTIFIER,node)->buffer, FIELD);
//		if(int_vec_contains(sym->tqualifiers,DEAD))
//		{
//			ASTNode* identifier = (ASTNode*) identifiers.data[field];
//			ASTNode* copy_node = astnode_dup(identifiers.data[field],NULL);
//			identifier->lhs=NULL;
//			identifier->rhs=NULL;
//			printf("FOUND DEAD FIELD\n");
//		}
//	}
//	free_node_vec(&identifiers);
//
//}
//
void
check_array_dim_identifiers(const char* id, const ASTNode* node)
{
	if(node->lhs) check_array_dim_identifiers(id,node->lhs);
	if(node->rhs) check_array_dim_identifiers(id,node->rhs);

	if(node->type & NODE_BINARY_EXPRESSION && node->rhs && node->rhs->lhs && get_node_by_token(IDENTIFIER,node))
	{
		fprintf(stderr,FATAL_ERROR_MESSAGE"Only arithmetic expressions consisting of const integers allowed in global array dimensions\n");
		fprintf(stderr,"Wrong expression (%s) in dimension of variable: %s\n\n",combine_all_new(node),id);
		exit(EXIT_FAILURE);
	}

	if(node->token != IDENTIFIER) return;
	const bool is_int_var = check_symbol(NODE_VARIABLE_ID,node->buffer,INT,0);
	if(!is_int_var)
	{
		fprintf(stderr,FATAL_ERROR_MESSAGE"Only dconst and const integer variables allowed in array dimensions\n");
		fprintf(stderr,"Wrong dimension (%s) for variable: %s\n\n",node->buffer,id);
		exit(EXIT_FAILURE);
	}

}
void
check_array_dimensions(const char* id, const ASTNode* node)
{
	if(node->lhs) check_array_dimensions(id,node->lhs);
	if(node->rhs) check_array_dimensions(id,node->rhs);

	if(!(node->type & NODE_ARRAY_ACCESS)) return;
	check_array_dim_identifiers(id,node->rhs);
}
void
check_global_array_dimensions(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(check_global_array_dimensions);
	if(!(node->type & NODE_DECLARATION)) return;
	if(!(node->type & NODE_GLOBAL)) return;
	node_vec vars = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < vars.size; ++i)
	{
		const ASTNode* first_access = get_node(NODE_ARRAY_ACCESS,vars.data[i]);
		if(!first_access) continue;
		check_array_dimensions(get_node_by_token(IDENTIFIER,first_access->lhs)->buffer, first_access);
	}
	free_node_vec(&vars);
}

void
debug_prints(const ASTNode* node)
{
	(void) node;
	/**
	TRAVERSE_PREAMBLE(debug_prints);
	if(node->type == NODE_ASSIGNMENT)
	{
		if(strstr(combine_all_new(node), "uu_addition"))
			printf("HMM: %s\n",combine_all_new(node));
	}
	**/
}
void
gen_stencils(const bool gen_mem_accesses, FILE* stream)
{
  const size_t num_stencils = count_stencils();
  if (gen_mem_accesses || !OPTIMIZE_MEM_ACCESSES) {
    FILE* tmp = fopen("stencil_accesses.h", "w+");
    assert(tmp);
    fprintf(tmp,
            "static int "
            "stencils_accessed[NUM_KERNELS][NUM_ALL_FIELDS][NUM_STENCILS] = {");
    for (size_t i = 0; i < num_kernels; ++i)
      for (size_t j = 0; j < num_fields; ++j)
        for (size_t k = 0; k < num_stencils; ++k)
          fprintf(tmp, "[%lu][%lu][%lu] = 1,", i, j, k);
    fprintf(tmp, "};");

    fprintf(tmp,
            "static int "
            "previous_accessed[NUM_KERNELS][NUM_ALL_FIELDS] = {");
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
              "stencils_accessed[NUM_KERNELS][NUM_ALL_FIELDS][NUM_STENCILS] = {");
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
}

char**
get_dfunc_strs(const ASTNode* root)
{
  char** dfunc_strs = malloc(sizeof(char*)*num_dfuncs);
  FILE* dfunc_fps[num_dfuncs];
  const ASTNode* dfunc_heads[num_dfuncs];
  size_t sizeloc;
  for(size_t i = 0; i < num_dfuncs; ++i)
  {

	const Symbol* dfunc_symbol = get_symbol_by_index(NODE_DFUNCTION_ID,i,0);
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
	fclose(dfunc_fps[i]);
  }
  return dfunc_strs;
}

void
generate(const ASTNode* root_in, FILE* stream, const bool gen_mem_accesses, const bool optimize_conditionals)
{ 
  (void)optimize_conditionals;
  symboltable_reset();
  ASTNode* root = astnode_dup(root_in,NULL);
  //preprocess(root);
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  gen_type_info(root);
  //Used to help in constexpr deduction
  if(gen_mem_accesses)
  {
  	//gen_ssa_in_basic_blocks(root);
  	gen_constexpr_info(root);
	//remove_dead_writes(root);
  }

  traverse(root, NODE_NO_OUT, NULL);
  check_global_array_dimensions(root);

  gen_multidimensional_field_accesses_recursive(root,gen_mem_accesses);



  // Fill the symbol table
  gen_user_taskgraphs(root);
  combinatorial_params_info info = get_combinatorial_params_info(root);
  gen_kernel_input_params(root,info.params.vals,info.kernels_with_input_params,info.kernel_combinatorial_params);
  replace_boolean_dconsts_in_optimized(root,info.params.vals,info.kernels_with_input_params,info.kernel_combinatorial_params);
  free_combinatorial_params_info(&info);
  gen_kernel_postfixes_and_reduce_outputs(root,gen_mem_accesses);


  // print_symbol_table();

  // Generate user_kernels.h
  fprintf(stream, "#pragma once\n");





  // Device constants
  // gen_dconsts(root, stream);
  traverse(root, NODE_NO_OUT, NULL);
  {
  	  const bool has_optimization_info = written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
	  //if(has_optimization_info) reorder_dead_fields_last(root);
          FILE* fp = fopen("fields_info.h","w");
          gen_field_info(fp);
          fclose(fp);

	  symboltable_reset();
  	  traverse(root, NODE_NO_OUT, NULL);
	  string_vec datatypes = get_all_datatypes();

  	  FILE* fp_info = fopen("array_info.h","w");
  	  fprintf(fp_info,"\n #ifdef __cplusplus\n");
  	  fprintf(fp_info,"\n#include <array>\n");
  	  fprintf(fp_info,"typedef struct {std::array<int,3>  len; std::array<bool,3> from_config;} AcArrayDims;\n");
  	  fprintf(fp_info,"typedef struct { bool is_dconst; int d_offset; int num_dims; AcArrayDims dims; const char* name; bool is_alive;} array_info;\n");
  	  for (size_t i = 0; i < datatypes.size; ++i)
  	  	  gen_array_info(fp_info,datatypes.data[i],root);
  	  fprintf(fp_info,"\n #endif\n");
  	  fclose(fp_info);

	  //TP: !IMPORTANT! gen_array_info will temporarily update the nodes to push DEAD type qualifiers to dead gmem arrays.
	  //This info is used in gen_gmem_array_declarations so they should be called after each other, maybe will simply combine them into a single function
  	  for (size_t i = 0; i < datatypes.size; ++i)
	  	gen_gmem_array_declarations(datatypes.data[i],root);
  }
  const char* array_datatypes[] = {"int","AcReal","bool","int3","AcReal3","long","long long"};
  for (size_t i = 0; i < sizeof(array_datatypes)/sizeof(array_datatypes[0]); ++i)
  	gen_array_reads(root,root,array_datatypes[i]);

  // Stencils

  // Stencil generator

  // Compile
  gen_stencils(gen_mem_accesses,stream);


  // Device functions
  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
  char** dfunc_strs = get_dfunc_strs(root);
	
  // Kernels
  symboltable_reset();
  gen_kernels(root, dfunc_strs, gen_mem_accesses);
  for(size_t i = 0; i < num_dfuncs; ++i)
  	free(dfunc_strs[i]);
  free(dfunc_strs);

  //gen_dfunc_internal_names(root);
  //inline_dfuncs(root);

  symboltable_reset();
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           stream);
  if(gen_mem_accesses)
  {
	  fflush(stream);
	  //This is used to eliminate known constexpr conditionals
	  //get_executed_conditionals();
	  //eliminate_conditionals(root);


	  //clean_stream(stream);

  	  //traverse(root,
          // NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
          //     NODE_HOSTDEFINE | NODE_NO_OUT,
          // stream);
  }

  // print_symbol_table();
  //free(written_fields);
  //free(read_fields);
  //free(field_has_stencil_op);
  //written_fields       = NULL;
  //read_fields          = NULL;
  //field_has_stencil_op = NULL;

  free_structs_info(&s_info);

}


void
compile_helper(void)
{
  format_source("user_kernels.h.raw","user_kernels.h");
  system("cp user_kernels.h user_kernels_backup.h");
  FILE* analysis_stencils = fopen("analysis_stencils.h", "w");
  gen_analysis_stencils(analysis_stencils);
  fclose(analysis_stencils);
  printf("Compiling %s...\n", STENCILACC_SRC);
#if AC_USE_HIP
  printf("--- USE_HIP: `%d`\n", AC_USE_HIP);
#else
  printf("--- USE_HIP not defined\n");
#endif
  printf("--- ACC_RUNTIME_API_DIR: `%s`\n", ACC_RUNTIME_API_DIR);
  printf("--- GPU_API_INCLUDES: `%s`\n", GPU_API_INCLUDES);

#if AC_USE_HIP
  const char* use_hip = "-DAC_USE_HIP=1 ";
#else
  const char* use_hip = "";
#endif
  char cmd[4096];
  const char* api_includes = strlen(GPU_API_INCLUDES) > 0 ? " -I " GPU_API_INCLUDES  " " : "";
  sprintf(cmd, "gcc -Wshadow -I. -I " ACC_RUNTIME_API_DIR " %s %s %s " 
	       STENCILACC_SRC " -lm -lstdc++ -o " STENCILACC_EXEC " "
  ,api_includes, use_hip, TWO_D ? "-DTWO_D=1 " : "-DTWO_D=0 "
  );
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
}

void
get_executed_conditionals(void)
{
	compile_helper();
  	FILE* proc = popen("./" STENCILACC_EXEC " -C", "r");
  	assert(proc);
  	pclose(proc);

  	free_int_vec(&executed_conditionals);
  	FILE* fp = fopen("executed_conditionals.bin","rb");
  	int size;
  	int tmp;
  	fread(&size, sizeof(int), 1, fp);
  	for(int i = 0; i < size; ++i)
  	{
  		fread(&tmp, sizeof(int), 1, fp);
  	      push_int(&executed_conditionals,tmp);
  	}
  	for(size_t i = 0; i < executed_conditionals.size; ++i)
  	        printf("HI: %d\n",executed_conditionals.data[i]);
  	fclose(fp);
}
void
generate_mem_accesses(void)
{
  compile_helper();
  // Generate memory accesses to a header
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
}


