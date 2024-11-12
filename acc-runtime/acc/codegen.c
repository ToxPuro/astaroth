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
#include <hashtable.h>
extern struct hashmap_s string_intern_hashmap;
#include <hash.h>
#include "codegen.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
static string_vec primitive_datatypes = VEC_INITIALIZER;
#define INT_STR       primitive_datatypes.data[0]
#define REAL_STR      primitive_datatypes.data[1]
#define BOOL_STR      primitive_datatypes.data[2]
#define LONG_STR      primitive_datatypes.data[3]
#define LONG_LONG_STR primitive_datatypes.data[4]
//TP: only one of these should be active at any time since either float or double equals AcReal
#define FLOAT_STR     primitive_datatypes.data[5]
#define DOUBLE_STR    primitive_datatypes.data[5]
#define MAX_ARRAY_RANK (10)
string_vec
get_prof_types()
{
  string_vec prof_types = VEC_INITIALIZER;
  push(&prof_types,intern("Profile<X>"));
  push(&prof_types,intern("Profile<Y>"));
  push(&prof_types,intern("Profile<Z>"));
  push(&prof_types,intern("Profile<XY>"));
  push(&prof_types,intern("Profile<XZ>"));
  push(&prof_types,intern("Profile<YX>"));
  push(&prof_types,intern("Profile<YZ>"));
  push(&prof_types,intern("Profile<ZX>"));
  push(&prof_types,intern("Profile<ZY>"));
  return prof_types;
}


static const char* REAL_ARR_STR = NULL;

static const char* REAL_PTR_STR = NULL;
static const char* REAL3_PTR_STR = NULL;
static const char* FIELD3_PTR_STR = NULL;
static const char* VTXBUF_PTR_STR = NULL;
static const char* FIELD_PTR_STR = NULL;
static const char* STENCIL_STR    = NULL;

static const char* MATRIX_STR   = NULL;
static const char* REAL3_STR    = NULL;
static const char* INT3_STR     = NULL;

static const char* COMPLEX_STR = NULL;

static const char* EQ_STR = NULL;
static const char* DOT_STR = NULL;
static const char* LESS_STR = NULL;
static const char* GREATER_STR  = NULL;

static const char* MULT_STR = NULL;
static const char* MINUS_STR = NULL;
static const char* PLUS_STR = NULL;
static const char* DIV_STR = NULL;

static const char* LEQ_STR = NULL;
static const char* GEQ_STR = NULL;
static const char* MEQ_STR= NULL;
static const char* AEQ_STR= NULL;
static const char* MINUSEQ_STR= NULL;
static const char* DEQ_STR= NULL;
static const char* PERIODIC = NULL;


static const char* VALUE_STR      = NULL;

static const char* DEAD_STR      = NULL;
static const char* AUXILIARY_STR      = NULL;
static const char* COMMUNICATED_STR      = NULL;

static const char* CONST_STR = NULL;
static const char* CONSTEXPR_STR = NULL;
static const char* OUTPUT_STR = NULL;
static const char* INPUT_STR = NULL;
static const char* GLOBAL_MEM_STR = NULL;
static const char* DYNAMIC_STR = NULL;
static const char* INLINE_STR = NULL;
static const char* UTILITY_STR = NULL;
static const char* ELEMENTAL_STR = NULL;
static const char* BOUNDCOND_STR = NULL;
static const char* FIXED_BOUNDARY_STR = NULL;
static const char* RUN_CONST_STR = NULL;
static const char* CONST_DIMS_STR = NULL;
static const char* DCONST_STR = NULL;

static const char* FIELD_STR      = NULL;
static const char* KERNEL_STR      = NULL;
static const char* FIELD3_STR      = NULL;
static const char* FIELD4_STR      = NULL;
static const char* PROFILE_STR      = NULL;

static const char* BOUNDARY_X_TOP_STR = NULL; 
static const char* BOUNDARY_X_BOT_STR = NULL; 

static const char* BOUNDARY_Y_BOT_STR = NULL; 
static const char* BOUNDARY_Y_TOP_STR = NULL; 

static const char* BOUNDARY_Z_BOT_STR = NULL; 
static const char* BOUNDARY_Z_TOP_STR = NULL; 


static const char* BOUNDARY_X_STR = NULL; 
static const char* BOUNDARY_Y_STR = NULL; 
static const char* BOUNDARY_Z_STR = NULL; 

static const char*  BOUNDARY_XY_STR  = NULL;   
static const char*  BOUNDARY_XZ_STR  = NULL;
static const char*  BOUNDARY_YZ_STR  = NULL;
static const char*  BOUNDARY_XYZ_STR = NULL;
const char*
type_output(const char* type)
{
	if(strstr(type,PROFILE_STR) && strstr(type,"<"))
		return PROFILE_STR;
	return type;
}

void
gen_dlsym(FILE* fp, const char* func_name)
{
	fprintf(fp,"*(void**)(&%s) = dlsym(handle,\"%s\");\n",func_name,func_name);
	fprintf(fp,"if(!%s) fprintf(stderr,\"Astaroth error was not able to load %s\\n\");\n",func_name,func_name);
}


void
get_executed_nodes(const int round);

#include "ast.h"
#include "tab.h"
#include <string.h>
#include <ctype.h>
extern string_vec const_ints;
extern string_vec const_int_values;
#include "expr.h"


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

static inline const char*
strupr(const char* src)
{
	char* res = strdup(src);
	int index = -1;
	while(res[++index] != '\0')
		res[index] = toupper(res[index]);
	return intern(res);
}
static inline const char*
to_upper_case(const char* src)
{
	char* res = strdup(src);
	res[0] = (res[0] == '\0') ? res[0] : toupper(res[0]);
	return intern(res);
}

static int* written_fields       = NULL;
static int* read_fields          = NULL;
static int* read_profiles        = NULL;
static int* reduced_profiles     = NULL;
static int* field_has_stencil_op = NULL;
static size_t num_fields   = 0;
static size_t num_profiles = 0;
static size_t num_kernels  = 0;
static size_t num_dfuncs   = 0;

static int_vec executed_nodes = VEC_INITIALIZER;
bool
is_called(const ASTNode* node)
{
	return int_vec_contains(executed_nodes,node->id);
}



#define STENCILGEN_HEADER "stencilgen.h"
#define STENCILGEN_SRC ACC_DIR "/stencilgen.c"
#define STENCILGEN_EXEC "stencilgen.out"
#define STENCILACC_SRC AC_BASE_PATH "/src/core/stencil_accesses.cpp"
#define STENCILACC_EXEC "acc_stencil_accesses.o"
#define ACC_RUNTIME_API_DIR ACC_DIR "/../api"
//



#define MAX_KERNELS (100)
#define MAX_FUNCS (1100)
#define MAX_COMBINATIONS (1000)
#define MAX_DFUNCS (1000)
// Symbols
#define MAX_ID_LEN (256)
typedef struct {
  NodeType type;
  //We keep this as int_vec since makes comparisons so much faster
  string_vec tqualifiers;
  const char* tspecifier;
  const char* identifier;
  } Symbol;


static string_vec tspecifier_mappings = VEC_INITIALIZER;


#define SYMBOL_TABLE_SIZE (65536)
static Symbol symbol_table[SYMBOL_TABLE_SIZE] = {};
#define MAX_NESTS (32)
static struct hashmap_s symbol_table_hashmap[MAX_NESTS];

static size_t num_symbols[MAX_NESTS] = {};
static int    nest_ids[MAX_NESTS] = {};
static size_t current_nest           = 0;


//arrays symbol table
#define MAX_NUM_ARRAYS (256)

static const Symbol*
symboltable_lookup_range(const char* identifier, const int start_range, const int end_range)
{
  if (!identifier)
    return NULL;

  {
	for(int i = start_range; i >= end_range; --i)
	{
  		Symbol* sym = (Symbol*)hashmap_get(&symbol_table_hashmap[i], identifier, strlen(identifier));
		if(sym) return sym;
	}
	return NULL;
  }
  return NULL;
}

static const Symbol*
symboltable_lookup(const char* identifier)
{
  return symboltable_lookup_range(identifier,current_nest,0);
}
static const Symbol*
symboltable_lookup_current_nest(const char* identifier)
{
  return symboltable_lookup_range(identifier,current_nest,current_nest);
}

static const Symbol*
symboltable_lookup_surrounding_scope(const char* identifier)
{
  return symboltable_lookup_range(identifier,current_nest-1,0);
}




static int
get_symbol_index(const NodeType type, const char* symbol, const char* tspecifier)
{

  int counter = 0;
  const char* search = symbol;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if ((!tspecifier || tspecifier == symbol_table[i].tspecifier) && symbol_table[i].type & type)
    {
	    if(symbol_table[i].identifier == search)
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
    if ((!tspecifier || symbol_table[i].tspecifier == tspecifier) && symbol_table[i].type & type)
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
  const Symbol* sym = (Symbol*)hashmap_get(&symbol_table_hashmap[0], symbol, strlen(symbol));
  if(sym && sym->type & type && (!tspecifier || sym->tspecifier == tspecifier))
	  return sym;
  return NULL;
}

static const Symbol*
get_symbol_token(const NodeType type, const char* symbol, const char* tspecifier)
{
  const Symbol* sym = (Symbol*)hashmap_get(&symbol_table_hashmap[0], symbol, strlen(symbol));
  if(sym && sym->type & type && (!tspecifier || sym->tspecifier == tspecifier))
	  return sym;
  return NULL;
}

#define REAL_SPECIFIER  (1 << 0)
#define INT_SPECIFIER   (1 << 1)
#define BOOL_SPECIFIER  (1 << 2)
#define REAL3_SPECIFIER (1 << 3)
#define REAL4_SPECIFIER (1 << 4)

static const char* EMPTY_STR      = NULL;

bool
has_optimization_info()
{
 return written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields;
}
string_vec
get_struct_field_types(const char* struct_name);
string_vec
get_struct_field_names(const char* struct_name);
bool
consists_of_types(const string_vec target_types, const char* curr_type)
{
	if(str_vec_contains(target_types,curr_type)) return true;
	string_vec types = get_struct_field_types(curr_type);
	if(types.size == 0)
	{
		free_str_vec(&types);
		return false;
	}
	bool res = true;
	for(size_t i = 0; i < types.size; ++i)
		res &= consists_of_types(target_types,types.data[i]);
	return res;

}
bool is_primitive_datatype(const char* type)
{
	return str_vec_contains(primitive_datatypes,type);
}
bool
all_real_struct(const char* struct_name)
{
	if(is_primitive_datatype(struct_name)) return false;
	string_vec target_types = VEC_INITIALIZER;
	push(&target_types,REAL_STR);
	const bool res = consists_of_types(target_types,struct_name);
	free_str_vec(&target_types);
	return res;
}
string_vec
get_allocating_types()
{
	static string_vec allocating_types = VEC_INITIALIZER;
	if(allocating_types.size == 0)
	{
		push(&allocating_types,intern("Field"));
		push(&allocating_types,intern("Field3"));
		push(&allocating_types,intern("Profile<X>"));
		push(&allocating_types,intern("Profile<Y>"));
		push(&allocating_types,intern("Profile<Z>"));
		push(&allocating_types,intern("Profile<XY>"));
		push(&allocating_types,intern("Profile<XZ>"));
		push(&allocating_types,intern("Profile<YX>"));
		push(&allocating_types,intern("Profile<YZ>"));
		push(&allocating_types,intern("Profile<ZX>"));
		push(&allocating_types,intern("Profile<ZY>"));
	}
	return allocating_types;
}
bool is_allocating_type(const char* type)
{
	return consists_of_types(get_allocating_types(),type);
}
static int 
add_symbol_base(const NodeType type, const char** tqualifiers, size_t n_tqualifiers, const char* tspecifier,
           const char* id, const char* postfix)
{
  if(is_number(id))
  {
	  printf("WRONG: %s\n",id);
	  exit(EXIT_FAILURE);
  }
  const char* dconst_ql[1] = {DCONST_STR};
  assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);
  symbol_table[num_symbols[current_nest]].type          = type;
  if(type == NODE_VARIABLE_ID && current_nest == 0 && n_tqualifiers == 0 && tspecifier)
  {
	  if(!is_allocating_type(tspecifier))
	  {
		  tqualifiers = dconst_ql;
		  n_tqualifiers = 1;
	  }
  }
  init_str_vec(&symbol_table[num_symbols[current_nest]].tqualifiers);
  for(size_t i = 0; i < n_tqualifiers; ++i)
      	push(&symbol_table[num_symbols[current_nest]].tqualifiers,tqualifiers[i]);


  if(tspecifier)
  	symbol_table[num_symbols[current_nest]].tspecifier =  tspecifier;
  else
	symbol_table[num_symbols[current_nest]].tspecifier = EMPTY_STR;
  symbol_table[num_symbols[current_nest]].identifier = id;
  if(postfix)
  {
	  char* full_name;
	  asprintf(&full_name,"%s_%s",id,postfix);
  	  symbol_table[num_symbols[current_nest]].identifier = intern(full_name);
	  free(full_name);
  }
  /**
  if(id != intern(id))
  {
	  printf("WRONG: %s\n",id);
	  char* str = NULL;
	  printf("HMM: %c\n",str[1000]);
  }
  **/
  hashmap_put(&symbol_table_hashmap[current_nest], id, strlen(id),(void*)&symbol_table[num_symbols[current_nest]]);




  const bool is_field_without_type_qualifiers = tspecifier && tspecifier == FIELD_STR && symbol_table[num_symbols[current_nest]].tqualifiers.size == 0;
  ++num_symbols[current_nest];
  if(!is_field_without_type_qualifiers)
  	return num_symbols[current_nest]-1;
	  
  if(!has_optimization_info())
  {
  	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, intern("Communicated"));
  	return num_symbols[current_nest]-1;
  } 



   const int field_index = get_symbol_index(NODE_VARIABLE_ID, id, FIELD_STR);
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
   	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, COMMUNICATED_STR);
   }
   if(is_auxiliary)
	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, AUXILIARY_STR);
   if(is_dead)
	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, DEAD_STR);


  //return the index of the lastly added symbol
  return num_symbols[current_nest]-1;
}
static int 
add_symbol(const NodeType type, const char** tqualifiers, const size_t n_tqualifiers, const char* tspecifier, const char* id)
{
	return add_symbol_base(type,tqualifiers,n_tqualifiers,tspecifier,id,NULL);
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

bool 
is_boundary_param(const char* param)
{
    return !strcmps(param,
            BOUNDARY_X_TOP_STR,
	    BOUNDARY_X_BOT_STR,
            BOUNDARY_Y_TOP_STR,
            BOUNDARY_Y_BOT_STR,
            BOUNDARY_Z_TOP_STR,
            BOUNDARY_Z_BOT_STR,
            BOUNDARY_X_STR,
            BOUNDARY_Y_STR,
            BOUNDARY_Z_STR,
            BOUNDARY_XY_STR,
            BOUNDARY_XZ_STR,
            BOUNDARY_YZ_STR,
            BOUNDARY_XYZ_STR
    );
}


string_vec
get_all_datatypes()
{
  string_vec datatypes = str_vec_copy(s_info.user_structs);

  for (size_t i = 0; i < primitive_datatypes.size; ++i)
	  push(&datatypes,primitive_datatypes.data[i]);

  user_enums_info enum_info = e_info;
  for (size_t i = 0; i < enum_info.names.size; ++i)
	  push(&datatypes,enum_info.names.data[i]);
  return datatypes;
}

const char*
convert_to_define_name(const char* name)
{
	if(name == LONG_LONG_STR)
		return intern("long_long");
	char* res = strdup(name);
	if(strlen(res) > 2 && res[0]  == 'A' && res[1] == 'c')
	{
		res = &res[2];
		res[0] = tolower(res[0]);
	}
	return intern(res);
}

static void
symboltable_reset(void)
{
  for(int i = 0; i < MAX_NESTS; ++i)
  {
  	hashmap_destroy(&symbol_table_hashmap[i]);
  	const unsigned initial_size = 2000;
  	hashmap_create(initial_size, &symbol_table_hashmap[i]);
  }
  for(size_t i = 0; i < SYMBOL_TABLE_SIZE ; ++i)
	  free_str_vec(&symbol_table[i].tqualifiers);

  current_nest              = 0;
  num_symbols[current_nest] = 0;

  const char* const_tq[1]  =  {intern("const")};
  const char* dynamic_tq[1] = {intern("dynamic")};


  // Add built-in variables (TODO consider NODE_BUILTIN)
  add_symbol(NODE_VARIABLE_ID, NULL, 0, NULL,  intern("stderr"));           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("print"));           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("fprintf"));           // TODO REMOVE

  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("threadIdx"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("blockIdx"));        // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("vertexIdx"));       // TODO REMOVE
  //add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("idx"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("tid"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("start"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("end"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR, intern("globalVertexIdx")); // TODO REMOVE

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("write_base"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_x"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_y"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_z"));  // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_xy"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_xz"));  // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_yx"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_yz"));  // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_zx"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("value_profile_zy"));  // TODO RECHECK
					      	 //
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_min_real"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_max_real"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real"));  // TODO RECHECK
					      			      //
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_x"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_y"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_z"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_xy"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_xz"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_yx"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_yz"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_zx"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_real_zy"));  // TODO RECHECK
					      			      //
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_min_int"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_max_int"));  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("reduce_sum_int"));  // TODO RECHECK
									      //
  add_symbol(NODE_FUNCTION_ID, NULL, 0,INT_STR, intern("size"));  // TODO RECHECK
  //In develop
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "read_w");
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write_w");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, INT_STR,   intern("len"));    // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, FIELD3_STR,intern("MakeField3")); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("uint64_t"));   // TODO RECHECK
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR, intern("UINT64_MAX")); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, REAL_STR, intern("rand_uniform"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, REAL_STR, REAL_STR);
  add_symbol(NODE_FUNCTION_ID, NULL, 0, REAL_STR, intern("previous_base"));  // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("multm2_sym"));   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("diagonal"));   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("sum"));   // TODO RECHECK

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_STR, intern("AC_REAL_PI"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_STR, intern("AC_REAL_EPSILON"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("NUM_FIELDS"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("NUM_PROFILES"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("NUM_VTXBUF_HANDLES"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("NUM_ALL_FIELDS"));

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("FIELD_IN"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("FIELD_OUT"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("IDX"));
  add_symbol(NODE_FUNCTION_ID, NULL, 0, REAL3_STR, REAL3_STR);

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, BOOL_STR, intern("true"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, BOOL_STR, intern("false"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("NGHOST"));

  // Astaroth 2.0 backwards compatibility START
  // (should be actually built-in externs in acc-runtime/api/acc-runtime.h)
  const char* tqualifiers[1] = {intern("dconst")};


  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR, intern("STENCIL_ORDER"));
  // Astaroth 2.0 backwards compatibility END
  int index = add_symbol(NODE_VARIABLE_ID, NULL, 0 , INT3_STR, intern("blockDim"));
  symbol_table[index].tqualifiers.size = 0;

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_X"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_Y"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_Z"));

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_XY"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_XZ"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_YZ"));

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR,  intern("BOUNDARY_XYZ"));

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("periodic"));

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
             symbol_table[i].tspecifier == KERNEL_STR ? "kernel" : "device");

    if (str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
      printf("(dconst)");

    if (symbol_table[i].tspecifier == STENCIL_STR)
      printf("(stencil)");

    printf("\n");
  }
  printf("---\n");
}



const char*
convert_to_enum_name(const char* name)
{
	static char res[4098];
	if(name == LONG_LONG_STR)
		return "AcLongLong";
	if(strstr(name,"Ac")) return name;
	sprintf(res,"Ac%s",to_upper_case(name));
	return res;
}
node_vec
get_array_accesses(const ASTNode* base)
{
	    node_vec dst = VEC_INITIALIZER;
	    if(!base) return dst;

	    node_vec nodes = VEC_INITIALIZER;
	    get_array_access_nodes(base,&dst);
	    return dst;
}
node_vec
get_array_var_dims(const char* var, const ASTNode* root)
{
	    const ASTNode* var_identifier = get_node_by_buffer_and_type(var,NODE_VARIABLE_ID,root);
	    const ASTNode* decl = get_parent_node(NODE_DECLARATION,var_identifier);

	    const ASTNode* access_start = get_node(NODE_ARRAY_ACCESS,decl);
	    return get_array_accesses(access_start);
}
const ASTNode*
get_var_val(const char* var, const ASTNode* node)
{
	if(node->lhs)
	{
		const ASTNode* lhs_val = get_var_val(var,node->lhs);
		if(lhs_val) return lhs_val;
	}
	if(node->type & NODE_ASSIGN_LIST && has_qualifier(node,"const") && get_node(NODE_TSPEC,node))
	{
		node_vec assignments = get_nodes_in_list(node->rhs);
		for(size_t i = 0; i < assignments.size; ++i)
		{
			const ASTNode* elem = assignments.data[i];
			const ASTNode* assignment = get_node(NODE_ASSIGNMENT,elem);
			const ASTNode* id = get_node_by_token(IDENTIFIER,assignment);
			if(!id) continue;
			const char* name = id->buffer;
			if(name == var)
				return assignment->rhs;
		}
		free_node_vec(&assignments);
	}
	if(node->rhs)
	{
		const ASTNode* rhs_val = get_var_val(var,node->rhs);
		if(rhs_val) return rhs_val;
	}
	return NULL;

}
const char*
get_const_int3_val(const char* name, const ASTNode* root, const char* member_id)
{
	const ASTNode* val = get_var_val(name,root);
	node_vec elems = get_nodes_in_list(get_node(NODE_STRUCT_INITIALIZER,val)->lhs);
	const char* res = NULL;
	if(!strcmp(member_id,"x"))
		res = combine_all_new(elems.data[0]);
	if(!strcmp(member_id,"y"))
		res = combine_all_new(elems.data[1]);
	if(!strcmp(member_id,"z"))
		res = combine_all_new(elems.data[2]);
	free_node_vec(&elems);
	return res;
}
const char*
get_expr_type(ASTNode* node);

bool
all_identifiers_are_constexpr(const ASTNode* node);

void
get_array_var_length(const char* var, const ASTNode* root, char* dst)
{
	    sprintf(dst,"%s","");
	    node_vec tmp = get_array_var_dims(var,root);
	    for(size_t i = 0; i < tmp.size; ++i)
	    {
		const char* all = intern(combine_all_new(tmp.data[i]));
	    	const char* val = NULL;
		if(all_identifiers_are_constexpr(tmp.data[i]) && (strstr(all,".y") || strstr(all,".x") || strstr(all,".z")))
		{
			const char* base   = get_node_by_token(IDENTIFIER,tmp.data[i])->buffer;
			const char* member = get_node(NODE_MEMBER_ID,tmp.data[i])->buffer;
			val = get_const_int3_val(base,root,member);
		}
		else
			val = all;
		strcatprintf(dst,"%s%s",(i) ? MULT_STR : "",val);
	    }
	    free_node_vec(&tmp);
}

int default_accesses[10000] = { [0 ... 10000-1] = 1};
int read_accesses[10000] = { [0 ... 10000-1] = 0};

const  int*
get_arr_accesses(const char* datatype_scalar)
{

	char* filename;
	const char* define_name =  convert_to_define_name(datatype_scalar);
	asprintf(&filename,"%s_arr_accesses",define_name);
  	if(!file_exists(filename) || !has_optimization_info() || !OPTIMIZE_ARRAYS)
		return default_accesses;

	FILE* fp = fopen(filename,"rb");
	int size = 1;
	fread(&size, sizeof(int), 1, fp);
	fread(read_accesses, sizeof(int), size, fp);
	fclose(fp);
	return read_accesses;
}


typedef struct
{
	bool is_dconst;
	node_vec dims;
	const char* AcArrayDimsStr;
	const char* name;
	bool is_alive;
	bool accessed;
} array_info;
bool
check_symbol(const NodeType type, const char* name, const char* tspecifier, const char* tqualifier)
{
  const Symbol* sym = get_symbol_token(type,name,tspecifier);
  return 
	  !sym ? false :
	  !tqualifier ? true :
	  str_vec_contains(sym->tqualifiers,tqualifier);
}
bool
check_symbol_string(const NodeType type, const char* name, const char* tspecifier, const char* tqualifier)
{
  const Symbol* sym = get_symbol(type,name,tspecifier);
  return 
	  !sym ? false :
	  !tqualifier ? true :
	  str_vec_contains(sym->tqualifiers,tqualifier);
}

bool
check_symbol_index(const NodeType type, const int index, const char* tspecifier, const char* tqualifier)
{
  const Symbol* sym = get_symbol_by_index(type,index,tspecifier);
  return 
	  !sym ? false :
	  !tqualifier ? true :
	  str_vec_contains(sym->tqualifiers,tqualifier);
}

bool
all_identifiers_are_const(const ASTNode* node)
{
	if(node->type & NODE_MEMBER_ID)
		return true;
	bool res = true;
	if(node->lhs)
		res &= all_identifiers_are_const(node->lhs);
	if(node->rhs)
		res &= all_identifiers_are_const(node->rhs);
	if(node->token != IDENTIFIER)
		return res;
	if(!node->buffer)
		return res;
	if(check_symbol(NODE_FUNCTION_ID,node->buffer,0,0)) return true;
	if(check_symbol(NODE_ANY,node->buffer,0,0)) return true;
	return false;
}

bool
all_identifiers_are_constexpr(const ASTNode* node)
{
	if(node->type & NODE_MEMBER_ID)
		return true;
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
	//TP: this should not happend but for now simply set the constexpr value to the correct value
	//TODO: fix
	if(!node->is_constexpr &&  check_symbol(NODE_ANY,node->buffer,0,CONST_STR))
	{
		ASTNode* hack = (ASTNode*)node;
		hack->is_constexpr = true;
	}
	res &= node->is_constexpr;
	return res;
}
array_info
get_array_info(Symbol* sym, const bool accessed, const ASTNode* root)
{

	array_info res;
	res.is_dconst = str_vec_contains(sym->tqualifiers,DCONST_STR);


	res.dims = get_array_var_dims(sym->identifier,root);
	res.name = sym->identifier;
	res.accessed = accessed;
	if (!accessed) push(&sym->tqualifiers,DEAD_STR);
	bool const_dims = true;
	for(size_t dim = 0; dim < MAX_ARRAY_RANK; ++dim) const_dims &= (dim >= res.dims.size || all_identifiers_are_constexpr(res.dims.data[dim]));
	if (const_dims) push(&sym->tqualifiers,CONST_DIMS_STR);

	return res;
}
void
output_array_info(FILE* fp, array_info info, const ASTNode* root)
{
	fprintf(fp,"%s","\n{");
	fprintf(fp,"%s,",info.is_dconst ? "true" : "false");
	fprintf(fp,"%zu,",info.dims.size);
	fprintf(fp,"%s","{{");
	for(size_t dim = 0; dim < MAX_ARRAY_RANK; ++dim)
	{
		const bool integer_dim = (dim >= info.dims.size || all_identifiers_are_constexpr(info.dims.data[dim]));
		const char* from_config = integer_dim ? "false" : "true";
		if(dim >= info.dims.size) 
		{
			fprintf(fp,"{-1,NULL,%s},",from_config);
			continue;
		}
		const ASTNode* base = get_node_by_token(IDENTIFIER,info.dims.data[dim]);
		const ASTNode* member = get_node(NODE_MEMBER_ID,info.dims.data[dim]);
		if(!base)
		{
			fprintf(fp,"{%s,NULL,%s},",combine_all_new(info.dims.data[dim]),from_config);
			continue;
		}
		if(!member)
			fprintf(fp,"{%s,NULL,%s},",base->buffer,from_config);
		else
		{
			if(integer_dim)
			{
				const char* res = get_const_int3_val(base->buffer,root,member->buffer);
				fprintf(fp,"{%s,\"%s\",%s},",res,member->buffer,from_config);
			}
			else
				fprintf(fp,"{%s,\"%s\",%s},",base->buffer,member->buffer,from_config);
		}
	}

	fprintf(fp,"%s","}},");
        fprintf(fp, "\"%s\",", info.name);
        fprintf(fp, "%s,",info.accessed ? "true" : "false");
	fprintf(fp,"%s","},");
	free_node_vec(&info.dims);
}
void
gen_array_info(FILE* fp, const char* datatype_scalar, const ASTNode* root)
{

 
  const int* accesses = get_arr_accesses(datatype_scalar);
  char tmp[1000];
  sprintf(tmp,"%s*",datatype_scalar);
  const char* datatype = intern(tmp);
  const char* define_name =  convert_to_define_name(datatype_scalar);
  int counter = 0;
  fprintf(fp, "\nstatic const array_info %s_array_info[] __attribute__((unused)) = {", define_name);
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        symbol_table[i].tspecifier == datatype)
    {

  	if(str_vec_contains(symbol_table[i].tqualifiers,CONST_STR, RUN_CONST_STR)) continue;
	output_array_info(fp,
		get_array_info(&symbol_table[i],accesses[counter],root),root
		);
	counter++;
    }
  }

  //runtime array lengths come after other arrays
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        symbol_table[i].tspecifier == datatype
	&& str_vec_contains(symbol_table[i].tqualifiers,RUN_CONST_STR)
	)
    {

    	array_info info = get_array_info(&symbol_table[i],true,root);
    	output_array_info(fp,info,root);
    }
	   
  }
  //pad one extra to silence warnings
  fprintf(fp,"\n{false,-1,{{");
  for(int i = 0; i  < MAX_ARRAY_RANK; ++i)
	  fprintf(fp,"{-1,NULL,false}%s",i < MAX_ARRAY_RANK-1 ? "," : "");
  fprintf(fp,"}},");
  fprintf(fp,"\"AC_EXTRA_PADDING\",true}");
  fprintf(fp, "\n};");
}


void
gen_gmem_array_declarations(const char* datatype_scalar, const ASTNode* root)
{
	const char* define_name = convert_to_define_name(datatype_scalar);
	const char* uppr_name =       strupr(define_name);
	const char* upper_case_name = to_upper_case(define_name);
	const char* enum_name = convert_to_enum_name(datatype_scalar);

	char tmp[4098];
	sprintf(tmp,"%s*",datatype_scalar);
	const char* datatype = intern(tmp);


	FILE* fp = fopen("memcpy_to_gmem_arrays.h","a");
	

	fprintf(fp,"void memcpy_to_gmem_array(const %sArrayParam param,%s* &ptr)\n"
        "{\n", enum_name,datatype_scalar);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR) && !str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
	  {
		  if (!str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
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
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR) && !str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
	  {
		  if (str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
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
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR) && !str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
	  {
		  if (str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
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
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR) && !str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
	  {
		  if (str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
		  {
			node_vec dims = get_array_var_dims(symbol_table[i].identifier,root);
                  	char array_length_str[100000];
                  	get_array_var_length(symbol_table[i].identifier,root,array_length_str);
			fprintf(fp,"DECLARE_CONST_DIMS_GMEM_ARRAY(%s,%s,%s,%s);\n",datatype_scalar, define_name, symbol_table[i].identifier,array_length_str);
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

	const char* define_name = convert_to_define_name(datatype_scalar);
	const char* uppr_name =       strupr(define_name);
	const char* upper_case_name = to_upper_case(define_name);
	const char* enum_name = convert_to_enum_name(datatype_scalar);
	char tmp[4098];
	sprintf(tmp,"%s*",datatype_scalar);
	const char* datatype = intern(tmp);
	fprintf_filename("info_access_operators.h","%s operator[](const %s param) const {return param;}\n",datatype_scalar,datatype_scalar);
	fprintf_filename("info_access_operators.h","const %s& operator[](const %sParam param) const {return %s_params[param];}\n"
		,datatype_scalar,enum_name,define_name);
	fprintf_filename("info_access_operators.h","const %s& operator[](const %sCompParam param) const {return run_consts.config.%s_params[param];}\n"
		,datatype_scalar,enum_name,define_name);
	fprintf_filename("info_access_operators.h","%s* const& operator[](const %sArrayParam param) const {return %s_arrays[param];}\n",datatype_scalar,enum_name,define_name);
	fprintf_filename("info_access_operators.h","%s& operator[](const %sParam param) {return %s_params[param];}\n",datatype_scalar,enum_name,define_name);
	fprintf_filename("info_access_operators.h","%s* & operator[](const %sArrayParam param) {return %s_arrays[param];}\n",datatype_scalar,enum_name,define_name);

	fprintf_filename("comp_info_access_operators.h","const %s& operator[](const %sCompParam param) const {return %s_params[param];}\n",datatype_scalar,enum_name,define_name);
	fprintf_filename("comp_info_access_operators.h","const %s* const& operator[](const %sCompArrayParam param) const {return %s_arrays[param];}\n",datatype_scalar,enum_name,define_name);
	fprintf_filename("comp_info_access_operators.h","%s& operator[](const %sCompParam param) {return %s_params[param];}\n",datatype_scalar,enum_name,define_name);
	fprintf_filename("comp_info_access_operators.h","const %s* & operator[](const %sCompArrayParam param) {return %s_arrays[param];}\n",datatype_scalar,enum_name,define_name);
	fprintf_filename("comp_info_access_operators.h","%s operator[](const %s param) const {return param;}\n",datatype_scalar,datatype_scalar);

	fprintf_filename("loaded_info_access_operators.h","const bool& operator[](const %sCompParam param) const {return %s_params[param];}\n",enum_name,define_name);
	fprintf_filename("loaded_info_access_operators.h","const bool& operator[](const %sCompArrayParam param) const {return %s_arrays[param];}\n",enum_name,define_name);
	fprintf_filename("loaded_info_access_operators.h","bool& operator[](const %sCompParam param) {return %s_params[param];}\n",enum_name,define_name);
	fprintf_filename("loaded_info_access_operators.h","bool& operator[](const %sCompArrayParam param) {return %s_arrays[param];}\n",enum_name,define_name);
	fprintf_filename("loaded_info_access_operators.h","bool operator[](const %sParam ) const {return false;}\n",enum_name);
	fprintf_filename("loaded_info_access_operators.h","bool operator[](const %sArrayParam ) const {return false;}\n",enum_name);
	fprintf_filename("loaded_info_access_operators.h","bool operator[](const %s) const {return false;}\n",datatype_scalar);

	fprintf_filename("array_decl.h","%s* %s_arrays[NUM_%s_ARRAYS+1];\n",datatype_scalar,define_name,uppr_name);

	fprintf_filename("get_vba_array.h","if constexpr(std::is_same<P,%sArrayParam>::value) return vba.%s_arrays[(int)param];\n",enum_name,define_name);

	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sCompParam>::value) return (%s){};\n",enum_name,datatype_scalar);

	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sCompArrayParam>::value) return (%s){};\n",enum_name,datatype_scalar);


	fprintf_filename("get_empty_pointer.h","if constexpr(std::is_same<P,%sArrayParam>::value) return (%s*){};\n",enum_name,datatype_scalar);



	fprintf_filename("get_param_name.h","if constexpr(std::is_same<P,%sCompParam>::value) return %s_comp_param_names[(int)param];\n",enum_name,define_name);


	fprintf_filename("get_num_params.h",
				" (std::is_same<P,%sParam>::value)      ? NUM_%s_PARAMS : \n"
				" (std::is_same<P,%sArrayParam>::value) ? NUM_%s_ARRAYS : \n"
				" (std::is_same<P,%sCompParam>::value)      ? NUM_%s_COMP_PARAMS : \n"
				" (std::is_same<P,%sCompArrayParam>::value) ? NUM_%s_COMP_ARRAYS : \n"
			,enum_name,uppr_name
			,enum_name,uppr_name
			,enum_name,uppr_name
			,enum_name,uppr_name
			);
	fprintf_filename("get_array_info.h",
			" if(std::is_same<P,%sArrayParam>::value) return %s_array_info[(int)array]; \n"
			" if(std::is_same<P,%sCompArrayParam>::value) return %s_array_info[(int)array + NUM_%s_ARRAYS]; \n"
			,enum_name,define_name
			,enum_name,define_name, uppr_name
			);
	

	fprintf_filename("device_set_input.h", "AcResult\nacDeviceSet%sInput(Device device, const %sInputParam param, const %s val)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_INPUT_PARAMS == 0) return AC_FAILURE;\n"
		    "\tdevice->input.%s_params[param] = val;\n"
		    "\treturn AC_SUCCESS;\n"
		    "}\n"
	,upper_case_name, enum_name, datatype_scalar, uppr_name, define_name);

        fprintf_filename("device_load_uniform.h","GEN_DEVICE_LOAD_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
	if(is_primitive_datatype(datatype_scalar))
	{
        	fprintf_filename("device_load_uniform.h","GEN_DEVICE_LOAD_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_load_uniform_decl.h","DEVICE_LOAD_ARRAY_DECL(%sArrayParam, %s)\n",enum_name,upper_case_name);
		fprintf_filename("device_load_uniform_overloads.h","OVERLOAD_DEVICE_LOAD_ARRAY(%sArrayParam, %s)\n",enum_name,upper_case_name);
		fprintf_filename("device_load_uniform_loads.h","LOAD_DSYM(acDeviceLoad%sArray)\n",upper_case_name);


		fprintf_filename("device_store_uniform.h","GEN_DEVICE_STORE_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_uniform.h","GEN_DEVICE_STORE_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_uniform_decl.h","DECL_DEVICE_STORE_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_uniform_decl.h","DECL_DEVICE_STORE_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_overloads.h","OVERLOAD_DEVICE_STORE_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_overloads.h","OVERLOAD_DEVICE_STORE_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
	}
	fprintf_filename("device_get_output.h", "%s\nacDeviceGet%sOutput(Device device, const %sOutputParam param)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_OUTPUTS == 0) return (%s){};\n"
		    "\treturn device->output.%s_outputs[param];\n"
		    "}\n"
	,datatype_scalar,upper_case_name, enum_name, uppr_name, datatype_scalar,define_name);

	fprintf_filename("device_get_input.h", "%s\nacDeviceGet%sInput(Device device, const %sInputParam param)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_INPUT_PARAMS == 0) return (%s){};\n"
		    "\treturn device->input.%s_params[param];\n"
		    "}\n"
	,datatype_scalar,upper_case_name, enum_name, uppr_name, datatype_scalar,define_name);

	fprintf_filename("device_set_input_decls.h","FUNC_DEFINE(AcResult, acDeviceSet%sInput,(Device device, const %sInputParam, const %s val));\n",upper_case_name,enum_name,datatype_scalar);	

	fprintf_filename("device_get_output_decls.h","FUNC_DEFINE(%s, acDeviceGet%sOutput,(Device device, const %sOutputParam));\n",datatype_scalar,upper_case_name,enum_name);	

	fprintf_filename("device_get_input_decls.h","FUNC_DEFINE(%s, acDeviceGet%sInput,(Device device, const %sInputParam));\n",datatype_scalar,upper_case_name,enum_name);	

	fprintf_filename("device_set_input_overloads.h","#ifdef __cplusplus\nstatic inline AcResult acDeviceSetInput(Device device, const %sInputParam& param, const %s& val){ return acDeviceSet%sInput(device,param,val); }\n#endif\n",enum_name, datatype_scalar, upper_case_name);	

	fprintf_filename("device_get_output_overloads.h","#ifdef __cplusplus\nstatic inline %s acDeviceGetOutput(Device device, const %sOutputParam& param){ return acDeviceGet%sOutput(device,param); }\n#endif\n",datatype_scalar,enum_name, upper_case_name);	

	fprintf_filename("device_get_input_overloads.h","#ifdef __cplusplus\nstatic inline %s acDeviceGetInput(Device device, const %sInputParam& param){ return acDeviceGet%sInput(device,param); }\n#endif\n",datatype_scalar,enum_name, upper_case_name);	
	
	{
		FILE* fp;
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
	}

	fprintf_filename("dconst_decl.h","%s DEVICE_INLINE  DCONST(const %sParam& param){return d_mesh_info.%s_params[(int)param];}\n"
			,datatype_scalar, enum_name, define_name);

	fprintf_filename("dconst_decl.h","%s DEVICE_INLINE  VAL(const %sParam& param){return d_mesh_info.%s_params[(int)param];}\n"
			,datatype_scalar, enum_name, define_name);

	fprintf_filename("dconst_decl.h","%s DEVICE_INLINE  VAL(const %s& val){return val;}\n"
			,datatype_scalar, datatype_scalar, define_name);

	fprintf_filename("rconst_decl.h","%s DEVICE_INLINE RCONST(const %sCompParam&){return d_mesh_info.%s_params[0];}\n"
			,datatype_scalar, enum_name, define_name);

	fprintf_filename("get_address.h","size_t  get_address(const %sParam& param){ return (size_t)&d_mesh_info.%s_params[(int)param];}\n"
			,enum_name, define_name);
	fprintf_filename("load_dconst_arrays.h","cudaError_t\n"
		   "load_array(const %s* values, const size_t bytes, const %sArrayParam arr)\n"
		    "{\n",
		     datatype_scalar, enum_name);

		     

  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	{
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
		  fprintf_filename("load_dconst_arrays.h","if (arr == %s)\n return cudaMemcpyToSymbol(AC_INTERNAL_d_%s_arrays_%s,values,bytes,0,cudaMemcpyHostToDevice);\n",symbol_table[i].identifier,define_name, symbol_table[i].identifier);
  	}
	fprintf_filename("load_dconst_arrays.h","(void)values;(void)bytes;(void)arr;\nreturn cudaSuccess;\n}\n");


	fprintf_filename("store_dconst_arrays.h","cudaError_t\n"
		  "store_array(%s* values, const size_t bytes, const %sArrayParam arr)\n"
		    "{\n",
	datatype_scalar, enum_name);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	{
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
	  {
		  fprintf_filename("store_dconst_arrays.h","if (arr == %s)\n return cudaMemcpyFromSymbol(values,AC_INTERNAL_d_%s_arrays_%s,bytes,0,cudaMemcpyDeviceToHost);\n",symbol_table[i].identifier,define_name, symbol_table[i].identifier);
	  }
  	}
	fprintf_filename("store_dconst_arrays.h","(void)values;(void)bytes;(void)arr;\nreturn cudaSuccess;\n}\n");






	fprintf_filename("load_and_store_uniform_overloads.h",
	        "static AcResult __attribute ((unused)) "
		"acLoadUniform(const cudaStream_t stream, const %sParam param, const %s value) { return acLoad%sUniform(stream,param,value);}\n"
	        "static AcResult __attribute ((unused)) "
		"acLoadUniform(const cudaStream_t stream, const %sArrayParam param, const %s* values, const size_t length) { return acLoad%sArrayUniform(stream,param,values,length);}\n"
		"static AcResult __attribute ((unused))"
	        "acStoreUniform(const cudaStream_t stream, const %sParam param, %s* value) { return acStore%sUniform(stream,param,value);}\n"
		"static AcResult __attribute ((unused))"
	        "acStoreUniform(const %sArrayParam param, %s* values, const size_t length) { return acStore%sArrayUniform(param,values,length);}\n"
		,enum_name, datatype_scalar, upper_case_name
		,enum_name, datatype_scalar, upper_case_name
		,enum_name, datatype_scalar, upper_case_name
		,enum_name, datatype_scalar, upper_case_name
		);


	fprintf_filename("load_and_store_uniform_funcs.h",
	 	"AcResult acLoad%sUniform(const cudaStream_t, const %sParam param, const %s value) { return acLoadUniform(param,value); }\n"
	 	"AcResult acLoad%sArrayUniform(const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length) { return acLoadArrayUniform(param ,values, length); }\n"
	 	"AcResult acStore%sUniform(const cudaStream_t, const %sParam param, %s* value) { return acStoreUniform(param,value); }\n"
	 	"AcResult acStore%sArrayUniform(const %sArrayParam param, %s* values, const size_t length) { return acStoreArrayUniform(param ,values, length); }\n"
	        ,upper_case_name, enum_name, datatype_scalar
	        ,upper_case_name, enum_name, datatype_scalar
	        ,upper_case_name, enum_name, datatype_scalar
	        ,upper_case_name, enum_name, datatype_scalar
	     );




	fprintf_filename("load_and_store_uniform_header.h",
		"FUNC_DEFINE(AcResult, acLoad%sUniform,(const cudaStream_t, const %sParam param, const %s value));\n"
		"FUNC_DEFINE(AcResult, acLoad%sArrayUniform, (const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length));\n"
		"FUNC_DEFINE(AcResult, acStore%sUniform,(const cudaStream_t, const %sParam param, %s* value));\n"
		"FUNC_DEFINE(AcResult, acStore%sArrayUniform, (const %sArrayParam param, %s* values, const size_t length));\n"
	    	,upper_case_name, enum_name, datatype_scalar
	    	,upper_case_name, enum_name, datatype_scalar
	    	,upper_case_name, enum_name, datatype_scalar
	    	,upper_case_name, enum_name, datatype_scalar
		);


	//we pad with 1 since zero sized arrays are not allowed with some CUDA compilers
        for (size_t i = 0; i < num_symbols[current_nest]; ++i)
        {
          if (symbol_table[i].type & NODE_VARIABLE_ID &&
              symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
          {
                  char array_length_str[4098];
                  get_array_var_length(symbol_table[i].identifier,root,array_length_str);
                  fprintf_filename("dconst_arrays_decl.h","__device__ __constant__ %s AC_INTERNAL_d_%s_arrays_%s[%s];\n",datatype_scalar, define_name,symbol_table[i].identifier,array_length_str);
          }
        }


	fprintf_filename("gmem_arrays_accessed_decl.h","int gmem_%s_arrays_accessed[NUM_%s_ARRAYS]{};\n",define_name,uppr_name);
	
	fprintf_filename("gmem_arrays_output_accesses.h",
			"{\nFILE* fp_arr_accesses = fopen(\"%s_arr_accesses\", \"wb\");"
			"int tmp = NUM_%s_ARRAYS; fwrite(&tmp,sizeof(int),1,fp_arr_accesses); fwrite(gmem_%s_arrays_accessed,sizeof(int),NUM_%s_ARRAYS,fp_arr_accesses);"
			"fclose(fp_arr_accesses);\n}\n"
			,define_name
			,uppr_name,define_name,uppr_name
			);


	fprintf_filename("load_comp_info.h",
		"static AcResult __attribute((unused)) acLoad%sCompInfo(const %sCompParam param, const %s val, AcCompInfo* info)      {\n"
		     "info->is_loaded.%s_params[(int)param] = true;\n"
		     "info->config.%s_params[(int)param] = val;\n"
		     "return AC_SUCCESS;\n"
		     "}\n"
		"static AcResult __attribute((unused)) acLoad%sArrayCompInfo(const %sCompArrayParam param, const %s* val, AcCompInfo* info)      {\n"
				"info->is_loaded.%s_arrays[(int)param] = true;\n"
				"info->config.%s_arrays[(int)param] = val;\n"
				"return AC_SUCCESS;\n"
				"}\n"
		,upper_case_name ,enum_name,datatype_scalar,define_name,define_name
		,upper_case_name ,enum_name,datatype_scalar,define_name,define_name
			);

	fprintf_filename("load_comp_info_overloads.h",
			"GEN_LOAD_COMP_INFO(%sCompParam,%s,%s)\n"
			"GEN_LOAD_COMP_INFO(%sCompArrayParam,%s*,%sArray)\n"
			, enum_name, datatype_scalar, upper_case_name
			, enum_name, datatype_scalar, upper_case_name
			);

	fprintf_filename("is_comptime_param.h",
		"constexpr static bool UNUSED IsCompParam(const %s&)               {return false;}\n"       
		"constexpr static bool UNUSED IsCompParam(const %sParam&)          {return false;}\n"  
		"constexpr static bool UNUSED IsCompParam(const %sArrayParam&)     {return false;}\n"
		"constexpr static bool UNUSED IsCompParam(const %sCompArrayParam&) {return true;}\n"
		"constexpr static bool UNUSED IsCompParam(const %sCompParam&)      {return true;}\n"
		,datatype_scalar
		,enum_name
		,enum_name
		,enum_name
		,enum_name
		);

	fprintf_filename("is_array_param.h",
		"constexpr static bool UNUSED IsArrayParam(const %s&)               {return false;}\n"       
		"constexpr static bool UNUSED IsArrayParam(const %sParam&)          {return false;}\n"  
		"constexpr static bool UNUSED IsArrayParam(const %sArrayParam&)     {return true;}\n"
		"constexpr static bool UNUSED IsArrayParam(const %sCompArrayParam&) {return true;}\n"
		"constexpr static bool UNUSED IsArrayParam(const %sCompParam&)      {return false;}\n"
		,datatype_scalar
		,enum_name
		,enum_name
		,enum_name
		,enum_name
		);

	if(datatype_scalar != INT_STR)
	{
		fprintf_filename("scalar_types.h","%sParam,\n",enum_name);
		fprintf_filename("scalar_comp_types.h","%sCompParam,\n",enum_name);
		fprintf_filename("array_types.h","%sArrayParam,\n",enum_name);
		fprintf_filename("array_comp_types.h","%sCompArrayParam,\n",enum_name);
	}
}


void
gen_comp_declarations(const char* datatype_scalar)
{
	const char* define_name = convert_to_define_name(datatype_scalar);
	const char* upper = strupr(define_name);
	FILE* fp = fopen("comp_decl.h","a");
	fprintf(fp,"%s %s_params[MAX_NUM_%s_COMP_PARAMS];\n",datatype_scalar,define_name,upper);
	fprintf(fp,"const %s* %s_arrays[MAX_NUM_%s_COMP_ARRAYS];\n",datatype_scalar,define_name,upper);
	fclose(fp);

	fp = fopen("comp_loaded_decl.h","a");
	fprintf(fp,"bool %s_params[MAX_NUM_%s_COMP_PARAMS];\n",define_name,upper);
	fprintf(fp,"bool  %s_arrays[MAX_NUM_%s_COMP_ARRAYS];\n",define_name,upper);
	fclose(fp);

	fopen("input_decl.h","a");
	fprintf(fp,"%s %s_params[NUM_%s_INPUT_PARAMS+1];\n",datatype_scalar,define_name,strupr(define_name));
	fclose(fp);

	fp = fopen("output_decl.h","a");
	fprintf(fp,"%s %s_outputs[NUM_%s_OUTPUTS+1];\n",datatype_scalar,define_name,strupr(define_name));
	fclose(fp);
}


void
gen_datatype_enums(FILE* fp, const char* datatype_scalar)
{
  char tmp[1000];
  sprintf(tmp,"%s*",datatype_scalar);
  const char* datatype_arr = intern(tmp);

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_PARAMS} %sParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_arr && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR,GLOBAL_MEM_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_ARRAYS} %sArrayParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_OUTPUTS} %sOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,INPUT_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_INPUT_PARAMS} %sInputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_arr && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_OUTPUT_ARRAYS} %sArrayOutputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,RUN_CONST_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_COMP_PARAMS} %sCompParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));


  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_arr && str_vec_contains(symbol_table[i].tqualifiers,RUN_CONST_STR))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_%s_COMP_ARRAYS} %sCompArrayParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));

  fprintf(fp,"\n");
  int counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
     counter  += (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,CONST_STR));
  fprintf(fp, "#define NUM_%s_CONSTS (%d)\n",strupr(convert_to_define_name(datatype_scalar)),counter);


  counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
     counter  += (symbol_table[i].tspecifier == datatype_arr&& str_vec_contains(symbol_table[i].tqualifiers,CONST_STR));
  fprintf(fp, "#define NUM_%s_ARR_CONSTS (%d)\n",strupr(convert_to_define_name(datatype_scalar)),counter);

  const char* uppr = strupr(convert_to_define_name(datatype_scalar));
  counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
     counter  += (symbol_table[i].tspecifier == datatype_scalar);
  fprintf(fp,"#define MAX_NUM_%s_COMP_PARAMS (%d)\n",uppr,counter);
  counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
     counter  += (symbol_table[i].tspecifier == datatype_arr);

  fprintf(fp,"#define MAX_NUM_%s_COMP_ARRAYS (%d)\n",uppr,counter);

  const char* upr_name = strupr(convert_to_define_name(datatype_scalar));

}
void
gen_param_names(FILE* fp, const char* datatype_scalar)
{
  char tmp[1000];
  sprintf(tmp,"%s*",datatype_scalar);
  const char* datatype_arr = intern(tmp);

  fprintf(fp, "static const char* %sparam_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_array_output_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_arr && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const char* %s_comp_param_names[] __attribute__((unused)) = {",convert_to_define_name(datatype_scalar));
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,RUN_CONST_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");


}



#include "create_node.h"





static int int_log2(int x)
{

	int res = 0;
	while (x >>= 1) ++res;
	return res;
}
static ASTNode*
create_primary_expression(const char* identifier)
{
	return astnode_create(NODE_PRIMARY_EXPRESSION,create_identifier_node(identifier),NULL);
}

ASTNode*
get_index_node(const ASTNode* array_access_start, const node_vec var_dims)
{
    	node_vec array_accesses = VEC_INITIALIZER;
	get_array_access_nodes(array_access_start,&array_accesses);
	
    	if(array_accesses.size != var_dims.size)
    	        return NULL;
	node_vec new_accesses = VEC_INITIALIZER;
    	for(size_t j = 0; j < array_accesses.size; ++j)
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
    			astnode_sprintf(prefix_node,"%s","");
			astnode_set_prefix("+(",prefix_node);
			node_vec dim_nodes = VEC_INITIALIZER;
    			for(size_t k = 0; k < j; ++k)
				push_node(&dim_nodes,var_dims.data[k]);
			ASTNode* dims_node = build_list_node(dim_nodes,MULT_STR);
			prefix_node->lhs = dims_node;
			dims_node->parent = dims_node;
			astnode_set_postfix(")*",prefix_node);
    		}
		astnode_set_prefix("(",node->rhs);
		astnode_set_postfix(")",node->rhs);
    	}
	ASTNode* res = build_list_node(new_accesses,"");
    	free_node_vec(&array_accesses);
    	free_node_vec(&new_accesses);
	return res;

}




#define TRAVERSE_PREAMBLE_PARAMS(FUNC_NAME, ...) \
	if(node->lhs) \
		FUNC_NAME(node->lhs,__VA_ARGS__); \
	if(node->rhs) \
		FUNC_NAME(node->rhs,__VA_ARGS__); 
void
gen_matrix_reads(ASTNode* node)
{
  TRAVERSE_PREAMBLE(gen_matrix_reads);
  if(!(node->type & NODE_ARRAY_ACCESS)) return;
  const char* base_type = get_expr_type(node->lhs);
  if(base_type != MATRIX_STR) return;
  astnode_sprintf_postfix(node->lhs,".data");
}
#define max(a,b) (a > b ? a : b)
#define min(a,b) (a < b ? a : b)
static int
count_nest(const ASTNode* node,const NodeType type)
{
	int lhs_res =  (node->lhs) ? count_nest(node->lhs,type) : 0;
	int rhs_res =  (node->rhs) ? count_nest(node->rhs,type) : 0;
	return max(lhs_res,rhs_res) + (node->type == type);
}
void
get_const_array_var_dims_recursive(const char* name, const ASTNode* node, node_vec* res)
{
	if(node->lhs)
		get_const_array_var_dims_recursive(name,node->lhs,res);
	if(node->rhs)
		get_const_array_var_dims_recursive(name,node->rhs,res);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	if(!has_qualifier(node,"const")) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	node_vec assignments = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < assignments.size; ++i)
	{
		const ASTNode* def = assignments.data[i];
        	const ASTNode* assignment = def->rhs;
		const char* def_name  = get_node_by_token(IDENTIFIER, def)->buffer;
		if(def_name != name) continue;
		const ASTNode* array_initializer  = get_node(NODE_ARRAY_INITIALIZER, assignment);
		const int array_dim = array_initializer ? count_nest(array_initializer,NODE_ARRAY_INITIALIZER) : 0;
		if(array_initializer)
		{
			const int num_of_elems = array_initializer ? count_num_of_nodes_in_list(array_initializer->lhs) : 0;
			push_node(res,create_primary_expression(intern(itoa(num_of_elems))));
			const ASTNode* second_array_initializer = get_node(NODE_ARRAY_INITIALIZER, array_initializer->lhs);
			if(second_array_initializer)
			{
				const int num_of_elems_in_list = count_num_of_nodes_in_list(second_array_initializer->lhs);
				push_node(res,create_primary_expression(intern(itoa(num_of_elems_in_list))));
			}
		}
	}
}
node_vec
get_const_array_var_dims(const char* name, const ASTNode* node)
{
	node_vec res = VEC_INITIALIZER;
	get_const_array_var_dims_recursive(name,node,&res);
	return res;

}
void
gen_array_reads(ASTNode* node, const ASTNode* root, const char* datatype_scalar)
{
  TRAVERSE_PREAMBLE_PARAMS(gen_array_reads,root,datatype_scalar);
  if(node->type != NODE_ARRAY_ACCESS)
	  return;
  if(!node->lhs) return;
  if(get_parent_node(NODE_VARIABLE,node)) return;
  const char* array_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
  const char* datatype = sprintf_intern("%s*",datatype_scalar);
  const int l_current_nest = 0;
  const Symbol* sym = get_symbol(NODE_VARIABLE_ID,intern(array_name),intern(datatype));
  if(!sym)
       return;

  {
    
    if(get_parent_node(NODE_ARRAY_ACCESS,node)) return;
    //TP: replace dead arr accesses with default initializer values
    //TP: Might not be needed after we have a conditional removal but for now need to care of dead reads to keep compiling the code
    if(str_vec_contains(sym->tqualifiers,DEAD_STR,RUN_CONST_STR))
    {
	    node = node->parent;
	    node->lhs = NULL;
	    node->rhs = NULL;
	    astnode_sprintf(node,"(%s){}",datatype_scalar);
	    return;
    }
    node_vec var_dims =  get_array_var_dims(array_name, root);
    if(var_dims.size == 0) var_dims = get_const_array_var_dims(array_name, root);
	
    ASTNode* elem_index         = get_index_node(node,var_dims);
    if(!elem_index)
	    fatal("Incorrect array access: %s,%ld\n",combine_all_new(node),var_dims.size);
    ASTNode* base = node;
    base->lhs=NULL;
    base->rhs=NULL;
    base->prefix=NULL;
    base->postfix=NULL;
    base->infix=NULL;

    ASTNode* identifier_node = create_primary_expression(
            str_vec_contains(sym->tqualifiers,CONST_STR) ? sym->identifier :
	    sprintf_intern(
		    "AC_INTERNAL_%s_%s_arrays_%s",
        	    str_vec_contains(sym->tqualifiers,DCONST_STR) ? "d" : "gmem",
        	    convert_to_define_name(datatype_scalar), sym->identifier
    ));
    identifier_node->parent = base;
    base->lhs =  identifier_node;
    ASTNode* pointer_access = astnode_create(NODE_UNKNOWN,NULL,NULL);
    ASTNode* elem_access_offset = astnode_create(NODE_UNKNOWN,NULL,NULL);
    ASTNode* elem_access        = astnode_create(NODE_UNKNOWN,elem_access_offset,elem_index); 
    ASTNode* access_node        = astnode_create(NODE_UNKNOWN,pointer_access,elem_access);
    base->rhs = access_node;
    access_node ->parent = base;

    astnode_set_prefix("[",elem_access);
    astnode_set_postfix("]",elem_access);
    if(str_vec_contains(sym->tqualifiers,DCONST_STR,CONST_STR))
    {
    }
    else if(str_vec_contains(sym->tqualifiers,GLOBAL_MEM_STR))
    {
	
    	if(!str_vec_contains(sym->tqualifiers,DYNAMIC_STR) && !is_left_child(NODE_ASSIGNMENT,node))
	{
		astnode_sprintf_prefix(base,"%s%s",
				 is_primitive_datatype(datatype_scalar) ? "__ldg(": "",
				 is_primitive_datatype(datatype_scalar) ? "&": ""
		       );
		if(is_primitive_datatype(datatype_scalar))
			astnode_set_postfix(")",base);

	}
    }
    else
    {
	    fprintf(stderr,"Fatal error: no case for array read: %s\n",combine_all_new(node));
	    exit(EXIT_FAILURE);
    }
    free_node_vec(&var_dims);
  }
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
	push(&(params->user_struct_field_types[struct_index]), get_node(NODE_TSPEC,field)->lhs->buffer);
	push(&(params->user_struct_field_names[struct_index]), get_node_by_token(IDENTIFIER,field->rhs)->buffer);
}
void
read_user_structs_recursive(const ASTNode* node, structs_info* params)
{
	TRAVERSE_PREAMBLE_PARAMS(read_user_structs_recursive,params);
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
mark_kernel_inputs(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(mark_kernel_inputs);
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

typedef struct
{
	string_vec types;
	string_vec expr;
	node_vec expr_nodes;
} func_params_info;

#define FUNC_PARAMS_INITIALIZER {.types = VEC_INITIALIZER, .expr = VEC_INITIALIZER, .expr_nodes = VEC_INITIALIZER}
void
free_func_params_info(func_params_info* info)
{
	free_str_vec(&info -> types);
	free_str_vec(&info -> expr);
	free_node_vec(&info -> expr_nodes);
}

void
get_function_params_info_recursive(const ASTNode* node, const char* func_name, func_params_info* dst)
{
	TRAVERSE_PREAMBLE_PARAMS(get_function_params_info_recursive,func_name,dst);
	if(!(node->type & NODE_FUNCTION))
		return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
	if(!fn_identifier)
		return;
	if(fn_identifier->buffer != func_name)
		return;
        ASTNode* param_list_head = node->rhs->lhs;
	if(!param_list_head) return;
	node_vec params = get_nodes_in_list(node->rhs->lhs);
	for(size_t i = 0; i < params.size; ++i)
	{
	  	const ASTNode* param = params.data[i];
		const ASTNode* tspec = get_node(NODE_TSPEC,param->lhs);
	  	push(&dst->types,(tspec) ? tspec->lhs->buffer : NULL);
	  	push(&dst->expr,param->rhs->buffer);
	  	push_node(&dst->expr_nodes,param->rhs);
	}
	free_node_vec(&params);
}
func_params_info
get_function_params_info(const ASTNode* node, const char* func_name)
{
	func_params_info res = FUNC_PARAMS_INITIALIZER;
	get_function_params_info_recursive(node,func_name,&res);
	return res;
}
void
gen_kernel_structs(const ASTNode* root)
{

	string_vec names = VEC_INITIALIZER;
	func_params_info infos[num_kernels];
	int kernel_index = 0;
	for(size_t sym = 0; sym< num_symbols[0]; ++sym)
	{
		if(symbol_table[sym].tspecifier != KERNEL_STR) continue;
		const char* name = symbol_table[sym].identifier;
		push(&names,name);
		infos[kernel_index] = get_function_params_info(root,name);
		kernel_index++;
	}
	string_vec unique_input_types[num_kernels];
	int num_unique_input_types = 0;
	memset(unique_input_types,0,sizeof(string_vec)*num_kernels);
	for(size_t k = 0; k < num_kernels; ++k)
	{
		string_vec types = infos[k].types;
		if(types.size == 0) continue;
		if(!str_vec_in(unique_input_types,num_unique_input_types,types))
		{
			unique_input_types[num_unique_input_types] = types;
			++num_unique_input_types;
		}
	}
	{
		FILE* fp = fopen("load_ac_kernel_params.h","w");
		for(int i = 0; i < num_unique_input_types; ++i)
		{
			fprintf(fp,"AcResult acLoadKernelParams(acKernelInputParams& params, const AcKernel kernel,");
			const string_vec types = unique_input_types[i];
			for(size_t j = 0; j < types.size; ++j)
				fprintf(fp,"%s p_%ld%s",types.data[j],j,(j < types.size-1) ? "," : "");
			fprintf(fp,"){\n");
			for(size_t k = 0; k < num_kernels; ++k)
			{
				const func_params_info info = infos[k];
				const char* name = names.data[k];
				if(!str_vec_eq(infos[k].types,types)) continue;
				fprintf(fp,"if(kernel == %s){ \n",name);
				for(size_t j = 0; j < types.size; ++j)
				{
					fprintf(fp,"params.%s.%s = p_%ld;\n",
							name,info.expr.data[j],j
							);
				}
				fprintf(fp,"return AC_SUCCESS;}\n");
			}
			fprintf(fp,"return AC_FAILURE;}\n");
		}
		//TP: you have to generate some load types always since the library assumes they exist
		const size_t num_always_produced = 5;
		string_vec always_produced_load_types[num_always_produced];
		memset(always_produced_load_types,0,sizeof(string_vec)*num_always_produced);
		push(&always_produced_load_types[0],FIELD_STR);
		push(&always_produced_load_types[0],REAL_PTR_STR);

		push(&always_produced_load_types[1],FIELD3_STR);
		push(&always_produced_load_types[1],REAL_PTR_STR);

		push(&always_produced_load_types[2],FIELD4_STR);
		push(&always_produced_load_types[2],REAL_PTR_STR);

		push(&always_produced_load_types[3],FIELD_STR);

		push(&always_produced_load_types[4],FIELD_STR);
		push(&always_produced_load_types[4],REAL_STR);
		for(size_t i = 0; i < num_always_produced; ++i)
		{
			string_vec types = always_produced_load_types[i];
			if(str_vec_in(unique_input_types,num_unique_input_types,types)) continue;
			fprintf(fp,"AcResult acLoadKernelParams(acKernelInputParams&, const AcKernel,");
			for(size_t j = 0; j < types.size; ++j)
				fprintf(fp,"%s %s",types.data[j],(j < types.size-1) ? "," : "");
			fprintf(fp,"){return AC_FAILURE;}\n");
		}
		fclose(fp);
		fp = fopen("load_ac_kernel_params_def.h","w");
		for(int i = 0; i < num_unique_input_types; ++i)
		{
			fprintf(fp,"AcResult acLoadKernelParams(acKernelInputParams& params, const AcKernel kernel,");
			const string_vec types = unique_input_types[i];
			for(size_t j = 0; j < types.size; ++j)
				fprintf(fp,"%s p_%ld%s",types.data[j],j,(j < types.size-1) ? "," : "");
			fprintf(fp,");");
		}
		for(size_t i = 0; i < num_always_produced; ++i)
		{
			string_vec types = always_produced_load_types[i];
			if(str_vec_in(unique_input_types,num_unique_input_types,types)) continue;
			fprintf(fp,"AcResult acLoadKernelParams(acKernelInputParams&, const AcKernel,");
			for(size_t j = 0; j < types.size; ++j)
				fprintf(fp,"%s %s",types.data[j],(j < types.size-1) ? "," : "");
			fprintf(fp,");\n");
			free_str_vec(&types);
		}
		fclose(fp);
	}
	for(size_t k = 0; k < num_kernels; ++k)
	{
		func_params_info info = infos[k];
		const char* name = names.data[k];
		FILE* fp = fopen("user_input_typedefs.h","a");
		fprintf(fp,"\ntypedef struct %sInputParams {", name);
		for(size_t i = 0; i < info.types.size; ++i)
			fprintf(fp,"%s %s;",info.types.data[i],info.expr.data[i]);
		fprintf(fp,"} %sInputParams;\n",name);
		fclose(fp);
		fp = fopen("safe_vtxbuf_input_params.h","a");
		fprintf(fp,"if(kernel == %s){ \n",name);
		for(size_t i = 0; i < info.types.size; ++i)
		{
			const char* param_name = info.expr.data[i];
			const char* param_type = info.types.data[i];
			if(strstr(param_type,"*"))
			{
				if(param_type != REAL_PTR_STR)
					fatal("How to handle non-real input ptr?\n");
				fprintf(fp,"vba.kernel_input_params.%s.%s = vba.reduce_scratchpads_real[0][0];\n",name,param_name);
			}
		}
		fprintf(fp,"}\n");
		fclose(fp);
	}
	FILE* fp = fopen("kernel_input_param_str.h","w");
	fprintf(fp,"const char* kernel_input_param_strs[] = {");
	for(size_t k = 0; k < num_kernels; ++k)
	{
		fprintf(fp,"\"");
		const string_vec types = infos[k].types;
		for(size_t i = 0; i < types.size; ++i)
			fprintf(fp,"%s%s",types.data[i],i < types.size -1 ? "," : "");
		fprintf(fp,"\",");
	}
	fprintf(fp,"};\n");
	fclose(fp);
	for(size_t k = 0; k < num_kernels; ++k)
		free_func_params_info(&infos[k]);

	FILE* stream = fopen("user_input_typedefs.h","a");
	fprintf(stream,"\ntypedef union acKernelInputParams {\n\n");
	for(size_t k = 0; k < num_kernels; ++k)
	{
		const char* name = names.data[k];
		fprintf(stream,"%sInputParams %s;\n", name,name);
	}
	fprintf(stream,"} acKernelInputParams;\n\n");
	fclose(stream);
	free_str_vec(&names);
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
create_broadcast_op(const structs_info info, const int i, const char* op, FILE* fp)
{
		const char* struct_name = info.user_structs.data[i];
		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator%s(const %s& a, const %s& b)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,op,struct_name,REAL_STR,struct_name);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\ta.%s %s b,\n",info.user_struct_field_names[i].data[j],op);
		fprintf(fp,"\t};\n}\n");


		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "operator%s(const %s& a, const %s& b)\n"
			   "{\n"
			   "\treturn (%s){\n"
			,struct_name,op,REAL_STR,struct_name,struct_name);

		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\t\ta %s b.%s,\n",op,info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");

		fprintf(fp,"static HOST_DEVICE_INLINE void\n"
			   "operator%s=(%s& a, const %s& b)\n"
			   "{\n"
			,op,struct_name,REAL_STR);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\ta.%s %s= b;\n",info.user_struct_field_names[i].data[j],op);
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
		if(struct_name != INT3_STR)
		{
			fprintf(struct_def,"typedef struct %s {",struct_name);
			for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
			{
				const char* type = s_info.user_struct_field_types[i].data[j];
				const char* name = s_info.user_struct_field_names[i].data[j];
				fprintf(struct_def, "%s %s;", type_output(type), name);
			}
			fprintf(struct_def, "} %s;\n", s_info.user_structs.data[i]);
        		fclose(struct_def);
		}

		bool all_reals = true;
		bool all_scalar_types = true;
		for(size_t j = 0; j < s_info.user_struct_field_types[i].size; ++j)
		{
			all_reals        &=  s_info.user_struct_field_types[i].data[j] == REAL_STR;
			all_scalar_types &= s_info.user_struct_field_types[i].data[j] == REAL_STR || s_info.user_struct_field_types[i].data[j] == INT_STR;
		}
		if(!all_scalar_types) continue;
		FILE* fp = fopen("user_typedefs.h","a");
		fprintf(fp,"#ifdef __cplusplus\n");



		if(struct_name != INT3_STR)
		{
			create_binary_op(s_info,i,MINUS_STR,fp);
			create_binary_op(s_info,i,PLUS_STR,fp);
			create_unary_op (s_info,i,MINUS_STR,fp);
			create_unary_op (s_info,i,PLUS_STR,fp);
		}

		
		if(!all_reals)  fprintf(fp,"#endif\n");
		if(!all_reals)  continue;
		create_broadcast_op(s_info,i,DIV_STR,fp);
		create_broadcast_op(s_info,i,MULT_STR,fp);
		create_broadcast_op(s_info,i,PLUS_STR,fp);
		create_broadcast_op(s_info,i,MINUS_STR,fp);
		if(struct_name == COMPLEX_STR)
		{
			fprintf(fp,"#endif\n");
			continue;
		}

		create_binary_op(s_info,i,DIV_STR,fp);
		create_binary_op(s_info,i,MULT_STR,fp);
		

		fprintf(fp,"#endif\n");

				   
		fclose(fp);
	}
}





func_params_info
get_func_call_params_info(const ASTNode* func_call);
void gen_loader(const ASTNode* func_call, const ASTNode* root)
{
		const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
		const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,0,0);
		if(func_name == PERIODIC)
			return;
		ASTNode* param_list_head = func_call->rhs;
		FILE* stream = fopen("user_loaders.h","a");
		if(!param_list_head)
		{
			fprintf(stream,"[](ParamLoadingInfo ){},");
			fclose(stream);
			return;
		}
		func_params_info call_info  = get_func_call_params_info(func_call);
		bool is_boundcond = false;
		for(size_t i = 0; i< call_info.expr.size; ++i)
			is_boundcond |= (strstr(call_info.expr.data[i],"BOUNDARY_") != NULL);

		func_params_info params_info =  get_function_params_info(root,func_name);

		fprintf(stream,"[](ParamLoadingInfo p){\n");
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
				fprintf(stream, "p.params -> %s.%s = %s;\n", func_name, params_info.expr.data[i], input_param);
			else
				fprintf(stream, "p.params -> %s.%s = acDeviceGetInput(acGridGetDevice(),%s); \n", func_name,params_info.expr.data[i], input_param);
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
		fprintf(stream,"},");
		fclose(stream);

		free_func_params_info(&params_info);
		free_func_params_info(&call_info);
}

static int_vec field_remappings = VEC_INITIALIZER;

const char*
get_field_name(const int field)
{
	const int correct_field_index = int_vec_get_index(field_remappings,field);
	return get_symbol_by_index(NODE_VARIABLE_ID,field,FIELD_STR)->identifier;
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
		fatal("incorrect boundary specification: %s\n",boundary_in);
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
	const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	bc_fields res;
	char* full_name = malloc((strlen(boundconds_name) + strlen(func_name) + 500)*sizeof(char));
	sprintf(full_name,"%s__%s",boundconds_name,func_name);
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID, intern(full_name), KERNEL_STR);
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
	const bool all_included =  func_name == PERIODIC;
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
	const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);
	const int num_boundaries =  6;


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
			if(!strcmp(func_name,PERIODIC))
				gen_loader(func_call,root);
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
	char* tmp = strdup(func_name);
	remove_suffix(tmp,"____");
	const char* dfunc_name = intern(tmp);
	free(tmp);
	func_params_info params_info = get_function_params_info(root,dfunc_name);
	if(call_info.expr.size-1 != params_info.expr.size)
		fatal("Number of inputs %lu for %s in BoundConds does not match the number of input params %lu \n", call_info.expr.size-1, dfunc_name, params_info.expr.size);
	const size_t num_of_rest_params = params_info.expr.size;
        free_func_params_info(&params_info);
	fprintf(fp,"Kernel %s_%s()\n{\n",prefix,func_name);
	fprintf(fp,"\t%s(",dfunc_name);
	for(size_t j = 0; j <num_of_rest_params; ++j)
	{
		fprintf(fp,"%s",call_info.expr.data[j+call_param_offset]);
		if(j < num_of_rest_params-1) fprintf(fp,",");
	}
	fprintf(fp,"%s\n",")");
	fprintf(fp,"%s\n","}");
}
void
gen_dfunc_bc_kernel(const ASTNode* func_call, FILE* fp, const ASTNode* root, const char* boundconds_name)
{

	const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
		
	if(!func_call->rhs)
		fatal("need to declare boundary in bc func call: %s\n",combine_all_new(func_call));
	const char* boundary = get_node_by_token(IDENTIFIER,func_call->rhs)->buffer;
	const int boundary_int = get_boundary_int(boundary);


	func_params_info call_info = get_func_call_params_info(func_call);

	if(func_name == PERIODIC)
		return;
	char* prefix;
	const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,0,0);

	if(is_dfunc) asprintf(&prefix,"%s_AC_KERNEL_",boundconds_name);
	else asprintf(&prefix,"%s_",boundconds_name);
	write_dfunc_bc_kernel(root,prefix,func_name,call_info,fp);

	free_func_params_info(&call_info);
}



bool
do_not_rename(const ASTNode* node, const char* str_to_check)
{
	if(!node->buffer)                                            return true;
	if(node->token != IDENTIFIER)                                return true;
	if(strcmp(node->buffer,str_to_check))                        return true;
	if(strstr(node->buffer,"AC_INTERNAL"))                       return true;
	if(strstr(node->buffer,"AC_MANGLED"))                        return true;
	if(node->type & NODE_MEMBER_ID)                              return true;
	if(symboltable_lookup(node->buffer))                         return true;
	return false;
}

void
rename_variables(ASTNode* node, const char* new_name, const char* new_expr, const char* old_name)
{
	TRAVERSE_PREAMBLE_PARAMS(rename_variables,new_name,new_expr,old_name)
	if(do_not_rename(node,old_name)) return;
	astnode_set_buffer(new_name,node);
	node->expr_type = new_expr;
}

void
rename_while(const NodeType type, ASTNode* head, const char* new_name, const char* new_expr, const char* old_name)
{
	while(head->type == type)
	{
		if(head->rhs)
			rename_variables(head->rhs,new_name,new_expr,old_name);
		else
			rename_variables(head->lhs,new_name,new_expr,old_name);
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
		if(func_name == PERIODIC) continue;
		astnode_sprintf(identifier,"%s____%zu",identifier->buffer,i);
	}
	free_node_vec(&func_calls);
}
void
gen_user_boundcond_calls(const ASTNode* node, const ASTNode* root, string_vec* names)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_user_boundcond_calls,root,names);
	if(node->type != NODE_BOUNDCONDS_DEF) return;
	const char* name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	push(names,name);
	fprintf_filename("taskgraph_bc_handles.h","%s,",name);
	{
		FILE* stream = fopen("taskgraph_kernels.h","a");
		fprintf(stream,"{");
		node_vec calls = get_nodes_in_list(node->rhs);
		for(size_t i = 0; i < calls.size; ++i)
		{
			const char* call_name = get_node_by_token(IDENTIFIER,calls.data[i]->lhs)->buffer;
			if(call_name != PERIODIC)
				fprintf(stream,"%s,",sprintf_intern("%s__%s",name,call_name));
			else
				fprintf(stream,"BOUNDCOND_PERIODIC,");
		}
		fprintf(stream,"},");
		fclose(stream);

		stream = fopen("taskgraph_kernel_bcs.h","a");
		fprintf(stream,"{");

		for(size_t i = 0; i < calls.size; ++i)
		{
			func_params_info info = get_func_call_params_info(calls.data[i]);
			const char* call_name = get_node_by_token(IDENTIFIER,calls.data[i]->lhs)->buffer;
			fprintf(stream,"%s,",info.expr.data[0]);
			free_func_params_info(&info);
		}
		fprintf(stream,"},");
		fclose(stream);
		fprintf_filename("user_loaders.h","{}");

		free_node_vec(&calls);
	}
}
void
gen_user_taskgraphs_recursive(const ASTNode* node, const ASTNode* root, string_vec* names)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_user_taskgraphs_recursive,root,names);
	if(node->type != NODE_TASKGRAPH_DEF)
		return;
	const char* boundconds_name = node->lhs->rhs->buffer;
	fprintf_filename("taskgraph_bc_handles.h","%s,",boundconds_name);

	const char* name = node->lhs->lhs->buffer;
	push(names,name);
	node_vec kernel_call_nodes = get_nodes_in_list(node->rhs);
	int_vec kernel_calls = VEC_INITIALIZER;
	for(size_t i = 0; i < kernel_call_nodes.size; ++i)
	{
		const char* func_name = get_node_by_token(IDENTIFIER,kernel_call_nodes.data[i])->buffer;
		push_int(&kernel_calls,get_symbol_index(NODE_FUNCTION_ID,func_name,KERNEL_STR));
	}

	{
		FILE* stream = fopen("taskgraph_kernels.h","a");
		fprintf(stream,"{");
		for(size_t i = 0; i < kernel_call_nodes.size; ++i)
		{
			const char* func_name = get_node_by_token(IDENTIFIER,kernel_call_nodes.data[i])->buffer;
			fprintf(stream,"%s,",func_name);
		}
		fprintf(stream,"},");
		fclose(stream);
	}
	{

		FILE* stream = fopen("taskgraph_kernel_bcs.h","a");
		fprintf(stream,"{");
		for(size_t i = 0; i < kernel_call_nodes.size; ++i)
			fprintf(stream,"BOUNDARY_NONE,");
		fprintf(stream,"},");
		fclose(stream);
	}
	fprintf_filename("user_loaders.h","{");
	for(size_t i = 0; i < kernel_call_nodes.size; ++i)
		gen_loader(kernel_call_nodes.data[i],root);
	fprintf_filename("user_loaders.h","},");
}


void
gen_input_enums(FILE* fp, string_vec input_symbols, string_vec input_types, const char* datatype)
{
  const char* datatype_scalar = datatype;
  fprintf(fp,"typedef enum {");
  for (size_t i = 0; i < input_types.size; ++i)
  {
	  if(datatype !=input_types.data[i])
		  continue;
	  fprintf(fp,"%s,",input_symbols.data[i]);
  }
  fprintf(fp, "NUM_%s_INPUT_PARAMS} %sInputParam;",strupr(convert_to_define_name(datatype_scalar)),convert_to_enum_name(datatype_scalar));
	
}
void
gen_dfunc_bc_kernels(const ASTNode* node, const ASTNode* root, FILE* fp)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_dfunc_bc_kernels,root,fp);
	if(node->type != NODE_BOUNDCONDS_DEF) return;
	const char* name = node->lhs->buffer;
	node_vec func_calls = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < func_calls.size; ++i)
		gen_dfunc_bc_kernel(func_calls.data[i],fp,root,name);
	free_node_vec(&func_calls);
}
void
gen_user_taskgraphs(const ASTNode* root)
{
	make_unique_bc_calls((ASTNode*) root);
	string_vec graph_names = VEC_INITIALIZER;
	fprintf_filename_w("taskgraph_bc_handles.h","const AcDSLTaskGraph DSLTaskGraphBCs[NUM_DSL_TASKGRAPHS] = { ");
	fprintf_filename_w("taskgraph_kernels.h","const std::vector<AcKernel> DSLTaskGraphKernels[NUM_DSL_TASKGRAPHS] = { ");
	fprintf_filename_w("taskgraph_kernel_bcs.h","const std::vector<AcBoundary> DSLTaskGraphKernelBoundaries[NUM_DSL_TASKGRAPHS] = { ");
	fprintf_filename_w("user_loaders.h","const std::vector<std::function<void(ParamLoadingInfo step_info)>> DSLTaskGraphKernelLoaders[NUM_DSL_TASKGRAPHS] = { ");

	gen_user_taskgraphs_recursive(root,root,&graph_names);
	string_vec bc_names = VEC_INITIALIZER;
	gen_user_boundcond_calls(root,root,&bc_names);

	FILE* fp = fopen("taskgraph_enums.h","w");
	fprintf(fp,"typedef enum {");
	for(size_t i = 0; i < graph_names.size; ++i)
		fprintf(fp,"%s,",graph_names.data[i]);
	for(size_t i = 0; i < bc_names.size; ++i)
		fprintf(fp,"%s,",bc_names.data[i]);
	fprintf(fp,"NUM_DSL_TASKGRAPHS} AcDSLTaskGraph;\n");

	fprintf(fp,"static const char* %s_names[] __attribute__((unused)) = {","taskgraph");
	for(size_t i = 0; i < graph_names.size; ++i)
		fprintf(fp,"\"%s\",",graph_names.data[i]);
	for(size_t i = 0; i < bc_names.size; ++i)
		fprintf(fp,"\"%s\",",bc_names.data[i]);
	fprintf(fp,"};\n");


	free_str_vec(&graph_names);
	free_str_vec(&bc_names);
	fclose(fp);
	fprintf_filename("taskgraph_bc_handles.h","};\n");
	fprintf_filename("taskgraph_kernels.h","};\n");
	fprintf_filename("taskgraph_kernel_bcs.h","};\n");
	fprintf_filename("user_loaders.h","};\n");
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
		const int param_index = push(&combinatorials.names[kernel_index],intern(full_name));

		const int enum_index = str_vec_get_index(e_info.names,var.type);
		string_vec options  = e_info.options[enum_index];
		for(size_t i = 0; i < options.size; ++i)
		{
			push(&combinatorials.options[kernel_index+100*param_index],options.data[i]);
		}
	}
	if(BOOL_STR == var.type)
	{
		const int param_index = push(&combinatorials.names[kernel_index],intern(full_name));
		push(&combinatorials.options[kernel_index+100*param_index],intern("false"));
		push(&combinatorials.options[kernel_index+100*param_index],intern("true"));
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
			push(&combinations.vals[kernel_index + MAX_KERNELS*combinations.nums[kernel_index]], intern(combinatorials.options[kernel_index+100*my_index].data[i]));
			++combinations.nums[kernel_index];

		}
		return;
	}
	else
	{
		for(size_t i = 0; i<combinatorials.options[kernel_index+100*my_index].size; ++i)
		{
			string_vec copy = str_vec_copy(res);
			push(&copy, intern(combinatorials.options[kernel_index+100*my_index].data[i]));
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
	TRAVERSE_PREAMBLE_PARAMS(add_kernel_bool_dconst_to_combinations,kernel_index,dst);
	if(node->token != IDENTIFIER) return;
  	if(!check_symbol(NODE_VARIABLE_ID, node->buffer, BOOL_STR, DCONST_STR)) return;
	if(str_vec_contains(dst.names[kernel_index], node->buffer)) return;
	add_param_combinations((variable){BOOL_STR,node->buffer}, kernel_index,"", dst);


}
void
gen_kernel_num_of_combinations_recursive(const ASTNode* node, param_combinations combinations, string_vec* user_kernels_with_input_params,combinatorial_params combinatorials)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_kernel_num_of_combinations_recursive,combinations,user_kernels_with_input_params,combinatorials);
	if(node->type & NODE_KFUNCTION && node->rhs->lhs)
	{
	   const char* kernel_name = get_node(NODE_FUNCTION_ID, node)->buffer;
	   const int kernel_index = push(user_kernels_with_input_params,kernel_name);
	   ASTNode* param_list_head = node->rhs->lhs;
	   func_params_info info = get_function_params_info(node,kernel_name);
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
	string_vec names;
	int_vec* called_funcs;
} funcs_calling_info;

static funcs_calling_info calling_info = {VEC_INITIALIZER, .called_funcs = NULL };


void
get_reduce_info_in_func(const ASTNode* node, node_vec* src)
{
	TRAVERSE_PREAMBLE_PARAMS(get_reduce_info_in_func,src);
	if(!(node->type & NODE_FUNCTION_CALL))
		return;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	if(!func_name || !(
				strstr(func_name,"reduce_sum_AC_MANGLED") 
				|| strstr(func_name,"reduce_min_AC_MANGLED") 
				|| strstr(func_name,"reduce_max_AC_MANGLED") 
				)) return;
	push_node(src,node);
}
ReduceOp
get_reduce_type(const ASTNode* node)
{
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	return
			strstr(func_name,"reduce_sum") ? REDUCE_SUM :
			strstr(func_name,"reduce_max") ? REDUCE_MAX :
			REDUCE_MIN;
}
const char*
get_reduce_dst(const ASTNode* node)
{
	return intern(combine_all_new(node->rhs->rhs));
}
const char*
get_reduce_dst_type(const ASTNode* node)
{
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	const char* type = 
		        strstr(func_name,"Profile") ? PROFILE_STR :
			strstr(func_name,"AcReal")  ? REAL_STR :
			strstr(func_name,"int")     ? INT_STR :
			NULL;
	if(!type) fatal("Was not able to get reduce type: %s\n",combine_all_new(node));
	return type;
}
static node_vec reduce_infos[MAX_FUNCS] = {[0 ... MAX_FUNCS -1] = VEC_INITIALIZER};
void
get_reduce_info(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(get_reduce_info);
	if(node->type & NODE_FUNCTION)
		get_reduce_info_in_func(node,&reduce_infos[str_vec_get_index(calling_info.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer)]);
}
void
gen_reduce_info(const ASTNode* root)
{
	get_reduce_info(root);
	for(size_t i = 0; i < calling_info.names.size; ++i)
		for(int j = i-1; j >= 0; --j)
		{
			if(!int_vec_contains(calling_info.called_funcs[i], j)) continue;
			for(size_t k = 0; k < reduce_infos[j].size; ++k)
				push_node(&reduce_infos[i],reduce_infos[j].data[k]);
		}
}
void
gen_kernel_postfixes_recursive(ASTNode* node, const bool gen_mem_accesses)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_kernel_postfixes_recursive,gen_mem_accesses);
	if(!(node->type & NODE_KFUNCTION))
		return;
	ASTNode* compound_statement = node->rhs->rhs;
	if(gen_mem_accesses)
	{
	  astnode_sprintf_postfix(compound_statement,"%s; (void)vba;}",compound_statement->postfix);
	  return;
	}
	const node_vec kernel_reduce_info = reduce_infos[str_vec_get_index(calling_info.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer)];
	if(kernel_reduce_info.size == 0)
	{
	  astnode_sprintf_postfix(compound_statement,"%s}",compound_statement->postfix);
	  return;
	}
	const ASTNode* fn_identifier = get_node(NODE_FUNCTION_ID,node);
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,fn_identifier->buffer,KERNEL_STR);

#if AC_USE_HIP
	const char* shuffle_instruction = "rocprim::warp_shuffle_down(";
	const char* warp_size  = "const size_t warp_size = rocprim::warp_size();";
	const char* warp_id= "const size_t warp_id = rocprim::warp_id();\n";
#else
	const char* shuffle_instruction = "__shfl_down_sync(0xffffffff,";
	const char* warp_size  = "constexpr size_t warp_size = 32;";
	const char* warp_id= "const size_t warp_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) / warp_size;";
#endif
	astnode_sprintf_postfix(compound_statement,"%s %s\n%s\n%s",compound_statement->postfix,
						warp_size,warp_id,
						"const size_t lane_id = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) % warp_size;"
						"const int warps_per_block = (blockDim.x*blockDim.y*blockDim.z + warp_size -1)/warp_size;"
						"const int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;"
						"const int out_index =  vba.reduce_offset + warp_id + block_id*warps_per_block;"
			);
	for(size_t i = 0; i < kernel_reduce_info.size; ++i)
	{
		const char* dst_type = get_reduce_dst_type(kernel_reduce_info.data[i]);
		if(strstr(dst_type,"Profile")) continue;
		ReduceOp reduce_op = get_reduce_type(kernel_reduce_info.data[i]);
		const char* output = get_reduce_dst(kernel_reduce_info.data[i]);
		const char* define_name = convert_to_define_name(dst_type);
	 	astnode_sprintf_postfix(compound_statement,"%sif(should_reduce_%s[(int)%s]){"
						"for(int offset = warp_size/2; offset > 0; offset /= 2){ \n"
						,compound_statement->postfix
						,define_name
						,output
				      );

		const char* array_name = sprintf_intern("%s_%s",
			reduce_op == REDUCE_SUM ? "reduce_sum_res" :
			reduce_op == REDUCE_MIN ? "reduce_min_res" :
			reduce_op == REDUCE_MAX ? "reduce_max_res" :
			NULL,
			define_name
			);


		const char* res_name = sprintf_intern("%s[(int)%s]",array_name,output);
	 	switch(reduce_op)
	 	{
		 	case(REDUCE_SUM):
				astnode_sprintf_postfix(compound_statement,"%s%s += %s%s,offset);\n",compound_statement->postfix,res_name,shuffle_instruction,res_name);
				break;
		 	case(REDUCE_MIN):
				astnode_sprintf_postfix(compound_statement,"%s"
						"const AcReal shuffle_tmp = %s%s,offset);"
						"%s = (shuffle_tmp < %s) ? shuffle_tmp : %s;\n"
						,compound_statement->postfix
						,shuffle_instruction,res_name
						,res_name,res_name,res_name
						);
				break;
		 	case(REDUCE_MAX):
				astnode_sprintf_postfix(compound_statement,"%s"
						"const AcReal shuffle_tmp = %s%s,offset);"
						"%s = (shuffle_tmp > %s) ? shuffle_tmp : %s;\n"
						,compound_statement->postfix
						,shuffle_instruction,res_name
						,res_name,res_name,res_name);
				break;
		 	case(NO_REDUCE):
				printf("WRONG!\n");
				printf("%s\n",fn_identifier->buffer);
      				exit(EXIT_FAILURE);
	 	}

	 	astnode_sprintf_postfix(compound_statement,
				"%s"
				"}\n"
				"if(lane_id == 0) {vba.reduce_scratchpads_%s[(int)%s][0][out_index] = %s;}}\n"
		,compound_statement->postfix
		,define_name
		,output,res_name);
	}
	astnode_sprintf_postfix(compound_statement,"%s}",compound_statement->postfix);
}




void
init_populate_in_func(const ASTNode* node, int_vec* src)
{
	TRAVERSE_PREAMBLE_PARAMS(init_populate_in_func,src);
	if(!(node->type & NODE_FUNCTION_CALL)) return;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	const int dfunc_index = get_symbol_index(NODE_DFUNCTION_ID,func_name,0);
	if(dfunc_index < 0) return;
	if(int_vec_contains(*src,dfunc_index))  return;
	push_int(src,dfunc_index);
}
void
init_populate(const ASTNode* node, funcs_calling_info* info, const NodeType func_type)
{
	TRAVERSE_PREAMBLE_PARAMS(init_populate,info,func_type);
	if(!(node->type & func_type)) return;
	push(&info->names,get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	init_populate_in_func(node,&info->called_funcs[info->names.size-1]);
}

void
gen_calling_info(const ASTNode* root)
{
	if(calling_info.names.size != 0) return;
	calling_info.called_funcs = malloc(sizeof(string_vec)*MAX_FUNCS);
	for(int i = 0; i < MAX_FUNCS; ++i) 
	{
		calling_info.called_funcs[i].data     = NULL;
		calling_info.called_funcs[i].size     = 0;
		calling_info.called_funcs[i].capacity = 0;
	}
	init_populate(root,&calling_info,NODE_DFUNCTION);
	init_populate(root,&calling_info,NODE_KFUNCTION);
	//TP: we depend on the fact that dfuncs have to be declared before used
	for(size_t i = 0; i < calling_info.names.size; ++i)
		for(int j = i-1; j >= 0; --j)
		{
			if(!int_vec_contains(calling_info.called_funcs[i], j)) continue;
			for(size_t k = 0; k < calling_info.called_funcs[j].size; ++k)
			{
				if(int_vec_contains(calling_info.called_funcs[i], calling_info.called_funcs[j].data[k])) continue;
				push_int(&calling_info.called_funcs[i],calling_info.called_funcs[j].data[k]);
			}
		}
	
}

void
gen_kernel_postfixes(ASTNode* root, const bool gen_mem_accesses)
{
	gen_kernel_postfixes_recursive(root,gen_mem_accesses);
}
void
gen_kernel_reduce_outputs(const bool gen_mem_accesses)
{
  FILE* fp = fopen("kernel_reduce_outputs.h","w");
  string_vec prof_types = get_prof_types();
  int num_reduce_outputs = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (((symbol_table[i].tspecifier == REAL_STR || symbol_table[i].tspecifier == INT_STR) && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
		    || str_vec_contains(prof_types,symbol_table[i].tspecifier)
	)
	    ++num_reduce_outputs;
  //extra padding to make the code compile on some compilers
  fprintf(fp,"%s","static const KernelReduceOutput kernel_reduce_outputs[NUM_KERNELS][NUM_OUTPUTS+1] = { ");
  string_vec kernel_reduce_output_entries[num_kernels];
  memset(kernel_reduce_output_entries,0,sizeof(kernel_reduce_output_entries));
  if(file_exists("reduce_dst_integers.h"))
  {
       FILE* csv_file = fopen("reduce_dst_integers.h","r");
       get_csv_entries(kernel_reduce_output_entries,csv_file);
       fclose(csv_file);
  }

  int kernel_index = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (symbol_table[i].tspecifier == KERNEL_STR)
    {
      const size_t index = str_vec_get_index(calling_info.names,symbol_table[i].identifier);
      const string_vec dst_ints = kernel_reduce_output_entries[kernel_index];
      int active_call_index = 0;
      fprintf(fp,"%s","{");
      for(int j = 0; j < num_reduce_outputs; ++j)
      {
        fprintf(fp,"%s","{");
      	if(reduce_infos[index].size < (size_t) j+1)
	        fprintf(fp,"%d,AC_REAL_TYPE,false",-1);
      	else
	{
		const char* var      = get_reduce_dst(reduce_infos[index].data[j]);
		const char* dst_type = get_reduce_dst_type(reduce_infos[index].data[j]);
	      	fprintf(fp,"(int)%s,",gen_mem_accesses || dst_ints.size == 0 ? "-1" : dst_ints.data[active_call_index]);
		if(dst_type == intern("Profile"))
        		fprintf(fp,"%s,","AC_PROF_TYPE");
		else
		{
			const char* define_name = convert_to_define_name(dst_type);
			const char* uppr_name =       strupr(define_name);
			fprintf(fp,"AC_%s_TYPE,",uppr_name);
		}
		if(gen_mem_accesses || !OPTIMIZE_MEM_ACCESSES)
		{
			fprintf(fp,"%s,","true");
			if(gen_mem_accesses)
			{
				astnode_sprintf_postfix((ASTNode*)reduce_infos[index].data[j],"%s;executed_nodes.push_back(%d)",reduce_infos[index].data[j]->postfix,reduce_infos[index].data[j]->id);
			}
		}
		else
		{
			fprintf(fp,"%s,",is_called(reduce_infos[index].data[j]) ? "true" : "false");
			active_call_index += is_called(reduce_infos[index].data[j]);
		}

	}
        fprintf(fp,"%s","},");
      }
      fprintf(fp,"{-1,AC_REAL_TYPE,false}");
      fprintf(fp,"%s","},");
      kernel_index++;
    }
  }
  fprintf(fp,"%s","};\n");
  for(size_t i = 0; i < num_kernels; ++i)
	  free_str_vec(&kernel_reduce_output_entries[i]);

  //extra padding to help some compilers
  fprintf(fp,"%s","static const AcReduceOp kernel_reduce_ops[NUM_KERNELS][NUM_OUTPUTS+1] = { ");
  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (symbol_table[i].tspecifier == KERNEL_STR)
    {
      const size_t index = str_vec_get_index(calling_info.names,symbol_table[i].identifier);
      fprintf(fp,"%s","{");
      for(int j = 0; j < num_reduce_outputs; ++j)
      {
      	if(reduce_infos[index].size < (size_t) j+1)
        	fprintf(fp,"%s,","NO_REDUCE");
	else
	{
      		switch(get_reduce_type(reduce_infos[index].data[j]))
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
    }
  fprintf(fp,"%s","};\n");
  fclose(fp);
  fp = fopen("kernel_reduce_info.h","w");
  fprintf(fp, "\nstatic const int kernel_calls_reduce[] = {");

  for (size_t i = 0; i < num_symbols[0]; ++i)
    if (symbol_table[i].tspecifier == KERNEL_STR)
    {
      const size_t index = str_vec_get_index(calling_info.names,symbol_table[i].identifier);
      const char* val = (reduce_infos[index].size == 0) ? "0" : "1";
      fprintf(fp,"%s,",val);
    }

  fprintf(fp, "};\n");
  fclose(fp);
}
void
gen_kernel_postfixes_and_reduce_outputs(ASTNode* root, const bool gen_mem_accesses)
{
  gen_kernel_postfixes(root,gen_mem_accesses);

  gen_kernel_reduce_outputs(gen_mem_accesses);
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
	ASTNode* head = astnode_create(NODE_UNKNOWN,node,NULL);
	replace_node(node,head);
	node_vec optimized_decls = VEC_INITIALIZER;
	for(int i = 0; i < combinations.nums[kernel_index]; ++i)
	{
		ASTNode* new_node = astnode_dup(node,NULL);
		ASTNode* function_id = (ASTNode*) get_node(NODE_FUNCTION_ID,new_node->lhs);
		astnode_sprintf(function_id,"%s_optimized_%d",get_node(NODE_FUNCTION_ID,node)->buffer,i);
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
		fprintf(fp,"if(kernel_enum == %s ",get_node(NODE_FUNCTION_ID,node)->buffer);
		for(size_t j = 0; j < combination_vals.size; ++j)
			fprintf(fp, " && vba.kernel_input_params.%s.%s ==  %s ",get_node(NODE_FUNCTION_ID,node)->buffer,combination_params.data[j],combination_vals.data[j]);

		fprintf(fp,
				")\n{\n"
				"\treturn %s_optimized_%d;\n}\n"
		,get_node(NODE_FUNCTION_ID,node)->buffer,i);
		fprintf(fp_defs,"%s_optimized_%d,",get_node(NODE_FUNCTION_ID,node)->buffer,i);
	}
	
	//printf("NUM of combinations: %d\n",combinations.nums[kernel_index]);
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
  	if(!check_symbol(NODE_VARIABLE_ID, node->buffer, BOOL_STR, DCONST_STR)) return;
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
	astnode_sprintf(node,"%s",combinations.data[param_index]);
	node->lhs = NULL;
	node->rhs = NULL;
}
void
gen_kernel_input_params(ASTNode* node, const string_vec* vals, string_vec user_kernels_with_input_params, string_vec* user_kernel_combinatorial_params, const bool gen_mem_accesses)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_kernel_input_params,vals,user_kernels_with_input_params,user_kernel_combinatorial_params,gen_mem_accesses);
	if(!(node->type & NODE_INPUT && node->buffer))
		return;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,function->lhs);
	char* kernel_name = strdup(fn_identifier->buffer);
	const int combinations_index = get_suffix_int(kernel_name,"_optimized_");
	remove_suffix(kernel_name,"_optimized_");
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,kernel_name);
	const char* type = get_expr_type(node);
	if(combinations_index == -1)
	{

		if(type && strstr(type,"*") && gen_mem_accesses)
		{
			if(type == REAL_PTR_STR)
			{
				astnode_sprintf(node,"vba.out[0]");
				return;
			}
			else
			{
				fatal("How to handle non-real input pointer?\n");
			}
		}
		astnode_sprintf(node,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
		return;
	}
	const string_vec combinations = vals[kernel_index + MAX_KERNELS*combinations_index];
	const int param_index = str_vec_get_index(user_kernel_combinatorial_params[kernel_index],node->buffer);
	if(param_index < 0)
	{
		astnode_sprintf(node,"vba.kernel_input_params.%s.%s",kernel_name,node->buffer);
		return;
	}
	astnode_sprintf(node->parent->parent->parent,"%s",combinations.data[param_index]);
	node->parent->parent->parent->infix= NULL;
	node->parent->parent->parent->lhs = NULL;
	node->parent->parent->parent->rhs = NULL;
}
static void
check_for_undeclared_use_in_range(const ASTNode* node)
{
	  const ASTNode* range_node = get_parent_node_by_token(RANGE,node);
	  if(range_node)
		fatal("Undeclared variable or function used on a range expression\n"
				"Range: %s\n"
				"Var: %s\n"
				"\n"
				,combine_all_new(range_node),node->buffer);
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
			const ASTNode* surrounding_func = get_parent_node(NODE_FUNCTION,node);
			if(surrounding_func && strstr(get_node_by_token(IDENTIFIER,surrounding_func)->buffer,"AC_INTERNAL_COPY"))
				return;
			if(func_name == PERIODIC) return;
			if(strstr(func_name,"AC_MANGLED_NAME")) return;
			fprintf(stderr,FATAL_ERROR_MESSAGE);
                        if(str_vec_contains(duplicate_dfuncs.names,func_name))
                                fprintf(stderr,"Unable to resolve overloaded function: %s\nIn:\t%s\n",func_name,tmp);
                        else
                                fprintf(stderr,"Undeclared function used: %s\nIn:\t%s\n",func_name,tmp);
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
		  fatal(
			"Undeclared variable or function used on the right hand side of an assignment\n"
			"Assignment: %s\n"
			"Var: %s\n"
			"\n"
				  ,combine_all_new(get_parent_node(NODE_ASSIGNMENT,node)),node->buffer
			);
}
bool
add_auto(const ASTNode* node)
{
	return !get_parent_node_exclusive(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID) &&
                 !(node->type & NODE_INPUT) &&
		 !(node->no_auto) &&
		 !(is_user_enum_option(node->buffer)) &&
		 !(strstr(node->buffer,"AC_INTERNAL_d_")) &&
		 !(strstr(node->buffer,"AC_INTERNAL_gmem_"));
}
static void
check_for_undeclared_conditional(const ASTNode* node)
{

         const bool used_in_conditional = is_left_child(NODE_IF,node);
         if(used_in_conditional)
                 fatal(
                       "Undeclared variable or function used on in a conditional\n"
                       "Conditional: %s\n"
                       "Var: %s\n"
                       "\n"
                                 ,combine_all_new(get_parent_node(NODE_IF,node)->lhs),node->buffer
                       );
}
static void
check_for_undeclared(const ASTNode* node)
{
	if(add_auto(node))
	{
	 check_for_undeclared_use_in_range(node);
	 check_for_undeclared_function(node);
	 check_for_undeclared_use_in_assignment(node);
	 check_for_undeclared_conditional(node);
	}

}
static void
translate_buffer_body(FILE* stream, const ASTNode* node)
{
  if (stream && node->buffer) {
    const Symbol* symbol = symboltable_lookup(node->buffer);
    if (symbol && symbol->type & NODE_VARIABLE_ID && str_vec_contains(symbol->tqualifiers,DCONST_STR))
      fprintf(stream, "DCONST(%s)", node->buffer);
    else if (symbol && symbol->type & NODE_VARIABLE_ID && str_vec_contains(symbol->tqualifiers,RUN_CONST_STR))
      fprintf(stream, "RCONST(%s)", node->buffer);
    else if (symbol && symbol->type & NODE_VARIABLE_ID && symbol->tspecifier == PROFILE_STR)
      fprintf(stream, "(NUM_FIELDS+%s)", node->buffer);
    else if(symbol && symbol->tspecifier == KERNEL_STR)
	    fprintf(stream,"KERNEL_%s",node->buffer);
    else
      fprintf(stream, "%s", node->buffer);
  }
}
static void
output_qualifiers(FILE* stream, const ASTNode* node, const char** tqualifiers, const size_t n_tqualifiers)
{
        const ASTNode* is_dconst = get_parent_node_exclusive(NODE_DCONST, node);
        if (is_dconst)
          fprintf(stream, "__device__ ");

        if (n_tqualifiers)
	  for(size_t i=0; i<n_tqualifiers;++i)
	  {
		if(tqualifiers[i] != BOUNDCOND_STR && tqualifiers[i] != FIXED_BOUNDARY_STR && tqualifiers[i] != ELEMENTAL_STR && tqualifiers[i] != UTILITY_STR)
          		fprintf(stream, "%s ", tqualifiers[i]);
	  }
}
typedef struct
{
	const char* id;
	int token;
} tspecifier;

static int
n_occurances(const char* str, const char test)
{
	int res = 0;
	int i = -1;
	while(str[++i] != '\0') res += str[i] == test;
	return res;
}
const char*
get_array_elem_type(char* arr_type)
{
	if(n_occurances(arr_type,'<') == 1)
	{
		int start = 0;
		while(arr_type[++start] != '<');
		int end = start;
		++start;
		while(arr_type[end] != ',' && arr_type[end] != ' ') ++end;
		arr_type[end] = '\0';
		char* tmp = malloc(sizeof(char)*1000);
		strcpy(tmp, &arr_type[start]);
		return intern(tmp);
	}
	return intern(arr_type);
}
const char*
get_array_elem_size(char* arr_type)
{
	if(n_occurances(arr_type,'<') == 1)
	{
		int start = 0;
		while(arr_type[++start] != '<');
		int end = start;
		++start;
		while(arr_type[end] != ',' && arr_type[end] != ' ') ++end;
		arr_type[end] = '\0';
		++end;
		start = end;
		while(arr_type[end] != ',' && arr_type[end] != '>' && arr_type[end] != ' ') ++end;
		arr_type[end] = '\0';
		char* tmp = malloc(sizeof(char)*1000);
		strcpy(tmp, &arr_type[start]);
		return intern(tmp);
	}
	return intern(arr_type);
}

const char*
output_specifier(FILE* stream, const tspecifier tspec, const ASTNode* node)
{
	const char* res = NULL;
        if (tspec.id) 
	{
          //TP: the pointer view is only internally used to mark arrays. for now simple lower to auto
	  if(tspec.id[strlen(tspec.id)-1] == '*' || tspec.id[strlen(tspec.id)-2] == '*')
            fprintf(stream, "%s ", "auto");
	  //TP: Hacks
	  else if(strstr(tspec.id,"WITH_INLINE"))
          	fprintf(stream, "%s ", "auto");
	  else if(strstr(tspec.id,"AcArray"))
	  {
		  fprintf(stream, "%s ", get_array_elem_type(strdup(tspec.id)));
		  res = get_array_elem_size(strdup(tspec.id));
	  }
	  else if(tspec.id != KERNEL_STR)
	  {
            fprintf(stream, "%s ", type_output(tspec.id));
	  }
        }
        else if (add_auto(node))
	{
	  if(node->is_constexpr && !(node->type & NODE_FUNCTION_ID)) fprintf(stream, " constexpr ");
	  if(node->expr_type)
	  {
          	//TP: the pointer view is only internally used to mark arrays. for now simple lower to auto
	  	if(node->expr_type[strlen(node->expr_type)-1] == '*' || node->expr_type[strlen(node->expr_type)-2] == '*')
            		fprintf(stream, "%s ", "auto");
		//TP: Hacks
		else if(strstr(node->expr_type,"WITH_INLINE"))
            		fprintf(stream, "%s ", "auto");
		else
		{
	  		if(strstr(node->expr_type,"AcArray"))
	  		{
	  		        fprintf(stream, "%s ", get_array_elem_type(strdup(node->expr_type)));
	  		        res = get_array_elem_size(strdup(node->expr_type));
	  		}
			else
		  		fprintf(stream, "%s ",type_output(node->expr_type));
		}
	  }
	  else
          	fprintf(stream, "auto ");
	}
	return res;
	//else if(node->expr_type && !(node->type & INPUT))
	//  fprintf(stream,"%s ",node->expr_type);
}

tspecifier
get_tspec(const ASTNode* decl)
{
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
get_qualifiers(const ASTNode* decl, const char** tqualifiers)
{
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
		  tqualifiers[n_tqualifiers] = tqual_list_node->rhs->lhs->buffer;
		  ++n_tqualifiers;
		  tqual_list_node = tqual_list_node->lhs;
	  }
	  tqualifiers[n_tqualifiers] = tqual_list_node->lhs->lhs->buffer;
	  ++n_tqualifiers;

      return n_tqualifiers;
}
static void
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
refresh_current_hashmap()
{
    hashmap_destroy(&symbol_table_hashmap[current_nest]);
    const unsigned initial_size = 2000;
    hashmap_create(initial_size, &symbol_table_hashmap[current_nest]);
}
const char* 
add_to_symbol_table(const ASTNode* node, const NodeType exclude, FILE* stream, bool do_checks, const ASTNode* decl, const char* postfix, const bool skip_global_dup_check)
{
  const char* res = NULL;
  if (node->buffer && node->token == IDENTIFIER && !(node->type & exclude)) {
    if (do_checks) check_for_shadowing(node);
    if (!symboltable_lookup(node->buffer)) {
      const char* tqualifiers[MAX_ID_LEN];
      size_t n_tqualifiers = get_qualifiers(decl,tqualifiers);
      tspecifier tspec = get_tspec(decl);


      if (stream) {
	output_qualifiers(stream,node,tqualifiers,n_tqualifiers);
	res = output_specifier(stream,tspec,node);
      }
      if (!(node->type & NODE_MEMBER_ID))
      {
        add_symbol_base(node->type, tqualifiers, n_tqualifiers, tspec.id,  node->buffer, postfix);
        if(do_checks && !tspec.id) check_for_undeclared(node);
      }
    }
    else if(do_checks && !skip_global_dup_check && current_nest == 0 && !(node->type & NODE_FUNCTION_ID))
	    fatal("Multiple declarations of %s\n",node->buffer);
  }
  return res;
}
static bool
is_left_child_of(const ASTNode* parent, const ASTNode* node_to_search)
{
	if(!parent) return false;
	if(!parent->lhs) return false;
	return get_node_by_id(node_to_search->id,parent->lhs) != NULL;
}
void
rename_scoped_variables(ASTNode* node, const ASTNode* decl, const ASTNode* func_body)
{
  FILE* stream = NULL;
  const bool do_checks = false;
  const NodeType exclude = 0;
  if(node->type == NODE_STRUCT_DEF) return;
  if(node->type & NODE_DECLARATION) decl = node;
  if(node->parent && node->parent->type & NODE_FUNCTION && node->parent->rhs->id == node->id) func_body = node;
  // Do not translate tqualifiers or tspecifiers immediately
  if (node->parent &&
      (node->parent->type & NODE_TQUAL || node->parent->type & NODE_TSPEC))
    	return;

  // Prefix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest < MAX_NESTS);

    ++current_nest;
    num_symbols[current_nest] = num_symbols[current_nest - 1];
    refresh_current_hashmap();
    nest_ids[current_nest]++;
  }

  // Traverse LHS
  //TP: skip the lhs of func_body since that is input params
  if (node->lhs)
    rename_scoped_variables(node->lhs, decl, func_body);

  //TP: do not rename func params since it is not really needed and does not gel well together with kernel params
  //TP: also skip func calls since they should anways always be in the symbol table except because of hacks
  //TP: skip also enums since they are anyways unique
  char* postfix = (is_left_child_of(func_body,node) || is_left_child(NODE_FUNCTION_CALL,node) || get_parent_node(NODE_ENUM_DEF,node)) ? NULL : itoa(nest_ids[current_nest]);

  add_to_symbol_table(node,exclude,stream,do_checks,decl,postfix,true);
  free(postfix);
  if(node->buffer) 
  {
	  const Symbol* sym= symboltable_lookup_range(node->buffer,current_nest,1);
	  if(sym)
		  node->buffer = sym->identifier;
  }

  // Traverse RHS
  if (node->rhs)
    rename_scoped_variables(node->rhs, decl, func_body);

  // Postfix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest > 0);
    --current_nest;
  }
}
void
traverse_base(const ASTNode* node, const NodeType return_on, const NodeType exclude, FILE* stream, bool do_checks, const ASTNode* decl, bool skip_global_dup_check)
{
  if(node->type == NODE_ENUM_DEF)   return;
  if(node->type == NODE_STRUCT_DEF) return;
  if(node->type & NODE_DECLARATION) decl = node;
  if (node->type & exclude)
	  stream = NULL;
  if(return_on != NODE_UNKNOWN && (node->type == return_on))
	  return;
  // Do not translate tqualifiers or tspecifiers immediately
  if (node->parent &&
      (node->parent->type & NODE_TQUAL || node->parent->type & NODE_TSPEC))
    return;

  // Prefix translation
  if (stream && node->prefix)
      fprintf(stream, "%s", node->prefix);

  // Prefix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest < MAX_NESTS);

    ++current_nest;
    num_symbols[current_nest] = num_symbols[current_nest - 1];
    refresh_current_hashmap();
  }

  // Traverse LHS
  if (node->lhs)
    traverse_base(node->lhs, return_on, exclude, stream, do_checks,decl,skip_global_dup_check);

  const char* size = add_to_symbol_table(node,exclude,stream,do_checks,decl,NULL,skip_global_dup_check);

  // Infix translation
  if (stream && node->infix) 
    fprintf(stream, "%s", node->infix);
  translate_buffer_body(stream, node);
  if(size)
	  fprintf(stream, "[%s] ",size);

  // Traverse RHS
  if (node->rhs)
  {
    skip_global_dup_check |= (node->type & NODE_ASSIGNMENT);
    skip_global_dup_check |= (node->type & NODE_ARRAY_ACCESS);
    traverse_base(node->rhs, return_on, exclude, stream,do_checks,decl,skip_global_dup_check);
  }

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
	traverse_base(node,NODE_UNKNOWN,exclude,stream,false,NULL,false);
}

func_params_info
get_func_call_params_info(const ASTNode* func_call)
{
		func_params_info res = FUNC_PARAMS_INITIALIZER;
		if(!func_call->rhs) return res;
		node_vec params = get_nodes_in_list(func_call->rhs);
		for(size_t i = 0; i < params.size; ++i)
		{

			push(&res.expr,intern(combine_all_new(params.data[i])));
			push(&res.types,intern(get_expr_type((ASTNode*) params.data[i])));
	        }	
		res.expr_nodes = params;
		return res;
}
string_vec
get_struct_field_types(const char* struct_name)
{
		const structs_info info = s_info;
		string_vec res = VEC_INITIALIZER;
		for(size_t i = 0; i < info.user_structs.size; ++i)
			if(info.user_structs.data[i] == struct_name) res = str_vec_copy(info.user_struct_field_types[i]);
		return res;
}
string_vec
get_struct_field_names(const char* struct_name)
{
		const structs_info info = s_info;
		string_vec res = VEC_INITIALIZER;
		for(size_t i = 0; i < info.user_structs.size; ++i)
			if(info.user_structs.data[i] == struct_name) res = str_vec_copy(info.user_struct_field_names[i]);
		return res;
}

size_t
get_number_of_members(const char* struct_name)
{
	string_vec types = get_struct_field_types(struct_name);
	size_t res = types.size;
	free_str_vec(&types);
	return res;
}

const char*
get_user_struct_member_expr(const ASTNode* node)
{
		char* res = NULL;
		const char* struct_type = get_expr_type(node->lhs);
		const char* field_name = get_node(NODE_MEMBER_ID,node)->buffer;
		if(!field_name) return NULL;
		const structs_info info = s_info;
		int index = -1;
		for(size_t i = 0; i < info.user_structs.size; ++i)
			if(info.user_structs.data[i] == struct_type) index = i;
		if(index == -1)
			return NULL;
		int field_index = -1;
		for(size_t i = 0; i < info.user_struct_field_names[index].size; ++i)
			if(info.user_struct_field_names[index].data[i] == field_name) field_index = i;
		if(field_index == -1)
			return NULL;
		return info.user_struct_field_types[index].data[field_index];
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
	if(!identifier_node || identifier_node->buffer != identifier) return;
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
		(get_node_by_token(REALNUMBER,node)) ? REAL_STR:
		(get_node_by_token(DOUBLENUMBER,node)) ? REAL_STR:
		(get_node_by_token(NUMBER,node)) ? INT_STR :
		(get_node_by_token(STRING,node)) ? "char*" :
	  (!sym) ? NULL :
	  sym->tspecifier;
}
const char*
get_array_access_type(const ASTNode* node)
{
	int counter = 1;
	ASTNode* array_access_base = node->lhs;
	while(array_access_base->type == NODE_ARRAY_ACCESS)
	{
		array_access_base = array_access_base->lhs;
		++counter;
	}
	const char* base_type = get_expr_type(array_access_base);
	return (!base_type)   ? NULL : 
		counter == 2 && base_type == MATRIX_STR ? REAL_STR :
		base_type == MATRIX_STR ? REAL_PTR_STR:
		strstr(base_type,MULT_STR) ? intern(remove_substring(strdup(base_type),MULT_STR)) :
		strstr(base_type,"AcArray") ? get_array_elem_type(strdup(base_type)) :
		base_type == FIELD_STR  ? REAL_STR :
		NULL;
}

const char*
get_struct_expr_type(const ASTNode* node)
{
	const char* base_type = get_expr_type(node->lhs);
	const ASTNode* left = get_node(NODE_MEMBER_ID,node);
	return
		!base_type ? NULL :
		get_user_struct_member_expr(node);

}
const char*
get_binary_expr_type(const ASTNode* node)
{
	const char* op = get_node_by_token(BINARY_OP,node->rhs->lhs)->buffer;
	if(op && !strcmps(op,"==",GREATER_STR,LESS_STR,GEQ_STR,LEQ_STR))
		return BOOL_STR;
	ASTNode* lhs_node = op == MULT_STR && node->lhs->rhs ? node->lhs->rhs
				                                            : node->lhs;
	const char* lhs_res = get_expr_type(lhs_node);
	const char* rhs_res = get_expr_type(node->rhs);
	if(!lhs_res || !rhs_res) return NULL;
	const bool lhs_real = lhs_res == REAL_STR;
	const bool rhs_real = rhs_res == REAL_STR;
	const bool lhs_int   = lhs_res == INT_STR;
	const bool rhs_int   = rhs_res == INT_STR;
	return
		op && !strcmps(op,PLUS_STR,MINUS_STR,MULT_STR,DIV_STR) && (!strcmp(lhs_res,FIELD_STR) || !strcmp(rhs_res,FIELD_STR))   ? REAL_STR  :
		op && !strcmps(op,PLUS_STR,MINUS_STR,MULT_STR,DIV_STR) && (!strcmp(lhs_res,FIELD3_STR) || !strcmp(rhs_res,FIELD3_STR)) ? REAL3_STR :
                (lhs_real || rhs_real) && (lhs_int || rhs_int) ? REAL_STR :
                !strcmp_null_ok(op,MULT_STR) && !strcmp(lhs_res,MATRIX_STR) &&  !strcmp(rhs_res,REAL3_STR) ? REAL3_STR :
		!strcmp(lhs_res,COMPLEX_STR) || !strcmp(rhs_res,COMPLEX_STR)   ? COMPLEX_STR  :
		lhs_real && !strcmps(rhs_res,INT_STR,LONG_STR,LONG_LONG_STR,DOUBLE_STR,FLOAT_STR)    ?  REAL_STR  :
		op && !strcmps(op,MULT_STR,DIV_STR,PLUS_STR,MINUS_STR)     && lhs_real && !rhs_int  ?  rhs_res   :
		op && !strcmps(op,MULT_STR,DIV_STR,PLUS_STR,MINUS_STR)  && rhs_real && !lhs_int  ?  lhs_res   :
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

bool is_first_decl(const ASTNode* node)
{
	return node->type & NODE_DECLARATION && node->token == FIRST;
}
const char*
get_assignment_expr_type(ASTNode* node)
{
	ASTNode* func_base = (ASTNode*) get_parent_node(NODE_FUNCTION,node);
	ASTNode* decl = (ASTNode*)get_node(NODE_DECLARATION,node->lhs);
	const ASTNode* tspec = get_node(NODE_TSPEC,decl);
	const char* var_name = get_node_by_token(IDENTIFIER,decl)->buffer;
	if(tspec)
	{
		
		if(test_type(node->rhs,tspec->lhs->buffer))
		{
			node->expr_type = tspec->lhs->buffer;
			decl->expr_type = tspec->lhs->buffer;
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
			//TP should only consider the first declaration since that sets the type
			if(is_first_decl(decl))
			{
		      		node->expr_type = get_expr_type(node->rhs);
		      		decl->expr_type = node->expr_type;
			}
		}
	}
	return node->expr_type;

}
const char*
get_type_declaration_type(ASTNode* node)
{
	node->expr_type = get_node(NODE_TSPEC,node)->lhs->buffer;
	const char* var_name = get_node_by_token(IDENTIFIER,node)->buffer;
	return node->expr_type;
}
void
get_nodes(const ASTNode* node, node_vec* nodes, string_vec* names, const NodeType type)
{
	if(node->lhs)
		get_nodes(node->lhs,nodes,names,type);
	if(node->rhs)
		get_nodes(node->rhs,nodes,names,type);
	if(!(node->type & type)) return;
	push_node(nodes,node);
	push(names,get_node_by_token(IDENTIFIER,node)->buffer);
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
gen_local_type_info(ASTNode* node);
bool
gen_declared_type_info(ASTNode* node);
const char*
get_func_call_expr_type(ASTNode* node)
{
	if(node->lhs->type == NODE_STRUCT_EXPRESSION)
	{
	       const ASTNode* struct_expr   = node->lhs;
               const char* struct_func_name = get_node(NODE_MEMBER_ID,struct_expr->rhs)->buffer;
               const char* base_type        = get_expr_type(struct_expr->lhs);
               if(!strcmp_null_ok(base_type,MATRIX_STR) && !strcmps(struct_func_name,"col","row"))
                       return REAL3_STR;
	}
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	Symbol* sym = (Symbol*)get_symbol(NODE_VARIABLE_ID | NODE_FUNCTION_ID ,func_name,NULL);
	if(sym && sym->tspecifier == STENCIL_STR)
		return REAL_STR;
	if(sym && sym->type & NODE_FUNCTION_ID)
	{
		const ASTNode* func = NULL;
		for(size_t i = 0; i < dfunc_nodes.size; ++i)
			if(dfunc_names.data[i] == func_name) func = dfunc_nodes.data[i];
		if(strlen(sym->tspecifier))
			return sym->tspecifier;
		else if(func && func->expr_type)
			return func->expr_type;
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
					func_params_info info = get_function_params_info(func,func_name);
					if(!str_vec_contains(duplicate_dfuncs.names,func_name) && call_info.types.size == info.expr.size)
					{
						ASTNode* func_copy = astnode_dup(func,NULL);
						for(size_t i = 0; i < info.expr.size; ++i)
							set_primary_expression_types(func_copy, call_info.types.data[i], info.expr.data[i]);
						gen_local_type_info(func_copy);
						if(func_copy->expr_type) 
							node->expr_type = func_copy -> expr_type;
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
	return intern(res);
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
		const char* res = (n_structs_having_types== 1 && index >= 0) ? info.user_structs.data[index] : NULL;
		free_str_vec(&types);
		free_node_vec(&nodes);
		astnode_sprintf_prefix(node->parent,"(%s)",res);
		return res;
	}
	return node->expr_type;
}
const char*
get_cast_expr_type(ASTNode* node)
{
	const char* res = intern(combine_all_new(node->lhs));
	test_type(node->rhs,res);
	return res;
}
const char*
get_declared_expr_type(ASTNode* node)
{

	if(node->expr_type) return node->expr_type;
	const char* res = node->expr_type;
	if(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node))
		get_type_declaration_type(node);
	else if(node->type & NODE_ASSIGNMENT && get_parent_node(NODE_FUNCTION,node) &&  !get_node(NODE_MEMBER_ID,node->lhs) && !get_node(NODE_ARRAY_ACCESS,node->lhs))
		res = get_assignment_expr_type(node);
	else
	{
		if(node->lhs && !res)
			res = get_declared_expr_type(node->lhs);
		if(node->rhs && !res)
			res = get_declared_expr_type(node->rhs);
	}
	return res;
}
const char*
get_in_range_expr_type(ASTNode* node)
{
	const char* base_type = get_expr_type(node->rhs);
	const char* res = (!base_type)   ? NULL : 
		strstr(base_type,MULT_STR) ? intern(remove_substring(strdup(base_type),MULT_STR)) :
		strstr(base_type,"AcArray") ? get_array_elem_type(strdup(base_type)) :
		NULL;
	if(!node->lhs->expr_type)
		node->lhs->expr_type = res;
	return res;
}
const char*
get_expr_type(ASTNode* node)
{

	//TP: Cast is special since it overwrites other rules
	if(node->token == CAST)
	{
		node->expr_type = get_cast_expr_type(node);
		return node->expr_type;
	}
	if(node->expr_type) return node->expr_type;
	const char* res = node->expr_type;
	if(node->token == IN_RANGE)
		res = get_in_range_expr_type(node);
	else if(node->type & NODE_ARRAY_INITIALIZER)
		res = get_array_initializer_type(node);
	else if(node->type == NODE_PRIMARY_EXPRESSION)
		res = get_primary_expr_type(node);
	else if(node->type & NODE_STRUCT_INITIALIZER)
		res = get_struct_initializer_type(node);
	else if(node->type & NODE_ARRAY_ACCESS)
		res = get_array_access_type(node);
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
		res = get_assignment_expr_type(node);
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
	if(!type || strcmps(type,FIELD_STR,"VertexBufferHandle"))
		return;
	ASTNode* array_access = (ASTNode*)get_parent_node(NODE_ARRAY_ACCESS,node);
	if(!array_access || !is_left_child(NODE_ARRAY_ACCESS,node))	return;
	while(get_parent_node(NODE_ARRAY_ACCESS,array_access)) array_access = (ASTNode*) get_parent_node(NODE_ARRAY_ACCESS,array_access);

	if(node->type & NODE_MEMBER_ID)
	{
		node = (ASTNode*)get_parent_node(NODE_STRUCT_EXPRESSION,node);
		while(node->parent->type & NODE_STRUCT_EXPRESSION) node = node->parent;
	}

	node_vec nodes = VEC_INITIALIZER;
	get_array_access_nodes(array_access,&nodes);
	if(nodes.size != 1 && nodes.size != 3)	
	{
		fprintf(stderr,"Fatal error: only 1 and 3 -dimensional reads/writes are allowed for VertexBuffers\n");
	}


	ASTNode* idx_node = astnode_create(NODE_UNKNOWN,NULL,NULL);
	if(!gen_mem_accesses)
	{
		astnode_set_prefix("DEVICE_VTXBUF_IDX(",idx_node);
		astnode_set_postfix(")",idx_node);
	}
	ASTNode* rhs = astnode_create(NODE_UNKNOWN, idx_node, NULL);
	ASTNode* indexes = build_list_node(nodes,",");
	idx_node->lhs = indexes;
	indexes->parent = idx_node;

	ASTNode* before_lhs = NULL;
	if(gen_mem_accesses && is_left_child(NODE_ASSIGNMENT,node))
	{
		before_lhs = astnode_create(NODE_UNKNOWN,astnode_dup(node,NULL),astnode_dup(idx_node,NULL));
		astnode_set_prefix("mark_as_written(",before_lhs);
		astnode_set_infix(",",before_lhs);
		astnode_set_postfix(");",before_lhs);

		astnode_set_prefix("DEVICE_VTXBUF_IDX(",idx_node);
		astnode_set_postfix(")",idx_node);
	}

	free_node_vec(&nodes);
	astnode_free(array_access);
        array_access->rhs = rhs;
	ASTNode* lhs = astnode_create(NODE_UNKNOWN, before_lhs, astnode_dup(node,NULL));
	array_access->lhs = lhs;
	if(gen_mem_accesses && !is_left_child(NODE_ASSIGNMENT,node))
	{
		astnode_set_postfix(")",rhs);

		astnode_set_infix("AC_INTERNAL_read_field(",lhs);
		astnode_set_postfix(",",lhs);
	}
	else
	{
		astnode_set_prefix("[",rhs);
		astnode_set_postfix("]",rhs);

		astnode_set_infix("vba.in[",lhs);
		astnode_set_postfix("]",lhs);
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



bool
is_builtin_constant(const char* name)
{
	return strlen(name) > 2 && name[0] == 'A' && name[1] == 'C';
}
void
gen_const_def(const ASTNode* def, const ASTNode* tspec, FILE* fp, FILE* fp_builtin, FILE* fp_non_scalar_constants, FILE* fp_non_scalar_builtin)
{
		const ASTNode* name_node = get_node_by_token(IDENTIFIER,def);
		if(!name_node)
		{
			fatal("Could not find name for const declaration: %s\n",combine_all_new(def));
		}
		const char* name  = get_node_by_token(IDENTIFIER, def)->buffer;
		if(!name) return;
        	const ASTNode* assignment = def->rhs;
		if(!assignment) return;
		const ASTNode* struct_initializer = get_node(NODE_STRUCT_INITIALIZER,assignment);
		const ASTNode* array_initializer  = get_node(NODE_ARRAY_INITIALIZER, assignment);
		const ASTNode* array_access       = get_node(NODE_ARRAY_ACCESS, def->lhs);
		const char* datatype = tspec->lhs->buffer;
		const char* datatype_scalar = intern(remove_substring(strdup(datatype),MULT_STR));
		//TP: the C++ compiler is not always able to use the structs if you don't have the conversion from the initializer list
		//TP: e.g. multiplying a matrix with a scalar won't work without the conversion
		if(struct_initializer && !array_initializer)
			if(!struct_initializer->parent->postfix)
				astnode_sprintf_prefix(struct_initializer->parent,"(%s)",datatype_scalar);
		const char* assignment_val = intern(combine_all_new(assignment));
		const int array_dim = array_initializer ? count_nest(array_initializer,NODE_ARRAY_INITIALIZER) : 0;
		if(array_initializer)
		{
			const int num_of_elems = array_initializer ? count_num_of_nodes_in_list(array_initializer->lhs) : 0;
			const ASTNode* second_array_initializer = get_node(NODE_ARRAY_INITIALIZER, array_initializer->lhs);
			if(array_dim == 1)
			{
				if(is_builtin_constant(name))
					fprintf(fp_non_scalar_builtin, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s[%d] = %s;\n#endif\n",datatype_scalar, name, num_of_elems, assignment_val);
				else
				{
					fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s[%d] = %s;\n#endif\n",datatype_scalar, name, num_of_elems, assignment_val);
					fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s[%d] = %s;\n#endif\n",datatype_scalar, name, num_of_elems, assignment_val);
				}
			}
			else if(array_dim == 2)
			{
				const int num_of_elems_in_list = count_num_of_nodes_in_list(second_array_initializer->lhs);
				if(is_builtin_constant(name))
					fprintf(fp_non_scalar_builtin, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s[%d*%d] = %s;\n#endif\n",datatype_scalar, name, num_of_elems_in_list, num_of_elems, assignment_val);
				else
				{
					fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s[%d*%d] = %s;\n#endif\n",datatype_scalar, name, num_of_elems_in_list, num_of_elems, assignment_val);
					fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s[%d][%d] = %s;\n#endif\n",datatype_scalar, name, num_of_elems_in_list, num_of_elems, assignment_val);
				}

			}
			else
				fatal("todo add 3d const arrays\n");
		}
		else if(array_access)
		{
			const char* num_of_elems = combine_all_new(array_access->rhs);
				fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] const %s %s[%s] = %s;\n#endif\n",datatype_scalar, name, num_of_elems, assignment_val);
				fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] const %s %s[%s] = %s;\n#endif\n",datatype_scalar, name, num_of_elems, assignment_val);
		}
		else
		{
		        //TP: define macros have greater portability then global constants, since they do not work on some CUDA compilers
			//TP: actually can not make macros since if the user e.g. writes const nx = 3 then that define would conflict with variables in hip
                        if(is_primitive_datatype(datatype_scalar))
			{
                                if(is_builtin_constant(name)) fprintf(fp_builtin, "#define %s ((%s)%s)\n",name, datatype_scalar, assignment_val);
				if(!is_builtin_constant(name))
				{
                                	fprintf(fp, "[[maybe_unused]] const constexpr %s %s = %s;\n", datatype_scalar, name, assignment_val);
                                	fprintf(fp_non_scalar_constants, "[[maybe_unused]] const constexpr %s %s = %s;\n", datatype_scalar, name, assignment_val);
				}
			}
                        else
			{
                               fprintf(
						is_builtin_constant(name) ? fp_non_scalar_builtin : fp_non_scalar_constants,  
						"\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s = %s;\n#endif\n",datatype_scalar, name, assignment_val
				);
			        if(!is_builtin_constant(name))
                               		fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] const constexpr %s %s = %s;\n#endif\n",datatype_scalar, name, assignment_val);
			}
		}
}

void
gen_const_variables(const ASTNode* node, FILE* fp, FILE* fp_bi,FILE* fp_non_scalars,FILE* fp_bi_non_scalars)
{
	if(node->lhs)
		gen_const_variables(node->lhs,fp,fp_bi,fp_non_scalars,fp_bi_non_scalars);
	if(node->rhs)
		gen_const_variables(node->rhs,fp,fp_bi,fp_non_scalars,fp_bi_non_scalars);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	if(!has_qualifier(node,"const")) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	node_vec assignments = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < assignments.size; ++i)
		gen_const_def(assignments.data[i],tspec,fp,fp_bi,fp_non_scalars,fp_bi_non_scalars);
	free_node_vec(&assignments);
}

static int curr_kernel = 0;

static void
gen_kernels_recursive(const ASTNode* node, char** dfunctions,
            const bool gen_mem_accesses)
{
  assert(node);

  if (node->lhs)
    gen_kernels_recursive(node->lhs, dfunctions, gen_mem_accesses);
  if (node->type & NODE_KFUNCTION) {

    const size_t len = 64 * 1024 * 1024;
    char* prefix     = malloc(len);
    assert(prefix);
    prefix[0] = '\0';

    const char* name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
    const int_vec called_dfuncs = calling_info.called_funcs[str_vec_get_index(calling_info.names,name)];

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
	    if(str_vec_contains(dfunc_symbol->tqualifiers,INLINE_STR)) continue;
	    if(int_vec_contains(called_dfuncs,i)) strcat(prefix,dfunctions[i]);
    }

    astnode_set_prefix(prefix, compound_statement);
    free(prefix);
    free(cmdoptions);
  }


  if (node->rhs)
    gen_kernels_recursive(node->rhs, dfunctions, gen_mem_accesses);
}
static void
gen_kernels(const ASTNode* node, char** dfunctions,
            const bool gen_mem_accesses)
{
  	traverse(node, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
	gen_kernels_recursive(node,dfunctions,gen_mem_accesses);
  	symboltable_reset();
}

string_vec
get_names(const char* type)
{
	string_vec res = VEC_INITIALIZER;
	for(size_t i = 0; i < num_symbols[0]; ++i)
		if(symbol_table[i].tspecifier == type)
			push(&res, symbol_table[i].identifier);
	return res;
}
string_vec
get_names_variadic(const string_vec types)
{
	string_vec res = VEC_INITIALIZER;
	for(size_t i = 0; i < num_symbols[0]; ++i)
		if(str_vec_contains(types,symbol_table[i].tspecifier))
			push(&res, symbol_table[i].identifier);
	return res;
}


void
gen_names(const char* datatype, const char* type, FILE* fp)
{
	string_vec names = get_names(type); 
	fprintf(fp,"static const char* %s_names[] __attribute__((unused)) = {",datatype);
	for(size_t i = 0; i < names.size; ++i)
  		fprintf(fp, "\"%s\",", names.data[i]);
	//TP: add padding that in case there are 0 instances of the datatype to not get constant compiler warnings
	fprintf(fp,"\"padding\",");
	fprintf(fp,"};\n");
	free_str_vec(&names);
}
void
gen_names_variadic(const char* datatype, const string_vec types, FILE* fp)
{
	string_vec names = get_names_variadic(types); 
	fprintf(fp,"static const char* %s_names[] __attribute__((unused)) = {",datatype);
	for(size_t i = 0; i < names.size; ++i)
  		fprintf(fp, "\"%s\",", names.data[i]);
	fprintf(fp,"};\n");
	free_str_vec(&names);
}
static size_t
count_symbols(const char* type)
{
	size_t res = 0;
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    		if(symbol_table[i].tspecifier == type)
      			++res;
	return res;
}
static size_t
count_symbols_type(const NodeType type)
{
	size_t res = 0;
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    		if(symbol_table[i].type & type)
      			++res;
	return res;
}
static size_t
count_variables(const char* datatype, const char* qual)
{
	size_t res = 0;
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    		if((!datatype || symbol_table[i].tspecifier == datatype) && str_vec_contains(symbol_table[i].tqualifiers,qual) && (symbol_table[i].type & NODE_VARIABLE_ID))
      			++res;
	return res;
}
static size_t
count_profiles()
{

  string_vec prof_types = get_prof_types();
  size_t res = 0;
  for(size_t i = 0; i < prof_types.size; ++i)
	  res += count_symbols(prof_types.data[i]);
  return res;
}

void
gen_enums(FILE* fp, const char* type, const char* prefix, const char* num_name, const char* name)
{
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier == type)

      fprintf(fp, "%s%s,",prefix, symbol_table[i].identifier);
  fprintf(fp, "%s} %s;",num_name,name);
}

void
gen_enums_variadic(FILE* fp, const string_vec  types, const char* prefix, const char* num_name, const char* name)
{
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(str_vec_contains(types,symbol_table[i].tspecifier))
      fprintf(fp, "%s%s,",prefix, symbol_table[i].identifier);
  fprintf(fp, "%s} %s;",num_name,name);
}
static void
gen_field_info(FILE* fp)
{
  num_fields   = count_symbols(FIELD_STR);

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
    if(symbol_table[i].tspecifier == FIELD_STR)
	    push(&original_names,symbol_table[i].identifier);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if(symbol_table[i].tspecifier == FIELD_STR){
      const bool is_dead = str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR);
      if(is_dead) continue;
      push(&field_names, symbol_table[i].identifier);
      const char* name = symbol_table[i].identifier;
      const bool is_aux  = str_vec_contains(symbol_table[i].tqualifiers,AUXILIARY_STR);
      const bool is_comm = str_vec_contains(symbol_table[i].tqualifiers,COMMUNICATED_STR);
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
    if(symbol_table[i].tspecifier == FIELD_STR){
      const bool is_dead = str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR);
      if(!is_dead) continue;
      push(&field_names, symbol_table[i].identifier);
      const char* name = symbol_table[i].identifier;
      const bool is_aux  = str_vec_contains(symbol_table[i].tqualifiers,AUXILIARY_STR);
      const bool is_comm = str_vec_contains(symbol_table[i].tqualifiers,COMMUNICATED_STR);
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
  //TP: IMPORTANT!! if there are dead fields NUM_VTXBUF_HANDLES is equal to alive fields not all fields.  
  //TP: the compiler is allowed to move dead field declarations till the end
  //TP: this way the user can easily loop all alive fields with the old 0:NUM_VTXBUF_HANDLES and same for the Astaroth library dead fields are skiped over automatically
  {
        string_vec prof_types = get_prof_types();
	FILE* fp_enums = fopen("builtin_enums.h","w");
  	fprintf(fp_enums,"#pragma once\n");
  	gen_enums(fp_enums,STENCIL_STR,"stencil_","NUM_STENCILS","Stencil");
  	gen_enums(fp_enums,intern("WorkBuffer"),"","NUM_WORK_BUFFERS","WorkBuffer");
  	gen_enums(fp_enums,KERNEL_STR,"","NUM_KERNELS","AcKernel");
  	gen_enums_variadic(fp_enums,prof_types,"","NUM_PROFILES","Profile");
  	fprintf(fp_enums, "typedef enum {");
  	for(size_t i = 0; i < num_of_fields; ++i)
  	        fprintf(fp_enums,"%s,",field_names.data[i]);

  	fprintf(fp_enums, "} Field;\n");
  	if(has_optimization_info())
  		fprintf(fp_enums, "#define NUM_FIELDS (%ld)\n", num_of_alive_fields);
  	else
  		fprintf(fp_enums, "#define NUM_FIELDS (%ld)\n", num_of_fields);
	fprintf(fp_enums, "#define NUM_ALL_FIELDS (%ld)\n",num_of_fields);
  	fprintf(fp_enums, "#define NUM_DEAD_FIELDS (%ld)\n", num_of_fields-num_of_alive_fields);
  	fprintf(fp_enums, "#define NUM_COMMUNICATED_FIELDS (%d)\n", num_of_communicated_fields);
	fclose(fp_enums);
  }

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
    fprintf(fp_vtxbuf_is_comm_func,"case(%s): return %s;\n", field_names.data[i], (field_is_communicated[i]) ? "true" : "false");

  fprintf(fp_vtxbuf_is_comm_func,"default: return false;\n");
  fprintf(fp_vtxbuf_is_comm_func, "}\n}\n");

  fclose(fp_vtxbuf_is_comm_func);

  fp = fopen("field_names.h","w");
  fprintf(fp,"static const char* field_names[] __attribute__((unused)) = {");
  for(size_t i=0;i<num_of_fields;++i)
	  fprintf(fp,"\"%s\",",field_names.data[i]);
  fprintf(fp,"};\n");
  fprintf(fp, "static const char** vtxbuf_names __attribute__((unused)) = field_names;\n");
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
const char*
to_dsl_type(const char* type)
{
	if(!strcmp(type,"AcReal2")) return "real2";
	if(!strcmp(type,"AcReal3")) return "real3";
	if(!strcmp(type,"AcReal4")) return "real4";
	if(!strcmp(type,COMPLEX_STR)) return "complex";
	if(!strcmp(type,"AcBool3")) return "bool3";
	return NULL;
}

void
remove_extra_braces_in_arr_initializers(ASTNode* node)
{
	TRAVERSE_PREAMBLE(remove_extra_braces_in_arr_initializers);
	if(!(node->type & NODE_ARRAY_INITIALIZER)) return;
	if(!get_parent_node(NODE_ARRAY_INITIALIZER,node)) return;
	astnode_set_postfix("",node);
	astnode_set_prefix("",node);
}

static void
gen_user_defines(const ASTNode* root_in, const char* out)
{
  ASTNode* root = astnode_dup(root_in,NULL);
  remove_extra_braces_in_arr_initializers(root);
  FILE* fp = fopen(out, "w");
  fprintf(fp, "#pragma once\n");
  assert(fp);


  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, fp);

  num_fields  = count_symbols(FIELD_STR);
  num_kernels = count_symbols(KERNEL_STR);
  num_dfuncs  = count_symbols_type(NODE_DFUNCTION_ID);

  

  //TP: fields info is generated separately since it is different between 
  //analysis generation and normal generation
  fprintf(fp,"\n#include \"fields_info.h\"\n");
  // Enums
  //
  string_vec prof_types = get_prof_types();



  {
	FILE* stream = fopen("profiles_info.h","w");
  	fprintf(stream,"static AcProfileType prof_types[NUM_PROFILES] = {");
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
	  if(str_vec_contains(prof_types,symbol_table[i].tspecifier))
  	  {
  	          if(symbol_table[i].tspecifier == intern("Profile<X>"))
  	      		    fprintf(stream,"PROFILE_X");
  	          else if(symbol_table[i].tspecifier == intern("Profile<Y>"))
  	      		    fprintf(stream,"PROFILE_Y");
  	          else if(symbol_table[i].tspecifier == intern("Profile<Z>"))
  	      		    fprintf(stream,"PROFILE_Z");
  	          else if(symbol_table[i].tspecifier == intern("Profile<XY>"))
  	      		    fprintf(stream,"PROFILE_XY");
  	          else if(symbol_table[i].tspecifier == intern("Profile<XZ>"))
  	      		    fprintf(stream,"PROFILE_XZ");
  	          else if(symbol_table[i].tspecifier == intern("Profile<YX>"))
  	      		    fprintf(stream,"PROFILE_YX");
  	          else if(symbol_table[i].tspecifier == intern("Profile<YZ>"))
  	      		    fprintf(stream,"PROFILE_YZ");
  	          else if(symbol_table[i].tspecifier == intern("Profile<ZX>"))
  	      		    fprintf(stream,"PROFILE_ZX");
  	          else if(symbol_table[i].tspecifier == intern("Profile<ZY>"))
  	      		    fprintf(stream,"PROFILE_ZY");
  	          else
  	          	fatal("Unknown profile type for: %s\n",symbol_table[i].identifier);
		  fprintf(stream,",");
  	  }
  	fprintf(stream,"};\n");
	fclose(stream);
  }


  fprintf(fp, "static const bool skip_kernel_in_analysis[NUM_KERNELS] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == KERNEL_STR)
    {
      if (str_vec_contains(symbol_table[i].tqualifiers,UTILITY_STR))
	      fprintf(fp,"true,");
      else
	      fprintf(fp,"false,");
    }
  fprintf(fp, "};");

  fprintf(fp, "static const bool kernel_has_fixed_boundary[NUM_KERNELS] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == KERNEL_STR)
    {
      if (str_vec_contains(symbol_table[i].tqualifiers,FIXED_BOUNDARY_STR))
	      fprintf(fp,"true,");
      else
	      fprintf(fp,"false,");
    }
  fprintf(fp, "};");

  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  // START---------------------------


  // Enum strings (convenience)
  gen_names("stencil",STENCIL_STR,fp);
  gen_names("work_buffer",intern("WorkBuffer"),fp);
  gen_names("kernel",KERNEL_STR,fp);
  gen_names_variadic("profile",prof_types,fp);
  //TP: field names have to be generated differently since they might get reorder because of dead fields
  fprintf(fp,"\n#include \"field_names.h\"\n");


  for (size_t i = 0; i < s_info.user_structs.size; ++i)
  {
	  fprintf_filename("to_str_funcs.h","std::string to_str(const %s value)\n"
		       "{\n"
		       "std::string res = \"{\";"
		       "std::string tmp;\n"
		       ,s_info.user_structs.data[i]);

	  for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
	  {
		const char* middle = (j < s_info.user_struct_field_names[i].size -1) ? "res += \",\";\n" : "";
		fprintf_filename("to_str_funcs.h","res += to_str(value.%s);\n"
				"%s"
		,s_info.user_struct_field_names[i].data[j],middle);
	  }
	  fprintf_filename("to_str_funcs.h",
			  "res += \"}\";\n"
			  "return res;\n"
			  "}\n"
	  );
	  const char* dsl_type = to_dsl_type(s_info.user_structs.data[i]);
	  const char* res = dsl_type ? dsl_type : s_info.user_structs.data[i];
	  fprintf_filename("to_str_funcs.h","template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n", s_info.user_structs.data[i], res);
  }


  string_vec datatypes = get_all_datatypes();

  for (size_t i = 0; i < datatypes.size; ++i)
  {
	  const char* datatype = datatypes.data[i];
	  gen_param_names(fp,datatype);
	  gen_datatype_enums(fp,datatype);

	  fprintf_filename("device_mesh_info_decl.h","%s %s_params[NUM_%s_PARAMS+1];\n",datatype,convert_to_define_name(datatype),strupr(convert_to_define_name(datatype)));
	  gen_array_declarations(datatype,root);
	  gen_comp_declarations(datatype);
  }

  const size_t num_real_outputs = count_variables(REAL_STR,OUTPUT_STR);
  fprintf(fp,"\n#define  NUM_OUTPUTS (NUM_REAL_OUTPUTS+NUM_INT_OUTPUTS+NUM_PROFILES)\n");
  fprintf(fp,"\n#define  PROF_SCRATCHPAD_INDEX(PROF) (NUM_REAL_OUTPUTS+PROF)\n");
  const size_t num_real_scratchpads = max(1,num_profiles+num_real_outputs);
  fprintf(fp,"\n#define  NUM_REAL_SCRATCHPADS (%zu)\n",num_real_scratchpads);
 
  const size_t num_dconsts = count_variables(NULL,DCONST_STR);
  fprintf(fp,"\n#define NUM_DCONSTS (%zu)\n",num_dconsts);

  fprintf(fp,"\n#include \"array_info.h\"\n");
  fprintf(fp,"\n#include \"taskgraph_enums.h\"\n");

  free_str_vec(&datatypes);
  free_structs_info(&s_info);



  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  fprintf(fp,
  	"\n// Redefined for backwards compatibility START\n"
  	"#define NUM_VTXBUF_HANDLES (NUM_FIELDS)\n"
  	"typedef Field VertexBufferHandle;\n"
	 );
  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  // END-----------------------------

  // Device constants
  // Would be cleaner to declare dconsts as extern and refer to the symbols
  // directly instead of using handles like above, but for backwards
  // compatibility and user convenience commented out for now
  /**
  for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
    if (!(symbol_table[i].type & NODE_FUNCTION_ID) &&
        !(symbol_table[i].type & NODE_FIELD_ID) &&
        !(symbol_table[i].type & NODE_PROFILE_ID) &&
        !(symbol_table[i].type & NODE_STENCIL_ID)) {
      fprintf(fp, "// extern __device__ %s %s;\n", symbol_table[i].tspecifier,
              symbol_table[i].identifier);
    }
  }
  **/

  // Stencil order
  fprintf(fp,
  	"#ifndef STENCIL_ORDER\n"
  	"#define STENCIL_ORDER (6)\n"
  	"#endif\n"
  	"#define STENCIL_HEIGHT (STENCIL_ORDER+1)\n"
  	"#define STENCIL_WIDTH  (STENCIL_ORDER+1)\n"
  	"#define STENCIL_DEPTH  (STENCIL_ORDER+1)\n"
  	"#define NGHOST (STENCIL_ORDER/2)\n"
  	"#define NGHOST_X (STENCIL_ORDER/2)\n"
  	"#define NGHOST_Y (STENCIL_ORDER/2)\n"
  	"#if TWO_D == 0\n"
	"#define NGHOST_Z (STENCIL_ORDER / 2)\n"
	"#else\n"
	"#define NGHOST_Z (0)\n"
	"#endif\n"
	 );

  char cwd[9000];
  cwd[0] = '\0';
  char* err = getcwd(cwd, sizeof(cwd));
  assert(err != NULL);
  char autotune_path[10004];
  sprintf(autotune_path,"%s/autotune.csv",cwd);
  fprintf(fp,
  	"__attribute__((unused)) static const char* autotune_csv_path= \"%s\";\n"
  	"__attribute__((unused)) static const char* runtime_astaroth_path = \"%s\";\n"
  	"__attribute__((unused)) static const char* runtime_astaroth_runtime_path = \"%s\";\n"
  	"__attribute__((unused)) static const char* runtime_astaroth_utils_path = \"%s\";\n"
  	"__attribute__((unused)) static const char* runtime_astaroth_build_path = \"%s\";\n"
  	"__attribute__((unused)) static const char* acc_compiler_path  = \"%s\";\n"
  	"__attribute__((unused)) static const char* astaroth_base_path = \"%s\";\n"
	,autotune_path
	,AC_BASE_PATH"/runtime_compilation/build/src/core/libastaroth_core.so"
	,AC_BASE_PATH"/runtime_compilation/build/src/core/kernels/libkernels.so"
	,AC_BASE_PATH"/runtime_compilation/build/src/utils/libastaroth_utils.so"
	,AC_BASE_PATH"/runtime_compilation/build"
	,ACC_COMPILER_PATH
	,AC_BASE_PATH
	);

  fclose(fp);

  //Done to refresh the autotune file when recompiling DSL code
  fp = fopen(autotune_path,"w");
  fclose(fp);

  fp = fopen("user_constants.h","w");
  FILE* fp_built_in  = fopen("user_built-in_constants.h","w");
  FILE* fp_non_scalar_constants = fopen("user_non_scalar_constants.h","w");
  FILE* fp_bi_non_scalar_constants = fopen("user_builtin_non_scalar_constants.h","w");

  fprintf(fp,"#pragma once\n");
  fprintf(fp_built_in,"#pragma once\n");
  //fprintf(fp_non_scalar_constants,"#pragma once\n");
  fprintf(fp_bi_non_scalar_constants,"#pragma once\n");

  gen_const_variables(root,fp,fp_built_in,fp_non_scalar_constants,fp_bi_non_scalar_constants);


  fclose(fp);
  fclose(fp_built_in);
  fclose(fp_non_scalar_constants);
  fclose(fp_bi_non_scalar_constants);
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
    if (symbol_table[i].tspecifier == KERNEL_STR)
      fprintf(fp_dec, "static void __global__ KERNEL_%s %s);\n", symbol_table[i].identifier, default_param_list);

  fprintf(fp_dec, "static const Kernel kernels[] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == KERNEL_STR)
      fprintf(fp_dec, "KERNEL_%s,", symbol_table[i].identifier);
  fprintf(fp_dec, "};");

  fclose(fp_dec);

  // Astaroth 2.0 backwards compatibility END

  fclose(fp);
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
	if(node->type & NODE_MEMBER_ID) return;
	if(strcmp(node->buffer,old_name)) return;
	if(check_symbol(NODE_FUNCTION_ID,node->buffer,0,0)) return;
	astnode_set_buffer(new_name,node);
}
void
append_to_identifiers(const char* str_to_append, ASTNode* node, const char* str_to_check)
{
	if(node->lhs)
		append_to_identifiers(str_to_append,node->lhs,str_to_check);
	if(node->rhs)
		append_to_identifiers(str_to_append,node->rhs,str_to_check);
	if(do_not_rename(node,str_to_check)) return;
	astnode_sprintf(node,"%s___AC_INTERNAL_%s",node->buffer,str_to_append);
}

void
rename_local_vars(const char* str_to_append, ASTNode* node, ASTNode* dfunc_start)
{
	if(node->lhs)
		rename_local_vars(str_to_append,node->lhs,dfunc_start);
	if(node->type & NODE_DECLARATION)
	{
		const char* name = get_node_by_token(IDENTIFIER,node)->buffer;
		if(!(node->parent->type & NODE_FUNCTION_CALL))
		append_to_identifiers(str_to_append,dfunc_start,name);
	}
	if(node->rhs)
		rename_local_vars(str_to_append,node->rhs,dfunc_start);
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
	const char* func_name = get_node_by_token(IDENTIFIER,node)->buffer;
	if(!func_name || !check_symbol(NODE_DFUNCTION_ID,func_name,0,INLINE_STR)) return;
	//to exclude input params
	//rename_local_vars(fn_identifier->buffer,node->rhs->rhs,node->rhs->rhs);
	char* prefix = NULL;
	asprintf(&prefix,"%s_REPLACE_WITH_INLINE_COUNTER",func_name);
	rename_local_vars(prefix,node->rhs,node);
	free(prefix);
}
const ASTNode*
get_dfunc(const char* name)
{
		for(size_t i = 0; i < dfunc_names.size; ++i)
			if(name == dfunc_names.data[i]) return dfunc_nodes.data[i];
		return NULL;
}
void
add_to_node_list(ASTNode* head, const ASTNode* new_node)
{
	while(head->rhs) head = head->lhs;
	ASTNode* last_elem = head->lhs;
	ASTNode* new_last = astnode_create(NODE_UNKNOWN,astnode_dup(new_node,NULL),NULL);
	new_last->type  = head->type;
	new_last->token = head->token;
	head->lhs = new_last;
	head->rhs = last_elem;
}
void
replace_substrings(ASTNode* node, const char* to_replace, const char* replacement)
{
	if(node->lhs)
		replace_substrings(node->lhs,to_replace,replacement);
	if(node->rhs)
		replace_substrings(node->rhs,to_replace,replacement);
	if(node->expr_type)
	{
		char* tmp = strdup(node->expr_type);
		replace_substring(&tmp,to_replace,replacement);
		node->expr_type = intern(tmp);
	}
	if(node->prefix)
	{
		char* tmp = strdup(node->prefix);
		replace_substring(&tmp,to_replace,replacement);
		astnode_set_prefix(tmp,node);
	}
	if(node->infix)
	{
		char* tmp = strdup(node->infix);
		replace_substring(&tmp,to_replace,replacement);
		astnode_set_infix(tmp,node);
	}
	if(node->buffer)
	{
		char* tmp = strdup(node->buffer);
		replace_substring(&tmp,to_replace,replacement);
		astnode_set_buffer(tmp,node);
	}
	if(node->postfix)
	{
		char* tmp = strdup(node->postfix);
		replace_substring(&tmp,to_replace,replacement);
		astnode_set_postfix(tmp,node);
	}
}
static
bool is_return_node(const ASTNode* node)
{
	const bool res = 
		!node->lhs ? false :
		node->lhs->token == RETURN;
	return res;
}

void replace_return_nodes(ASTNode* node, const ASTNode* decl_node)
{
	TRAVERSE_PREAMBLE_PARAMS(replace_return_nodes,decl_node);
	if(!is_return_node(node)) return;
	replace_node(node,create_assignment(decl_node,node->rhs,EQ_STR));
}

ASTNode*
inline_returning_function(const ASTNode* node, int counter)
{
	ASTNode* func_node = (ASTNode*) get_node(NODE_FUNCTION_CALL,node);
	ASTNode* decl_node = (ASTNode*) get_node(NODE_DECLARATION,node);
	if(!func_node || !func_node->lhs) return NULL;
	const char* func_name = get_node_by_token(IDENTIFIER,func_node->lhs)->buffer;
	if(!func_name || !check_symbol(NODE_DFUNCTION_ID,func_name,0,INLINE_STR)) return NULL;
	const ASTNode* dfunc = get_dfunc(func_name);
	if(!dfunc) return NULL;
	if(!dfunc->rhs->rhs->lhs) return NULL;
	ASTNode* new_dfunc = astnode_dup(dfunc,NULL);
	ASTNode* dfunc_statements = new_dfunc->rhs->rhs->lhs;
	
	replace_return_nodes(dfunc_statements,decl_node);	
	const char* full = combine_all_new(decl_node);

	node_vec params = VEC_INITIALIZER;
	if(func_node->rhs) params = get_nodes_in_list(func_node->rhs);
	func_params_info params_info = get_function_params_info(dfunc,func_name);
	for(size_t i = 0; i < params.size; ++i)
	{

		const bool is_constexpr = all_identifiers_are_constexpr(params.data[i]);
		const char* type = params_info.types.data[i] ? sprintf_intern("%s%s",
				params_info.types.data[i], is_constexpr ? "" : "&")
			: NULL;
		ASTNode* alias = create_assignment(create_declaration(params_info.expr.data[i],type,CONST_STR),params.data[i],EQ_STR);
		add_to_node_list(dfunc_statements,alias);
	}
	char* replacement = itoa(counter);
	replace_substrings(new_dfunc,"REPLACE_WITH_INLINE_COUNTER",replacement);
	free(replacement);
	free_func_params_info(&params_info);
	free_node_vec(&params);
	return new_dfunc->rhs->rhs->lhs;
}
const ASTNode* get_return_node(const ASTNode* node)
{
	const ASTNode* res = NULL;
	if(node->lhs) res = get_return_node(node->lhs);
	if(is_return_node(node)) return node;
	if(!res && node->rhs) return get_return_node(node->rhs);
	return res;
}
ASTNode*
inline_non_returning_function(const ASTNode* node, int counter)
{
	ASTNode* func_node = (ASTNode*) get_node(NODE_FUNCTION_CALL,node);
	ASTNode* decl_node = (ASTNode*) get_node(NODE_DECLARATION,node);
	if(!func_node || !func_node->lhs) return NULL;
	const char* func_name = get_node_by_token(IDENTIFIER,func_node->lhs)->buffer;
	const ASTNode* dfunc = get_dfunc(func_name);
	if(!dfunc) return NULL;
	if(!dfunc->rhs->rhs->lhs) return NULL;
	ASTNode* new_dfunc = astnode_dup(dfunc,NULL);
	if(get_return_node(new_dfunc->rhs))
		fatal("Not capturing return value of inlined value not allowed\n");
	ASTNode* dfunc_statements = new_dfunc->rhs->rhs->lhs;
	node_vec params = VEC_INITIALIZER;
	if(func_node->rhs) params = get_nodes_in_list(func_node->rhs);
	func_params_info params_info = get_function_params_info(dfunc,func_name);
	for(size_t i = 0; i < params.size; ++i)
	{
		const bool is_constexpr = all_identifiers_are_constexpr(params.data[i]);
		const char* type = params_info.types.data[i] ? sprintf_intern("%s%s",
				params_info.types.data[i], is_constexpr ? "" : "&")
			: NULL;
		ASTNode* alias = create_assignment(create_declaration(params_info.expr.data[i],type,CONST_STR),params.data[i],EQ_STR);
		add_to_node_list(dfunc_statements,alias);
	}
	char* replacement = itoa(counter);
	replace_substrings(new_dfunc,"REPLACE_WITH_INLINE_COUNTER",replacement);
	const char* debug = combine_all_new(new_dfunc);
	if(strstr(debug,"REPLACE_WITH_INLINE_COUNTER")) printf("WRONG!\n");
	free(replacement);
	return new_dfunc->rhs->rhs->lhs;
}
bool
inline_dfuncs_recursive(ASTNode* node, int* counter)
{

	bool res = false;
	if(node->type == NODE_BOUNDCONDS_DEF) return res;
	if(node->lhs)
		res |= inline_dfuncs_recursive(node->lhs,counter);
	if(node->rhs)
		res |= inline_dfuncs_recursive(node->rhs,counter);

	if(!(node->type & NODE_ASSIGNMENT) && node->token != NON_RETURNING_FUNC_CALL) return res;
	ASTNode* func_node = (ASTNode*) get_node(NODE_FUNCTION_CALL,node);
	if(!func_node || !func_node->lhs) return NULL;
	const char* func_name = get_node_by_token(IDENTIFIER,func_node)->buffer;
	if(!func_name || !check_symbol(NODE_DFUNCTION_ID,func_name,0,INLINE_STR)) return NULL;
	ASTNode* res_node = node->type & NODE_ASSIGNMENT
			    ? inline_returning_function(node,*counter)
			    : inline_non_returning_function(node,*counter);
	if(!res_node) return res;
	(*counter)++;

	astnode_sprintf_prefix (res_node,"//start of inlined %s\n",combine_all_new(func_node));
	astnode_sprintf_postfix(res_node,"//endof inlined %s\n",combine_all_new(func_node));
	replace_node(node,res_node);

	return true;
}
const ASTNode* 
push_to_statement_list(ASTNode* head, ASTNode* res)
{

	if(!(head->type & NODE_STATEMENT_LIST_HEAD)) fatal("WRONG\n");
	if(!head->rhs)
	{
		const char* debug =combine_all_new(res);
		head->rhs = head->lhs;
		head->type = NODE_STATEMENT_LIST_HEAD;
		ASTNode* statement = astnode_create(NODE_UNKNOWN, res, NULL);
		head->lhs=  statement;
		statement->parent = head;
	}
	else
	{
		const char* debug =combine_all_new(res);
		ASTNode* new_lhs = astnode_create(NODE_STATEMENT_LIST_HEAD,head->lhs, res);
		head->lhs = new_lhs;
		new_lhs->parent = head;
	}
	return head;
}
//TP: canonalizes inline func calls to always assign their return value i.e.  var = f() e.g. h(f()) ---> var = f(); h(var);
bool
turn_inline_function_calls_to_assignments_in_statement(ASTNode* node, ASTNode* base_statement, int* counter)
{
	bool res = false;
	if(node->lhs)
		res |= turn_inline_function_calls_to_assignments_in_statement(node->lhs,base_statement,counter);
	if(node->rhs)
		res |= turn_inline_function_calls_to_assignments_in_statement(node->rhs,base_statement,counter);
	if(!(node->type & NODE_FUNCTION_CALL) || !node->lhs ) return res;
	if(node->parent->token == NON_RETURNING_FUNC_CALL) return res;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	if(!func_name || !check_symbol(NODE_DFUNCTION_ID,func_name,0,INLINE_STR)) return res;
	const ASTNode* unary_op = get_parent_node_by_token(UNARY,node);
	const bool in_unary_op = unary_op && unary_op->rhs;
	if(!(in_unary_op || get_parent_node(NODE_FUNCTION_CALL,node) || get_parent_node_by_token(BINARY,node) || get_parent_node_by_token(CAST,node))) return res;
	char* inline_var_name;
	asprintf(&inline_var_name,"inlined_var_return_value_%d",*counter);
	ASTNode* decl  = create_declaration(inline_var_name,NULL,NULL);
	ASTNode* node_res = create_assignment(decl,node,"=");
	replace_node(node,create_primary_expression(inline_var_name));
	ASTNode* head = (ASTNode*)get_parent_node_inclusive(NODE_STATEMENT_LIST_HEAD,base_statement);
	//if(!head) fatal("NO HEAD!: %s\n",combine_all_new(node));
	push_to_statement_list(head,node_res);
	(*counter)++;
	free(inline_var_name);
	return true;
}
bool
turn_inline_function_calls_to_assignments(ASTNode* node, int* counter)
{
	bool res = false;
	if(node->lhs)
		res |= turn_inline_function_calls_to_assignments(node->lhs,counter);
	if(node->rhs)
		res |= turn_inline_function_calls_to_assignments(node->rhs,counter);

	if(node->token != STATEMENT)
		return false;
	ASTNode* head = (ASTNode*)get_parent_node_inclusive(NODE_STATEMENT_LIST_HEAD,node);
	if(!head) fatal("NO HEAD!: %s\n",combine_all_new(node));

	ASTNode* basic_statement = get_node_by_token(BASIC_STATEMENT,node);
	if(!basic_statement) return res;
	const bool did_something = turn_inline_function_calls_to_assignments_in_statement(node,node,counter);
	res |= did_something;
	return res;
}
void
inline_dfuncs(ASTNode* node)
{
  traverse(node,NODE_NO_OUT,NULL);
  bool inlined_something = true;
  int var_counter = 0;
  int counter = 0;
  while(inlined_something)
  {
  	inlined_something  = turn_inline_function_calls_to_assignments(node,&var_counter);
  	inlined_something |= inline_dfuncs_recursive(node, &counter);
  }

}
void
transform_arrays_to_std_arrays_in_func(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_arrays_to_std_arrays_in_func);
	if(!(node->type & NODE_DECLARATION))
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
	const char* dim = combine_all_new(node->rhs->lhs->rhs);
	//TP: at least some CUDA compilers do not allow zero-sized objects in device code so have to pad the array length
	if(!strcmp(dim,"(0)"))
	{
		dim = "(1)";
	}
	astnode_sprintf(tspec->lhs,"AcArray<%s,%s>",tspec->lhs->buffer,dim);
	node->rhs->lhs->infix = NULL;
	node->rhs->lhs->postfix= NULL;
	node->rhs->lhs->rhs = NULL;
	//remove unneeded braces if assignment
	if(node->parent->type & NODE_ASSIGNMENT && node->parent->rhs)
		node->parent->rhs->prefix = NULL;
}
void
transform_arrays_to_std_arrays(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_arrays_to_std_arrays);
	if(node->type & NODE_FUNCTION)
		transform_arrays_to_std_arrays_in_func(node);
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
void
eval_ands(ASTNode* node)
{
	const char* lhs_val = combine_all_new(node->lhs);
	if(!strcmp(lhs_val,"false") || !strcmp(lhs_val,"(false)"))
	{
		replace_node(node,create_primary_expression("false"));
		return;
	}
	/**
	if(strstr(lhs_val,"false") && !strstr(lhs_val,"!"))
	{
		printf("HI: %s\n",lhs_val);
		printf("HI: %s\n",combine_all_new(node));
		exit(EXIT_FAILURE);
	}
	**/
}
void
eval_comparisons(ASTNode* node, const ASTNode* root, const char* op)
{
	if(!node->lhs->lhs) return;
	if(!node->lhs->lhs->lhs) return;
	if(!node->lhs->lhs->lhs->lhs) return;
	if(!(node->lhs->lhs->lhs->lhs->type & NODE_PRIMARY_EXPRESSION)) return;
	ASTNode* primary_expr = node->lhs->lhs->lhs->lhs;
	if(primary_expr->lhs->token != IDENTIFIER) return;
	const char* lhs_var = primary_expr->lhs->buffer;
	if(!check_symbol(NODE_VARIABLE_ID,lhs_var,REAL_STR,CONST_STR)) return;
	if(strcmp_null_ok(get_expr_type(primary_expr),"AcReal")) return;
	if(!node->rhs->rhs->lhs) return;
	if(!node->rhs->rhs->lhs->lhs) return;
	if(!(node->rhs->rhs->lhs->lhs->type & NODE_PRIMARY_EXPRESSION)) return;
	if(node->rhs->rhs->lhs->lhs->lhs->token != REALNUMBER) return;
	const char* real_number = node->rhs->rhs->lhs->lhs->lhs->buffer;
	const ASTNode* lhs_val = get_var_val(lhs_var,root);
	double lhs_double = atof(combine_all_new(lhs_val));
	double rhs_double = atof(real_number);
	const bool success = 
			!strcmp(op,"!=") ? lhs_double != rhs_double :
			!strcmp(op,"<")  ? lhs_double < rhs_double :
			!strcmp(op,">")  ? lhs_double > rhs_double : false;

	replace_node(node,create_primary_expression(success ? "true" : "false"));
}
void
eval_conditionals(ASTNode* node, const ASTNode* root)
{
	if(node->lhs)
		eval_conditionals(node->lhs,root);
	if(node->rhs)
		eval_conditionals(node->rhs,root);
	if(node->token == IDENTIFIER && node->buffer && !get_parent_node(NODE_DECLARATION,node))
	{
		if(node->buffer != intern("false") && node->buffer != intern("true"))
		{
			const char* id = node->buffer;
			if(check_symbol_string(NODE_VARIABLE_ID,id,"bool",CONST_STR))
			{
				const char* val = intern(combine_all_new(get_var_val(id,root)));
				if(val)
				{
					astnode_set_buffer(val,node);
					return;
				}
			}
			/**
			**/
		}
	}
	if(!node_is_binary_expr(node)) return;
	const char* op = get_node_by_token(BINARY_OP,node->rhs->lhs)->buffer;
	if(!op) return;
	if(op == intern("!=") || op == LESS_STR ||op == GREATER_STR) eval_comparisons(node,root,op);
	if(op == intern("&&")) eval_ands(node);
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
	if(node->buffer != identifier_name)
		return;
	node->is_constexpr |= is_constexpr;
}

void
count_num_of_assignments(const ASTNode* node, string_vec* names, int_vec* counts)
{
	if(node->lhs)
		count_num_of_assignments(node->lhs,names,counts);
	if(node->rhs)
		count_num_of_assignments(node->rhs,names,counts);
	if(node->type == NODE_ASSIGNMENT && node->rhs)
	{
	  const char* lhs = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	  const int index = str_vec_get_index(*names,lhs);
	  if(index == -1)
	  {
		  push(names,lhs);
		  push_int(counts,1);
	  }
	  else
	  {
		  counts->data[index]++;
	  }
	}

}
bool
inside_conditional_scope(const ASTNode* node)
{
                const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
		while(!(begin_scope->parent->parent->type & NODE_FUNCTION))
		{
			//Else node	
			if(begin_scope->parent->parent->lhs && begin_scope->parent->parent->lhs->token == ELSE) return true;
			//If and Elif node
			if(begin_scope->parent->parent->type & NODE_IF) return true;
                	begin_scope = get_parent_node(NODE_BEGIN_SCOPE,begin_scope);
		}
		return false;
}
typedef struct
{
	bool in_array_access;
	ASTNode* func_base;
} gen_constexpr_params;

bool
gen_constexpr_in_func(ASTNode* node, const bool gen_mem_accesses, const string_vec names, const int_vec assign_counts, gen_constexpr_params params)
{
	bool res = false;
	params.in_array_access |= (node->type & NODE_ARRAY_ACCESS);
	if(node->type & NODE_FUNCTION) params.func_base = node;
	if(node->lhs)
		res |= gen_constexpr_in_func(node->lhs,gen_mem_accesses,names,assign_counts,params);
	if(node->rhs)
		res |= gen_constexpr_in_func(node->rhs,gen_mem_accesses,names,assign_counts,params);
	if(node->token == IDENTIFIER && node->buffer && !node->is_constexpr)
	{
		node->is_constexpr |= check_symbol(NODE_ANY,node->buffer,0,CONST_STR);
		//if array access that means we are accessing the vtxbuffer which obviously is not constexpr
 		if(!params.in_array_access)
			node->is_constexpr |= check_symbol(NODE_VARIABLE_ID,node->buffer,FIELD_STR,0);
		node->is_constexpr |= check_symbol(NODE_DFUNCTION_ID,node->buffer,FIELD_STR,CONSTEXPR_STR);
		res |= node->is_constexpr;
	}
	if(node->type & NODE_IF && all_identifiers_are_constexpr(node->lhs) && !node->is_constexpr)
	{
		//TP: we only consider conditionals that are not inside other conditionals
		if(!inside_conditional_scope(node))
		{
			node->is_constexpr = true;
			res = true;
			astnode_set_prefix(" constexpr (",node);
			if(node->rhs->lhs->type & NODE_BEGIN_SCOPE && gen_mem_accesses)
			{
				astnode_sprintf_prefix(node->rhs->lhs,"{executed_nodes.push_back(%d);",node->id);
			}
		}
	}
	//TP: below sets the constexpr value of lhs the same as rhs for: lhs = rhs
	//TP: we restrict to the case that lhs is assigned only once in the function since full generality becomes too hard 
	//TP: However we get sufficiently far with this approach since we turn many easy cases to SSA form which this check covers
	if((node->type & NODE_ASSIGNMENT) && node->rhs && !node->is_constexpr)
	{
	  bool is_constexpr = all_identifiers_are_constexpr(node->rhs);
	  ASTNode* lhs_identifier = get_node_by_token(IDENTIFIER,node->lhs);
	  const int index = str_vec_get_index(names,lhs_identifier->buffer);
	  if(assign_counts.data[index] != 1)
		  return res;
	  if(lhs_identifier->is_constexpr == is_constexpr)
		  return res;
	  res |= is_constexpr;
	  node->is_constexpr |= is_constexpr;
	  lhs_identifier->is_constexpr |= is_constexpr;
	  set_identifier_constexpr(params.func_base,lhs_identifier->buffer,is_constexpr);
	  if(is_constexpr && get_node(NODE_TSPEC,node->lhs))
	  {
		  ASTNode* tspec = (ASTNode*) get_node(NODE_TSPEC,node->lhs);
		  astnode_sprintf(tspec->lhs," constexpr %s",tspec->lhs->buffer);
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
				push(&func_sym->tqualifiers,CONSTEXPR_STR);
				node->is_constexpr = true;
				res |= node->is_constexpr;
			}
		}
	}
	return res;
}
bool
gen_constexpr_info_base(ASTNode* node, const bool gen_mem_accesses)
{
	bool res = false;
	if(node->type & NODE_GLOBAL)
		return res;
	if(node->lhs)
		res |= gen_constexpr_info_base(node->lhs,gen_mem_accesses);
	if(node->rhs)
		res |= gen_constexpr_info_base(node->rhs,gen_mem_accesses);
	if(node->type & NODE_FUNCTION)
	{
		string_vec names = VEC_INITIALIZER;
		int_vec    n_assignments = VEC_INITIALIZER;
		count_num_of_assignments(node,&names,&n_assignments);
		res |= gen_constexpr_in_func(node,gen_mem_accesses,names,n_assignments,(gen_constexpr_params){false,NULL});
	}
	return res;
}

void
gen_constexpr_info(ASTNode* root, const bool gen_mem_accesses)
{
	bool has_changed = true;
	while(has_changed) 
		has_changed = gen_constexpr_info_base(root,gen_mem_accesses);

}

bool
gen_declared_type_info(ASTNode* node)
{
	bool res = false;
	if(node->type & NODE_GLOBAL)
		return res;
	if(node->lhs)
		res |= gen_declared_type_info(node->lhs);
	if(node->rhs)
		res |= gen_declared_type_info(node->rhs);
	if(node->expr_type) return res;
	if(is_return_node(node))
		return res;
	if(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node))
		get_expr_type(node);
	res |=  node -> expr_type != NULL;
	return res;
}
bool
gen_local_type_info(ASTNode* node)
{
	bool res = false;
	if(node->type & NODE_GLOBAL)
		return res;
	if(node->lhs)
		res |= gen_local_type_info(node->lhs);
	if(node->rhs)
		res |= gen_local_type_info(node->rhs);
	if(node->expr_type) return res;
	if(is_return_node(node))
	{
		const char* expr_type = get_expr_type(node->rhs);
		if(expr_type)
		{
			node->expr_type = expr_type;
			ASTNode* dfunc_start = (ASTNode*) get_parent_node(NODE_DFUNCTION,node);
			const char* func_name = get_node(NODE_DFUNCTION_ID,dfunc_start)->buffer;
			Symbol* func_sym = (Symbol*)get_symbol(NODE_DFUNCTION_ID,func_name,NULL);
			if(func_sym && !str_vec_contains(func_sym->tqualifiers,expr_type))
				dfunc_start -> expr_type = expr_type;
		}
	}
	if(node->type & (NODE_PRIMARY_EXPRESSION | NODE_FUNCTION_CALL) ||
		(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node)) ||
		(node->type & NODE_EXPRESSION && all_primary_expressions_and_func_calls_have_type(node)) ||
		(node->type & NODE_ASSIGNMENT && node->rhs && get_parent_node(NODE_FUNCTION,node) &&  !get_node(NODE_MEMBER_ID,node->lhs))
		|| (node->token == IN_RANGE)
	)
		get_expr_type(node);
	res |=  node -> expr_type != NULL;
	return res;
}
bool
flow_type_info_in_func(ASTNode* node,string_vec* names, string_vec* types)
{
	bool res = false;
	if(node->lhs)
		res |= flow_type_info_in_func(node->lhs,names,types);
	if(node->type & NODE_DECLARATION)
	{


		const char* var_name = get_node_by_token(IDENTIFIER,node)->buffer;
		const int index = str_vec_get_index(*names,var_name);
		//const char* func_name = get_node_by_token(IDENTIFIER,get_parent_node(NODE_FUNCTION,node))->buffer;
		//if(strstr(func_name,"rk3") && node->expr_type)
			//printf("VAR: %s,%s\n",var_name,node->expr_type);
		if(node->expr_type)
		{
			if(index == -1)
			{
				push(names,var_name);
				push(types,intern(node->expr_type));
			}
			else
			{
				if(node->expr_type != types->data[index])
				{

					printf("WRONG DOES NOT MATCH\n");
					printf("%s %s\n",node->expr_type,types->data[index]);
					printf("VAR: %s\n",var_name);
					exit(EXIT_FAILURE);
				}
			}
		}
	}
	if(node->type == NODE_PRIMARY_EXPRESSION && !node->expr_type)
	{
		ASTNode* identifier = (ASTNode*)get_node_by_token(IDENTIFIER,node);
		if(identifier)
		{
			const int index = str_vec_get_index(*names,identifier->buffer);
			if(index != -1)
			{
				res = true;
				node->expr_type = types->data[index];
				identifier->expr_type = types->data[index];
			}
		}
	}
	if(node->type == NODE_ASSIGNMENT && !node->expr_type)
	{

		const int n_lhs = count_num_of_nodes_in_list(node->lhs->rhs);
		if(n_lhs == 1)
		{
			ASTNode* decl = (ASTNode*)get_node(NODE_DECLARATION,node->lhs);
			const char* var_name = get_node_by_token(IDENTIFIER,decl)->buffer;
			const int index = str_vec_get_index(*names,var_name);
			if(index != -1)
			{
				node->expr_type = types->data[index];
			}
		}
	}
	if(node->rhs)
		res |= flow_type_info_in_func(node->rhs,names,types);
	return res;
}
bool
flow_type_info(ASTNode* node)
{
	bool res = false;
	if(node->lhs)
		res |= flow_type_info(node->lhs);
	if(node->rhs)
		res |= flow_type_info(node->rhs);
	if(node->type & NODE_FUNCTION)
	{
		string_vec names = VEC_INITIALIZER;
		string_vec types = VEC_INITIALIZER;
		res |= flow_type_info_in_func(node,&names,&types);
		free_str_vec(&names);
		free_str_vec(&types);
	}
	return res;
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
create_func_call_expr(const char* func_name, const ASTNode* param)
{
	ASTNode* func_call = create_func_call(func_name,param);
	ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
	ASTNode* expression         = astnode_create(NODE_EXPRESSION,unary_expression,NULL);
	return expression;
}
bool
field_to_real_conversion(ASTNode* node, const ASTNode* root)
{
	bool res = false;
	if(node->lhs)
		res |= field_to_real_conversion(node->lhs,root);
	if(node->rhs)
		res |= field_to_real_conversion(node->rhs,root);
	if(!(node->type & NODE_FUNCTION_CALL)) return res;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	if(!check_symbol(NODE_DFUNCTION_ID,func_name,0,0)) return res;
	if(str_vec_contains(duplicate_dfuncs.names,func_name)) return res;

	func_params_info params_info = get_function_params_info(root,func_name);
	func_params_info call_info = get_func_call_params_info(node);
	if(call_info.types.size == 0) return res;
	const int offset = is_boundary_param(call_info.expr.data[0]) ? 1 : 0;
	if(params_info.types.size != call_info.types.size-offset)
		fatal("number of parameters does not match, expected %zu but got %zu in %s\n",params_info.types.size, call_info.types.size, combine_all_new(node));
	for(size_t i = offset; i < call_info.types.size; ++i)
	{
		if(!call_info.types.data[i]) continue;
		if(
		      (params_info.types.data[i-offset] == REAL3_STR     && call_info.types.data[i] == FIELD3_STR)
		   || (params_info.types.data[i-offset] == REAL_STR      && call_info.types.data[i] == FIELD_STR)
		   || (params_info.types.data[i-offset] == REAL_PTR_STR  && call_info.types.data[i] == VTXBUF_PTR_STR)
		   || (params_info.types.data[i-offset] == REAL_PTR_STR  && call_info.types.data[i] == FIELD_PTR_STR)
		   || (params_info.types.data[i-offset] == REAL3_PTR_STR && call_info.types.data[i] == FIELD3_PTR_STR)
		   || (params_info.types.data[i-offset] == REAL_STR      && strstr(call_info.types.data[i],"Profile"))
		  )
		{
			ASTNode* expr = (ASTNode*)call_info.expr_nodes.data[i];
			expr->expr_type = params_info.types.data[i-offset];
			replace_node(
					expr,
					create_func_call_expr(VALUE_STR,expr)
				     );
		}
	}
	free_func_params_info(&call_info);
	free_func_params_info(&params_info);
	return res;
}


void
gen_type_info(ASTNode* root)
{
  	transform_arrays_to_std_arrays(root);
  	if(dfunc_nodes.size == 0)
  		get_nodes(root,&dfunc_nodes,&dfunc_names,NODE_DFUNCTION);
	bool has_changed = true;
	int iter = 0;
	//TP: this is done first so that e.g. real3 = real broadcast assignment won't be understood incorrectly
	gen_declared_type_info(root);
	flow_type_info(root);
	while(has_changed)
	{
		has_changed = gen_local_type_info(root);
		has_changed |= field_to_real_conversion(root,root);
		has_changed |= flow_type_info(root);
	}
}
const ASTNode*
find_dfunc_start(const ASTNode* node, const char* dfunc_name)
{
	if(node->type & NODE_DFUNCTION && get_node(NODE_DFUNCTION_ID,node) && get_node(NODE_DFUNCTION_ID,node)->buffer && get_node(NODE_DFUNCTION_ID,node)->buffer == dfunc_name) return node;
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
			asprintf(&tmp,"%s_%s",tmp,
					types.data[i] ? types.data[i] : "Auto");
		tmp = realloc(tmp,sizeof(char)*(strlen(tmp) + 5*types.size));
		replace_substring(&tmp,MULT_STR,"ARRAY");
		replace_substring(&tmp,"<","_");
		replace_substring(&tmp,">","_");
		return tmp;
}
void
mangle_dfunc_names(ASTNode* node, string_vec* dst_types, const char** dst_names,int* counters)
{
	TRAVERSE_PREAMBLE_PARAMS(mangle_dfunc_names,dst_types,dst_names,counters);
	if(!(node->type & NODE_DFUNCTION))
		return;
	const int dfunc_index = str_vec_get_index(duplicate_dfuncs.names,get_node_by_token(IDENTIFIER,node)->buffer);
	if(dfunc_index == -1)
		return;
	const char* dfunc_name = duplicate_dfuncs.names.data[dfunc_index];
	const int overload_index = counters[dfunc_index];
	func_params_info params_info = get_function_params_info(node,dfunc_name);
	counters[dfunc_index]++;
	for(size_t i = 0; i < params_info.types.size; ++i)
		push(&dst_types[overload_index + MAX_DFUNCS*dfunc_index], params_info.types.data[i]);
	dst_names[overload_index + MAX_DFUNCS*dfunc_index]  = dfunc_name;
	ASTNode* identifier = get_node_by_token(IDENTIFIER,node->lhs);
	astnode_set_buffer(get_mangled_name(dfunc_name,params_info.types),identifier);
	free_func_params_info(&params_info);
}
static bool
is_subtype(const char* a, const char* b)
{
	if(a != PROFILE_STR) return false;
	bool res = false;
	string_vec prof_types = get_prof_types();
	for(size_t i = 0; i < prof_types.size; ++i)
		res |= b == prof_types.data[i];
	free_str_vec(&prof_types);
	return res;
}
static bool
compatible_types(const char* a, const char* b)
{
	if(is_subtype(a,b)) return true;
	const bool res = !strcmp(a,b) 
	       || (!strcmp(a,REAL_PTR_STR) && strstr(b,"AcArray") && strstr(b,REAL_STR)) ||
	          (!strcmp(b,REAL_PTR_STR) && strstr(a,"AcArray") && strstr(a,REAL_STR)) ||
                  (!strcmp(a,FIELD_STR) && !strcmp(b,"VertexBufferHandle"))  ||
		  (!strcmp(b,FIELD_STR) && !strcmp(a,"VertexBufferHandle"))  ||
                  (!strcmp(a,"Field*") && !strcmp(b,VTXBUF_PTR_STR))  ||
		  (!strcmp(b,"Field*") && !strcmp(a,VTXBUF_PTR_STR))
		  || (a == REAL_STR && b == FIELD_STR)
		  || (a == REAL3_STR && b == FIELD3_STR)
		  || (a == REAL_PTR_STR && b == VTXBUF_PTR_STR)
		  || (a == REAL_PTR_STR && b == FIELD_PTR_STR)
		  || (a == REAL3_PTR_STR && b == FIELD3_PTR_STR)
		;
	if(!res)
	{
		if(a == REAL_PTR_STR && get_array_elem_type(strdup(b)) == REAL_STR)
			return true;
		if(a == REAL3_PTR_STR && get_array_elem_type(strdup(b)) == REAL3_STR)
			return true;
	}
	return res;
}
typedef struct 
{
	const string_vec* types;
	const char*const * names;
} dfunc_possibilities;
static int_vec
get_possible_dfuncs(const func_params_info call_info, const dfunc_possibilities possibilities, const int dfunc_index, const bool strict, const bool use_auto, const char* func_name)
{
	int overload_index = MAX_DFUNCS*dfunc_index-1;
	int_vec possible_indexes = VEC_INITIALIZER;
    	//TP: ugly hack to resolve calls in BoundConds
	const int param_offset = (call_info.expr.size > 0 && is_boundary_param(call_info.expr.data[0])) ? 1 : 0;
	while(possibilities.names[++overload_index] == func_name)
	{
		bool possible = true;
		if(call_info.types.size - param_offset != possibilities.types[overload_index].size) continue;
		bool all_types_specified = true;
		for(size_t i = param_offset; i < call_info.types.size; ++i)
			all_types_specified &= (possibilities.types[overload_index].data[i-param_offset] != NULL);
		if(use_auto && all_types_specified) continue;
		for(size_t i = param_offset; i < call_info.types.size; ++i)
		{
			const char* func_type = possibilities.types[overload_index].data[i-param_offset];
			const char* call_type = call_info.types.data[i];
			if(strict)
				possible &= !call_type || !func_type || !strcmp(func_type,call_type);
			else
				possible &= !call_type || !func_type || compatible_types(func_type,call_type);
			if(!use_auto) possible &= func_type != NULL;
			if(!use_auto) possible &= call_type != NULL;

		}
		if(possible)
		{
			push_int(&possible_indexes,overload_index);
		}
	}
	return possible_indexes;
}
bool
resolve_overloaded_calls(ASTNode* node, const dfunc_possibilities possibilities)
{
	bool res = false;
	if(node->lhs)
		res |= resolve_overloaded_calls(node->lhs,possibilities);
	if(node->rhs)
		res |= resolve_overloaded_calls(node->rhs,possibilities);
	if(!(node->type & NODE_FUNCTION_CALL))
		return res;
	if(!get_node_by_token(IDENTIFIER,node->lhs))
		return res;
	const int dfunc_index = str_vec_get_index(duplicate_dfuncs.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	if(dfunc_index == -1)
		return res;
	const char* dfunc_name = duplicate_dfuncs.names.data[dfunc_index];
	func_params_info call_info = get_func_call_params_info(node);
	int correct_types = -1;
	int_vec possible_indexes_conversion              = get_possible_dfuncs(call_info,possibilities,dfunc_index,false,true,dfunc_name);
	int_vec possible_indexes_conversion_no_auto      = get_possible_dfuncs(call_info,possibilities,dfunc_index,false,false,dfunc_name);
	int_vec possible_indexes_strict          = get_possible_dfuncs(call_info,possibilities,dfunc_index,true,true,dfunc_name);
	int_vec possible_indexes_strict_no_auto  = get_possible_dfuncs(call_info,possibilities,dfunc_index,true,false,dfunc_name);
	//TP: by default use the strict rules but if there no suitable ones use conversion rules
	//TP: if there are multiple possibilities pick the one with all specified parameters
	const int_vec possible_indexes = 
					possible_indexes_strict_no_auto.size      > 0 ? possible_indexes_strict_no_auto :
					possible_indexes_conversion_no_auto.size  > 0 ? possible_indexes_conversion_no_auto :
					possible_indexes_strict.size              > 0 ? possible_indexes_strict :
					possible_indexes_conversion;
	bool able_to_resolve = possible_indexes.size == 1;
	if(!able_to_resolve) { 
		//if(!strcmp(dfunc_name,"dot"))
		//{
		//	char my_tmp[10000];
		//	my_tmp[0] = '\0';
		//	combine_all(node->rhs,my_tmp); 
		//	printf("Not able to resolve: %s\n",my_tmp); 
		//	printf("Not able to resolve: %s,%zu\n",call_info.types.data[1],possible_indexes_conversion.size); 
		//}
		return res;
	}
	{
		const string_vec types = possibilities.types[possible_indexes.data[0]];
		astnode_set_buffer(get_mangled_name(dfunc_name,types), get_node_by_token(IDENTIFIER,node->lhs));

	}

	free_str_vec(&call_info.expr);
	free_str_vec(&call_info.types);
	free_int_vec(&possible_indexes_strict);
	free_int_vec(&possible_indexes_conversion);
	return true;
}

void
gen_overloads(ASTNode* root)
{
  bool overloaded_something = true;
  string_vec dfunc_possible_types[MAX_DFUNCS * duplicate_dfuncs.names.size];
  const char* dfunc_possible_names[MAX_DFUNCS * duplicate_dfuncs.names.size];
  memset(dfunc_possible_types,0,sizeof(string_vec)*MAX_DFUNCS*duplicate_dfuncs.names.size);
  memset(dfunc_possible_names,0,sizeof(char*)*MAX_DFUNCS*duplicate_dfuncs.names.size);
  int counters[duplicate_dfuncs.names.size];
  memset(counters,0,sizeof(int)*duplicate_dfuncs.names.size);
  mangle_dfunc_names(root,dfunc_possible_types,dfunc_possible_names,counters);

  
  //TP: refresh the symbol table with the mangled names
  symboltable_reset();
  traverse(root, NODE_NO_OUT, NULL);

  //TP we have to create the internal names here since it has to be done after name mangling but before transformation to AcArray
  gen_dfunc_internal_names(root);


  const dfunc_possibilities overload_possibilities = {dfunc_possible_types,dfunc_possible_names}; 
  while(overloaded_something)
  {
	overloaded_something = false;
  	gen_type_info(root);
	overloaded_something |= resolve_overloaded_calls(root,overload_possibilities);
  	//for(size_t i = 0; i < duplicate_dfuncs.size; ++i)
  	        //overloaded_something |= resolve_overloaded_calls(root,duplicate_dfuncs.data[i],dfunc_possible_types,i);
  }
  for(size_t i = 0; i < MAX_DFUNCS*duplicate_dfuncs.names.size; ++i)
	  free_str_vec(&dfunc_possible_types[i]);
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
	if(!str_vec_contains(sym -> tqualifiers,REAL_STR)) return;
	func_params_info param_info = get_func_call_params_info(node);
	if(param_info.types.data[0] == FIELD_STR && param_info.expr.size == 1)
	{
		ASTNode* expression         = create_func_call_expr(VALUE_STR,node->rhs);
		ASTNode* expression_list = astnode_create(NODE_UNKNOWN,expression,NULL);
		node->rhs = expression_list;
	}
	free_func_params_info(&param_info);
}

bool
is_field_expr(const char* expr)
{
	return expr && (expr == FIELD_STR || expr == FIELD3_STR || !strcmp(expr,FIELD_PTR_STR) || !strcmp(expr,VTXBUF_PTR_STR) || !strcmp(expr,FIELD3_PTR_STR));
}
bool
is_value_applicable_type(const char* expr)
{
	return is_field_expr(expr) || is_subtype(PROFILE_STR,expr);
}

void
transform_field_unary_ops(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_field_unary_ops);
	if(!node_is_unary_expr(node)) return;
	const char* base_type= get_expr_type(node->rhs);
	const char* unary_op = get_node_by_token(UNARY_OP,node->lhs)->buffer;
	if(strcmps(unary_op,PLUS_STR,MINUS_STR)) return;
	if(!base_type) return;
	if(!is_value_applicable_type(base_type)) return;

	ASTNode*  func_call = create_func_call(VALUE_STR,node->rhs);
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
        if(strcmps(op,PLUS_STR,MINUS_STR,DIV_STR,MULT_STR)) return;


	if(is_value_applicable_type(lhs_expr))
	{
		ASTNode* func_call = create_func_call(VALUE_STR,node->lhs);
		ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
		ASTNode* expression         = astnode_create(NODE_EXPRESSION,unary_expression,NULL);
		node->lhs = expression;
	}
	if(is_value_applicable_type(rhs_expr))
	{

		ASTNode* func_call = create_func_call(VALUE_STR,node->rhs->rhs);
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
	func_params_info info = get_function_params_info(node,dfunc_name);
	if(!has_qualifier(node,"elemental")) return;
	if(info.expr.size == 1 && info.types.data[0] == REAL_STR && !strstr(dfunc_name,"AC_INTERNAL_COPY"))
	{
		const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);

		fprintf(stream,"%s_AC_INTERNAL_COPY (real %s){%s}\n",dfunc_name,info.expr.data[0],func_body);
		fprintf(stream,"%s (real3 v){return real3(%s_AC_INTERNAL_COPY(v.x), %s_AC_INTERNAL_COPY(v.y), %s_AC_INTERNAL_COPY(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		fprintf(stream,"inline %s(real[] arr){\nreal res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s_AC_INTERNAL_COPY(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
		fprintf(stream,"inline %s(real3[] arr){\nreal3 res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
	}

	else if(info.expr.size == 1 && info.types.data[0] == FIELD_STR && !strstr(dfunc_name,"AC_INTERNAL_COPY"))
	{
		if(intern(node->expr_type) == REAL_STR)
		{
			const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);
			fprintf(stream,"%s_AC_INTERNAL_COPY (Field %s){%s}\n",dfunc_name,info.expr.data[0],func_body);
			fprintf(stream,"%s (Field3 v){return real3(%s_AC_INTERNAL_COPY(v.x), %s_AC_INTERNAL_COPY(v.y), %s_AC_INTERNAL_COPY(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
			fprintf(stream,"inline %s(Field[] arr){\nreal res[size(arr)]\n for i in 0:size(arr)\n   res[i] = %s_AC_INTERNAL_COPY(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
			fprintf(stream,"inline %s(Field3[] arr){\nreal3 res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
		}
		else if(intern(node->expr_type) == REAL3_STR)
		{
			const char* func_body = combine_all_new_with_whitespace(node->rhs->rhs->lhs);

			fprintf(stream,"%s_AC_INTERNAL_COPY (Field %s){%s}\n",dfunc_name,info.expr.data[0],func_body);
			fprintf(stream,"%s (Field3 v){return Matrix(%s_AC_INTERNAL_COPY(v.x), %s_AC_INTERNAL_COPY(v.y), %s_AC_INTERNAL_COPY(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		}
		else
			fatal("Missing elemental case for func: %s\nReturn type: %s\n",dfunc_name,node->expr_type);
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
mark_first_declarations_in_funcs(ASTNode* node, string_vec* names)
{
	if(node->lhs)
		mark_first_declarations_in_funcs(node->lhs,names);
	if(node->type & NODE_DECLARATION)
	{
		const char* var_name = get_node_by_token(IDENTIFIER,node)->buffer;
		if(!str_vec_contains(*names,var_name))
		{
			push(names,var_name);
			node->token = FIRST;
		}
	}
	if(node->rhs)
		mark_first_declarations_in_funcs(node->rhs,names);

}
void
mark_first_declarations(ASTNode* node)
{
	if(node->lhs)
		mark_first_declarations(node->lhs);
	if(node->type & NODE_FUNCTION)
	{
		string_vec names = VEC_INITIALIZER;
		mark_first_declarations_in_funcs(node,&names);
		free_str_vec(&names);
	}
	if(node->rhs)
		mark_first_declarations(node->rhs);
}
void
gen_global_strings()
{
	push(&primitive_datatypes,intern("int"));
	push(&primitive_datatypes,intern("AcReal"));
	push(&primitive_datatypes,intern("bool"));
	push(&primitive_datatypes,intern("long"));
	push(&primitive_datatypes,intern("long long"));

	if(AC_DOUBLE_PRECISION)
		push(&primitive_datatypes,intern("float"));
	else
		push(&primitive_datatypes,intern("double"));

	VALUE_STR = intern("value");

	COMPLEX_STR= intern("AcComplex");
	REAL3_STR= intern("AcReal3");
	REAL_ARR_STR = intern("AcRealArray");

	REAL_PTR_STR = intern("AcReal*");
	REAL3_PTR_STR = intern("AcReal3*");
	FIELD3_PTR_STR = intern("Field3*");
	VTXBUF_PTR_STR = intern("VertexBufferHandle*");
	FIELD_PTR_STR = intern("Field*");

	MATRIX_STR = intern("AcMatrix");
	INT3_STR = intern("int3");
	EQ_STR = intern("=");

	DOT_STR = intern("dot");

	LESS_STR = intern("<");
	GREATER_STR = intern(">");
	LEQ_STR = intern("<=");
	GEQ_STR = intern(">=");

	MEQ_STR= intern("*=");
	AEQ_STR= intern("+=");
	MINUSEQ_STR= intern("-=");
	DEQ_STR= intern("/=");
	PERIODIC = intern("periodic");


	EMPTY_STR = intern("\0");
	DEAD_STR = intern("dead");
	INLINE_STR = intern("inline");
	UTILITY_STR = intern("utility");
	BOUNDCOND_STR = intern("boundcond");
	FIXED_BOUNDARY_STR = intern("fixed_boundary");
	ELEMENTAL_STR = intern("elemental");
	AUXILIARY_STR = intern("auxiliary");
	COMMUNICATED_STR = intern("communicated");
	CONST_STR  = intern("const");
	DCONST_STR = intern("dconst");
	CONSTEXPR_STR = intern("constexpr");
	GLOBAL_MEM_STR  = intern("gmem");
	DYNAMIC_STR  = intern("dynamic");
	OUTPUT_STR  = intern("output");
	INPUT_STR  = intern("input");
	RUN_CONST_STR = intern("run_const");
	CONST_DIMS_STR= intern("const_dims");
	FIELD_STR = intern("Field");
	STENCIL_STR = intern("Stencil");
	KERNEL_STR = intern("Kernel");
	FIELD3_STR = intern("Field3");
	FIELD4_STR = intern("Field4");
	PROFILE_STR = intern("Profile");

	MULT_STR = intern("*");
	PLUS_STR = intern("+");
	MINUS_STR = intern("-");
	DIV_STR = intern("/");


 	BOUNDARY_X_TOP_STR = intern("BOUNDARY_X_TOP"); 
 	BOUNDARY_X_BOT_STR = intern("BOUNDARY_X_BOT"); 

 	BOUNDARY_Y_BOT_STR = intern("BOUNDARY_Y_BOT"); 
 	BOUNDARY_Y_TOP_STR = intern("BOUNDARY_Y_TOP"); 

 	BOUNDARY_Z_BOT_STR = intern("BOUNDARY_Z_BOT"); 
 	BOUNDARY_Z_TOP_STR = intern("BOUNDARY_Z_TOP"); 


 	BOUNDARY_X_STR = intern("BOUNDARY_X"); 
 	BOUNDARY_Y_STR = intern("BOUNDARY_Y"); 
 	BOUNDARY_Z_STR = intern("BOUNDARY_Z"); 

	BOUNDARY_XY_STR  = intern("BOUNDARY_XY");   
	BOUNDARY_XZ_STR  = intern("BOUNDARY_XZ");
	BOUNDARY_YZ_STR  = intern("BOUNDARY_YZ");
	BOUNDARY_XYZ_STR = intern("BOUNDARY_XYZ");
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
	if(strcmps(op,MEQ_STR,MINUSEQ_STR,AEQ_STR,DEQ_STR))   return;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return;
	ASTNode* assign_expression = node->rhs->rhs->lhs;
	remove_substring(op,EQ_STR);
	ASTNode* binary_expression = create_binary_expression(node->lhs, assign_expression, op);
	ASTNode* assignment        = create_assignment(node->lhs, binary_expression, EQ_STR); 
	assignment->parent = node->parent;
	node->parent->lhs = assignment;
}
ASTNode*
create_struct_tspec(const char* datatype)
{
	ASTNode* struct_type = astnode_create(NODE_UNKNOWN,NULL,NULL); 
	astnode_set_buffer(datatype,struct_type);
	struct_type->token = STRUCT_TYPE;
	return astnode_create(NODE_TSPEC,struct_type,NULL);
}
ASTNode*
create_broadcast_initializer(const ASTNode* expr, const char* datatype)
{
	node_vec nodes = VEC_INITIALIZER;


	const size_t n_initializer= get_number_of_members(datatype);
	for(size_t i = 0; i < n_initializer; ++i)
		push_node(&nodes,expr);

	ASTNode* expression_list = build_list_node(nodes,",");
	ASTNode* initializer = astnode_create(NODE_STRUCT_INITIALIZER, expression_list,NULL); 
	astnode_set_prefix("{",initializer);
	astnode_set_postfix("}",initializer);
	free_node_vec(&nodes);


	ASTNode* tspec = create_struct_tspec(datatype);
	ASTNode* res = astnode_create(NODE_UNKNOWN,tspec,initializer);
	astnode_set_prefix("(",res);
	astnode_set_infix(")",res);
	res->token     = CAST;
	res->lhs->type ^= NODE_TSPEC;
	return res;
}
void
transform_broadcast_assignments(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_broadcast_assignments);
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
	if(!function) return;
	const char* function_name = get_node_by_token(IDENTIFIER,function->lhs)->buffer;
	const char* op = node->rhs->lhs->buffer;
	if(op != EQ_STR) return;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return;
	const char* lhs_type = get_expr_type(node->lhs);
	const char* rhs_type = get_expr_type(node->rhs);
	if(!lhs_type || !rhs_type) return;
	if(all_real_struct(lhs_type) && rhs_type == REAL_STR)
	{
		const ASTNode* expr = node->rhs->rhs->lhs;
		node->rhs->rhs->lhs = create_broadcast_initializer(expr,lhs_type);
	}
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
	if(node->rhs->lhs->buffer == EQ_STR)   return false;
	if(!(if_node->rhs->lhs->type & NODE_BEGIN_SCOPE) && is_first_decl(node->lhs))
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
	if(node->rhs->lhs->buffer ==EQ_STR)   return;
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
		if(second_var_name == var_name) return;
		if(count_num_of_nodes_in_list(second_assign->rhs->rhs) != 1)   return;
		if(second_assign->rhs->lhs->buffer == EQ_STR)   return;
		ASTNode* second_assign_expr = second_assign->rhs->rhs->lhs;
		//same as below except now : condition is the else condition
		//ASTNode* ternary_expr = create_ternary_expr(conditional, assign_expression ,second_assign_expr);
		ASTNode* ternary_expr = create_ternary_expr(conditional, assign_expression,second_assign_expr);
		ASTNode* assignment =   create_assignment(node->lhs,ternary_expr,EQ_STR);
		assignment->parent = if_node->parent->parent;
		if_node->parent->parent->lhs = assignment;

		return;
	}


	ASTNode* ternary_expr = create_ternary_expr(conditional, assign_expression ,create_primary_expression(var_name));
	ASTNode* assignment =   create_assignment(node->lhs,ternary_expr,EQ_STR);
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
	const char* var_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	const ASTNode* begin_scope = get_parent_node(NODE_BEGIN_SCOPE,node);
	//TP: only consider writes in the first nest
	if(begin_scope -> id != func->rhs->rhs->id) return;
	const ASTNode* head = get_parent_node(NODE_STATEMENT_LIST_HEAD,node);
	const bool final_node = is_left_child(NODE_STATEMENT_LIST_HEAD,node);
	const bool is_used_in_rest = is_used_in_statements(final_node ? head : head->parent,var_name);
	ASTNode* primary_expr = get_node_by_token(IDENTIFIER,node->lhs)->parent;
	const char* expr_type = get_expr_type(primary_expr);
	const char* primary_identifier = primary_expr->lhs->buffer;
	if(!is_used_in_rest && expr_type && 
	   //exclude written fields since they write to the vertex buffer
	   expr_type == FIELD_STR && 
	   //exclude writes to dynamic global arrays since they persist after the kernel
	   !check_symbol(NODE_VARIABLE_ID,primary_identifier,0,DYNAMIC_STR)
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
	const char* var_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;

	//TP: checks is the var declared before this scope
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
	astnode_set_buffer(new_name,get_node_by_token(IDENTIFIER,node->lhs));
	free(new_name);
}
void
canonalize(ASTNode* node)
{
	canonalize_assignments(node);
	//canonalize_if_assignments(node);
}
void
get_overrided_vars(ASTNode* node, string_vec* overrided_vars)
{
	TRAVERSE_PREAMBLE_PARAMS(get_overrided_vars,overrided_vars);
	if(!(node->type & NODE_DECLARATION)) return;
	if(!has_qualifier(node,"override") || !has_qualifier(node,"const")) return;
	node_vec assignments = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < assignments.size; ++i)
	{
		const char* name  = get_node_by_token(IDENTIFIER, assignments.data[i])->buffer;
		if(!str_vec_contains(*overrided_vars,name))
			push(overrided_vars,name);
	}
	free_node_vec(&assignments);

}
void
check_for_writes_to_const_variables_in_func(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(check_for_writes_to_const_variables_in_func);
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const ASTNode* id = get_node_by_token(IDENTIFIER,node);
	if(!id) return;
	if(check_symbol(NODE_ANY,id->buffer,0,DCONST_STR))
		fatal("Write to dconst variable: %s\n",combine_all_new(node));
	if(check_symbol(NODE_ANY,id->buffer,0,CONST_STR))
		fatal("Write to const variable: %s\n",combine_all_new(node));
	if(check_symbol(NODE_ANY,id->buffer,0,RUN_CONST_STR))
		fatal("Write to run_const variable: %s\n",combine_all_new(node));
}
void
check_for_writes_to_const_variables(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(check_for_writes_to_const_variables);
	if(!(node->type & NODE_FUNCTION)) return;
	check_for_writes_to_const_variables_in_func(node);
}
void
process_overrides_recursive(ASTNode* node, const string_vec overrided_vars)
{
	TRAVERSE_PREAMBLE_PARAMS(process_overrides_recursive,overrided_vars);
	if(!((node->type & NODE_DECLARATION))) return;
	const ASTNode* id = get_node_by_token(IDENTIFIER,node);
	if(!id) return;
	if(str_vec_contains(overrided_vars,id->buffer) && !has_qualifier(node,"override"))
	{
		if(!(node->type & NODE_GLOBAL))
				fatal("WRONG: %s\n",combine_all_new(node));
		if(node->parent->lhs->id == node->id)
			node->parent->lhs = NULL;
		else
			node->parent->rhs = NULL;
	}
}
void
process_overrides(ASTNode* root)
{
	string_vec overrided_vars = VEC_INITIALIZER;
	get_overrided_vars(root,&overrided_vars);
	process_overrides_recursive(root,overrided_vars);
	free_str_vec(&overrided_vars);
}
static void
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
bool
expand_allocating_types_base(ASTNode* node)
{
	bool res = false;
	if(node->lhs)
		res |= expand_allocating_types_base(node->lhs);
	if(node->rhs)
		res |= expand_allocating_types_base(node->rhs);
	if(node->type  != (NODE_DECLARATION | NODE_GLOBAL)) return res;
	const ASTNode* type_node = get_node(NODE_TSPEC,node);
	if(!type_node) return res;
	//const ASTNode* qualifier = get_node(NODE_TQUAL,node);
	////TP: do this only for no qualifier declarations
	//if(qualifier) return;
	const char* type = intern(combine_all_new(type_node));
	const ASTNode* tquals = node->lhs->rhs ? node->lhs->lhs : NULL;
	if(!is_allocating_type(type)) return res;
	const bool array_decl = get_node(NODE_ARRAY_ACCESS,node);
	ASTNode* declaration_list_head = node->rhs;
	if(array_decl)
	{
		const int num_elems = eval_int(declaration_list_head->lhs->rhs,true,NULL);
		ASTNode* field_name = declaration_list_head->lhs->lhs->lhs;
                const char* field_name_str = strdup(field_name->buffer);
		node_vec id_nodes = VEC_INITIALIZER;
		for(int i=0; i<num_elems;++i)
		{
			ASTNode* identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
			identifier->token = IDENTIFIER;
			astnode_sprintf(identifier,"%s_%d",field_name_str,i);
			ASTNode* primary_expression = astnode_create(NODE_PRIMARY_EXPRESSION,identifier,NULL);
			ASTNode* res_node = astnode_create(NODE_UNKNOWN,primary_expression,NULL);
			set_identifier_type(NODE_VARIABLE_ID,res_node);
			push_node(&id_nodes,res_node);
		}
		node_vec decl_nodes = VEC_INITIALIZER;
		for(int i=0; i<num_elems;++i)
		{
			ASTNode* decl = astnode_create((NODE_DECLARATION | NODE_GLOBAL),
					create_type_declaration_with_qualifiers(tquals,type),
					astnode_dup(id_nodes.data[i],NULL)
					);
			ASTNode* res_node = astnode_create(NODE_UNKNOWN,decl,NULL);
			push_node(&decl_nodes,res_node);
		}


		ASTNode* elems = build_list_node(id_nodes,",");
		ASTNode* decls = build_list_node(decl_nodes,"");
		free_node_vec(&id_nodes);
		free_node_vec(&decl_nodes);

		//TP: create the equivalent of const Field CHEMISTRY = [CHEMISTRY_0, CHEMISTRY_1,...,CHEMISTRY_N]
		ASTNode* arr_initializer = create_arr_initializer(elems);
		ASTNode* type_declaration = create_type_declaration("const",sprintf_intern("%s*",type));
		ASTNode* const_declaration = create_const_declaration(arr_initializer,field_name_str,type_declaration);

		//TP: replace the original field identifier e.g. CHEMISTRY with the generated list of handles
		//node->lhs = astnode_dup(elems,NULL);
		//node->rhs = var_definitions;
		//node->rhs = astnode_dup(elems,NULL);
		node->type = NODE_UNKNOWN;
		node->lhs  = decls;
		node->rhs  = const_declaration;
		//ASTNode* res_node = astnode_create(NODE_UNKNOWN,astnode_dup(node,NULL), const_declaration);
		//if(node->parent->rhs && node->parent->rhs->id)
		//	node->parent->rhs = res_node;
		//else
		//	node->parent->lhs = res_node;
		return true;
	}
	string_vec types = get_struct_field_types(type);
	string_vec names = get_struct_field_names(type);
	if(types.size > 0)
	{
		const char* var_name = get_node_by_token(IDENTIFIER,node->rhs)->buffer;
		node_vec id_nodes = VEC_INITIALIZER;
		for(size_t i = 0; i < types.size; ++i)
		{
			ASTNode* identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
			identifier->token = IDENTIFIER;
			astnode_sprintf(identifier,"%s_%s",var_name,to_upper_case(names.data[i]));
			ASTNode* primary_expression = astnode_create(NODE_PRIMARY_EXPRESSION,identifier,NULL);
			ASTNode* id_node = astnode_create(NODE_UNKNOWN,primary_expression,NULL);
			set_identifier_type(NODE_VARIABLE_ID,id_node);
			push_node(&id_nodes,id_node);
		}
		node_vec decl_nodes = VEC_INITIALIZER;
		for(size_t i = 0; i < types.size; ++i)
		{
			ASTNode* decl = astnode_create((NODE_DECLARATION | NODE_GLOBAL),
					create_type_declaration_with_qualifiers(tquals,types.data[i]),
					astnode_dup(id_nodes.data[i],NULL)
					);
			ASTNode* res_node = astnode_create(NODE_UNKNOWN,decl,NULL);
			push_node(&decl_nodes,res_node);
		}
		ASTNode* elems = build_list_node(id_nodes,",");
		ASTNode* decls = build_list_node(decl_nodes,"");
		ASTNode* struct_initializer = create_struct_initializer(elems);
		ASTNode* type_declaration = create_type_declaration("const",type);
		ASTNode* const_declaration = create_const_declaration(struct_initializer,var_name,type_declaration);
		const ASTNode* res_name = get_node_by_token(IDENTIFIER,const_declaration->rhs);
		ASTNode* new_lhs = astnode_create(NODE_DECLARATION | NODE_GLOBAL, astnode_dup(node->lhs,NULL),astnode_dup(elems,NULL));
		node->type = NODE_UNKNOWN;
		node->lhs = decls;
		node->rhs = const_declaration;
		free_node_vec(&id_nodes);
		free_node_vec(&decl_nodes);
		free_str_vec(&types);
		free_str_vec(&names);
		return true;
	}
	return res;
}
void
expand_allocating_types(ASTNode* root)
{
	bool expand = true;
	while(expand)
	{
		expand = expand_allocating_types_base(root);
		//printf("Expanded: %d\n",expand);
	}

}
void
preprocess(ASTNode* root, const bool optimize_conditionals)
{
  remove_extra_braces_in_arr_initializers(root);
  symboltable_reset();
  rename_scoped_variables(root,NULL,NULL);
  symboltable_reset();
  free_node_vec(&dfunc_nodes);
  free_str_vec(&dfunc_names);
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  expand_allocating_types(root);
  canonalize(root);
  mark_kernel_inputs(root);


  traverse(root, 0, NULL);
  check_for_writes_to_const_variables(root);
  process_overrides(root);
  transform_field_intrinsic_func_calls_and_ops(root);
  //We use duplicate dfuncs from gen_boundcond_kernels
  //duplicate_dfuncs = get_duplicate_dfuncs(root);
  mark_first_declarations(root);
  gen_overloads(root);
  eval_conditionals(root,root);
  transform_broadcast_assignments(root);
  gen_kernel_combinatorial_optimizations_and_input(root,optimize_conditionals);
  free_structs_info(&s_info);
  gen_calling_info(root);
  symboltable_reset();
  traverse_base(root, 0, 0, NULL, true,NULL,false);
}


void
gen_extra_funcs(const ASTNode* root_in, FILE* stream)
{
  	for(int i = 0; i < MAX_NESTS; ++i)
  	{
  		const unsigned initial_size = 2000;
  		hashmap_create(initial_size, &symbol_table_hashmap[i]);
  	}

	gen_global_strings();


	push(&tspecifier_mappings,INT_STR);
	push(&tspecifier_mappings,REAL_STR);
	push(&tspecifier_mappings,BOOL_STR);
  	ASTNode* root = astnode_dup(root_in,NULL);
  	s_info = read_user_structs(root);
	e_info = read_user_enums(root);
	expand_allocating_types(root);

	symboltable_reset();
	rename_scoped_variables(root,NULL,NULL);
	symboltable_reset();

  	process_overrides(root);
  	traverse_base(root, 0, 0, NULL, true,NULL,false);
        duplicate_dfuncs = get_duplicate_dfuncs(root);

	mark_first_declarations(root);

  	assert(root);
	gen_type_info(root);
	gen_extra_func_definitions_recursive(root,root,stream);
	free_str_vec(&duplicate_dfuncs.names);
	free_int_vec(&duplicate_dfuncs.counts);
  	free_structs_info(&s_info);
}


void
stencilgen(ASTNode* root)
{
  const size_t num_stencils = count_symbols(STENCIL_STR);
  //const size_t num_profiles = count_symbols(PROFILE);

  // Device constants
  // gen_dconsts(root, stream);

  // Stencils

  // Stencil generator
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
       if(symbol.tspecifier == STENCIL_STR) {
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
  char* stencil_coeffs;
  size_t file_size;
  FILE* stencil_coeffs_fp = open_memstream(&stencil_coeffs, &file_size);
  //Fill symbol table
  traverse(root,
           NODE_VARIABLE_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT | NODE_FUNCTION_ID,
           stencil_coeffs_fp);
  fflush(stencil_coeffs_fp);

  	fprintf(stencilgen, "static char* "
                      "dynamic_coeffs[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = { %s };\n", stencil_coeffs);
  	fprintf(stencilgen, "static char* "
                      "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = {");
  traverse(root,
           NODE_VARIABLE_ID | NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT | NODE_FUNCTION_ID,
           stencilgen);
  fprintf(stencilgen, "};");
  fclose(stencilgen);
}

//These are the same for mem_accesses pass and normal pass
void
gen_output_files(ASTNode* root)
{

  //TP: Get number of run_const variable by skipping overrides
  traverse_base(root, NODE_ASSIGN_LIST, NODE_ASSIGN_LIST, NULL, false, NULL,false);
  num_profiles = count_profiles();
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  symboltable_reset();
  traverse(root, 0, NULL);
  process_overrides(root);

  file_append("user_typedefs.h","#include \"func_attributes.h\"\n");
  gen_user_enums();
  gen_user_structs();
  replace_const_ints(root,const_int_values,const_ints);
  gen_user_defines(root, "user_defines.h");
  gen_kernel_structs(root);
  FILE* fp = fopen("user_kernel_declarations.h","w");
  fclose(fp);
  fp = fopen("user_kernel_declarations.h","a");
  fclose(fp);
  stencilgen(root);
  gen_user_kernels("user_declarations.h");
  fp = fopen("user_typedefs.h","a");
  fprintf(fp,"typedef enum{\n");
  string_vec datatypes = get_all_datatypes();
  for (size_t i = 0; i < datatypes.size; ++i)
  {
	const char* define_name = convert_to_define_name(datatypes.data[i]);
	const char* uppr_name =       strupr(define_name);
	fprintf(fp,"  AC_%s_TYPE,\n",uppr_name);

  }
  free_str_vec(&datatypes);
  fprintf(fp,"  AC_PROF_TYPE\n");
  fprintf(fp,"} AcType;\n\n");
  fclose(fp);


}
bool
eliminate_conditionals_base(ASTNode* node)
{
	bool res = false;
	if(node->lhs)
		res |= eliminate_conditionals_base(node->lhs);
	if((node->type & NODE_IF && node->is_constexpr))
	{
		const bool is_executed = int_vec_contains(executed_nodes,node->id);
		const bool is_elif = get_node_by_token(ELIF,node->parent->lhs)  != NULL;
		if(is_executed)
		{
			//TP: now we know that this constexpr conditional is taken
			//TP: this means that its condition has to be always true given its constexpr nature, thus if the previous conditionals were not taken this is always taken
			//TP: since we iterate the conditionals in order we can remove the conditionals on the right that can not be taken
			node->rhs->rhs = NULL;
			//TP: if is not elif this is the base case and there is only a single redundant check left
			if(!is_elif)
			{
				node->rhs->lhs->parent = node->parent->parent;
				node->parent->parent->lhs = node->rhs->lhs;
			}
			node->type ^= NODE_IF;
			return true;
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
			else if(node->rhs->rhs && node->rhs->rhs->lhs->token == ELSE)
			{
				ASTNode* else_node = node->rhs->rhs;
				ASTNode* statement = node->parent->parent;
				statement->lhs = else_node->rhs;
				else_node->rhs->parent = statement;
			}
			//Conditional with only a single case that is not taken, simple remove the whole conditional
			else
			{
				ASTNode* statement = node->parent->parent;
				statement->lhs = NULL;
			}
			//node->type ^= NODE_IF;
			return true;
		}
	}
	if(node->rhs)
		res |= eliminate_conditionals_base(node->rhs);
	return res;

}
bool
eliminate_conditionals(ASTNode* node)
{
	bool process = true;
	bool eliminated_something = false;
	int round_num = 0;
	while(process)
	{
		const bool eliminated_something_this_round = eliminate_conditionals_base(node);
		process = eliminated_something_this_round;
		eliminated_something = eliminated_something || eliminated_something_this_round;
		printf("ELIMINATED SOMETHING THIS ROUND: %d,%d\n",eliminated_something_this_round,round_num++);
	}
	return eliminated_something;
}


void
clean_stream(FILE* stream)
{
	freopen(NULL,"w",stream);
}


void
gen_analysis_stencils(FILE* stream)
{
  string_vec stencil_names = get_names(STENCIL_STR);
  for (size_t i = 0; i < stencil_names.size; ++i)
    fprintf(stream,"AcReal %s(const Field& field_in)"
           "{stencils_accessed[field_in][stencil_%s]=1;return AcReal(1.0);};\n",
           stencil_names.data[i], stencil_names.data[i]);
  free_str_vec(&stencil_names);
}

void
check_array_dim_identifiers(const char* id, const ASTNode* node)
{
	if(node->lhs) check_array_dim_identifiers(id,node->lhs);
	if(node->rhs) check_array_dim_identifiers(id,node->rhs);

	if(node->type & NODE_BINARY_EXPRESSION && node->rhs && node->rhs->lhs && get_node_by_token(IDENTIFIER,node))
		fatal(
				"Only arithmetic expressions consisting of const integers allowed in global array dimensions\n"
				"Wrong expression (%s) in dimension of variable: %s\n\n"
				,combine_all_new(node),id
		);
	if(node->token != IDENTIFIER)     return;
	if(node->type  & NODE_MEMBER_ID) return;
	const bool is_int_var = check_symbol(NODE_VARIABLE_ID,node->buffer,INT_STR,0) || check_symbol(NODE_VARIABLE_ID,node->buffer,INT3_STR,0);
	if(!is_int_var)
		fatal(
			"Only dconst and const integer variables allowed in array dimensions\n"
			"Wrong dimension (%s) for variable: %s\n\n"
			,node->buffer,id
		);

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
  const size_t num_stencils = count_symbols(STENCIL_STR);
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
    fprintf(tmp,
            "static int "
            "write_tmp_called[NUM_KERNELS][NUM_ALL_FIELDS] = {");
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

const char*
get_cached_var_name(const int cache_index)
{
	return sprintf_intern("cached_value_%d",cache_index);
}

void
cache_func_calls_in_function(ASTNode* node, string_vec* cached_calls_expr, node_vec* cached_calls)
{
	if(node->lhs)
		cache_func_calls_in_function(node->lhs,cached_calls_expr,cached_calls);

	if(node->type & NODE_FUNCTION_CALL && node->lhs && get_node_by_token(IDENTIFIER,node->lhs) && node->rhs)
	{
		const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
		if(!strcmp(func_name,"value_AC_MANGLED_NAME__FieldARRAY") || !strcmp(func_name,"value_AC_MANGLED_NAME__Field3ARRAY"))
		{
			if(all_identifiers_are_const(node->rhs))
			{
				const char* call = intern(combine_all_new(node));
				int cache_index = str_vec_get_index(*cached_calls_expr,call);
				if(cache_index == -1)
				{
					push(cached_calls_expr,call);	
					push_node(cached_calls,node);	
					cache_index = cached_calls->size-1;
				}
				replace_node(node,create_primary_expression(get_cached_var_name(cache_index)));

			}
		}
	}
	if(node->rhs)
		cache_func_calls_in_function(node->rhs,cached_calls_expr,cached_calls);
}
//TP: for now only cache func calls of value(arr) where arr is an array of Field3 or Field.
//TP: the motivation is that since you can have arbitrary amount of (field_arr + arr_2 ...) where you always call value(field_arr) seems wasteful to create the array each time even though it is exactly the same
void
cache_func_calls(ASTNode* node)
{
	TRAVERSE_PREAMBLE(cache_func_calls);
	if(!node->rhs) return;
	if(!node->rhs->rhs) return;
	if(!node->rhs->rhs->lhs) return;
	if(node->type & NODE_FUNCTION && node->rhs->rhs->lhs)
	{
		string_vec cached_calls_expr = VEC_INITIALIZER;
		node_vec   cached_calls      = VEC_INITIALIZER;
		ASTNode* head = node->rhs->rhs->lhs;
		while(head->lhs && head->lhs->type & NODE_STATEMENT_LIST_HEAD) head = head->lhs;
		cache_func_calls_in_function(node, &cached_calls_expr, &cached_calls);
		for(size_t i = 0; i < cached_calls.size; ++i)
		{
			ASTNode* res = create_assignment(
								create_declaration(get_cached_var_name(i),NULL,NULL),
								cached_calls.data[i],
								"="
							);
			push_to_statement_list(head,res);
		}
		free_str_vec(&cached_calls_expr);
		free_node_vec(&cached_calls);
	}
}
void
rename_kernels(ASTNode* node)
{
	TRAVERSE_PREAMBLE(rename_kernels);
	if(node->type & NODE_KFUNCTION || node->type & NODE_FUNCTION_CALL)
	{
		ASTNode* id_node = get_node_by_token(IDENTIFIER,node->lhs);
		if(!id_node) return;
		if(!check_symbol(NODE_FUNCTION_ID,id_node->buffer,KERNEL_STR,0)) return;
		astnode_sprintf(id_node,"AC_ANALYSIS_%s",id_node->buffer);
		printf("RENAMED\n");
	}
}
const ASTNode*
get_func_head(const char* func_name, const ASTNode* node)
{
	if(node->lhs)
	{
		const ASTNode* lhs_res = get_func_head(func_name, node->lhs);
		if(lhs_res) return lhs_res;
	}
	if(node->rhs)
	{
		const ASTNode* rhs_res = get_func_head(func_name, node->rhs);
		if(rhs_res) return rhs_res;
	}
	if(!(node->type & NODE_FUNCTION)) return NULL;
	if (func_name == get_node_by_token(IDENTIFIER,node)->buffer)
		return node;
	return NULL;
}
void replace_write_calls(ASTNode* node, const ASTNode* decl_node)
{
	TRAVERSE_PREAMBLE_PARAMS(replace_return_nodes,decl_node);
	if(!is_return_node(node)) return;
	replace_node(node,create_assignment(decl_node,node->rhs,EQ_STR));
}

void fuse_kernels(const char* a, const char* b, const ASTNode* root)
{
	if(!has_optimization_info()) return;
	const int a_index = get_symbol_index(NODE_FUNCTION_ID, a, KERNEL_STR);
	const int b_index = get_symbol_index(NODE_FUNCTION_ID, b, KERNEL_STR);
	const ASTNode* a_head = get_func_head(a,root);
	const ASTNode* b_head = get_func_head(b,root);
	if(!a_head) return;
	if(!b_head) return;
	//TP: for now do not fuse kernels that have stencil ops
	for(size_t field_index = 0; field_index < num_fields; ++field_index)
	{
		if(field_has_stencil_op[field_index + num_fields*a_index]) return;
		if(field_has_stencil_op[field_index + num_fields*b_index]) return;
	}
	////TP: fusing makes only sense if the inputs overlap
	bool inputs_overlap = false;
	for(size_t field_index = 0; field_index < num_fields; ++field_index)
	{
		inputs_overlap |= (read_fields[field_index + num_fields*a_index] && read_fields[field_index + num_fields*b_index]);
		inputs_overlap |= (written_fields[field_index + num_fields*a_index] && read_fields[field_index + num_fields*b_index]);
	}
	if(!inputs_overlap) return;
	const ASTNode* a_body = a_head->rhs->rhs->lhs;
	const ASTNode* b_body = b_head->rhs->rhs->lhs;
	FILE* fp = fopen("fused_kernels.h","a");
	fprintf(fp,"Kernel %s_FUSED_%s () {\n",a,b);
	fprintf(fp,"%s",combine_all_new_with_whitespace(a_body));
	fprintf(fp,"%s",combine_all_new_with_whitespace(b_body));
	fprintf(fp,"}\n");
	fclose(fp);
}
void
generate(const ASTNode* root_in, FILE* stream, const bool gen_mem_accesses, const bool optimize_conditionals)
{ 

  (void)optimize_conditionals;
  symboltable_reset();
  ASTNode* root = astnode_dup(root_in,NULL);
  replace_const_ints(root,const_int_values,const_ints);
  gen_reduce_info(root);
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  gen_type_info(root);
  gen_constexpr_info(root,gen_mem_accesses);
  if(gen_mem_accesses)
  {
  	//gen_ssa_in_basic_blocks(root);
	//remove_dead_writes(root);
  }

  traverse_base(root, 0, NODE_NO_OUT, NULL,true,NULL,false);
  num_profiles = count_profiles();
  check_global_array_dimensions(root);

  gen_multidimensional_field_accesses_recursive(root,gen_mem_accesses);



  // Fill the symbol table
  gen_user_taskgraphs(root);
  combinatorial_params_info info = get_combinatorial_params_info(root);
  gen_kernel_input_params(root,info.params.vals,info.kernels_with_input_params,info.kernel_combinatorial_params,gen_mem_accesses);
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
          FILE* fp = fopen("fields_info.h","w");
          gen_field_info(fp);
          fclose(fp);

	  symboltable_reset();
  	  traverse(root, NODE_NO_OUT, NULL);
	  string_vec datatypes = get_all_datatypes();

  	  FILE* fp_info = fopen("array_info.h","w");
  	  fprintf(fp_info,"\n #ifdef __cplusplus\n");
  	  fprintf(fp_info,"\n#include <array>\n");
  	  fprintf(fp_info,"typedef struct {int base; const char* member; bool from_config;} AcArrayLen;\n");
  	  fprintf(fp_info,"typedef struct { bool is_dconst; int num_dims; std::array<AcArrayLen,%d> dims; const char* name; bool is_alive;} array_info;\n",MAX_ARRAY_RANK);
  	  for (size_t i = 0; i < datatypes.size; ++i)
  	  	  gen_array_info(fp_info,datatypes.data[i],root);
  	  fprintf(fp_info,"\n #endif\n");
  	  fclose(fp_info);

	  //TP: !IMPORTANT! gen_array_info will temporarily update the nodes to push DEAD_STR type qualifiers to dead gmem arrays.
	  //This info is used in gen_gmem_array_declarations so they should be called after each other, maybe will simply combine them into a single function
  	  for (size_t i = 0; i < datatypes.size; ++i)
	  	gen_gmem_array_declarations(datatypes.data[i],root);
  }
  //TP: currently only support scalar arrays
  for(size_t i = 0; i  < primitive_datatypes.size; ++i)
  	gen_array_reads(root,root,primitive_datatypes.data[i]);
  gen_matrix_reads(root);

  // Stencils

  // Stencil generator

  // Compile
  gen_stencils(gen_mem_accesses,stream);


  traverse(root,NODE_NO_OUT,NULL);
  cache_func_calls(root);
  inline_dfuncs(root);
  gen_type_info(root);
  gen_constexpr_info(root,gen_mem_accesses);

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

  symboltable_reset();
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           stream);
  if(gen_mem_accesses)
  {
	  fflush(stream);
	  //This is used to eliminate known constexpr conditionals
	  //TP: for know set code elimination off
	  //bool eliminated_something = true;
	  bool eliminated_something = false;

	  int round = 0;
  	  gen_constexpr_info(root,gen_mem_accesses);
	  while(eliminated_something)
	  {
		printf("ELIMINATION ROUND: %d\n",round++);
	  	clean_stream(stream);

		symboltable_reset();
  	  	traverse(root,
           		NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               		NODE_HOSTDEFINE | NODE_NO_OUT,
           	stream);
	  	fflush(stream);
	  	get_executed_nodes(round-1);
	  	eliminated_something = eliminate_conditionals(root);
		gen_constexpr_info(root,gen_mem_accesses);
	  }


	  clean_stream(stream);

	  symboltable_reset();
  	  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE | NODE_NO_OUT,
           stream);
	  fflush(stream);
	  get_executed_nodes(0);
  }

  // print_symbol_table();
  //free(written_fields);
  //free(read_fields);
  //free(field_has_stencil_op);
  //written_fields       = NULL;
  //read_fields          = NULL;
  //field_has_stencil_op = NULL;

  free_structs_info(&s_info);
  //fuse_kernels(intern("reduce_aa_x"),intern("reduce_aa_y"),root);
  //fuse_kernels(intern("remove_aa_x"),intern("remove_aa_y"),root);
  memset(reduce_infos,0,sizeof(node_vec)*MAX_FUNCS);
}


void
compile_helper(const bool log)
{
  format_source("user_kernels.h.raw","user_kernels.h");
  system("cp user_kernels.h user_kernels_backup.h");
  system("cp user_kernels.h user_cpu_kernels.h");
  FILE* analysis_stencils = fopen("analysis_stencils.h", "w");
  gen_analysis_stencils(analysis_stencils);
  fclose(analysis_stencils);
  if(log)
  {
  	printf("Compiling %s...\n", STENCILACC_SRC);
#if AC_USE_HIP
  	printf("--- USE_HIP: `%d`\n", AC_USE_HIP);
#else
  	printf("--- USE_HIP not defined\n");
#endif
  	printf("--- ACC_RUNTIME_API_DIR: `%s`\n", ACC_RUNTIME_API_DIR);
  	printf("--- GPU_API_INCLUDES: `%s`\n", GPU_API_INCLUDES);
  }
#if AC_USE_HIP
  const char* use_hip = "-DAC_USE_HIP=1 ";
#else
  const char* use_hip = "";
#endif
  char cmd[4096];
  const char* api_includes = strlen(GPU_API_INCLUDES) > 0 ? " -I " GPU_API_INCLUDES  " " : "";
  sprintf(cmd, "gcc -Wshadow -I. -I " ACC_RUNTIME_API_DIR " %s %s -DAC_DOUBLE_PRECISION=%d " 
	       STENCILACC_SRC " -lm -lstdc++ -o " STENCILACC_EXEC" "
  ,api_includes, use_hip, AC_DOUBLE_PRECISION
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
  if(log)
  	printf("Compile command: %s\n", cmd);
  {
	const int retval = system(cmd);
  	if (retval != 0) {
  	  fprintf(stderr, "Catastrophic error: could not compile the stencil access "
  	                  "generator.\n");
  	  fprintf(stderr, "Compiler error code: %d\n",retval);
  	  exit(EXIT_FAILURE);
  	}
  }
  if(log)
  	printf("%s compilation done\n", STENCILACC_SRC);
}

void
get_executed_nodes(const int round)
{
	compile_helper(false);
	char cmd[4096];
	sprintf(cmd,"cp user_kernels.h user_kernels_round_%d.h",round);
        system(cmd);
  	FILE* proc = popen("./" STENCILACC_EXEC " -C", "r");
  	assert(proc);
  	pclose(proc);

  	free_int_vec(&executed_nodes);
  	FILE* fp = fopen("executed_nodes.bin","rb");
  	int size;
  	int tmp;
  	fread(&size, sizeof(int), 1, fp);
  	for(int i = 0; i < size; ++i)
  	{
  		fread(&tmp, sizeof(int), 1, fp);
  	      push_int(&executed_nodes,tmp);
  	}
  	fclose(fp);
}
void
generate_mem_accesses(void)
{
  compile_helper(true);
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

  fp = fopen("user_read_profiles.bin","rb");
  read_profiles     = (int*)malloc(num_kernels*num_profiles*sizeof(int));
  fread(read_profiles, sizeof(int), num_kernels*num_profiles, fp);
  fclose(fp);

  fp = fopen("user_reduced_profiles.bin","rb");
  reduced_profiles  = (int*)malloc(num_kernels*num_profiles*sizeof(int));
  fread(reduced_profiles, sizeof(int), num_kernels*num_profiles, fp);
  fclose(fp);
}


/**
int_vec
get_read_fields(const char* func_name)
{
	int_vec res = VEC_INITIALIZER;
  	const bool has_optimization_info = has_optimization_info();
	if(!has_optimization_info) return res;
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,func_name,KERNEL_STR);
	if(kernel_index == -1) return res;
	for(size_t i = 0; i < num_fields; ++i)
		if (read_fields[i + num_fields*kernel_index]) push_int(&res,i);
	return res;
}
**/
