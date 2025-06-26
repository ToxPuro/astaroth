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
#define SIZE_T_STR    primitive_datatypes.data[6]
#define MAX_ARRAY_RANK (10)
#if AC_USE_HIP
const bool HIP_ON = true;
#else
const bool HIP_ON = false;
#endif
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
static const char* INTRINSIC_STR = NULL;
static const char* CHAR_PTR_STR = NULL;
static const char* REAL_PTR_STR = NULL;
static const char* BOOL_PTR_STR = NULL;
static const char* REAL3_PTR_STR = NULL;
static const char* FIELD3_PTR_STR = NULL;
static const char* VTXBUF_PTR_STR = NULL;
static const char* FIELD_PTR_STR = NULL;
static const char* STENCIL_STR    = NULL;

static const char* MATRIX_STR   = NULL;
static const char* TENSOR_STR   = NULL;
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
static const char* MODEQ_STR= NULL;
static const char* MINUSEQ_STR= NULL;
static const char* DEQ_STR= NULL;
static const char* PERIODIC = NULL;


static const char* VALUE_STR      = NULL;
static const char* OUTPUT_VALUE_STR      = NULL;

static const char* DEAD_STR      = NULL;
static const char* AUXILIARY_STR      = NULL;
static const char* COMMUNICATED_STR      = NULL;
static const char* DEVICE_ONLY_STR       = NULL;
static const char* DIMS_STR = NULL;
static const char* HALO_STR = NULL;
static const char* FIELD_ORDER_STR = NULL;

static const char* CONST_STR = NULL;
static const char* CONSTEXPR_STR = NULL;
static const char* OUTPUT_STR = NULL;
static const char* INPUT_STR = NULL;
static const char* GLOBAL_STR = NULL;
static const char* GLOBAL_MEM_STR = NULL;
static const char* DYNAMIC_STR = NULL;
static const char* INLINE_STR = NULL;
static const char* UTILITY_STR = NULL;
static const char* ELEMENTAL_STR = NULL;
static const char* BOUNDCOND_STR = NULL;
static const char* FIXED_BOUNDARY_STR = NULL;
static const char* RAYTRACE_STR = NULL;
static const char* RUN_CONST_STR = NULL;
static const char* CONST_DIMS_STR = NULL;
static const char* DCONST_STR = NULL;

static const char* FIELD_STR      = NULL;
static const char* KERNEL_STR      = NULL;
static const char* FIELD3_STR      = NULL;
static const char* FIELD4_STR      = NULL;
static const char* PROFILE_STR      = NULL;
static const char* COMPLEX_FIELD_STR  = NULL;


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
	fprintf(fp,"LOAD_DSYM(%s,stream)\n",func_name);
}


void
get_executed_nodes(const int round);

static int monomorphization_index = 0;

#include "ast.h"
#include "tab.h"
#include <string.h>
#include <ctype.h>
extern string_vec const_ints;
extern string_vec const_int_values;
extern string_vec run_const_ints;
extern string_vec run_const_int_values;
#include "expr.h"


#define TRAVERSE_PREAMBLE(FUNC_NAME) \
	if(node->lhs) \
		FUNC_NAME(node->lhs); \
	if(node->rhs) \
		FUNC_NAME(node->rhs); 

static node_vec    dfunc_nodes      = VEC_INITIALIZER;
static string_vec  dfunc_names      = VEC_INITIALIZER;

static node_vec    kfunc_nodes      = VEC_INITIALIZER;
static string_vec  kfunc_names      = VEC_INITIALIZER;
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

static int* written_fields           = NULL;
static int* read_fields              = NULL;
static int* read_profiles            = NULL;
static int* reduced_profiles         = NULL;
static int* reduced_reals            = NULL;
static int* reduced_ints             = NULL;
static int* reduced_floats           = NULL;
static int* field_has_stencil_op     = NULL;
static int* field_has_previous_call  = NULL;
static size_t num_fields   = 0;
static size_t num_complex_fields = 0;
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

bool skip_kernel_in_analysis[MAX_KERNELS] = {};
#define MAX_FUNCS (1100)
#define MAX_COMBINATIONS (1000)
static int MAX_DFUNCS = 0;
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
get_symbol_by_index_and_qualifier(const NodeType type, const int index, const char* tspecifier, const char* tqual)
{
  int counter = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if ((!tspecifier || symbol_table[i].tspecifier == tspecifier) && symbol_table[i].type & type && (str_vec_contains(symbol_table[i].tqualifiers,tqual)))
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
 return written_fields && read_fields && field_has_stencil_op && num_kernels && num_fields && field_has_previous_call && reduced_reals && reduced_ints && reduced_floats && reduced_profiles;
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
all_same_struct(const char* struct_name, const char* type)
{
	if(is_primitive_datatype(struct_name)) return false;
	string_vec target_types = VEC_INITIALIZER;
	push(&target_types,type);
	const bool res = consists_of_types(target_types,struct_name);
	free_str_vec(&target_types);
	return res;
}

typedef struct
{
	string_vec names;
	int_vec* called_funcs;
	int* topological_index;
} funcs_calling_info;

static funcs_calling_info calling_info = {VEC_INITIALIZER, .called_funcs = NULL, .topological_index = NULL};
static node_vec reduce_infos[MAX_FUNCS] = {[0 ... MAX_FUNCS -1] = VEC_INITIALIZER};

void
generate_error_messages()
{
	if(!has_optimization_info()) return;
	for(size_t kernel = 0; kernel < num_kernels; ++kernel)
	{
		const char* kernel_name = get_symbol_by_index(NODE_FUNCTION_ID, kernel, KERNEL_STR)->identifier;
		//These are empty Kernels by intent
		if(kernel_name == intern("AC_NULL_KERNEL")) continue;
		if(kernel_name == intern("BOUNDCOND_PERIODIC")) continue;
		bool updates_something = false;
		for(size_t j = 0; j < num_fields; ++j)
		{
			updates_something |= written_fields[j + num_fields*kernel];
			updates_something |= reduced_profiles[j + num_profiles*kernel];
		}
      		const size_t index = str_vec_get_index(calling_info.names,kernel_name);
		updates_something |= (reduce_infos[index].size != 0);
		if(!updates_something)
		{
			//TP: suppress not updating warning since not updating might be totally intentional
			//as an example with PC-A you can have some kernels be dummy kernels based on the input flags
			//
			//printf("\n\n");
			//printf("AC WARNING: Kernel %s does not update anything!!!\n",kernel_name);
			//printf("AC WARNING: Kernel %s does not update anything!!!\n",kernel_name);
			//printf("AC WARNING: Kernel %s does not update anything!!!\n",kernel_name);
			//printf("\n\n");
		}
	}
}
string_vec
get_allocating_types()
{
	static string_vec allocating_types = VEC_INITIALIZER;
	if(allocating_types.size == 0)
	{
		push(&allocating_types,intern("Field"));
		push(&allocating_types,intern("Field3"));
		push(&allocating_types,intern("ComplexField"));
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
static int_vec user_remappings = VEC_INITIALIZER; 
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
  if(type == NODE_VARIABLE_ID && current_nest == 0 && n_tqualifiers == 0 && tspecifier && tspecifier != RAYTRACE_STR)
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
	  my_asprintf(&full_name,"%s_%s",id,postfix);
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





  ++num_symbols[current_nest];
  bool is_field_without_comm_and_aux_qualifiers = tspecifier && tspecifier == FIELD_STR && current_nest == 0;

  for(size_t i = 0; i < symbol_table[num_symbols[current_nest]-1].tqualifiers.size; ++i)
	  is_field_without_comm_and_aux_qualifiers &= (symbol_table[num_symbols[current_nest]-1].tqualifiers.data[i] != COMMUNICATED_STR) && (symbol_table[num_symbols[current_nest]-1].tqualifiers.data[i] != AUXILIARY_STR);

  if(!is_field_without_comm_and_aux_qualifiers)
  	return num_symbols[current_nest]-1;
	  
  if(!has_optimization_info())
  {
  	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, intern("Communicated"));
  	return num_symbols[current_nest]-1;
  } 



   const int field_index = int_vec_get_index(user_remappings,get_symbol_index(NODE_VARIABLE_ID, id, FIELD_STR));

   bool is_auxiliary = true;
   bool is_communicated = false;
   bool is_dead         = true;
   //TP: a field is dead if its existence does not have any observable effect on the DSL computation
   //For now it means that the field is not read, no stencils called on it and not written out
   for(size_t k = 0; k < num_kernels; ++k)
   {
	   if(skip_kernel_in_analysis[k]) continue;
	   const int written        = written_fields[field_index + num_fields*k];
	   const int input_accessed = (read_fields[field_index + num_fields*k] || field_has_stencil_op[field_index + num_fields*k]);
	   is_auxiliary    &=  OPTIMIZE_FIELDS && (!written  || !field_has_stencil_op[field_index + num_fields*k]);
	   is_communicated |=  !OPTIMIZE_FIELDS || field_has_stencil_op[field_index + num_fields*k];
	   const bool should_be_alive = (!OPTIMIZE_FIELDS) || written  || input_accessed;
	   is_dead      &= !should_be_alive;

   }
   if(is_communicated)
   {
   	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, COMMUNICATED_STR);
   }
   if(is_auxiliary)
	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, AUXILIARY_STR);
   if(is_dead && ALLOW_DEAD_VARIABLES)
   {
	push(&symbol_table[num_symbols[current_nest]-1].tqualifiers, DEAD_STR);
   }


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
	string_vec values[100];
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
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("threadIdx"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("blockIdx"));        // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("vertexIdx"));       // TODO REMOVE
  //add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("idx"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("tid"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("start"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("end"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR,  intern("break"));       // TODO REMOVE
  add_symbol(NODE_VARIABLE_ID, dynamic_tq, 1, INT3_STR, intern("globalVertexIdx")); // TODO REMOVE

  //In develop
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "read_w");
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write_w");
  add_symbol(NODE_FUNCTION_ID, NULL, 0, FIELD3_STR,intern("MakeField3")); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL, intern("uint64_t"));   // TODO RECHECK
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR, intern("UINT64_MAX")); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, 0, REAL_STR, REAL_STR);


  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_STR, intern("AC_REAL_PI"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_STR, intern("AC_REAL_EPSILON"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_STR, intern("AC_REAL_MIN"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_STR, intern("AC_REAL_MAX"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR, intern("INT_MAX"));
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

  add_symbol(NODE_VARIABLE_ID, const_tq, 1, INT_STR, intern("STENCIL_ORDER"));
  // Astaroth 2.0 backwards compatibility END
  int index = add_symbol(NODE_VARIABLE_ID, NULL, 0 , INT3_STR, intern("blockDim"));
  symbol_table[index].tqualifiers.size = 0;

  add_symbol(NODE_FUNCTION_ID, NULL, 0, NULL,  intern("periodic"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_PTR_STR, intern("AC_INTERNAL_run_const_AcReal_array_here"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, BOOL_PTR_STR, intern("AC_INTERNAL_run_const_bool_array_here"));


  add_symbol(NODE_VARIABLE_ID, const_tq, 1, REAL_PTR_STR, intern("AC_INTERNAL_run_const_AcReal_array_here"));
  add_symbol(NODE_VARIABLE_ID, const_tq, 1, BOOL_PTR_STR, intern("AC_INTERNAL_run_const_bool_array_here"));

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

	    get_array_access_nodes(base,&dst);
	    return dst;
}
node_vec
get_array_var_dims(const char* var, const ASTNode* root)
{

	    static string_vec cache_keys = VEC_INITIALIZER;
	    static node_vec   cache[10000];

	    int cache_index = str_vec_get_index(cache_keys,var);
	    if(cache_index != -1)
	    {
	    	return cache[cache_index];
	    }

	    const ASTNode* var_identifier = get_node_by_buffer_and_type(var,NODE_VARIABLE_ID,root);
	    const ASTNode* decl = get_parent_node(NODE_DECLARATION,var_identifier);

	    const ASTNode* access_start = get_node(NODE_ARRAY_ACCESS,decl);
	    node_vec res = get_array_accesses(access_start);

	    push(&cache_keys,var);
	    cache_index = str_vec_get_index(cache_keys,var);
	    cache[cache_index] = res;

	    return res;
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
}

int default_accesses[10000] = { [0 ... 10000-1] = 1};
int read_accesses[10000] = { [0 ... 10000-1] = 0};

bool
fread_errchk(int* dst, const size_t bytes, const size_t count, FILE* stream)
{
	return fread(dst,bytes,count,stream) == count;
}

const  int*
get_arr_accesses(const char* datatype_scalar)
{

	char* filename;
	const char* define_name =  convert_to_define_name(datatype_scalar);
	my_asprintf(&filename,"%s_arr_accesses",define_name);
  	if(!file_exists(filename) || !has_optimization_info())
		return default_accesses;

	FILE* fp = fopen(filename,"rb");
	int size = 1;
	bool reading_successful =  fread_errchk(&size, sizeof(int), 1, fp);
	reading_successful      &= fread_errchk(read_accesses, sizeof(int), size, fp);
	fclose(fp);
	if(!reading_successful) fatal("Was not able to read array accesses!\n");
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
	if(!strcmp(node->buffer,"AC_ITERATION_NUMBER")) return true;
	//TP: this should not happend but for now simply set the constexpr value to the correct value
	//TODO: fix
	if(!node->is_constexpr &&  check_symbol(NODE_ANY,node->buffer,0,CONST_STR)) {
		ASTNode* hack = (ASTNode*)node;
		hack->is_constexpr = true;
	}
	if(!node->is_constexpr)
	{
		if(node->buffer == intern("any_AC"))
		{
			ASTNode* hack = (ASTNode*)node;
			hack->is_constexpr = true;
		}
		for(size_t i = 0; i < e_info.names.size; ++i)
		{
			for(size_t option = 0; option < e_info.options[i].size; ++option)
			{
				if(node->buffer == e_info.options[i].data[option])
				{
					ASTNode* hack = (ASTNode*)node;
					hack->is_constexpr = true;
				}
			}
		}
	}
			
	res &= node->is_constexpr;
	return res;
}
array_info
get_array_info(const Symbol* sym, const bool accessed, const ASTNode* root)
{
	array_info res;
	res.is_dconst = str_vec_contains(sym->tqualifiers,DCONST_STR);


	res.dims = get_array_var_dims(sym->identifier,root);
	res.name = sym->identifier;
	res.accessed = accessed;

	return res;
}
void
propagate_array_info(Symbol* sym, const bool accessed, const ASTNode* root)
{
	const array_info info = get_array_info(sym,accessed,root);
	if (!accessed && ALLOW_DEAD_VARIABLES && OPTIMIZE_ARRAYS) push(&sym->tqualifiers,DEAD_STR);
	bool const_dims = true;
	for(size_t dim = 0; dim < MAX_ARRAY_RANK; ++dim) const_dims &= (dim >= info.dims.size || all_identifiers_are_constexpr(info.dims.data[dim]));
	const bool is_gmem = str_vec_contains(sym->tqualifiers,GLOBAL_MEM_STR);
	//TP: for now suppress the knowledge that some gmems are of known sizes since can run of out memory otherwise
	if (const_dims && !is_gmem) push(&sym->tqualifiers,CONST_DIMS_STR);
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
        fprintf(fp, "%s,",(!info.accessed  && ALLOW_DEAD_VARIABLES && OPTIMIZE_ARRAYS) ? "false" : "true");
        fprintf(fp, "%s,",info.accessed ? "true" : "false");
	fprintf(fp,"%s","},");
}
void
gen_array_qualifiers_for_datatype(const char* datatype_scalar, const ASTNode* root)
{
  const int* accesses = get_arr_accesses(datatype_scalar);
  char tmp[1000];
  sprintf(tmp,"%s*",datatype_scalar);
  const char* datatype = intern(tmp);
  int counter = 0;
  for (size_t i = 0; i < num_symbols[0]; ++i)
  {
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        symbol_table[i].tspecifier == datatype)
    {

  	if(str_vec_contains(symbol_table[i].tqualifiers,CONST_STR, RUN_CONST_STR)) continue;
	propagate_array_info(&symbol_table[i],accesses[counter],root);
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

	propagate_array_info(&symbol_table[i],accesses[counter],root);
    }
	   
  }
}
void
gen_array_qualifiers(const ASTNode* root)
{
	string_vec datatypes = get_all_datatypes();
	for(size_t i = 0; i < datatypes.size; ++i)
	{
		gen_array_qualifiers_for_datatype(datatypes.data[i],root);
	}
	free_str_vec(&datatypes);
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
  fprintf(fp,"\"AC_EXTRA_PADDING\",true,true}");
  fprintf(fp, "\n};");
}


void
gen_gmem_array_declarations(const char* datatype_scalar, const ASTNode* root)
{
	const char* define_name = convert_to_define_name(datatype_scalar);
	const char* enum_name = convert_to_enum_name(datatype_scalar);

	char tmp[4098];
	sprintf(tmp,"%s*",datatype_scalar);
	const char* datatype = intern(tmp);


	FILE* fp = fopen("memcpy_to_gmem_arrays.h","a");
	

	fprintf(fp,"void memcpy_to_gmem_array(const %sArrayParam param,%s* &ptr)\n"
        "{\n", enum_name,datatype_scalar);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR))
	  {
		  if (!str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
		  {
			if(str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
			{
				fprintf(fp,"if (param == %s) \n{//%s is dead\n return;}\n",symbol_table[i].identifier,symbol_table[i].identifier);
			}
			else
		  		fprintf(fp,"if (param == %s) {ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(AC_INTERNAL_gmem_%s_arrays_%s,&ptr,sizeof(ptr),0,cudaMemcpyHostToDevice)); return;} \n",symbol_table[i].identifier,define_name,symbol_table[i].identifier);
	  	  }
	  }
	fprintf(fp,"fprintf(stderr,\"FATAL AC ERROR from memcpy_to_gmem_array\\n\");\n");
	fprintf(fp,"\n(void)param;(void)ptr;}\n");



	fprintf(fp,"void memcpy_to_const_dims_gmem_array(const %sArrayParam param,const %s* ptr)\n"
	"{\n", enum_name,datatype_scalar);
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].type & NODE_VARIABLE_ID &&
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR))
	  {
		  if (str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
		  {
			if(str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
			{
				fprintf(fp,"if (param == %s) \n{//%s is dead\n return;}\n",symbol_table[i].identifier,symbol_table[i].identifier);
			}
			else
			{
		  		fprintf(fp,"if (param == %s) {ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(AC_INTERNAL_gmem_%s_arrays_%s,ptr,sizeof(ptr[0])*get_const_dims_array_length(param),0,cudaMemcpyHostToDevice)); return;}\n",symbol_table[i].identifier,define_name,symbol_table[i].identifier);
			}
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
  	      symbol_table[i].tspecifier == datatype && str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR))
	  {
		  if(str_vec_contains(symbol_table[i].tqualifiers,DEAD_STR))
		  {
		  	fprintf(fp,"if (param == %s) {fprintf(stderr,\"Can not read since %s is dead!\\n\"); exit(EXIT_FAILURE);}\n",symbol_table[i].identifier,symbol_table[i].identifier);
		  }
		  else if (str_vec_contains(symbol_table[i].tqualifiers,CONST_DIMS_STR))
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
                  	char array_length_str[100000];
                  	get_array_var_length(symbol_table[i].identifier,root,array_length_str);
			if(!strcmp(array_length_str,"0"))
				sprintf(array_length_str,"1");
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


	fprintf_filename("info_loaded_operator_decl.h","const bool& operator[](const %sParam param) const {return %s_params[param];}\n",enum_name,define_name);
	fprintf_filename("info_loaded_operator_decl.h","const bool& operator[](const %sArrayParam param) const {return %s_arrays[param];}\n",enum_name,define_name);
	fprintf_filename("info_loaded_operator_decl.h","bool& operator[](const %sParam param) {return %s_params[param];}\n",enum_name,define_name);
	fprintf_filename("info_loaded_operator_decl.h","bool& operator[](const %sArrayParam param) {return %s_arrays[param];}\n",enum_name,define_name);
	fprintf_filename("info_loaded_operator_decl.h","bool operator[](const %sCompParam ) const {return false;}\n",enum_name);
	fprintf_filename("info_loaded_operator_decl.h","bool operator[](const %sCompArrayParam ) const {return false;}\n",enum_name);
	fprintf_filename("info_loaded_operator_decl.h","bool operator[](const %s) const {return false;}\n",datatype_scalar);

	fprintf_filename("array_decl.h","%s* %s_arrays[NUM_%s_ARRAYS+1];\n",datatype_scalar,define_name,uppr_name);

	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sParam>::value) return (%s){};\n",enum_name,datatype_scalar);
	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sArrayParam>::value) return (%s){};\n",enum_name,datatype_scalar);
	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sCompParam>::value) return (%s){};\n",enum_name,datatype_scalar);
	fprintf_filename("get_default_value.h","if constexpr(std::is_same<P,%sCompArrayParam>::value) return (%s){};\n",enum_name,datatype_scalar);


	fprintf_filename("get_empty_pointer.h","if constexpr(std::is_same<P,%sArrayParam>::value) return (%s*){};\n",enum_name,datatype_scalar);



	fprintf_filename("get_param_name.h","if constexpr(std::is_same<P,%sParam>::value) return %sparam_names[(int)param];\n",enum_name,define_name);
	fprintf_filename("get_param_name.h","if constexpr(std::is_same<P,%sCompParam>::value) return %s_comp_param_names[(int)param];\n",enum_name,define_name);


	fprintf_filename("get_num_params.h",
				" (std::is_same<P,%sParam>::value)      ? NUM_%s_PARAMS : \n"
				" (std::is_same<P,%sArrayParam>::value) ? NUM_%s_ARRAYS : \n"
				" (std::is_same<P,%sCompParam>::value)      ? NUM_%s_COMP_PARAMS : \n"
				" (std::is_same<P,%sCompArrayParam>::value) ? NUM_%s_COMP_ARRAYS : \n"
				" (std::is_same<P,%sOutputParam>::value) ? NUM_%s_OUTPUTS: \n"
			,enum_name,uppr_name
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
	


        fprintf_filename("device_load_uniform.h","GEN_DEVICE_LOAD_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);

	fprintf_filename("device_store_uniform.h","GEN_DEVICE_STORE_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
	fprintf_filename("device_store_uniform_decl.h","DECL_DEVICE_STORE_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
	fprintf_filename("device_store_overloads.h","OVERLOAD_DEVICE_STORE_UNIFORM(%sParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);

	if(is_primitive_datatype(datatype_scalar))
	{
        	fprintf_filename("device_load_uniform.h","GEN_DEVICE_LOAD_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_load_uniform_decl.h","DEVICE_LOAD_ARRAY_DECL(%sArrayParam, %s)\n",enum_name,upper_case_name);
		fprintf_filename("device_load_uniform_overloads.h","OVERLOAD_DEVICE_LOAD_ARRAY(%sArrayParam, %s)\n",enum_name,upper_case_name);
		fprintf_filename("device_load_uniform_loads.h","LOAD_DSYM(acDeviceLoad%sArray,stream)\n",upper_case_name);


		fprintf_filename("device_store_uniform.h","GEN_DEVICE_STORE_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_uniform_decl.h","DECL_DEVICE_STORE_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
		fprintf_filename("device_store_overloads.h","OVERLOAD_DEVICE_STORE_ARRAY(%sArrayParam, %s, %s)\n",enum_name,datatype_scalar,upper_case_name);
	}
	fprintf_filename("device_get_output.h", "%s\nacDeviceGet%sOutput(Device device, const %sOutputParam param)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_OUTPUTS == 0) return (%s){};\n"
		    "\treturn device->output.%s_outputs[param];\n"
		    "}\n"
	,datatype_scalar,upper_case_name, enum_name, uppr_name, datatype_scalar,define_name);


	fprintf_filename("device_set_input.h", "AcResult \nacDeviceSet%sInput(Device device, const %sInputParam param, const %s val)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_INPUT_PARAMS == 0) return AC_FAILURE;\n"
		    "\tdevice->input.%s_params[param] = val; return AC_SUCCESS;\n"
		    "}\n"
	,upper_case_name, enum_name, datatype_scalar, uppr_name, define_name);


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


	fprintf_filename("device_set_output.h", "AcResult\nacDeviceSet%sOutput(Device device, const %sOutputParam param, const %s val)\n"
		    "{\n"
		    "\tif constexpr(NUM_%s_OUTPUTS == 0) return AC_FAILURE;\n"
		    "\tdevice->output.%s_outputs[param] = val;\n"
		    "\treturn AC_SUCCESS;\n"
		    "}\n"
	,upper_case_name, enum_name, datatype_scalar, uppr_name, define_name);

	fprintf_filename("device_set_output_decl.h", "AcResult\nacDeviceSet%sOutput(Device device, const %sOutputParam param, const %s val);\n"
	,upper_case_name, enum_name, datatype_scalar);

	fprintf_filename("device_set_output_overloads.h","#ifdef __cplusplus\nstatic inline AcResult acDeviceSetOutput(Device device, const %sOutputParam& param, const %s val){ return acDeviceSet%sOutput(device,param,val); }\n#endif\n",enum_name, datatype_scalar, upper_case_name);	

	fprintf_filename("device_get_input_overloads.h","#ifdef __cplusplus\nstatic inline %s acDeviceGetInput(Device device, const %sInputParam& param){ return acDeviceGet%sInput(device,param); }\n#endif\n",datatype_scalar,enum_name, upper_case_name);	
	
	{
		FILE* fp;
		char* func_name;
		fp = fopen("device_set_input_loads.h","a");
		my_asprintf(&func_name,"acDeviceSet%sInput",upper_case_name);
		gen_dlsym(fp,func_name);
		fclose(fp);

		fp = fopen("device_get_input_loads.h","a");
		my_asprintf(&func_name,"acDeviceGet%sInput",upper_case_name);
		gen_dlsym(fp,func_name);
		fclose(fp);

		fp = fopen("device_get_output_loads.h","a");
		my_asprintf(&func_name,"acDeviceGet%sOutput",upper_case_name);
		gen_dlsym(fp,func_name);
		fclose(fp);

		free(func_name);
	}

	if(datatype_scalar == REAL_STR || datatype_scalar == INT_STR || (datatype_scalar == FLOAT_STR && AC_DOUBLE_PRECISION))
	{
		fprintf_filename("device_finalize_reduce.h",
				"AcResult\n"
				"acDeviceFinishReduce%s(Device device, const Stream stream, %s* result, const AcKernel kernel, const AcReduceOp reduce_op, const %sOutputParam output)\n"
				"{\n"
				"if constexpr (NUM_%s_OUTPUTS == 0) return AC_FAILURE;\n"
				"ERRCHK(stream < NUM_STREAMS);\n"
				"acReduce(device->streams[stream],reduce_op,device->vba.reduce_buffer_%s[output],acGetKernelReduceScratchPadSize(kernel));\n"
				"cudaMemcpyAsync(result,device->vba.reduce_buffer_%s[output].res,sizeof(result[0]),cudaMemcpyDeviceToHost,device->streams[stream]);\n"
				"return AC_SUCCESS;\n"
				"}\n"
				,upper_case_name,datatype_scalar,enum_name,uppr_name,define_name,define_name
				);
		
		fprintf_filename("device_finalize_reduce.h",
				"AcResult\n"
				"acDeviceFinishReduce%sStream(Device device, const cudaStream_t stream, %s* result, const AcKernel kernel, const AcReduceOp reduce_op, const %sOutputParam output)\n"
				"{\n"
				"if constexpr (NUM_%s_OUTPUTS == 0) return AC_FAILURE;\n"
				"acReduce(stream,reduce_op,device->vba.reduce_buffer_%s[output],acGetKernelReduceScratchPadSize(kernel));\n"
				"cudaMemcpyAsync(result,device->vba.reduce_buffer_%s[output].res,sizeof(result[0]),cudaMemcpyDeviceToHost,stream);\n"
				"return AC_SUCCESS;\n"
				"}\n"
				,upper_case_name,datatype_scalar,enum_name,uppr_name,define_name,define_name
				,define_name
				,upper_case_name
				);

		fprintf_filename("scalar_reduce_buffer_defs.h",
				"typedef struct\n"
				"{\n"
				"%s** src;\n"
				"%s** cub_tmp;\n"
				"size_t* cub_tmp_size;\n"
				"size_t* buffer_size;\n"
				"%s* res;\n"
				"} Ac%sScalarReduceBuffer;\n\n"
				,datatype_scalar,datatype_scalar,datatype_scalar,upper_case_name
				);
		fprintf_filename("scalar_reduce_buffers_in_vba.h",
				 "Ac%sScalarReduceBuffer reduce_buffer_%s[NUM_%s_OUTPUTS+1];"
				 ,upper_case_name,define_name,uppr_name
				);

		if(datatype_scalar != REAL_STR)
		{
			fprintf_filename("reduce_helpers.h",
					"__device__  __constant__ %s* d_symbol_reduce_scratchpads_%s[NUM_%s_OUTPUTS+1];\n"
					"static %s* d_reduce_scratchpads_%s[NUM_%s_OUTPUTS+1];\n"
					"static size_t d_reduce_scratchpads_size_%s[NUM_%s_OUTPUTS+1];\n"
					"__device__ __constant__ %s d_reduce_%s_res_symbol[NUM_%s_OUTPUTS+1];\n"
					,datatype_scalar,define_name,uppr_name,datatype_scalar,define_name,uppr_name,define_name,uppr_name
					,datatype_scalar,define_name,uppr_name
					);
		}
		fprintf_filename("reduce_helpers.h",
				"void \n"
				"allocate_scratchpad_%s(const size_t i, const size_t bytes, const AcReduceOp state)\n"
				"{\n"
				"ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&d_reduce_scratchpads_%s[i],bytes));\n"
				"ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(d_symbol_reduce_scratchpads_%s,&d_reduce_scratchpads_%s[i],sizeof(%s*),sizeof(%s*)*i,cudaMemcpyHostToDevice));\n"
				"d_reduce_scratchpads_size_%s[i] = bytes;\n"
				"acKernelFlush(0,d_reduce_scratchpads_%s[i], bytes/sizeof(%s),get_reduce_state_flush_var_%s(state));\n"
				"}\n\n"
				,define_name,define_name,define_name,define_name,datatype_scalar,datatype_scalar,define_name,define_name,datatype_scalar,define_name
				);
		fprintf_filename(
				"reduce_helpers.h",
				"void\n"
				"free_scratchpad_%s(const size_t i)\n"
				"{\n"
				"d_reduce_scratchpads_%s[i] = NULL;\n"
				"ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(d_symbol_reduce_scratchpads_%s,&d_reduce_scratchpads_%s[i],sizeof(%s*),sizeof(%s*)*i,cudaMemcpyHostToDevice));\n"
				"d_reduce_scratchpads_size_%s[i] = 0;\n"
				"}\n"
				,define_name,define_name,define_name,define_name,datatype_scalar,datatype_scalar,define_name
				);
		fprintf_filename(
				"reduce_helpers.h",
				"void\n"
				"resize_scratchpad_%s(const size_t i, const size_t new_bytes, const AcReduceOp state)\n"
				"{\n"
				"if(d_reduce_scratchpads_size_%s[i] >= new_bytes) return;\n"
				"free_scratchpad_%s(i);\n"
				"allocate_scratchpad_%s(i,new_bytes,state);\n"
				"}\n\n"
				,define_name,define_name,define_name,define_name
				);

		fprintf_filename(
				"reduce_helpers.h",
				"void\n"
				"resize_%ss_to_fit(const size_t n_elems, VertexBufferArray vba, const AcKernel kernel)\n"
				"{\n"
				"bool var_reduced = false;\n"
				"for(int i = 0; i < NUM_%s_OUTPUTS; ++i) var_reduced |= reduced_%ss[kernel][i];\n"
				"if(var_reduced)\n"
				"{\n"
				"	const size_t size = n_elems*sizeof(%s);\n"
				"	for(int i = 0; i < NUM_%s_OUTPUTS; ++i)\n"
				"	{\n"
				"		if(!reduced_%ss[kernel][i]) continue;\n"
				"		resize_scratchpad_%s(i, size, vba.scratchpad_states->%ss[i]);\n"
				"	}\n"
				"}\n"
				"}\n\n"
				,define_name,uppr_name,define_name,datatype_scalar,uppr_name,define_name,define_name,define_name,define_name
				);

	}

	fprintf_filename("dconst_decl.h","%s DEVICE_INLINE  DCONST(const %sParam& param){return d_mesh_info.%s_params[(int)param];}\n"
			,datatype_scalar, enum_name, define_name);

	//TP: TODO: compare the performance of having this one level of indirection vs. simply loading the value to dconst and using it from there
	if(datatype_scalar == REAL_STR || datatype_scalar == INT_STR || (datatype_scalar == FLOAT_STR && AC_DOUBLE_PRECISION))
	{
		fprintf_filename("output_value_decl.h","%s DEVICE_INLINE  output_value(const %sOutputParam& param){return d_reduce_%s_res_symbol[(int)param];}\n"
			,datatype_scalar, enum_name, define_name);
	}

	fprintf_filename("dconst_decl.h","%s DEVICE_INLINE  VAL(const %sParam& param){return d_mesh_info.%s_params[(int)param];}\n"
			,datatype_scalar, enum_name, define_name);

	fprintf_filename("dconst_decl.h","%s DEVICE_INLINE  VAL(const %s& val){return val;}\n"
			,datatype_scalar, datatype_scalar, define_name);

	fprintf_filename("rconst_decl.h","%s DEVICE_INLINE RCONST(const %sCompParam&){return d_mesh_info.%s_params[0];}\n"
			,datatype_scalar, enum_name, define_name);

	fprintf_filename("get_address.h","size_t  get_address(const %sParam& param){ return (size_t)&d_mesh_info.%s_params[(int)param];}\n"
			,enum_name, define_name);

	//fprintf_filename("get_address.h","size_t  get_address(const %sOutputParam& param){ return (size_t)&d_output.%s_outputs[(int)param];}\n"
	//		,enum_name, define_name);

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
	        //"static AcResult __attribute ((unused)) "
		//"acLoadOutput(const cudaStream_t stream, const %sOutputParam param, const %s value) { return acLoad%sOutput(stream,param,value);}\n"
		,enum_name, datatype_scalar, upper_case_name
		,enum_name, datatype_scalar, upper_case_name
		,enum_name, datatype_scalar, upper_case_name
		,enum_name, datatype_scalar, upper_case_name
		//,enum_name, datatype_scalar, upper_case_name
		);


	fprintf_filename("load_and_store_uniform_funcs.h",
	 	"AcResult acLoad%sUniform(const cudaStream_t, const %sParam param, const %s value) { return acLoadUniform(param,value); }\n"
	 	"AcResult acLoad%sArrayUniform(const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length) { return acLoadArrayUniform(param ,values, length); }\n"
	 	"AcResult acStore%sUniform(const cudaStream_t, const %sParam param, %s* value) { return acStoreUniform(param,value); }\n"
	 	"AcResult acStore%sArrayUniform(const %sArrayParam param, %s* values, const size_t length) { return acStoreArrayUniform(param ,values, length); }\n"
	 	//"AcResult acLoad%sOutput (const cudaStream_t stream, const %sOutputParam param, const %s value) { return acLoadOutput(stream,param,value); }\n"
	        ,upper_case_name, enum_name, datatype_scalar
	        ,upper_case_name, enum_name, datatype_scalar
	        ,upper_case_name, enum_name, datatype_scalar
	        ,upper_case_name, enum_name, datatype_scalar
	        //,upper_case_name, enum_name, datatype_scalar
	     );




	fprintf_filename("load_and_store_uniform_header.h",
		"FUNC_DEFINE(AcResult, acLoad%sUniform,(const cudaStream_t, const %sParam param, const %s value));\n"
		"FUNC_DEFINE(AcResult, acLoad%sOutput ,(const cudaStream_t, const %sOutputParam param, const %s value));\n"
		"FUNC_DEFINE(AcResult, acLoad%sArrayUniform, (const cudaStream_t, const %sArrayParam param, const %s* values, const size_t length));\n"
		"FUNC_DEFINE(AcResult, acStore%sUniform,(const cudaStream_t, const %sParam param, %s* value));\n"
		"FUNC_DEFINE(AcResult, acStore%sArrayUniform, (const %sArrayParam param, %s* values, const size_t length));\n"
	    	,upper_case_name, enum_name, datatype_scalar
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

	fprintf_filename("is_output_param.h",
		"constexpr static bool UNUSED IsOutputParam(const %s&)               {return false;}\n"       
		"constexpr static bool UNUSED IsOutputParam(const %sParam&)          {return false;}\n"  
		"constexpr static bool UNUSED IsOutputParam(const %sOutputParam&)     {return true;}\n"
		,datatype_scalar
		,enum_name
		,enum_name
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

	//Based on naming these should not be here
	//TODO: move them to an appropriately named function
	fp = fopen("info_loaded_decl.h","a");
	fprintf(fp,"bool  %s_params[NUM_%s_PARAMS];\n",define_name,upper);
	fprintf(fp,"bool  %s_arrays[NUM_%s_ARRAYS];\n",define_name,upper);
	fclose(fp);


	fp = fopen("input_decl.h","a");
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
  {
     counter  += (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,CONST_STR));
  }
  fprintf(fp, "#define NUM_%s_CONSTS (%d)\n",strupr(convert_to_define_name(datatype_scalar)),counter);


  counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
     counter  += (symbol_table[i].tspecifier == datatype_arr&& str_vec_contains(symbol_table[i].tqualifiers,CONST_STR));
  fprintf(fp, "#define NUM_%s_ARR_CONSTS (%d)\n",strupr(convert_to_define_name(datatype_scalar)),counter);

  const char* uppr = strupr(convert_to_define_name(datatype_scalar));
  counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
     counter  += (symbol_table[i].tspecifier == datatype_scalar && symbol_table[i].type == NODE_VARIABLE_ID);
  }
  fprintf(fp,"#define MAX_NUM_%s_COMP_PARAMS (%d)\n",uppr,counter);
  counter = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
     counter  += (symbol_table[i].tspecifier == datatype_arr && symbol_table[i].type == NODE_VARIABLE_ID);

  fprintf(fp,"#define MAX_NUM_%s_COMP_ARRAYS (%d)\n",uppr,counter);
}
void
gen_param_names(FILE* fp, const char* datatype_scalar, const char* place_attribute)
{
  char tmp[1000];
  sprintf(tmp,"%s*",datatype_scalar);
  const char* datatype_arr = intern(tmp);

  fprintf(fp, "%s static const char* %sparam_names%s[] __attribute__((unused)) = {",place_attribute,convert_to_define_name(datatype_scalar),place_attribute);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "\"padding\"};");

  fprintf(fp, "%s static const char* %s_output_names%s[] __attribute__((unused)) = {",place_attribute,convert_to_define_name(datatype_scalar),place_attribute);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "\"padding\"};");

  fprintf(fp, "%s static const char* %s_array_names%s[] __attribute__((unused)) = {",place_attribute,convert_to_define_name(datatype_scalar),place_attribute);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_arr && (str_vec_contains(symbol_table[i].tqualifiers,DCONST_STR) ||str_vec_contains(symbol_table[i].tqualifiers,GLOBAL_MEM_STR)))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "\"padding\"};");

  fprintf(fp, "%s static const char* %s_array_output_names%s[] __attribute__((unused)) = {",place_attribute,convert_to_define_name(datatype_scalar),place_attribute);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_arr && str_vec_contains(symbol_table[i].tqualifiers,OUTPUT_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "\"padding\"};");

  fprintf(fp, "%s static const char* %s_comp_param_names%s[] __attribute__((unused)) = {",place_attribute,convert_to_define_name(datatype_scalar),place_attribute);
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == datatype_scalar && str_vec_contains(symbol_table[i].tqualifiers,RUN_CONST_STR))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "\"padding\"};");


}



#include "create_node.h"





static ASTNode*
create_primary_expression(const char* identifier)
{
	return astnode_create(NODE_PRIMARY_EXPRESSION,create_identifier_node(identifier),NULL);
}
ASTNode*
build_product_node(const node_vec elems)
{
	return build_list_node(elems,MULT_STR);
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
  if(base_type != MATRIX_STR && base_type != TENSOR_STR) return;
  const char* lhs = combine_all_new(node->lhs);
  //TP: stupid hack!!
  if(lhs && strstr(lhs,".data")) return;
  if(node->lhs->postfix == intern("]"))
       astnode_sprintf_postfix(node->lhs,"].data");
  else
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
	static string_vec cache_keys = VEC_INITIALIZER;
	static node_vec   cache[10000];

	int cache_index = str_vec_get_index(cache_keys,name);
	if(cache_index != -1)
	{
		return cache[cache_index];
	}

	node_vec res = VEC_INITIALIZER;
	get_const_array_var_dims_recursive(name,node,&res);

	push(&cache_keys,name);
	cache_index = str_vec_get_index(cache_keys,name);
	cache[cache_index] = res;

	return res;

}

const char*
get_internal_array_name(const Symbol* sym)
{
	const char* datatype_scalar = intern(remove_substring(strdup(sym->tspecifier),"*"));
    	if(str_vec_contains(sym->tqualifiers,DEAD_STR,RUN_CONST_STR))
    	{
	    return sprintf_intern("AC_INTERNAL_run_const_%s_array_here",datatype_scalar);
	}
        if(str_vec_contains(sym->tqualifiers,CONST_STR)) return sym->identifier;
	return
	    sprintf_intern(
		    "AC_INTERNAL_%s_%s_arrays_%s",
        	    str_vec_contains(sym->tqualifiers,DCONST_STR) ? "d" : "gmem",
        	    convert_to_define_name(datatype_scalar), sym->identifier);
}

ASTNode*
create_struct_tspec(const char* datatype)
{
	ASTNode* struct_type = astnode_create(NODE_UNKNOWN,NULL,NULL); 
	astnode_set_buffer(datatype,struct_type);
	struct_type->token = STRUCT_TYPE;
	return astnode_create(NODE_TSPEC,struct_type,NULL);
}

static int
n_occurances(const char* str, const char test)
{
	int res = 0;
	int i = -1;
	while(str[++i] != '\0') res += str[i] == test;
	return res;
}

const char*
get_array_elem_type(const char* arr_type_in)
{
	char* arr_type = strdup(arr_type_in);
	if(!strstr(arr_type,"AcArray")) return NULL;
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
void
preprocess_array_reads(ASTNode* node, const ASTNode* root, const char* datatype_scalar, const bool gen_mem_accesses)
{
  TRAVERSE_PREAMBLE_PARAMS(preprocess_array_reads,root,datatype_scalar,gen_mem_accesses);
  if(node->type != NODE_ARRAY_ACCESS)
	  return;
  if(!node->lhs) return;
  if(get_parent_node(NODE_VARIABLE,node)) return;
  const char* array_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
  const char* datatype = sprintf_intern("%s*",datatype_scalar);
  const bool is_global = check_symbol(NODE_VARIABLE_ID,intern(array_name),0,0);
  if(datatype_scalar == REAL3_STR && !is_global)
  {
	  const char* expr_type = get_expr_type(node->lhs);
	  if(expr_type == datatype_scalar)
	  {
		if(gen_mem_accesses)
		{
			//TP: on comment for now since causes seg fault for some reason
	    		//node = node->parent;
	    		//node->lhs = NULL;
	    		//node->rhs = NULL;
	    		//astnode_sprintf(node,"(AcReal){}");
			astnode_sprintf_prefix(node->lhs,"reinterpret_cast<AcReal*>(&");
			astnode_sprintf_postfix(node->lhs,")");
		}
		else
		{
			astnode_sprintf_prefix(node->lhs,"reinterpret_cast<AcReal*>(&");
			astnode_sprintf_postfix(node->lhs,")");
		}
	  }
  }
  if(datatype_scalar == REAL_STR && !is_global)
  {
          const char* expr_type = get_expr_type(node->lhs);
	  if(expr_type && strstr(expr_type,"AcArray"))
	  {
		expr_type = get_array_elem_type(expr_type);
	  }
          if(expr_type == datatype_scalar || expr_type == datatype)
          {
        	if(gen_mem_accesses)
        	{
			if(node->rhs)
			{
				//TP: on comment for now since causes seg fault for some reason
	    			//node = node->parent;
	    			//node->lhs = NULL;
	    			//node->rhs = NULL;
	    			//astnode_sprintf(node,"(AcReal){}");
			}
        	}
          }
  }
  const Symbol* sym = get_symbol(NODE_VARIABLE_ID,intern(array_name),intern(datatype));
  if(!sym)
       return;
  {
    
    if(get_parent_node(NODE_ARRAY_ACCESS,node)) return;
    node_vec var_dims =  get_array_var_dims(array_name, root);
    if(var_dims.size == 0) var_dims = get_const_array_var_dims(array_name, root);
    node_vec array_accesses = VEC_INITIALIZER;
    get_array_access_nodes(node,&array_accesses);
    const char* expr_type = get_expr_type(node);
    if(expr_type == REAL_STR && var_dims.size == 2 && var_dims.size == array_accesses.size+1 && !strcmp(combine_all_new(var_dims.data[1]),"3"))
    {
	    node_vec nodes = VEC_INITIALIZER;
	    const size_t n_initializer = 3;
	    for(size_t i = 0; i < n_initializer; ++i)
	    {
		ASTNode* indexing = astnode_create(NODE_UNKNOWN,NULL,NULL);
		astnode_sprintf(indexing,"%zu",i);
		ASTNode* new_node = astnode_create(NODE_ARRAY_ACCESS,astnode_dup(array_accesses.data[0]->parent,NULL),indexing);
		astnode_set_infix("[",new_node);
		astnode_set_postfix("]",new_node);
	    	push_node(&nodes,new_node);
	    }

	    ASTNode* expression_list = build_list_node(nodes,",");
	    ASTNode* initializer = astnode_create(NODE_STRUCT_INITIALIZER, expression_list,NULL); 
	    astnode_set_prefix("{",initializer);
	    astnode_set_postfix("}",initializer);
	    free_node_vec(&nodes);


	    ASTNode* tspec = create_struct_tspec(REAL3_STR);
	    ASTNode* res = astnode_create(NODE_UNKNOWN,tspec,initializer);
	    astnode_set_prefix("(",res);
	    astnode_set_infix(")",res);
	    res->token     = CAST;
	    res->lhs->type ^= NODE_TSPEC;
	    replace_node(node,res);
    }
  }
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
    {
          //TP: 1d reads don't need further posprocessing so allow them even for multidimensional arrays
          node_vec array_accesses = VEC_INITIALIZER;
          get_array_access_nodes(node,&array_accesses);
          if(array_accesses.size == 1)
          {
                  free_node_vec(&array_accesses);
                  return;
          }
	  fatal("Incorrect array access: %s,%ld\n",combine_all_new(node),var_dims.size);
    }
    ASTNode* base = node;
    base->lhs=NULL;
    base->rhs=NULL;
    base->prefix=NULL;
    base->postfix=NULL;
    base->infix=NULL;

    const bool check_access_bounds = datatype_scalar == REAL_STR && ACC_ARRAY_BOUND_CHECKS;

    ASTNode* identifier_node = create_primary_expression(get_internal_array_name(sym));
    identifier_node->parent = base;
    ASTNode* pointer_access = astnode_create(NODE_UNKNOWN,NULL,NULL);
    ASTNode* elem_access_offset = astnode_create(NODE_UNKNOWN,NULL,NULL);
    ASTNode* elem_access        = astnode_create(NODE_UNKNOWN,elem_access_offset,elem_index); 
    astnode_set_prefix("[",elem_access);
    astnode_set_postfix("]",elem_access);
    if(check_access_bounds) elem_access = NULL;
    ASTNode* access_node        = astnode_create(NODE_UNKNOWN,pointer_access,elem_access);
    if(check_access_bounds)
    {
    	base->lhs =  identifier_node;
	ASTNode* a = build_product_node(var_dims);
	ASTNode* b = astnode_create(NODE_UNKNOWN,astnode_dup(elem_index,NULL), astnode_dup(create_primary_expression(sym->identifier),NULL));
	astnode_sprintf_prefix(a,",");
	astnode_sprintf_prefix(b,",");
	astnode_sprintf_infix(b,",");
        if(str_vec_contains(sym->tqualifiers,DCONST_STR,CONST_STR))
	{
		 astnode_sprintf_prefix(b->rhs,"\"");
		 astnode_sprintf_infix(b->rhs,"\"");
	}
	ASTNode* c = astnode_create(NODE_UNKNOWN,a,b);
	base->rhs = astnode_create(NODE_UNKNOWN,astnode_dup(access_node,NULL),c);
    }
    else
    {
    	base->lhs =  identifier_node;
    	base->rhs = access_node;
    }
    access_node ->parent = base;

    
    if(str_vec_contains(sym->tqualifiers,DCONST_STR,CONST_STR))
    {
		if(check_access_bounds)
		{
			{
				astnode_sprintf_prefix(base,"safe_access(");
				astnode_sprintf_postfix(base,")");
			}
		}
    }
    else if(str_vec_contains(sym->tqualifiers,GLOBAL_MEM_STR))
    {
	
    	if(!str_vec_contains(sym->tqualifiers,DYNAMIC_STR) && !is_left_child(NODE_ASSIGNMENT,node))
	{
		if(check_access_bounds)
		{
			{
				astnode_sprintf_prefix(base,"safe_access(");
				astnode_sprintf_postfix(base,")");
			}
		}
		else
		{
			//TP: on cuda bool -> __nv_bool and there is no __ldg for __nv_bool for some reason
			{
				astnode_sprintf_prefix(base,"%s%s",
					 is_primitive_datatype(datatype_scalar) && datatype_scalar != BOOL_STR ? "__ldg(": "",
					 is_primitive_datatype(datatype_scalar) && datatype_scalar != BOOL_STR ? "&": ""
			       	);
			}
			if(is_primitive_datatype(datatype_scalar) && datatype_scalar != BOOL_STR)
				astnode_set_postfix(")",base);
		}

	}
    }
    else
    {
	    fprintf(stderr,"Fatal error: no case for array read: %s\n",combine_all_new(node));
	    exit(EXIT_FAILURE);
    }
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

bool 
is_boundary_param(const char* param)
{
   return (is_user_enum_option(param) && get_enum(param) == intern("AcBoundary"));
}

void
read_user_enums_recursive(const ASTNode* node,string_vec* user_enums, string_vec* user_enum_options, string_vec* user_enum_values)
{
	if(node->type == NODE_ENUM_DEF)
	{
		const int enum_index = push(user_enums,node->lhs->buffer);
		node_vec nodes = get_nodes_in_list(node->rhs);
		for(size_t i = 0; i < nodes.size; ++i)
		{
			const char* name = get_node_by_token(IDENTIFIER,nodes.data[i])->buffer;
			const ASTNode* assignment = get_node(NODE_ASSIGNMENT,nodes.data[i]);
			push(&user_enum_options[enum_index],name);
			const char* value = assignment ? intern(combine_all_new(assignment->rhs)) : NULL;
			push(&user_enum_values[enum_index],value);
		}
		free_node_vec(&nodes);
	}
	if(node->lhs)
		read_user_enums_recursive(node->lhs,user_enums,user_enum_options,user_enum_values);
	if(node->rhs)
		read_user_enums_recursive(node->rhs,user_enums,user_enum_options,user_enum_values);
}

user_enums_info
read_user_enums(const ASTNode* node)
{
        string_vec user_enum_options[100] = { [0 ... 100-1] = VEC_INITIALIZER};
        string_vec user_enum_values[100] = { [0 ... 100-1] = VEC_INITIALIZER};
	string_vec user_enums = VEC_INITIALIZER;
	read_user_enums_recursive(node,&user_enums,user_enum_options,user_enum_values);
	user_enums_info res;
	res.names = user_enums;
	memcpy(&res.options,&user_enum_options,sizeof(string_vec)*100);
	memcpy(&res.values,&user_enum_values,sizeof(string_vec)*100);
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
	if(!get_node(NODE_TSPEC,field))
	{
		fatal("Was not able to get the type of the struct member: %s\n",combine_all_new(field));
	}
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

bool
is_returning_function(const ASTNode* node, const char* func)
{
	bool res = false;
	if(node->lhs) 
	{
		res = is_returning_function(node->lhs,func);
	}
	if(node->rhs) 
	{
		res = is_returning_function(node->rhs,func);
	}
	if(!(node->type & NODE_FUNCTION)) return res;
	{

		const ASTNode* fn_identifier = get_node_by_token(IDENTIFIER,node);
		if(!fn_identifier || !fn_identifier->buffer) return res;
		if(fn_identifier->buffer != func) return res;
		return get_node_by_token(RETURN,node) != NULL;
	}
}

void
get_function_params_info_recursive(const ASTNode* node, const char* func_name, func_params_info* dst)
{
	//TP: speed optim to end recursion if the params are already found
	if(dst->types.size || node->type & (NODE_DEF | NODE_GLOBAL)) return;
	//TP: speed optim no need to traverse into the function itself
	if(!(node->type & NODE_FUNCTION))
	{
		if(node->lhs) get_function_params_info_recursive(node->lhs,func_name,dst);
		if(dst->types.size > 0) return;
		if(node->rhs) 
		{
			if(node->token != PROGRAM || node->rhs->type & NODE_FUNCTION)
				get_function_params_info_recursive(node->rhs,func_name,dst);
		}
		if(dst->types.size > 0) return;
	}
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
remove_suffix(char *str, const char* suffix_match) {
    char *optimizedPos = strstr(str, suffix_match);
    if (optimizedPos != NULL) {
        *optimizedPos = '\0'; // Replace suffix_match with null character
    }
}

static func_params_info kernel_params_info[MAX_FUNCS];

void
gen_kernel_params_info(const ASTNode* root)
{
  memset(kernel_params_info,0,sizeof(kernel_params_info));
  int kernel_index = 0;
  for(size_t sym = 0; sym< num_symbols[0]; ++sym)
  {
  	if(symbol_table[sym].tspecifier != KERNEL_STR) continue;
  	kernel_params_info[kernel_index] = get_function_params_info(root,symbol_table[sym].identifier);
  	kernel_index++;
  }
}

void
gen_kernel_structs(ASTNode* root)
{

  	gen_kernel_params_info(root);
	string_vec names = VEC_INITIALIZER;
  	for(size_t sym = 0; sym< num_symbols[0]; ++sym)
  	{
  		if(symbol_table[sym].tspecifier != KERNEL_STR) continue;
		push(&names,symbol_table[sym].identifier);
  	}
	string_vec unique_input_types[num_kernels];
	const func_params_info* infos = kernel_params_info;
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
				char* params_name = strdup(name);
	        		remove_suffix(params_name,"_optimized_");
				if(!str_vec_eq(infos[k].types,types)) continue;
				fprintf(fp,"if(kernel == %s){ \n",name);
				for(size_t j = 0; j < types.size; ++j)
				{
					fprintf(fp,"params.%s.%s = p_%ld;\n",
							params_name,info.expr.data[j],j
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
		{
			FILE* fp = fopen("user_input_typedefs.h","a");
			fprintf(fp,"\ntypedef struct %sInputParams {", name);
			for(size_t i = 0; i < info.types.size; ++i)
				fprintf(fp,"%s %s;",info.types.data[i],info.expr.data[i]);
			fprintf(fp,"} %sInputParams;\n",name);
			fclose(fp);
		}

		{
			//FILE* fp = fopen("safe_vtxbuf_input_params.h","a");
			//fprintf(fp,"if(kernel == %s){ \n",name);
			//for(size_t i = 0; i < info.types.size; ++i)
			//{
			//	const char* param_name = info.expr.data[i];
			//	const char* param_type = info.types.data[i];
			//	if(strstr(param_type,"*"))
			//	{
			//		if(param_type == REAL_PTR_STR)
			//		{
			//			fprintf(fp,"vba.on_device.kernel_input_params.%s.%s = vba.on_device.out[0];\n",name,param_name);
			//		}
			//	}
			//}
			//fprintf(fp,"}\n");
			//fclose(fp);
		}
	}
	{
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
	}
	{
		FILE* stream = fopen("user_input_typedefs.h","a");
		fprintf(stream,"\ntypedef union acKernelInputParams {\n\n");
		for(size_t k = 0; k < num_kernels; ++k)
		{
			const char* name = names.data[k];
			fprintf(stream,"%sInputParams %s;\n", name,name);
		}
		fprintf(stream,"} acKernelInputParams;\n\n");
		fclose(stream);
	}

	free_str_vec(&names);
}


static void
create_comp_op(const structs_info info, const int i, FILE* fp)
{
		const char* struct_name = info.user_structs.data[i];
		fprintf(fp,"static HOST_DEVICE_INLINE bool\n"
			   "operator==(const %s& a, const %s& b)\n"
			   "{\n"
			   "return\n"
			,struct_name,struct_name);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\ta.%s == b.%s %s\n",info.user_struct_field_names[i].data[j],info.user_struct_field_names[i].data[j], j < info.user_struct_field_names[i].size-1 ? "&&" : "");
		fprintf(fp,";}\n");

		fprintf(fp,"static HOST_DEVICE_INLINE bool\n"
			   "operator!=(const %s& a, const %s& b)\n"
			   "{\n"
			   "return\n"
			,struct_name,struct_name);
		for(size_t j = 0; j < info.user_struct_field_names[i].size; ++j)
			fprintf(fp,"\ta.%s != b.%s %s\n",info.user_struct_field_names[i].data[j],info.user_struct_field_names[i].data[j], j < info.user_struct_field_names[i].size-1 ? "||" : "");
		fprintf(fp,";}\n");
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
			fprintf(fp,"\t\t%sa.%s,\n",op,info.user_struct_field_names[i].data[j]);
		fprintf(fp,"\t};\n}\n");
}
static void
create_comp_ops(FILE* fp, const string_vec elems, const size_t index, const char* op)
{
	if(index == elems.size) return;
	if(index < elems.size-1) fprintf(fp,"__ac__%s(a.%s,",op,elems.data[index]);
	else fprintf(fp,"a.%s",elems.data[index]);
	create_comp_ops(fp,elems,index+1,op);
	if(index < elems.size-1) fprintf(fp,")");
}

static void
create_comp_operator(const structs_info info, const int i, const char* op, FILE* fp)
{
		if(info.user_struct_field_names[i].size < 2) return;
		const char* struct_name = info.user_structs.data[i];
		fprintf(fp,"static HOST_DEVICE_INLINE %s\n"
			   "%s(const %s& a)\n"
			   "{\n"
			,info.user_struct_field_types[i].data[0],op,struct_name);
		fprintf(fp,"return ");
		create_comp_ops(fp,info.user_struct_field_names[i],0,op);
		fprintf(fp,";\n}\n");
}

void
gen_user_enums()
{
  user_enums_info enum_info = e_info;
  for (size_t i = 0; i < enum_info.names.size; ++i)
  {
	  fprintf_filename("user_typedefs.h","%s {\n","typedef enum");
	  for(size_t j = 0; j < enum_info.options[i].size; ++j)
	  {
		  const char* separator = (j < enum_info.options[i].size - 1) ? ",\n" : "";
		  fprintf_filename("user_typedefs.h","%s",enum_info.options[i].data[j]);
		  if(enum_info.values[i].data[j])
		  {
		  	fprintf_filename("user_typedefs.h"," = %s",enum_info.values[i].data[j]);
		  }
		  fprintf_filename("user_typedefs.h","%s",separator);
	  }
	  fprintf_filename("user_typedefs.h","} %s;\n",enum_info.names.data[i]);

	  fprintf_filename("to_str_funcs.h","std::string to_str(const %s value)\n"
		       "{\n"
		       "switch(value)\n"
		       "{\n"
		       ,enum_info.names.data[i]);

	  for(size_t j = 0; j < enum_info.options[i].size; ++j)
		  fprintf_filename("to_str_funcs.h","case %s: return \"%s\";\n",enum_info.options[i].data[j],enum_info.options[i].data[j]);
	  fprintf_filename("to_str_funcs.h","}return \"\";\n}\n");

	  fprintf_filename("to_str_funcs.h","template <>\n std::string\n get_datatype<%s>() {return \"%s\";};\n",enum_info.names.data[i], enum_info.names.data[i]);
  }
}
void
gen_user_structs()
{
	FILE* fp = fopen("user_typedefs.h","a");
	FILE* fp_comp = fopen("generated_comp_funcs.h","w");
	for(size_t i = 0; i < s_info.user_structs.size; ++i)
	{
		const char* struct_name = s_info.user_structs.data[i];
		//TP: we use the struct coming from HIP/CUDA
		if(struct_name != INT3_STR)
		{
			fprintf(fp,"typedef struct %s {",struct_name);
			for(size_t j = 0; j < s_info.user_struct_field_names[i].size; ++j)
			{
				const char* type = s_info.user_struct_field_types[i].data[j];
				const char* name = s_info.user_struct_field_names[i].data[j];
				fprintf(fp, "%s %s;", type_output(type), name);
			}
			fprintf(fp, "} %s;\n", s_info.user_structs.data[i]);
		}

		bool all_reals = true;
		bool all_ints  = true;
		bool all_scalar_types = true;
		bool all_fields = true;
		for(size_t j = 0; j < s_info.user_struct_field_types[i].size; ++j)
		{
			all_reals        &=  s_info.user_struct_field_types[i].data[j] == REAL_STR;
			all_ints         &=  s_info.user_struct_field_types[i].data[j] == INT_STR;
			all_fields       &=  s_info.user_struct_field_types[i].data[j] == FIELD_STR;
			all_scalar_types &= s_info.user_struct_field_types[i].data[j] == REAL_STR || s_info.user_struct_field_types[i].data[j] == INT_STR;
		}

		if(all_fields)
		{
			fprintf(fp,"#ifdef __cplusplus\n");
			create_comp_op(s_info,i,fp);
			fprintf(fp,"#endif\n");
		}

		if(!all_scalar_types) continue;
		fprintf(fp,"#ifdef __cplusplus\n");



		create_comp_op(s_info,i,fp);
		if(struct_name != INT3_STR)
		{
			create_binary_op(s_info,i,MINUS_STR,fp);
			create_binary_op(s_info,i,PLUS_STR,fp);
			create_unary_op (s_info,i,MINUS_STR,fp);
			create_unary_op (s_info,i,PLUS_STR,fp);
		}
		if(all_reals || all_ints)
		{
			create_comp_operator(s_info,i,"max",fp_comp);
			create_comp_operator(s_info,i,"min",fp_comp);
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
	}
	fclose(fp);
	fclose(fp_comp);
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


bool
is_field_expr(const char* expr)
{
	return expr && (expr == COMPLEX_FIELD_STR || expr == FIELD_STR || expr == FIELD3_STR || !strcmp(expr,FIELD_PTR_STR) || !strcmp(expr,VTXBUF_PTR_STR) || !strcmp(expr,FIELD3_PTR_STR));
}
bool
is_value_applicable_type(const char* expr)
{
	return is_field_expr(expr) || is_subtype(PROFILE_STR,expr);
}

bool
is_arr_type(const char* var)
{
	const Symbol* sym = get_symbol(NODE_VARIABLE_ID, intern(var), NULL);
	if(!sym) return false;
	if(sym->tspecifier && strstr(sym->tspecifier,"*")) return true;
	return false;
}

bool
is_output_type(const char* var)
{
	const Symbol* sym = get_symbol(NODE_VARIABLE_ID, intern(var), NULL);
	if(!sym) return false;
	if(str_vec_contains(sym->tqualifiers,OUTPUT_STR)) return true;
	if(sym->tspecifier && strstr(sym->tspecifier,"OutputParam")) return true;
	return false;
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
			is_boundcond |= is_boundary_param(call_info.expr.data[i]);

		func_params_info params_info =  get_function_params_info(root,func_name);

		fprintf(stream,"[](ParamLoadingInfo p){\n");
		fprintf(stream,"#include \"user_constants.h\"\n");
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
			if (all_identifiers_are_constexpr(call_info.expr_nodes.data[i]) || !strcmp(input_param,"p.step_number"))
				fprintf(stream, "p.params -> %s.%s = %s;\n", func_name, params_info.expr.data[i], input_param);
			else if(is_value_applicable_type(call_info.types.data[i]))
				fprintf(stream, "p.params -> %s.%s = %s;\n", func_name, params_info.expr.data[i], input_param);
			else if(check_symbol(NODE_VARIABLE_ID,call_info.expr.data[i],NULL,DCONST_STR))
				fprintf(stream, "p.params -> %s.%s = acDeviceGetLocalConfig(acGridGetDevice())[%s]; \n", func_name,params_info.expr.data[i], input_param);
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


const char*
get_field_name(const int field)
{
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

void
remove_ending_symbols(char* str, const char symbol)
{
	int len = strlen(str);
	while(str[--len] == symbol) str[len] = '\0';
}
static string_vec ray_func_names = VEC_INITIALIZER;
void
gen_ray_names()
{
	free_str_vec(&ray_func_names);
	for(size_t i = 0; i < num_symbols[0]; ++i)
	{
		if(symbol_table[i].tspecifier == RAYTRACE_STR)
		{
			push(&ray_func_names,sprintf_intern("incoming_%s",symbol_table[i].identifier));
			push(&ray_func_names,sprintf_intern("outgoing_%s",symbol_table[i].identifier));
		}
	}
}
void
check_for_undeclared_functions(const ASTNode* node, const ASTNode* root)
{
	TRAVERSE_PREAMBLE_PARAMS(check_for_undeclared_functions,root);
	if(get_node(NODE_MEMBER_ID,node)) return;
	if(!(node->type & NODE_FUNCTION_CALL)) return;

	char* tmp = strdup(get_node_by_token(IDENTIFIER,node->lhs)->buffer);
        remove_suffix(tmp,"____");
        const char* func_name = intern(tmp);

	if(check_symbol(NODE_FUNCTION_ID,func_name,NULL,NULL)) return;

	const Symbol* sym = get_symbol(NODE_FUNCTION_ID, func_name, NULL);
	if(sym && str_vec_contains(sym->tqualifiers,intern("intrinsic"))) return;
	if(sym && sym->tspecifier == STENCIL_STR) return;

	if(str_vec_contains(duplicate_dfuncs.names,func_name)) 
	{
		fprintf(stderr,FATAL_ERROR_MESSAGE);
		fprintf(stderr,"Was not able to resolve overloaded function in call:\n%s\n",combine_all_new(node));
		{
			func_params_info info = get_func_call_params_info(node);
                	fprintf(stderr,"Types: ");
                	for(size_t i = 0; i < info.types.size; ++i)
                	  fprintf(stderr,"%s%s",info.types.data[i] ? info.types.data[i] : "unknown"
					  ,i < info.types.size-1 ? "," : "");
		}
                fprintf(stderr,"\n");
                fprintf(stderr,"\n");
		fprintf(stderr,"Possibilities:\n");
		for(size_t i = 0; i < num_symbols[0]; ++i)
		{
			if(!symbol_table[i].identifier || 
			   !strstr(symbol_table[i].identifier,func_name)) continue;

			func_params_info info = get_function_params_info(root, symbol_table[i].identifier);
			fprintf(stderr,"%s(",func_name);
			for(size_t type = 0; type < info.types.size; ++type)
			{
                  		fprintf(stderr,"%s%s",info.types.data[type] ? info.types.data[type] : "auto"
					  ,type < info.types.size-1 ? "," : "");
			}
			fprintf(stderr,")\n");
		}
                fprintf(stderr,"\n");
                fprintf(stderr,"\n");
		exit(EXIT_FAILURE);


	}
	if(str_vec_contains(ray_func_names,func_name)) return;
	fatal("Undeclared function %s in call: %s\n",func_name,combine_all_new(node));
}

void
write_dfunc_bc_kernel(const ASTNode* root, const char* prefix, const char* func_name,const func_params_info call_info,FILE* fp)
{

	//TP: in bc call params jump over boundary
	const int call_param_offset = 0;
	char* tmp = strdup(func_name);
	remove_suffix(tmp,"____");
	const char* dfunc_name = intern(tmp);
	free(tmp);
	func_params_info params_info = get_function_params_info(root,dfunc_name);
	if(call_info.expr.size != params_info.expr.size)
		fatal("Number of inputs %lu for %s in BoundConds does not match the number of input params %lu \n", call_info.expr.size, dfunc_name, params_info.expr.size);
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
	func_params_info call_info = get_func_call_params_info(func_call);

	if(func_name == PERIODIC)
		return;
	char* prefix;
	const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,func_name,0,0);

	if(is_dfunc) my_asprintf(&prefix,"%s_AC_KERNEL_",boundconds_name);
	else my_asprintf(&prefix,"%s_",boundconds_name);
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
			fprintf(stream,"%s,",info.expr.data[0]);
			free_func_params_info(&info);
		}
		fprintf(stream,"},");
		fclose(stream);
		fprintf_filename("user_loaders.h","{");
		for(size_t i = 0; i < calls.size; ++i)
		{
			fprintf_filename("user_loaders.h","[](ParamLoadingInfo ){},");
		}
		fprintf_filename("user_loaders.h","},");

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
	check_for_undeclared_functions(node,root);
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
	fprintf_filename_w("taskgraph_bc_handles.h","static const AcDSLTaskGraph DSLTaskGraphBCs[NUM_DSL_TASKGRAPHS+1] = { ");
	fprintf_filename_w("taskgraph_kernels.h","static const std::vector<AcKernel> DSLTaskGraphKernels[NUM_DSL_TASKGRAPHS+1] = { ");
	fprintf_filename_w("taskgraph_kernel_bcs.h","static const std::vector<AcBoundary> DSLTaskGraphKernelBoundaries[NUM_DSL_TASKGRAPHS+1] = { ");
	fprintf_filename_w("user_loaders.h","static const std::vector<std::function<void(ParamLoadingInfo step_info)>> DSLTaskGraphKernelLoaders[NUM_DSL_TASKGRAPHS+1] = { ");

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
	fprintf(fp,"\"out_of_bounds\"");
	fprintf(fp,"};\n");


	free_str_vec(&graph_names);
	free_str_vec(&bc_names);
	fclose(fp);


	//TP: pad by one to suppress compiler warnings in case of no compute steps
	fprintf_filename("taskgraph_bc_handles.h","{}");
	fprintf_filename("taskgraph_kernels.h"   ,"{}");
	fprintf_filename("taskgraph_kernel_bcs.h","{}");
	fprintf_filename("user_loaders.h","{}");

	fprintf_filename("taskgraph_bc_handles.h","};\n");
	fprintf_filename("taskgraph_kernels.h"   ,"};\n");
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

bool
is_enum_type(const char* type)
{
	return str_vec_contains(e_info.names,type);
}


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
	if(is_enum_type(var.type))
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
	   func_params_info info = get_function_params_info(node,kernel_name);
	   for(size_t i = 0; i < info.expr.size; ++i)
	   {
		   const char* type = info.types.data[i]; 
		   const char* name = info.expr.data[i];
	           add_param_combinations((variable){type,name},kernel_index,"",combinatorials);
	   }
	   free_func_params_info(&info);
	   //add_kernel_bool_dconst_to_combinations(node,kernel_index,combinatorials);
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

void
get_stencil_calling_info_in_func(const ASTNode* node, string_vec* src)
{
	TRAVERSE_PREAMBLE_PARAMS(get_stencil_calling_info_in_func,src);
	if(!(node->type & NODE_FUNCTION_CALL))
		return;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	if(str_vec_contains(*src,func_name)) return;
	Symbol* sym = (Symbol*)get_symbol(NODE_VARIABLE_ID | NODE_FUNCTION_ID ,func_name,NULL);
	if(sym && sym->tspecifier == STENCIL_STR) 
	{
		push(src,func_name);
	}
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
void
get_reduce_info(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(get_reduce_info);
	if(node->type & NODE_FUNCTION)
		get_reduce_info_in_func(node,&reduce_infos[str_vec_get_index(calling_info.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer)]);
}

void
get_stencil_calling_info(const ASTNode* node, string_vec* stencils_called)
{
	TRAVERSE_PREAMBLE_PARAMS(get_stencil_calling_info,stencils_called);
	if(node->type == NODE_TASKGRAPH_DEF) return;
	if(node->type & NODE_FUNCTION)
		get_stencil_calling_info_in_func(node,&stencils_called[str_vec_get_index(calling_info.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer)]);
}
void
gen_reduce_info(const ASTNode* root)
{
	get_reduce_info(root);
	bool updated = true;
	while(updated)
	{
		updated = false;
		for(size_t i = 0; i < calling_info.names.size; ++i)
			for(size_t j = 0; j  < calling_info.names.size; ++j)
			{
				if(!int_vec_contains(calling_info.called_funcs[i], j)) continue;
				for(size_t k = 0; k < reduce_infos[j].size; ++k)
				{
					push_node(&reduce_infos[i],reduce_infos[j].data[k]);
				}
			}
	}
}

string_vec*
gen_stencil_calling_info(const ASTNode* root)
{
	string_vec* stencils_called = (string_vec*)malloc(sizeof(string_vec)*calling_info.names.size);
	for(size_t i = 0; i < calling_info.names.size; ++i) memset(&stencils_called[i],0,sizeof(string_vec));
	get_stencil_calling_info(root,stencils_called);
	bool updated = true;
	while(updated)
	{
		updated = false;
		for(size_t i = 0; i < calling_info.names.size; ++i)
			for(size_t j = 0; j  < calling_info.names.size; ++j)
			{
				if(!int_vec_contains(calling_info.called_funcs[i], j)) continue;
				for(size_t k = 0; k < stencils_called[j].size; ++k)
				{
					if(!str_vec_contains(stencils_called[i],stencils_called[j].data[k]))
					{
						push(&stencils_called[i],stencils_called[j].data[k]);
						updated = true;
					}
				}
			}
	}
	return stencils_called;
}

bool
has_buffered_writes(const char* kernel_name)
{
	return strstr(kernel_name,"FUSED") != NULL;
}
bool
has_profile_reductions(const int kernel)
{
	if(!has_optimization_info()) return false;
	for(size_t profile_index = 0; profile_index < num_profiles; ++profile_index)
		if(reduced_profiles[profile_index + num_profiles*kernel]) return true;
	return false;
}
bool
has_stencil_ops(const int kernel)
{
	if(!has_optimization_info()) return true;
	for(size_t field_index = 0; field_index < num_fields; ++field_index)
		if(field_has_stencil_op[field_index + num_fields*kernel]) return true;
	return false;
}
bool
is_pure_reduce_kernel(const int kernel)
{
	return has_profile_reductions(kernel) && !has_stencil_ops(kernel);
}

bool
has_block_loops(const int kernel)
{
	return has_profile_reductions(kernel);
}

#include "warp_reduce.h"
static size_t
count_variables(const char* datatype, const char* qual)
{
	size_t res = 0;
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    		if((!datatype || symbol_table[i].tspecifier == datatype) && str_vec_contains(symbol_table[i].tqualifiers,qual) && (symbol_table[i].type & NODE_VARIABLE_ID))
      			++res;
	return res;
}
void
gen_final_reductions(const char* datatype, const int kernel_index, ASTNode* compound_statement, const int* reduced, const bool prefix)
{
		const size_t n_elems = count_variables(datatype,OUTPUT_STR);
		const char* define_name = convert_to_define_name(datatype);
		if(reduced == NULL) fatal("BUFFERED_REDUCTIONS requires OPTIMIZE_MEM_ACCESSES=ON!\n");
		for(size_t i = 0; i < n_elems; ++i)
		{
			if(!reduced[i + n_elems*kernel_index]) continue;
			const Symbol* sym = get_symbol_by_index_and_qualifier(NODE_VARIABLE_ID, i, datatype ,OUTPUT_STR);
			const ReduceOp op = (ReduceOp)reduced[i + n_elems*kernel_index];
			if(prefix)
			{
				astnode_sprintf_postfix(compound_statement,
						"{"
						"const auto val = warp_reduce_%s_%s(%s_reduce_output);"
					        "if(lane_id == warp_leader_id) d_symbol_reduce_scratchpads_%s[%s][warp_out_index] = val;"
						"}"
						"%s"
				,reduce_op_to_name(op)
				,define_name
				,sym->identifier
				,define_name
				,sym->identifier
				,compound_statement->postfix
				);
			}
			else
			{
				astnode_sprintf_postfix(compound_statement,"%s"
						"{"
						"const auto val = warp_reduce_%s_%s(%s_reduce_output);"
					        "if(lane_id == warp_leader_id) d_symbol_reduce_scratchpads_%s[%s][warp_out_index] = val;"
						"}"
				,compound_statement->postfix
				,reduce_op_to_name(op)
				,define_name
				,sym->identifier
				,define_name
				,sym->identifier
				);
			}
		}

}
bool
kernel_calls_reduce(const char* kernel)
{
      const int index = str_vec_get_index(calling_info.names,kernel);
      if(index == -1) fatal("Could not find %s in calling info\n",kernel);
      return reduce_infos[index].size != 0;
}

void
gen_kernel_reduce_outputs()
{
  //extra padding to help some compilers
  FILE* fp = fopen("kernel_reduce_info.h","w");
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
gen_all_final_reductions(ASTNode* compound_statement, const int kernel_index, const bool prefix)
{
	gen_final_reductions(REAL_STR,kernel_index,compound_statement,reduced_reals,prefix);
	gen_final_reductions(INT_STR,kernel_index,compound_statement,reduced_ints,prefix);
	if(AC_DOUBLE_PRECISION)
	{
		gen_final_reductions(FLOAT_STR,kernel_index,compound_statement,reduced_floats,prefix);
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
	  astnode_sprintf_postfix(compound_statement,"%s; (void)vba; (void)current_block_idx;} } } }",compound_statement->postfix);
	  return;
	}

	const char* fn_name = get_node(NODE_FUNCTION_ID,node)->buffer;
	const int kernel_index = get_symbol_index(NODE_FUNCTION_ID,fn_name,KERNEL_STR);
	if(has_optimization_info() && has_buffered_writes(fn_name))
	{
		for(size_t i = 0; i < num_fields; ++i)
		{
			if(!written_fields[i + num_fields*kernel_index]) continue;
	  		const char* name = get_symbol_by_index(NODE_VARIABLE_ID,i,FIELD_STR)->identifier;
			astnode_sprintf_postfix(compound_statement,"vba.out[%s][idx] = f%s_svalue_stencil;\n%s",name,name,compound_statement->postfix);
		}
	}
	if(kernel_calls_reduce(fn_name) && BUFFERED_REDUCTIONS && !has_block_loops(kernel_index))
	{
		gen_all_final_reductions(compound_statement,kernel_index,true);
	}
	astnode_sprintf_postfix(compound_statement,"%s} } }",compound_statement->postfix);
	if(has_block_loops(kernel_index)  && kernel_calls_reduce(fn_name))
	{
#if AC_USE_HIP
	       astnode_sprintf_postfix(compound_statement,"%s AC_INTERNAL_active_threads = __ballot(1);",compound_statement->postfix);
#else
	       astnode_sprintf_postfix(compound_statement,"%s AC_INTERNAL_active_threads = __ballot_sync(0xffffffff,1);",compound_statement->postfix);
#endif
    		astnode_sprintf_postfix(compound_statement,"%s AC_INTERNAL_active_threads_are_contiguos = !(AC_INTERNAL_active_threads & (AC_INTERNAL_active_threads+1));",compound_statement->postfix);
    		//TP: if all threads are active can skip checks checking if target tid is active in reductions
    		astnode_sprintf_postfix(compound_statement,"%s AC_INTERNAL_all_threads_active = AC_INTERNAL_active_threads+1 == 0;",compound_statement->postfix);

		gen_all_final_reductions(compound_statement,kernel_index,false);
	}
	astnode_sprintf_postfix(compound_statement,"%s"
			"}"
	,compound_statement->postfix);
}




void
init_populate_in_func(const ASTNode* node, const string_vec names, int_vec* src)
{
	TRAVERSE_PREAMBLE_PARAMS(init_populate_in_func,names,src);
	if(!(node->type & NODE_FUNCTION_CALL)) return;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	const int dfunc_index = str_vec_get_index(names,func_name);
	if(dfunc_index < 0) return;
	if(int_vec_contains(*src,dfunc_index))  return;
	push_int(src,dfunc_index);
}
void
init_populate_names(const ASTNode* node, funcs_calling_info* info, const NodeType func_type)
{
	if(node->type == NODE_TASKGRAPH_DEF) return;
	TRAVERSE_PREAMBLE_PARAMS(init_populate_names,info,func_type);
	if(!(node->type & func_type)) return;
	const char* name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	push(&info->names,name);
}

void
init_populate_calls(const ASTNode* node, funcs_calling_info* info, const NodeType func_type)
{
	if(node->type == NODE_TASKGRAPH_DEF) return;
	TRAVERSE_PREAMBLE_PARAMS(init_populate_calls,info,func_type);
	if(!(node->type & func_type)) return;
	const char* name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	const int index = str_vec_get_index(info->names,name);
	init_populate_in_func(node,info->names,&info->called_funcs[index]);
}



void
gen_calling_info(const ASTNode* root)
{
	memset(&calling_info,0,sizeof(calling_info));
	calling_info.called_funcs = malloc(sizeof(string_vec)*MAX_FUNCS);
	for(int i = 0; i < MAX_FUNCS; ++i) 
	{
		calling_info.called_funcs[i].data     = NULL;
		calling_info.called_funcs[i].size     = 0;
		calling_info.called_funcs[i].capacity = 0;
	}
	init_populate_names(root,&calling_info,NODE_DFUNCTION);
	init_populate_names(root,&calling_info,NODE_KFUNCTION);

	init_populate_calls(root,&calling_info,NODE_DFUNCTION);
	init_populate_calls(root,&calling_info,NODE_KFUNCTION);
	
        bool updated_calling_info = true;
        while(updated_calling_info)
        {
                updated_calling_info = false;
                for(size_t i = 0; i < calling_info.names.size; ++i)
                        for(size_t j = 0; j < calling_info.names.size; ++j)
                        {
                                if(!int_vec_contains(calling_info.called_funcs[i], j)) continue;
                                for(size_t k = 0; k < calling_info.called_funcs[j].size; ++k)
                                {
					const int child_index = calling_info.called_funcs[j].data[k];
                                        if(int_vec_contains(calling_info.called_funcs[i],child_index)) continue;
                                        push_int(&calling_info.called_funcs[i],child_index);
                                        updated_calling_info = true;
                                }
                        }
        }
	calling_info.topological_index = (int*)malloc(sizeof(int)*calling_info.names.size);
	memset(calling_info.topological_index,-1,sizeof(int)*calling_info.names.size);
	int topological_index = 0;	
	int last_index = -1;
	while(topological_index < (int)calling_info.names.size)
	{
		if(last_index == topological_index) fatal("Can't make progress in topological sort!\n");
		last_index = topological_index;
		for(size_t i = 0; i < calling_info.names.size; ++i)
		{
			if(calling_info.topological_index[i] != -1) continue;
			bool i_call_someone = false;
			for(size_t j = 0; j < calling_info.names.size; ++j)
			{
				if(i == j) continue;
				if(calling_info.topological_index[j] != -1) continue;
				i_call_someone |= int_vec_contains(calling_info.called_funcs[i],j);
			}
			if(!i_call_someone)
			{
				calling_info.topological_index[i] = topological_index;
				++topological_index;
			}
		}
	}
}

void
gen_kernel_postfixes(ASTNode* root, const bool gen_mem_accesses)
{
	gen_kernel_postfixes_recursive(root,gen_mem_accesses);
}
void
append_to_tail_node(ASTNode* tail_node, ASTNode* new_node)
{
	////printf("HMM: %d\n",tail_node->token == PROGRAM);
	while(tail_node->lhs) tail_node = tail_node->lhs;
	tail_node->lhs = new_node;
}

ASTNode*
get_tail_node(ASTNode* node)
{
	if(node->rhs) return get_tail_node(node->rhs);
	ASTNode* new_node = astnode_create(NODE_UNKNOWN,NULL,NULL);
	node->rhs = new_node;
	new_node->parent = node;
	return new_node;
}

void
gen_optimized_kernel_decls(ASTNode* node, const param_combinations combinations, const string_vec user_kernels_with_input_params,string_vec* const user_kernel_combinatorial_params, ASTNode* tail_node)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_optimized_kernel_decls,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params,tail_node);
	if(!(node->type & NODE_KFUNCTION))
		return;
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,get_node(NODE_FUNCTION_ID,node)->buffer);
	if(kernel_index == -1)
		return;
	string_vec combination_params = user_kernel_combinatorial_params[kernel_index];
	if(combination_params.size == 0)
		return;
	ASTNode* head = astnode_create(NODE_UNKNOWN,NULL,NULL);
	node_vec optimized_decls = VEC_INITIALIZER;
	for(int i = 0; i < combinations.nums[kernel_index]; ++i)
	{
		ASTNode* new_node = astnode_dup(node,NULL);
		make_ids_unique(new_node);
		ASTNode* function_id = (ASTNode*) get_node(NODE_FUNCTION_ID,new_node->lhs);
		astnode_sprintf(function_id,"%s_optimized_%d",get_node(NODE_FUNCTION_ID,node)->buffer,i);
		push_node(&optimized_decls,new_node);
	}
	ASTNode* declarations = build_list_node(optimized_decls,"");
	free_node_vec(&optimized_decls);
	head->rhs = declarations;
	append_to_tail_node(tail_node,head);
}
void
gen_kernel_ifs_base(ASTNode* node, const param_combinations combinations, const string_vec user_kernels_with_input_params,string_vec* const user_kernel_combinatorial_params)
{
	if(node->lhs)
		gen_kernel_ifs_base(node->lhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
	if(node->rhs)
		gen_kernel_ifs_base(node->rhs,combinations,user_kernels_with_input_params,user_kernel_combinatorial_params);
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
		{
			fprintf(fp, " && vba.on_device.kernel_input_params.%s.%s ==  %s ",get_node(NODE_FUNCTION_ID,node)->buffer,combination_params.data[j],combination_vals.data[j]);
		}

		fprintf(fp,
				")\n{\n"
				"\treturn %s_optimized_%d;\n}\n"
		,get_node(NODE_FUNCTION_ID,node)->buffer,i);
		fprintf(fp_defs,"%s_optimized_%d,",get_node(NODE_FUNCTION_ID,node)->buffer,i);
	}
	
	fclose(fp);
	fprintf(fp_defs,"}\n");
	fclose(fp_defs);
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
gen_kernel_ifs(ASTNode* root, const bool optimize_input_params)
{
	FILE* fp = fopen("user_kernels_ifs.h","w");
	fclose(fp);
	if(!optimize_input_params) return;
	//TP: should be in reset_all_files but does not work for some reason
        combinatorial_params_info info = get_combinatorial_params_info(root);
 	gen_kernel_ifs_base(root,info.params,info.kernels_with_input_params,info.kernel_combinatorial_params);
        free_combinatorial_params_info(&info);
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
	if(strstr(kernel_name,"MONOMORPHIZED")) return;
	const int combinations_index = get_suffix_int(kernel_name,"_optimized_");
	remove_suffix(kernel_name,"_optimized_");
	const int kernel_index = str_vec_get_index(user_kernels_with_input_params,intern(kernel_name));
	const char* type = get_expr_type(node);
	if(combinations_index == -1)
	{
		if(type && gen_mem_accesses && !strstr(type,"*"))
		{
			astnode_sprintf(node,"(%s){}",type);
			return;
		}

		if(type && strstr(type,"*") && gen_mem_accesses)
		{
			const char* datatype_scalar = intern(remove_substring(strdup(type),MULT_STR));
			astnode_sprintf(node,"AC_INTERNAL_run_const_%s_array_here",datatype_scalar);
			return;
		}
		{
			if(node->prefix != NULL) fatal("Need prefix to be NULL!\n");
			astnode_sprintf_prefix(node,"vba.kernel_input_params.%s.",kernel_name);
		}
		return;
	}
	if(kernel_index == -1)
	{
		if(node->prefix != NULL) fatal("Need prefix to be NULL!\n");
		astnode_sprintf(node,"vba.kernel_input_params.%s.",kernel_name);
		return;
	}
	const string_vec combinations = vals[kernel_index + MAX_KERNELS*combinations_index];
	const int param_index = str_vec_get_index(user_kernel_combinatorial_params[kernel_index],node->buffer);
	if(param_index < 0)
	{
		if(node->prefix != NULL) fatal("Need prefix to be NULL!\n");
		astnode_sprintf_prefix(node,"vba.kernel_input_params.%s.",kernel_name);
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
	  {
		const ASTNode* func = get_parent_node(NODE_FUNCTION,node);
		if(func)
		{
			fatal("Undeclared variable or function used on a range expression\n"
					"Range: %s\n"
					"Var: %s\n"
					"In function: %s\n"
					"\n"
					,combine_all_new(range_node),node->buffer
					,get_node_by_token(IDENTIFIER,func)->buffer);
		}
		else
		{
			fatal("Undeclared variable or function used on a range expression\n"
					"Range: %s\n"
					"Var: %s\n"
					"\n"
					,combine_all_new(range_node),node->buffer);
		}
	  }
}	
static void
check_for_undeclared_use_in_assignment(const ASTNode* node)
{
	
	  const bool used_in_assignment = is_right_child(NODE_ASSIGNMENT,node);
	  if(used_in_assignment)
	  {
		  const ASTNode* func = get_parent_node(NODE_FUNCTION,node);
		  const char* assignment = combine_all_new(get_parent_node(NODE_ASSIGNMENT,node));
		  if(func)
		  {
		  	fatal(
				"Undeclared variable or function used on the right hand side of an assignment\n"
				"Assignment: %s\n"
				"Var: %s\n"
				"In function: %s\n"
				"\n"
					  ,assignment
					  ,node->buffer
					  ,get_node_by_token(IDENTIFIER,func)->buffer
				);
		  }
		  else
		  {
		  	fatal(
				"Undeclared variable or function used on the right hand side of an assignment\n"
				"Assignment: %s\n"
				"Var: %s\n"
				"\n"
					  ,assignment
					  ,node->buffer
				);
		  }
	  }
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
	 check_for_undeclared_use_in_assignment(node);
	 check_for_undeclared_conditional(node);
	}

}
static void
translate_buffer_body(FILE* stream, const ASTNode* node, const bool to_DSL)
{
  if (stream && node->buffer) {
    const Symbol* symbol = symboltable_lookup(node->buffer);
    if (symbol && symbol->type & NODE_VARIABLE_ID && str_vec_contains(symbol->tqualifiers,DCONST_STR))
    {
      if(!strstr(symbol->tspecifier,"*") && !strstr(symbol->tspecifier,"AcArray"))
      {
	if(to_DSL) fprintf(stream, "%s", node->buffer);
	else       fprintf(stream, "DCONST(%s)", node->buffer);
      }
      else
      {
	if(to_DSL) fprintf(stream, "%s", node->buffer);
	else       fprintf(stream, "(%s)", node->buffer);
      }
    }
    else if (symbol && symbol->type & NODE_VARIABLE_ID && str_vec_contains(symbol->tqualifiers,RUN_CONST_STR))
    {
      if(to_DSL) fprintf(stream, "%s", node->buffer);
      else       fprintf(stream, "RCONST(%s)", node->buffer);
    }
    else if(symbol && symbol->tspecifier == KERNEL_STR)
    {
	   if(to_DSL) fprintf(stream,"%s",node->buffer);
	   else       fprintf(stream,"KERNEL_%s",node->buffer);
    }
    else
    {
      if(to_DSL) 
      {
	char* out = strdup(node->buffer);
	remove_suffix(out,"_AC_MANGLED");
      	fprintf(stream, "%s", out);
	free(out);
      }
      else
      {
      	fprintf(stream, "%s", node->buffer);
      }
    }
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

string_vec
get_array_elem_size(const char* arr_type_in)
{
	string_vec res = VEC_INITIALIZER;
	char* arr_type = strdup(arr_type_in);
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
		const bool two_dimensional = arr_type[end] == ',';
		arr_type[end] = '\0';
		char* tmp = malloc(sizeof(char)*1000);
		strcpy(tmp, &arr_type[start]);
		push(&res,intern(tmp));

		if(two_dimensional)
		{
			++end;
			start = end;
			while(arr_type[end] != ',' && arr_type[end] != '>' && arr_type[end] != ' ') ++end;
			arr_type[end] = '\0';
			strcpy(tmp, &arr_type[start]);
			push(&res,intern(tmp));
		}
		
		return res;
	}
	push(&res,intern(arr_type));
	return res;
}


const char*
output_specifier(FILE* stream, const tspecifier tspec, const ASTNode* node)
{
	const char* res = NULL;
	const char* tspecifier_out = NULL;
        if (tspec.id)
	{
	  tspecifier_out = tspec.id;
        }
        else if (add_auto(node))
	{
	  if(node->is_constexpr && !(node->type & NODE_FUNCTION_ID)) fprintf(stream, " constexpr ");
	  if(node->expr_type)
	  {
		tspecifier_out = node->expr_type;
	  }
	  else
          	fprintf(stream, "auto ");
	}
	if(tspecifier_out)
	{
	  const bool is_reference = tspecifier_out[strlen(tspecifier_out)-1] == '&';
          //TP: the pointer view is only internally used to mark arrays. for now simple lower to auto
	  if(tspecifier_out[strlen(tspecifier_out)-1] == '*' || tspecifier_out[strlen(tspecifier_out)-2] == '*')
            fprintf(stream, "%s%s ", "auto", is_reference ? "&" : "");
	  //TP: Hacks
	  else if(strstr(tspecifier_out,"WITH_INLINE"))
            fprintf(stream, "%s%s ", "auto", is_reference ? "&" : "");
	  else if(strstr(tspecifier_out,"AcArray"))
	  {
		  if(is_reference)
		  {
			  fprintf(stream, "auto& ");
		  }
		  else
		  {
		  	fprintf(stream, "%s ", get_array_elem_type(tspecifier_out));
		  	string_vec sizes = get_array_elem_size(tspecifier_out);
			if(sizes.size == 1)
				res = sprintf_intern("[%s]",sizes.data[0]);
			else if(sizes.size == 2)
				//TP: even though it is not intended rest of the dims come from traversing
				//TP: works for now but a bit hacky
				res = sprintf_intern("[%s]",sizes.data[0]);
			else
				fatal("Add missing dimensionality initialization!\n");
			free_str_vec(&sizes);
		  }
	  }
	  else if(tspecifier_out != KERNEL_STR)
	  {
            fprintf(stream, "%s ", type_output(tspecifier_out));
	  }
	}
	return res;
}

tspecifier
get_tspec(const ASTNode* decl)
{
      if(!decl) return (tspecifier){NULL, 0};
      const ASTNode* tspec_node = get_node(NODE_TSPEC, decl);
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
      const ASTNode* tqual_list_node = tqual_node->parent;
      while(tqual_list_node->parent->rhs && tqual_list_node->parent->rhs->type & NODE_TQUAL) tqual_list_node = tqual_list_node->parent;
      node_vec tquals = get_nodes_in_list(tqual_list_node);
      size_t n_tqualifiers = 0;
      for(size_t i = 0; i < tquals.size; ++i)
      {
              tqualifiers[n_tqualifiers] = tquals.data[i]->lhs->buffer;
              ++n_tqualifiers;
      }
      free_node_vec(&tquals);
      return n_tqualifiers;
}
static void
check_for_shadowing(const ASTNode* node)
{
    if(symboltable_lookup_surrounding_scope(node->buffer)  && is_right_child(NODE_DECLARATION,node) && get_node(NODE_TSPEC,get_parent_node(NODE_DECLARATION,node)->lhs))
    {
      // Do not allow shadowing.
      //
      // Note that if we want to allow shadowing, then the symbol table must
      // be searched in reverse order
      fprintf(stderr,
              "Error! Symbol '%s' already present in symbol table. Shadowing "
              "is not allowed.\n",
	       node->buffer
              );
      exit(EXIT_FAILURE);
      assert(0);
    }
}

bool
is_enum_option(const char* var)
{
	for(size_t i = 0; i < e_info.names.size; ++i)
		if (str_vec_contains(e_info.options[i],var)) return true;
	return false;
}

void
refresh_current_hashmap()
{
    hashmap_destroy(&symbol_table_hashmap[current_nest]);
    const unsigned initial_size = 2000;
    hashmap_create(initial_size, &symbol_table_hashmap[current_nest]);
}
const char* 
add_to_symbol_table(const ASTNode* node, const NodeType exclude, FILE* stream, bool do_checks, const ASTNode* decl, const char* postfix, const bool skip_global_dup_check, const bool skip_shadowing_check)
{
  const char* res = NULL;
  if (node->buffer && node->token == IDENTIFIER && !(node->type & exclude)) {
    if (do_checks && !skip_shadowing_check) check_for_shadowing(node);
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
	if(!is_enum_option(node->buffer))
        	add_symbol_base(node->type, tqualifiers, n_tqualifiers, tspec.id,  node->buffer, postfix);
        if(!tspec.id && do_checks) check_for_undeclared(node);
      }
    }
   else if(do_checks && !skip_global_dup_check && current_nest == 0)
   {
         if(node->type & NODE_FUNCTION_ID)
         {
                 if(symboltable_lookup(node->buffer)->tspecifier == STENCIL_STR)
                      fatal("Multiple declarations of %s\n",node->buffer);
                 else if(get_tspec(decl).id == STENCIL_STR)
                      fatal("Multiple declarations of %s\n",node->buffer);
         }
         else
              fatal("Multiple declarations of %s\n",node->buffer);

   }
  }
  return res;
}
//static bool
//is_left_child_of(const ASTNode* parent, const ASTNode* node_to_search)
//{
//	if(!parent) return false;
//	if(!parent->lhs) return false;
//	return get_node_by_id(node_to_search->id,parent->lhs) != NULL;
//}
void
rename_scoped_variables_base(ASTNode* node, const ASTNode* decl, const ASTNode* func_body, const bool lhs_of_func_body, const bool lhs_of_func_call, bool child_of_enum, bool skip_shadowing_check, bool skip_dup_check)
{
  FILE* stream = NULL;
  const bool do_checks = true;
  const NodeType exclude = 0;
  if(node->type == NODE_STRUCT_DEF) return;
  if(node->type & NODE_DECLARATION) decl = node;
  if(node->parent && node->parent->type & NODE_FUNCTION && node->parent->rhs->id == node->id) func_body = node;
  child_of_enum |= (node->type & NODE_ENUM_DEF);
  if(node->type & NODE_ARRAY_ACCESS) skip_shadowing_check = true;
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
  {  
    rename_scoped_variables_base(node->lhs, decl, func_body, lhs_of_func_body || func_body != NULL, lhs_of_func_call || node->type & NODE_FUNCTION_CALL, child_of_enum,skip_shadowing_check,skip_dup_check);
  }

  //TP: do not rename func params since it is not really needed and does not gel well together with kernel params
  //TP: also skip func calls since they should anyways always be in the symbol table except because of hacks
  //TP: skip also enums since they are anyways unique
  char* postfix = (lhs_of_func_call || child_of_enum) ? NULL : itoa(nest_ids[current_nest]);

  if(!lhs_of_func_call)
  	add_to_symbol_table(node,exclude,stream,do_checks,decl,postfix,true,skip_shadowing_check);
  free(postfix);
  if(node->buffer) 
  {
	  const Symbol* sym= symboltable_lookup_range(node->buffer,current_nest,1);
	  if(sym)
		  node->buffer = sym->identifier;
  }

  // Traverse RHS
  if (node->rhs)
  {
    if(node->type & NODE_ASSIGNMENT) skip_dup_check = true;
    rename_scoped_variables_base(node->rhs, decl, func_body,lhs_of_func_body,lhs_of_func_call,child_of_enum,skip_shadowing_check,skip_dup_check);
  }

  // Postfix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest > 0);
    --current_nest;
  }
}
void
rename_scoped_variables(ASTNode* node, const ASTNode* decl, const ASTNode* func_body)
{
	rename_scoped_variables_base(node,decl,func_body,false,false,false,false,false);
}
typedef struct
{
	bool do_checks;
	const ASTNode* decl;
	bool skip_global_dup_check;
	bool skip_shadowing_check;
	const ASTNode* func_call;
	bool do_not_add_to_symbol_table;
	bool to_DSL;
	NodeType return_on;
} traverse_base_params;
void
traverse_base(const ASTNode* node, const NodeType exclude, FILE* stream, traverse_base_params params)
{
  if(node->type == NODE_ENUM_DEF)   return;
  if(node->type == NODE_STRUCT_DEF) return;
  if(node->type & NODE_DECLARATION)   params.decl       = node;
  if(node->type & NODE_FUNCTION_CALL) params.func_call = node;
  if (node->type & exclude)
	  stream = NULL;
  if(params.return_on != NODE_UNKNOWN && (node->type == params.return_on))
	  return;
  // Do not translate tqualifiers or tspecifiers immediately
  if (node->parent &&
      (node->parent->type & NODE_TQUAL || (params.decl && node->parent->type & NODE_TSPEC)))
    return;

  params.skip_shadowing_check |= (node->type & NODE_ARRAY_ACCESS);
  // Prefix translation
  if (stream && node->prefix)
  {
	if(!params.to_DSL || !strstr(node->prefix,"vba.kernel_input_params"))
	{
		if(params.to_DSL && !strcmp(node->prefix,KERNEL_PREFIX))
		{
      			fprintf(stream, "\nKernel ");
		}
 		else if(params.to_DSL && strstr(node->prefix,"AC_INTERNAL_ARRAY_LOOP_INDEX"))
                {
                }

		else
		{
      			fprintf(stream, "%s", node->prefix);
		}
	}
  }

  // Prefix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest < MAX_NESTS);

    ++current_nest;
    num_symbols[current_nest] = num_symbols[current_nest - 1];
    refresh_current_hashmap();
  }

  // Traverse LHS
  if (node->lhs)
    traverse_base(node->lhs, exclude, stream, params);

  const char* size = !params.func_call && !params.do_not_add_to_symbol_table ?
		  add_to_symbol_table(node,exclude,stream,params.do_checks,params.decl,NULL,params.skip_global_dup_check,params.skip_shadowing_check)
		: NULL;
  // Infix translation
  if (stream && node->infix) 
    fprintf(stream, "%s", node->infix);
  translate_buffer_body(stream, node, params.to_DSL);
  if(size)
	  fprintf(stream, "%s ",size);

  // Traverse RHS
  if (node->rhs)
  {
    params.skip_global_dup_check |= (node->type & NODE_ASSIGNMENT);
    params.skip_global_dup_check |= (node->type & NODE_ARRAY_ACCESS);
    params.do_not_add_to_symbol_table |= (node->type & NODE_ASSIGNMENT);
    traverse_base(node->rhs, exclude, stream,params);
  }

  // Postfix logic
  if (node->type & NODE_BEGIN_SCOPE) {
    assert(current_nest > 0);
    --current_nest;
  }

  // Postfix translation
  if (stream && node->postfix) 
  {
    if(params.to_DSL && strstr(node->postfix,"AC_INTERNAL_ARRAY_LOOP_INDEX"))
    {
    }
    else if(params.to_DSL && node->prefix && strstr(node->prefix,"AC_INTERNAL_ARRAY_LOOP_INDEX"))
    {
    }
    else
    {
       fprintf(stream, "%s", node->postfix);
    }
  }
}
static inline void
traverse(const ASTNode* node, const NodeType exclude, FILE* stream)
{
	traverse_base(node,exclude,stream,(traverse_base_params){});
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
		const char* struct_type = get_expr_type(node->lhs);
		const char* field_name = get_node(NODE_MEMBER_ID,node->rhs)->buffer;
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
		{
			fatal("NO %s member in %s: %s\n",field_name,struct_type,combine_all_new(node));
			return NULL;
		}
		const char* res = info.user_struct_field_types[index].data[field_index];
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
//static bool
//node_is_function_param(const ASTNode* node)
//{
//	return (node->type & NODE_DECLARATION) && is_left_child(NODE_FUNCTION,node);
//}
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

//static bool
//node_is_struct_access_expr(const ASTNode* node)
//{
//	return node->type == NODE_STRUCT_EXPRESSION && node->rhs && get_node(NODE_MEMBER_ID, node->rhs);
//}

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
		(get_node_by_token(STRING,node)) ? CHAR_PTR_STR :
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
		strstr(base_type,"AcArray") ? get_array_elem_type(base_type) :
		base_type == FIELD_STR  ? REAL_STR :
		base_type == REAL3_STR ? REAL_STR :
		NULL;
}

const char*
get_struct_expr_type(const ASTNode* node)
{
	const char* base_type = get_expr_type(node->lhs);
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
	ASTNode* lhs_node = node->lhs;
	const char* lhs_res = get_expr_type(lhs_node);
	const char* rhs_res = get_expr_type(node->rhs);
	if(!lhs_res || !rhs_res) return NULL;
	const bool lhs_real = lhs_res == REAL_STR;
	const bool rhs_real = rhs_res == REAL_STR;
	const bool lhs_int   = lhs_res == INT_STR;
	const bool rhs_int   = rhs_res == INT_STR;
	const char* res = 	
		op && !strcmps(op,PLUS_STR,MINUS_STR,MULT_STR,DIV_STR) && (!strcmp(lhs_res,FIELD_STR) || !strcmp(rhs_res,FIELD_STR))   ? REAL_STR  :
		op && !strcmps(op,PLUS_STR,MINUS_STR,MULT_STR,DIV_STR) && (!strcmp(lhs_res,FIELD3_STR) || !strcmp(rhs_res,FIELD3_STR)) ? REAL3_STR :
		op && rhs_real && !strcmp(lhs_res,REAL3_STR) ? REAL3_STR :
                (lhs_real || rhs_real) && (lhs_int || rhs_int) ? REAL_STR :
                !strcmp_null_ok(op,MULT_STR) && !strcmp(lhs_res,MATRIX_STR) &&  !strcmp(rhs_res,REAL3_STR) ? REAL3_STR :
		!strcmp(lhs_res,COMPLEX_STR) || !strcmp(rhs_res,COMPLEX_STR)   ? COMPLEX_STR  :
		lhs_real && !strcmps(rhs_res,INT_STR,LONG_STR,LONG_LONG_STR,DOUBLE_STR,FLOAT_STR)    ?  REAL_STR  :
		op && !strcmps(op,MULT_STR,DIV_STR,PLUS_STR,MINUS_STR)     && lhs_real && !rhs_int  ?  rhs_res   :
		op && !strcmps(op,MULT_STR,DIV_STR,PLUS_STR,MINUS_STR)  && rhs_real && !lhs_int  ?  lhs_res   :
		!strcmp(lhs_res,rhs_res) ? lhs_res :
		//TP: we lose size information but it's not that crucial for now
		strstr(lhs_res,"AcArray") && strstr(rhs_res,"*") ? rhs_res :
		strstr(rhs_res,"AcArray") && strstr(lhs_res,"*") ? lhs_res:
		strstr(lhs_res,"AcArray") && strstr(rhs_res,"AcArray") ? rhs_res :
		NULL;

	//TP: prints for debugging
	//if(strstr(combine_all_new(node),"R-L"))
	//{
	//	printf("HI: %s\n",combine_all_new_with_whitespace(node));
	//	printf("LHS: %s %s\n",lhs_res,combine_all_new(node->lhs));
	//	printf("RHS: %s %s\n",rhs_res,combine_all_new(node->rhs));
	//	printf("RES: %s\n",res);
	//}
	return res;
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
			if(decls.size != types.size) fatal("Cannot destructure %ld elements to %ld elements in %s\n",types.size,decls.size,combine_all_new(node));
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

bool
know_all_types(const string_vec types)
{
	for(size_t i = 0; i < types.size; ++i)
		if(types.data[i] == NULL) return false;
	return true;
}
const char*
get_func_call_expr_type(ASTNode* node)
{
	//TP: hack for Matrix member function calls
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
			func_params_info call_info = get_func_call_params_info(node);
			//TP: this does not scale: TODO think about how to generalize
			if((func_name == intern("min") || func_name == intern("max")) && know_all_types(call_info.types))
			{
				if(call_info.types.data[0] == FIELD_STR)
				{
					node->expr_type = REAL_STR;
				}
				else
				{
					node->expr_type = call_info.types.data[0];
				}
			}
			else if(func)
			{
				if(know_all_types(call_info.types))
				{
					if(!str_vec_contains(duplicate_dfuncs.names,func_name))
					{
						func_params_info info = get_function_params_info(func,func_name);
						if(call_info.types.size == info.expr.size)
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
				}
			}
			free_func_params_info(&call_info);
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
		if(res) astnode_sprintf_prefix(node->parent,"(%s)",res);
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
		strstr(base_type,"AcArray") ? get_array_elem_type(base_type) :
		NULL;
	if(!node->lhs->expr_type)
		node->lhs->expr_type = res;
	return res;
}
const char*
get_identifier_expr_type(ASTNode* node)
{
  	const Symbol* sym = get_symbol_token(NODE_VARIABLE_ID,node->buffer,NULL);
	if(sym)
		node->expr_type = sym->tspecifier;
	return node->expr_type;
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
	else if(node->token == IDENTIFIER)
		res = get_identifier_expr_type(node);
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
gen_profile_reads(ASTNode* node, const bool gen_mem_accesses)
{
	if(!(node->type & NODE_ARRAY_ACCESS))
	{
		TRAVERSE_PREAMBLE_PARAMS(gen_profile_reads,gen_mem_accesses)
	}
	if(!(node->type & NODE_ARRAY_ACCESS)) return;
	const char* type = get_expr_type(node->lhs);
	if(!type)
	{
		ASTNode* id = get_node_by_token(IDENTIFIER, node);
		if(id)
		{
			type = get_expr_type(id);
		}
	}
	if(!type || !strstr(type,"Profile"))
		return;
	node_vec nodes = VEC_INITIALIZER;
	get_array_access_nodes(node,&nodes);
	if(!strcmps(type,"Profile<X>","Profile<Y>","Profile<Z>") && nodes.size != 1)	
	{
		fatal("Fatal error: only 1-dimensional reads are allowed for 1d Profiles: %s\n",combine_all_new(node));
	}
	if(!strcmps(type,"Profile<XY>","Profile<YX>",
			"Profile<YZ>","Profile<ZY>",
			"Profile<XZ>","Profile<ZX>"
			) 
			&& nodes.size != 2)	
	{
		fatal("Fatal error: only 2-dimensional reads are allowed for 2d Profiles: %s\n",combine_all_new(node));
	}
	if(is_left_child(NODE_ASSIGNMENT,node))
	{
		fatal("Explicit writes to Profiles not allowed!\n");
	}

	ASTNode* idx_node = astnode_create(NODE_UNKNOWN,NULL,NULL);
	if(!gen_mem_accesses)
	{
		if(!strcmps(type,"Profile<XY>","Profile<XZ>"))
		{
			astnode_set_prefix("PROFILE_X_Y_OR_Z_INDEX(",idx_node);
		}
		else if(!strcmps(type,"Profile<YX>","Profile<YZ>"))
		{
			astnode_set_prefix("PROFILE_Y_X_OR_Z_INDEX(",idx_node);
		}
		else if(!strcmps(type,"Profile<ZX>","Profile<ZY>"))
		{
			astnode_set_prefix("PROFILE_Z_X_OR_Y_INDEX(",idx_node);
		}
		else
			astnode_set_prefix("(",idx_node);
		astnode_set_postfix(")",idx_node);
	}
	ASTNode* rhs = astnode_create(NODE_UNKNOWN, idx_node, NULL);
	ASTNode* indexes = build_list_node(nodes,",");
	idx_node->lhs = indexes;
	indexes->parent = idx_node;

	free_node_vec(&nodes);
	ASTNode* before_lhs = NULL;
	ASTNode* lhs = astnode_create(NODE_UNKNOWN, before_lhs, astnode_dup(node->lhs,NULL));
	astnode_free(node);

        node->rhs = rhs;
	node->lhs = lhs;
	if(gen_mem_accesses && !is_left_child(NODE_ASSIGNMENT,node))
	{
		astnode_set_postfix(")",rhs);

		astnode_set_infix("AC_INTERNAL_read_profile(",lhs);
		astnode_set_postfix(",",lhs);
		ASTNode* lhs_arr_access = (ASTNode*) get_node(NODE_ARRAY_ACCESS,lhs);
		if(lhs_arr_access)
		{
			lhs_arr_access->rhs = NULL;
			lhs_arr_access->prefix  = NULL;
			lhs_arr_access->postfix = NULL;
			lhs_arr_access->infix   = NULL;
		}
	}
	else
	{
		astnode_set_prefix("[",rhs);
		astnode_set_postfix("]",rhs);

		astnode_set_infix("vba.profiles.in[",lhs);
		astnode_set_postfix("]",lhs);

		ASTNode* lhs_arr_access = (ASTNode*) get_node(NODE_ARRAY_ACCESS,lhs);
		if(lhs_arr_access)
		{
			lhs_arr_access->rhs = NULL;
			lhs_arr_access->prefix  = NULL;
			lhs_arr_access->postfix = NULL;
			lhs_arr_access->infix   = NULL;
		}
	}
	lhs->parent = node;
}

void
turn_assignment_to_comma(ASTNode* node)
{
	ASTNode* assignment = (ASTNode*)get_parent_node(NODE_ASSIGNMENT,node);
	astnode_set_buffer(",",assignment->rhs->lhs);
}

void
gen_multidimensional_field_accesses_recursive(ASTNode* node, const bool gen_mem_accesses, const string_vec field_dims)
{
	if(!(node->type & NODE_STRUCT_EXPRESSION) && node->lhs)
	{
		gen_multidimensional_field_accesses_recursive(node->lhs,gen_mem_accesses,field_dims);
	}
	if(node->rhs)
	{
		gen_multidimensional_field_accesses_recursive(node->rhs,gen_mem_accesses,field_dims);
	}
	
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
	if(!type || strcmps(type,FIELD_STR,FIELD3_STR,FIELD4_STR,"VertexBufferHandle"))
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

	}

	free_node_vec(&nodes);
	astnode_free(array_access);
        array_access->rhs = rhs;
	ASTNode* lhs = astnode_create(NODE_UNKNOWN, before_lhs, astnode_dup(node,NULL));
	array_access->lhs = lhs;
	{

		if(is_left_child(NODE_ASSIGNMENT,node))
		{
			const char* func =  		
						type == FIELD4_STR ? "AC_INTERNAL_write_vtxbuf4(" :
						type == FIELD3_STR ? "AC_INTERNAL_write_vtxbuf3(" : 
						"AC_INTERNAL_write_vtxbuf(";
			astnode_set_infix(func,lhs);
			ASTNode* assignment = (ASTNode*)get_parent_node(NODE_ASSIGNMENT,node);
			//TP: has to be the most rhs since code elimination leaves a trailing ) otherwise
			ASTNode* most_rhs = assignment->rhs;
                        while(most_rhs->rhs) most_rhs = most_rhs->rhs;
                        astnode_set_postfix(")",most_rhs);
			turn_assignment_to_comma(node);
		}
		else
		{
			const char* func =
						type == FIELD4_STR ? "AC_INTERNAL_read_vtxbuf4(" :
						type == FIELD3_STR ? "AC_INTERNAL_read_vtxbuf3(" :
						"AC_INTERNAL_read_vtxbuf(";
			astnode_set_infix(func,lhs);
			astnode_set_postfix(")",rhs);
		}
		astnode_set_postfix(",",lhs);
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
print_const_array(FILE* fp, const char* datatype_scalar, const char* name, const int num_of_elems, const char* assignment_val, const char* static_str)
{
	fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] %s const constexpr %s %s[%d] = %s;\n#endif\n",static_str,datatype_scalar, name, num_of_elems, assignment_val);
}

void
gen_const_def(const ASTNode* def, const ASTNode* tspec, FILE* fp, FILE* fp_builtin, FILE* fp_non_scalar_constants, FILE* fp_non_scalar_builtin, const bool static_definitions)
{
		const char* static_str = static_definitions ? "static" : "";
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
		const char* assignment_val = intern(combine_all_new_with_whitespace(assignment));
		const int array_dim = array_initializer ? count_nest(array_initializer,NODE_ARRAY_INITIALIZER) : 0;
		if(array_initializer)
		{
			const int num_of_elems = array_initializer ? count_num_of_nodes_in_list(array_initializer->lhs) : 0;
			const ASTNode* second_array_initializer = get_node(NODE_ARRAY_INITIALIZER, array_initializer->lhs);
			if(array_dim == 1)
			{
				if(is_builtin_constant(name))
					print_const_array(fp_non_scalar_builtin,datatype_scalar,name,num_of_elems,assignment_val,static_str);
				else
				{
					print_const_array(fp,datatype_scalar,name,num_of_elems,assignment_val,static_str);
					print_const_array(fp_non_scalar_constants,datatype_scalar,name,num_of_elems,assignment_val,static_str);
				}
			}
			else if(array_dim == 2)
			{
				const int num_of_elems_in_list = count_num_of_nodes_in_list(second_array_initializer->lhs);
				if(is_builtin_constant(name))
					print_const_array(fp_non_scalar_builtin,datatype_scalar,name,num_of_elems_in_list*num_of_elems,assignment_val,static_str);
				else
				{
					print_const_array(fp,datatype_scalar,name,num_of_elems_in_list*num_of_elems,assignment_val,static_str);
					print_const_array(fp_non_scalar_constants,datatype_scalar,name,num_of_elems_in_list*num_of_elems,assignment_val,static_str);
				}

			}
			else
				fatal("todo add 3d const arrays\n");
		}
		else if(array_access)
		{
			const char* num_of_elems = combine_all_new(array_access->rhs);
				fprintf(fp_non_scalar_constants, "\n#ifdef __cplusplus\n[[maybe_unused]] %s const %s %s[%s] = %s;\n#endif\n",static_str,datatype_scalar, name, num_of_elems, assignment_val);
				fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] %s const %s %s[%s] = %s;\n#endif\n",static_str,datatype_scalar, name, num_of_elems, assignment_val);
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
                                	fprintf(fp, "[[maybe_unused]] %s const constexpr %s %s = %s;\n",static_str,datatype_scalar, name, assignment_val);
                                	fprintf(fp_non_scalar_constants, "[[maybe_unused]] %s const constexpr %s %s = %s;\n",static_str,datatype_scalar, name, assignment_val);
				}
			}
                        else
			{
                               fprintf(
						is_builtin_constant(name) ? fp_non_scalar_builtin : fp_non_scalar_constants,  
						"\n#ifdef __cplusplus\n #define %s (%s)\n#endif\n", name, assignment_val
				);
			        if(!is_builtin_constant(name))
                               		fprintf(fp, "\n#ifdef __cplusplus\n[[maybe_unused]] %s const constexpr %s %s = %s;\n#endif\n",static_str,datatype_scalar, name, assignment_val);
			}
		}
}
void
gen_var_loader(const ASTNode* assignment, FILE* fp)
{
	const char* id = get_node_by_token(IDENTIFIER, assignment->lhs)->buffer;
        if(!check_symbol(NODE_VARIABLE_ID,id,0,DCONST_STR)  && !check_symbol(NODE_VARIABLE_ID,id,0,RUN_CONST_STR)) return;
	size_t size = 0;
	char* rhs = NULL;
	FILE* stream = open_memstream(&rhs,&size);

  	traverse_base_params params;
  	memset(&params,0,sizeof(params));
  	params.do_checks = true;
	params.skip_global_dup_check = true;
	params.skip_shadowing_check = true;
  	traverse_base(assignment->rhs, 0, stream, params);
	fclose(stream);
	fprintf(fp,"push_val(%s,%s);\n",id,rhs);
	free(rhs);
}
void
gen_config_loaders(const ASTNode* node, FILE* fp)
{
	if(node->type & NODE_FUNCTION) return;
	TRAVERSE_PREAMBLE_PARAMS(gen_config_loaders,fp);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	node_vec assignments = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < assignments.size; ++i)
	{
		const ASTNode* assignment = get_node(NODE_ASSIGNMENT,assignments.data[i]);
		if(assignment) gen_var_loader(assignment,fp);
	}
	free_node_vec(&assignments);
}

void
gen_const_variables(const ASTNode* node, FILE* fp, FILE* fp_bi,FILE* fp_non_scalars,FILE* fp_bi_non_scalars, const bool static_definitions)
{
	TRAVERSE_PREAMBLE_PARAMS(gen_const_variables,fp,fp_bi,fp_non_scalars,fp_bi_non_scalars,static_definitions);
	if(!(node->type & NODE_ASSIGN_LIST)) return;
	if(!has_qualifier(node,"const")) return;
	const ASTNode* tspec = get_node(NODE_TSPEC,node);
	if(!tspec) return;
	node_vec assignments = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < assignments.size; ++i)
		gen_const_def(assignments.data[i],tspec,fp,fp_bi,fp_non_scalars,fp_bi_non_scalars,static_definitions);
	free_node_vec(&assignments);
}


int_vec
dfuncs_in_topological_order(void)
{
	int_vec res = VEC_INITIALIZER;
	for(size_t i = 0; i< calling_info.names.size; ++i)
	{
		for(size_t j = 0; j < calling_info.names.size; ++j)
		{
			const bool is_dfunc = check_symbol(NODE_DFUNCTION_ID,calling_info.names.data[j],NULL,NULL);
			if(!is_dfunc) continue;
			if(calling_info.topological_index[j] == (int)i) 
			{
				push_int(&res, get_symbol_index(NODE_DFUNCTION_ID,calling_info.names.data[j],0));
			}
		}
	}
	return res;
}

static void
write_calling_info(FILE* fp, const char* func, const char* arr_name)
{
    for(size_t index = 0; index < num_dfuncs; ++index)
    {
	    const Symbol* dfunc_symbol = get_symbol_by_index(NODE_DFUNCTION_ID,index,0);
	    if(!dfunc_symbol) continue;
	    const char* dfunc_name = dfunc_symbol->identifier;
	    if(dfunc_name == func)
	    {
		    fprintf(fp,"const bool %s[] = {",arr_name);
		    for(size_t kernel = 0; kernel < num_kernels; ++kernel)
		    {
			    	const char* name = get_symbol_by_index(NODE_FUNCTION_ID,kernel,KERNEL_STR)->identifier;
    				const int_vec called_dfuncs = calling_info.called_funcs[str_vec_get_index(calling_info.names,name)];
	    			const int call_index = str_vec_get_index(calling_info.names,dfunc_name);
		            	fprintf(fp,"%d,",int_vec_contains(called_dfuncs,call_index));
		    }
		    fprintf(fp,"};\n");
	    }
    }
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
count_profiles()
{

  string_vec prof_types = get_prof_types();
  size_t res = 0;
  for(size_t i = 0; i < prof_types.size; ++i)
	  res += count_symbols(prof_types.data[i]);
  return res;
}

static void
write_calling_info_for_stencilgen(const string_vec* stencils_called)
{
    FILE* fp = fopen("stencilgen_calling_info.h","w");
    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_X_"),"reduce_sum_real_x_called");
    write_calling_info(fp,intern("value_AC_MANGLED_NAME__ComplexField"),"value_complex_called_statically");
    write_calling_info(fp,intern("write_AC_MANGLED_NAME__ComplexField_AcComplex"),"write_complex_called_statically");
    write_calling_info(fp,intern("write_AC_MANGLED_NAME__Field_AcReal"),"write_base_called_statically");
    write_calling_info(fp,intern("previous_AC_MANGLED_NAME__Field"),"previous_called_statically");
    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_Y_"),"reduce_sum_real_y_called");
    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_Z_"),"reduce_sum_real_z_called");

    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_XY_"),"reduce_sum_real_xy_called");
    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_XZ_"),"reduce_sum_real_xz_called");

    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_YX_"),"reduce_sum_real_yx_called");
    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_YZ_"),"reduce_sum_real_yz_called");

    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_ZX_"),"reduce_sum_real_zx_called");
    write_calling_info(fp,intern("reduce_sum_AC_MANGLED_NAME__AcReal_Profile_ZY_"),"reduce_sum_real_zy_called");

    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_X_"),"value_profile_x_called");
    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_Y_"),"value_profile_y_called");
    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_Z_"),"value_profile_z_called");

    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_XY_"),"value_profile_xy_called");
    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_XZ_"),"value_profile_xz_called");

    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_YX_"),"value_profile_yx_called");
    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_YZ_"),"value_profile_yz_called");

    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_ZX_"),"value_profile_zx_called");
    write_calling_info(fp,intern("value_AC_MANGLED_NAME__Profile_ZY_"),"value_profile_zy_called");
    fprintf(fp,"const bool stencils_called[NUM_KERNELS][NUM_STENCILS] = {");
    for(size_t k = 0; k < num_kernels; ++k)
    {
	    fprintf(fp,"{");
	    const char* name = get_symbol_by_index(NODE_FUNCTION_ID,k,KERNEL_STR)->identifier;
	    const int call_index = str_vec_get_index(calling_info.names,name);
  	    const size_t num_stencils = count_symbols(STENCIL_STR);
	    for(size_t s = 0; s < num_stencils; ++s)
	    {
			const Symbol* sym = get_symbol_by_index(NODE_FUNCTION_ID, s, STENCIL_STR);
			fprintf(fp,"%d,",str_vec_contains(stencils_called[call_index],sym->identifier));
	    }
	    fprintf(fp,"},");
    }
    fprintf(fp,"};\n");
    fclose(fp);
}
static void
gen_kernels_recursive(const ASTNode* node, char** dfunctions,
            const bool gen_mem_accesses)
{
  static int curr_kernel = 0;
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
    int_vec topological_order = dfuncs_in_topological_order();
    for(size_t index = 0; index < num_dfuncs; ++index)
    {
	    const int i = topological_order.data[index];
	    const Symbol* dfunc_symbol = get_symbol_by_index(NODE_DFUNCTION_ID,i,0);
	    if(!dfunc_symbol) continue;
	    if(str_vec_contains(dfunc_symbol->tqualifiers,INLINE_STR)) continue;
	    const int call_index = str_vec_get_index(calling_info.names,dfunc_symbol->identifier);
	    if(int_vec_contains(called_dfuncs,call_index)) 
	    {
		    strcat(prefix,dfunctions[i]);
	    }
    }
    free_int_vec(&topological_order);

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
get_field_qualifier_recursive(const ASTNode* node,string_vec* dst, const char* qualifier)
{
	TRAVERSE_PREAMBLE_PARAMS(get_field_qualifier_recursive,dst,qualifier);
        if(node->type  != (NODE_DECLARATION | NODE_GLOBAL)) return;
        const ASTNode* type_node = get_node(NODE_TSPEC,node);
        if(!type_node) return;
        const char* type = intern(combine_all_new(type_node));
	if(type != FIELD_STR) return;
	const int n_fields = count_num_of_nodes_in_list(node->rhs);
        const ASTNode* tquals = node->lhs->rhs ? node->lhs->lhs : NULL;
	bool found = false;
	if(tquals)
	{
		node_vec tquals_vec = get_nodes_in_list(tquals);
		for(size_t i = 0; i < tquals_vec.size; ++i)
		{
			const ASTNode* tqual = get_node(NODE_TQUAL,tquals_vec.data[i]);
			if(tqual->lhs && tqual->lhs->buffer && tqual->lhs->buffer == qualifier)
			{
				for(int j = 0; j < n_fields; ++j)
					push(dst,intern(combine_all_new(tqual->rhs)));
				found = true;
			}
		}
		free_node_vec(&tquals_vec);
	}
	for(int j = 0; j < n_fields; ++j)
	{
		if(!found && qualifier == DIMS_STR) push(dst,intern("AC_mlocal"));
		if(!found && qualifier == HALO_STR) push(dst,intern("AC_nmin"));
		if(!found && qualifier == FIELD_ORDER_STR) push(dst,sprintf_intern("%d",-1));
	}
}

static  string_vec
get_field_dims(const ASTNode* node)
{
	string_vec res = VEC_INITIALIZER;
	get_field_qualifier_recursive(node,&res,DIMS_STR);
	return res;
}

static  string_vec
get_field_halos(const ASTNode* node)
{
	string_vec res = VEC_INITIALIZER;
	get_field_qualifier_recursive(node,&res,HALO_STR);
	return res;
}
static void
get_field_order(const ASTNode* node)
{
	string_vec tmp = VEC_INITIALIZER;
	int_vec tmp2 = VEC_INITIALIZER;
	get_field_qualifier_recursive(node,&tmp,FIELD_ORDER_STR);
	for(size_t i = 0; i < tmp.size; ++i)
		push_int(&tmp2,atoi(tmp.data[i]));
        free_int_vec(&user_remappings);
	for(size_t i = 0; i < tmp.size; ++i)
	{
		int next = -1;
		for(size_t j = 0; j < tmp2.size; ++j)
		{
			if(tmp2.data[j] == (int)i)
			{
				next = (int)j;
			}
		}
		if(next < 0)
		{
			for(size_t j = 0; j < tmp2.size; ++j)
			{
				if(tmp2.data[j] == -1 && next == -1 && !int_vec_contains(user_remappings,(int)j))
				{
					next = (int)j;
				}
			}
		}
		push_int(&user_remappings,next);
	}

	free_str_vec(&tmp);
}

static void
get_ray_directions_recursive(const ASTNode* node,string_vec* dst)
{
	TRAVERSE_PREAMBLE_PARAMS(get_ray_directions_recursive,dst);
        if(node->type  != (NODE_DECLARATION | NODE_GLOBAL)) return;
        const ASTNode* type_node = get_node(NODE_TSPEC,node);
        if(!type_node) return;
        //const ASTNode* qualifier = get_node(NODE_TQUAL,node);
        ////TP: do this only for no qualifier declarations
        //if(qualifier) return;
        const char* type = type_node->lhs->buffer;
	if(type != RAYTRACE_STR) return;
	push(dst,intern(combine_all_new(type_node->rhs)));
}


static string_vec
get_ray_directions(const ASTNode* node)
{
	string_vec res = VEC_INITIALIZER;
	get_ray_directions_recursive(node,&res);
	return res;
}

static void
gen_field_info(FILE* fp)
{
  num_fields   = count_symbols(FIELD_STR);
  num_complex_fields   = count_symbols(COMPLEX_FIELD_STR);

  // Enums
  int num_of_communicated_fields=0;
  size_t num_of_fields=0;
  bool field_is_auxiliary[256];
  bool field_is_communicated[256];
  bool field_is_device_only[256];
  bool field_is_dead[256];
  bool field_has_variable_dims[256];
  size_t num_of_alive_fields=0;
  string_vec field_names = VEC_INITIALIZER;
  string_vec original_names = VEC_INITIALIZER;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(symbol_table[i].tspecifier == FIELD_STR)
	    push(&original_names,symbol_table[i].identifier);
  string_vec names = VEC_INITIALIZER;
  for (size_t i = 0; i < original_names.size; ++i)
  {
	push(&names,(user_remappings.data[i] == -1) ? original_names.data[i] : original_names.data[user_remappings.data[i]]);
  }

  for (size_t i = 0; i < names.size; ++i)
  {
    const Symbol* sym = (Symbol*)get_symbol(NODE_VARIABLE_ID,names.data[i],NULL);
    const bool is_dead = str_vec_contains(sym->tqualifiers,DEAD_STR);
    if(is_dead) continue;
    push(&field_names,sym->identifier);
    const bool is_aux  = str_vec_contains(sym->tqualifiers,AUXILIARY_STR);
    const bool is_comm = str_vec_contains(sym->tqualifiers,COMMUNICATED_STR);
    const bool has_variable_dims = str_vec_contains(sym->tqualifiers,DIMS_STR);
    const bool is_device_only = str_vec_contains(sym->tqualifiers,DEVICE_ONLY_STR);
    field_is_auxiliary[num_of_fields]    = is_aux;
    field_is_communicated[num_of_fields] = is_comm;
    field_has_variable_dims[num_of_fields] = has_variable_dims;
    num_of_communicated_fields           += is_comm;
    num_of_alive_fields                  += (!is_dead);
    field_is_dead[num_of_fields]         = is_dead;
    field_is_device_only[num_of_fields] = is_device_only;
    ++num_of_fields;
  }
  for (size_t i = 0; i < names.size; ++i)
  {
    const Symbol* sym = (Symbol*)get_symbol(NODE_VARIABLE_ID,names.data[i],NULL);
    const bool is_dead = str_vec_contains(sym->tqualifiers,DEAD_STR);
    if(!is_dead) continue;
    push(&field_names,sym->identifier);
    const bool is_aux  = str_vec_contains(sym->tqualifiers,AUXILIARY_STR);
    const bool is_comm = str_vec_contains(sym->tqualifiers,COMMUNICATED_STR);
    const bool has_variable_dims = str_vec_contains(sym->tqualifiers,DIMS_STR);
    const bool is_device_only = str_vec_contains(sym->tqualifiers,DEVICE_ONLY_STR);
    field_is_auxiliary[num_of_fields]    = is_aux;
    field_is_communicated[num_of_fields] = is_comm;
    field_has_variable_dims[num_of_fields] = has_variable_dims;
    num_of_communicated_fields           += is_comm;
    num_of_alive_fields                  += (!is_dead);
    field_is_dead[num_of_fields]         = is_dead;
    field_is_device_only[num_of_fields] = is_device_only;
    ++num_of_fields;
  }
  string_vec complex_field_names = VEC_INITIALIZER;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    	if(symbol_table[i].tspecifier == COMPLEX_FIELD_STR){
	  push(&complex_field_names,symbol_table[i].identifier);
	}
  }
  fprintf(fp,"static const int field_remappings[] = {");
  for(size_t field = 0; field < num_of_fields; ++field)
  {
          const int old_index  = str_vec_get_index(names,field_names.data[field]);
  	  fprintf(fp,"%d,",old_index);
  }
  fprintf(fp,"};");
  //TP: IMPORTANT!! if there are dead fields NUM_VTXBUF_HANDLES is equal to alive fields not all fields.  
  //TP: the compiler is allowed to move dead field declarations till the end
  //TP: this way the user can easily loop all alive fields with the old 0:NUM_VTXBUF_HANDLES and same for the Astaroth library dead fields are skiped over automatically
  {
        string_vec prof_types = get_prof_types();
	FILE* fp_enums = fopen("builtin_enums.h","w");
  	fprintf(fp_enums,"#pragma once\n");
  	gen_enums(fp_enums,STENCIL_STR,"stencil_","NUM_STENCILS","Stencil");
  	gen_enums(fp_enums,RAYTRACE_STR,"ray_","NUM_RAYS","AcRay");
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
  	fprintf(fp_enums, "typedef enum {");
  	for(size_t i = 0; i < num_complex_fields; ++i)
  	        fprintf(fp_enums,"%s,",complex_field_names.data[i]);

  	fprintf(fp_enums, "NUM_COMPLEX_FIELDS} ComplexField;\n");
	fclose(fp_enums);
  }
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

  fprintf(fp, "static const bool vtxbuf_has_variable_dims[] = {");
  for(size_t i = 0; i < num_of_fields; ++i)
    if(field_has_variable_dims[i])
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

  fprintf(fp, "static const bool vtxbuf_is_device_only[] = {");

  for(size_t i = 0; i < num_of_fields; ++i)
    if(field_is_device_only[i])
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
  fclose(fp);

  fp = fopen("get_vtxbufs_funcs.h","w");
  for(size_t i = 0; i < num_of_fields; ++i)
  	fprintf(fp,"VertexBufferHandle acGet%s() {return %s;}\n", field_names.data[i], field_names.data[i]);
  fclose(fp);
  fp = fopen("get_vtxbufs_declares.h","w");
  for(size_t i = 0; i < num_of_fields; ++i)
	fprintf(fp,"FUNC_DEFINE(VertexBufferHandle, acGet%s,());\n",field_names.data[i]);	
  fclose(fp);
  fp = fopen("get_vtxbufs_loads.h","w");
  for(size_t i = 0; i < num_of_fields; ++i)
  {
	char* func_name;
	my_asprintf(&func_name,"acGet%s",field_names.data[i]);
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


  {
  	fprintf(fp, "static const bool skip_kernel_in_analysis[NUM_KERNELS] = {");
  	int k_counter = 0;
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].tspecifier == KERNEL_STR)
  	  {
  	    if (str_vec_contains(symbol_table[i].tqualifiers,UTILITY_STR))
  	    {
  	            skip_kernel_in_analysis[k_counter] = 1;
  	            fprintf(fp,"true,");
  	    }
  	    else
  	    {
  	            skip_kernel_in_analysis[k_counter] = 0;
  	            fprintf(fp,"false,");
  	    }
  	    k_counter++;
  	  }
  	fprintf(fp, "};");
  }

  {
  	fprintf(fp, "static const AcKernel kernel_enums[NUM_KERNELS] = {");
  	for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  	  if (symbol_table[i].tspecifier == KERNEL_STR)
  	  {
	    fprintf(fp,"%s,",symbol_table[i].identifier);
  	  }
  	fprintf(fp, "};");
  }


  fprintf(fp, "static const bool is_boundcond_kernel[NUM_KERNELS] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].tspecifier == KERNEL_STR)
    {
      if (str_vec_contains(symbol_table[i].tqualifiers,BOUNDCOND_STR))
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
  gen_names("ray",RAYTRACE_STR,fp);
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


  fprintf_filename("is_comptime_param.h","%s","#pragma once\n");
  {
	FILE* fp_runtime = fopen("user_defines_runtime_lib.h","w");
  	for (size_t i = 0; i < datatypes.size; ++i)
	{
		gen_param_names(fp_runtime,datatypes.data[i],"__device__");
	}
	fclose(fp_runtime);
  }
  for (size_t i = 0; i < datatypes.size; ++i)
  {
	  const char* datatype = datatypes.data[i];
	  gen_param_names(fp,datatype,"");
	  gen_datatype_enums(fp,datatype);

	  fprintf_filename("device_mesh_info_decl.h","%s %s_params[NUM_%s_PARAMS+1];\n",datatype,convert_to_define_name(datatype),strupr(convert_to_define_name(datatype)));
	  gen_array_declarations(datatype,root);
	  gen_comp_declarations(datatype);

	  fprintf(fp,"static bool %s_output_is_global [NUM_%s_OUTPUTS+1] __attribute__((unused)) = {",convert_to_define_name(datatype), strupr(convert_to_define_name(datatype)));
	  for(size_t symbol  = 0; symbol < num_symbols[0]; ++symbol)
	  {
	  	if(symbol_table[symbol].tspecifier == datatype && str_vec_contains(symbol_table[symbol].tqualifiers,OUTPUT_STR))
	  		fprintf(fp,"%s,\n",str_vec_contains(symbol_table[symbol].tqualifiers,GLOBAL_STR) ? "true" : "false");
	  }
	  fprintf(fp,"false};");
  }

  const size_t num_real_outputs = count_variables(REAL_STR,OUTPUT_STR);

#if AC_DOUBLE_PRECISION
  fprintf(fp,"\n#define  NUM_OUTPUTS (NUM_REAL_OUTPUTS+NUM_INT_OUTPUTS+NUM_FLOAT_OUTPUTS+NUM_PROFILES)\n");
#else
  fprintf(fp,"\n#define  NUM_OUTPUTS (NUM_REAL_OUTPUTS+NUM_INT_OUTPUTS+NUM_PROFILES)\n");
#endif
  fprintf(fp,"\n#define  PROF_SCRATCHPAD_INDEX(PROF) (NUM_REAL_OUTPUTS+PROF)\n");
  const size_t num_real_scratchpads = num_profiles+num_real_outputs;
  fprintf(fp,"\n#define  NUM_REAL_SCRATCHPADS (%zu)\n",num_real_scratchpads);
 
  const size_t num_dconsts = count_variables(NULL,DCONST_STR);
  fprintf(fp,"\n#define NUM_DCONSTS (%zu)\n",num_dconsts);

  fprintf(fp,"\n#include \"array_info.h\"\n");
  fprintf(fp,"\n#include \"taskgraph_enums.h\"\n");
  fprintf(fp,"\n#include \"fields_info.h\"\n");

  free_str_vec(&datatypes);
  free_structs_info(&s_info);

  fprintf(fp,"const int3 ray_directions[] = {");
  string_vec ray_directions = get_ray_directions(root_in);
  for(size_t ray = 0; ray < ray_directions.size; ++ray)
	  fprintf(fp,"{%s},",ray_directions.data[ray]);
  fprintf(fp,"{0,0,0}};");



  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK
  fprintf(fp,
  	"\n// Redefined for backwards compatibility START\n"
  	"#define NUM_VTXBUF_HANDLES (NUM_FIELDS)\n"
  	"typedef Field VertexBufferHandle;\n"
	 );
  fprintf(fp, "static const char** vtxbuf_names __attribute__((unused)) = "
              "field_names;\n");
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
	 );

  char cwd[9000];
  cwd[0] = '\0';
  const char* err = getcwd(cwd, sizeof(cwd));
  if(err == NULL) fatal("Was not able to get current working directory\n");
  char autotune_path[10004];
  sprintf(autotune_path,"%s/autotune.csv",cwd);
  fprintf(fp,
  	"__attribute__((unused)) static const char* autotune_csv_path= \"%s\";\n"
  	"__attribute__((unused)) static const char* astaroth_binary_path = \"%s\";\n"
  	"__attribute__((unused)) static const char* astaroth_base_path = \"%s\";\n"
	,autotune_path
	,AC_BINARY_PATH
	,AC_BASE_PATH
	);

  fclose(fp);

  //Done to refresh the autotune file when recompiling DSL code
  fp = fopen(autotune_path,"w");
  fprintf(fp,"### Implementation,Enum,Dims.x,Dims.y,Dims.z,Tpb.x,Tpb.y,Tpb.z,Time,Kernel,BlockFactor.x,BlockFactor.y,BlockFactor.z,RayTracingFactor.x,RayTracingFactor.y,RayTracingFactor.z,Sparse ###\n");
  fclose(fp);

  fp = fopen("user_constants.h","w");
  FILE* fp_built_in  = fopen("user_built-in_constants.h","w");
  FILE* fp_non_scalar_constants = fopen("user_non_scalar_constants.h","w");
  FILE* fp_bi_non_scalar_constants = fopen("user_builtin_non_scalar_constants.h","w");

  //fprintf(fp,"#pragma once\n");
  //fprintf(fp_non_scalar_constants,"#pragma once\n");
  fprintf(fp_bi_non_scalar_constants,"#pragma once\n");
  fprintf(fp_built_in,"#pragma once\n");

  gen_const_variables(root,fp,fp_built_in,fp_non_scalar_constants,fp_bi_non_scalar_constants,true);

  fclose(fp);
  fclose(fp_built_in);
  fclose(fp_non_scalar_constants);
  fclose(fp_bi_non_scalar_constants);

  fp = fopen("kernel_user_constants.h","w");
  fp_built_in  = fopen("kernel_user_built-in_constants.h","w");
  fp_non_scalar_constants = fopen("kernel_user_non_scalar_constants.h","w");
  fp_bi_non_scalar_constants = fopen("kernel_user_builtin_non_scalar_constants.h","w");

  gen_const_variables(root,fp,fp_built_in,fp_non_scalar_constants,fp_bi_non_scalar_constants,false);

  fclose(fp);
  fclose(fp_built_in);
  fclose(fp_non_scalar_constants);
  fclose(fp_bi_non_scalar_constants);

  fp = fopen("user_config_loader.h","w");
  gen_config_loaders(root,fp);
  fclose(fp);
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

  const char* default_param_list=  "(const int3 start, const int3 end, DeviceVertexBufferArray vba";
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

  fprintf(fp, "\n"); // Add newline at EOF
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
	my_asprintf(&prefix,"%s_REPLACE_WITH_INLINE_COUNTER",func_name);
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

const ASTNode*
get_kfunc(const char* name)
{
		for(size_t i = 0; i < kfunc_names.size; ++i)
			if(name == kfunc_names.data[i]) return kfunc_nodes.data[i];
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
bool
should_be_reference(const ASTNode* node)
{
	const bool is_constexpr = all_identifiers_are_constexpr(node);
	const char* param_type  = get_expr_type((ASTNode*)node);
	const bool param_is_arr = param_type && (strstr(param_type,"*") || strstr(param_type,"AcArray"));
	const bool is_reference = !is_constexpr || param_is_arr;
	return is_reference;
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

	node_vec params = VEC_INITIALIZER;
	if(func_node->rhs) params = get_nodes_in_list(func_node->rhs);
	func_params_info params_info = get_function_params_info(dfunc,func_name);
	for(size_t i = 0; i < params.size; ++i)
	{

		const char* type = params_info.types.data[i] ? sprintf_intern("%s%s",
				params_info.types.data[i], should_be_reference(params.data[i]) ? "&" : "")
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
		const char* type = params_info.types.data[i] ? sprintf_intern("%s%s",
				params_info.types.data[i], should_be_reference(params.data[i]) ? "&" : "")
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
	if(!func_node || !func_node->lhs) return res;
	const char* func_name = get_node_by_token(IDENTIFIER,func_node)->buffer;
	if(!func_name || !check_symbol(NODE_DFUNCTION_ID,func_name,0,INLINE_STR)) return res;
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
		head->rhs = head->lhs;
		head->type = NODE_STATEMENT_LIST_HEAD;
		ASTNode* statement = astnode_create(NODE_UNKNOWN, res, NULL);
		head->lhs=  statement;
		statement->parent = head;
	}
	else
	{
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
	my_asprintf(&inline_var_name,"inlined_var_return_value_%d",*counter);
	const char* type = get_expr_type(node) ? sprintf_intern("%s%s",
			get_expr_type(node), should_be_reference(node) ? "&" : "")
		: NULL;
	ASTNode* decl  = create_declaration(inline_var_name,type,CONST_STR);
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
	//ASTNode* head = (ASTNode*)get_parent_node_inclusive(NODE_STATEMENT_LIST_HEAD,node);
	//if(!head) fatal("NO HEAD!: %s\n",combine_all_new(node));
	if(node->lhs->token != BASIC_STATEMENT) return res;
	res |= turn_inline_function_calls_to_assignments_in_statement(node,node,counter);
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
        TRAVERSE_PREAMBLE(transform_arrays_to_std_arrays_in_func)
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
        const ASTNode* access_start = get_node(NODE_ARRAY_ACCESS,node);
        node_vec dims = get_array_accesses(access_start);
        const char* dim = combine_all_new(node->rhs->lhs->rhs);
        //TP: at least some CUDA compilers do not allow zero-sized objects in device code so have to pad the array length
        if(!strcmp(dim,"(0)"))
        {
                dim = "(1)";
                astnode_sprintf(tspec->lhs,"AcArray<%s,%s>",tspec->lhs->buffer,dim);
        }
        else if(dims.size == 1)
        {
                astnode_sprintf(tspec->lhs,"AcArray<%s,%s>",tspec->lhs->buffer,combine_all_new(dims.data[0]));
        }
        else if(dims.size == 2)
        {
                astnode_sprintf(tspec->lhs,"AcArray<%s,%s,%s>",tspec->lhs->buffer,strdup(combine_all_new(dims.data[0])),strdup(combine_all_new(dims.data[1])));
        }
        free_node_vec(&dims);
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
void
gen_kernel_combinatorial_optimizations_and_input(ASTNode* root, const bool optimize_input_params)
{
  combinatorial_params_info info = get_combinatorial_params_info(root);
  if(optimize_input_params)
  {
	gen_optimized_kernel_decls(root,info.params,info.kernels_with_input_params,info.kernel_combinatorial_params, get_tail_node(root));
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
	  const ASTNode* struct_expr = get_node(NODE_STRUCT_EXPRESSION,node->lhs);
	  if(struct_expr)
	  {
		  lhs = intern(combine_all_new(struct_expr));
	  }
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
		}
	}
	//TP: below sets the constexpr value of lhs the same as rhs for: lhs = rhs
	//TP: we restrict to the case that lhs is assigned only once in the function since full generality becomes too hard 
	//TP: However we get sufficiently far with this approach since we turn many easy cases to SSA form which this check covers
	if((node->type & NODE_ASSIGNMENT) && node->rhs && !node->is_constexpr)
	{
	  bool is_constexpr = all_identifiers_are_constexpr(node->rhs);
	  ASTNode* lhs_identifier = get_node_by_token(IDENTIFIER,node->lhs);
	  const char* lhs = lhs_identifier->buffer;
	  const ASTNode* struct_expr = get_node(NODE_STRUCT_EXPRESSION,node->lhs);
	  if(struct_expr)
	  {
		  lhs = intern(combine_all_new(struct_expr));
	  	  lhs_identifier = get_node_by_token(IDENTIFIER,struct_expr->lhs);
	  }

	  const int index = str_vec_get_index(names,lhs);
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
			if(!dfunc_start)
			{
				ASTNode* kernel_start = (ASTNode*) get_parent_node(NODE_FUNCTION,node);
				if(kernel_start)
				{
					fatal("Can not have return statements inside Kernels; In %s: %s\n",get_node_by_token(IDENTIFIER,kernel_start)->buffer,combine_all_new(node));
				}
			}
			dfunc_start -> expr_type = expr_type;
		}
	}
	if(node->type & (NODE_PRIMARY_EXPRESSION | NODE_FUNCTION_CALL) ||
		(node->type & NODE_DECLARATION && get_node(NODE_TSPEC,node)) ||
		(node->type & NODE_EXPRESSION && all_primary_expressions_and_func_calls_have_type(node)) ||
		(node->type & NODE_ASSIGNMENT && node->rhs && get_parent_node(NODE_FUNCTION,node) &&  !get_node(NODE_MEMBER_ID,node->lhs))
		|| (node->token == IN_RANGE)
		|| (node->token == CAST)
	)
		get_expr_type(node);
	res |=  node -> expr_type != NULL;
	return res;
}
bool
flow_type_info_in_func(ASTNode* node,string_vec* names, string_vec* types, const string_vec func_names, const string_vec func_return_types)
{
	bool res = false;
	if(node->lhs)
		res |= flow_type_info_in_func(node->lhs,names,types,func_names,func_return_types);
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
	if(node->type & NODE_FUNCTION_CALL && !node->expr_type)
	{
		const char* func_name = get_node_by_token(IDENTIFIER,node)->buffer;
		const int index = str_vec_get_index(func_names,func_name);
		if(index >= 0)
			node->expr_type = func_return_types.data[index];
	}
	if(node->rhs)
		res |= flow_type_info_in_func(node->rhs,names,types,func_names,func_return_types);
	return res;
}
bool
flow_type_info_base(ASTNode* node, const string_vec func_names, const string_vec func_return_types)
{
	bool res = false;
	if(node->lhs)
		res |= flow_type_info_base(node->lhs,func_names,func_return_types);
	if(node->rhs)
		res |= flow_type_info_base(node->rhs,func_names,func_return_types);
	if(node->type & NODE_FUNCTION)
	{
		string_vec names = VEC_INITIALIZER;
		string_vec types = VEC_INITIALIZER;
		res |= flow_type_info_in_func(node,&names,&types,func_names,func_return_types);
		free_str_vec(&names);
		free_str_vec(&types);
	}
	return res;
}
void
gather_dfunc_return_types(ASTNode* node, string_vec* func_names, string_vec* func_return_types)
{
	if(node->type & NODE_FUNCTION)
	{
		if(node->type & NODE_DFUNCTION)
		{
			if(node->expr_type == NULL) return;
			const char* name = get_node_by_token(IDENTIFIER,node)->buffer;
			if(str_vec_contains(*func_names,name)) return;
			push(func_names,name);
			push(func_return_types,node->expr_type);
		}
		return;
	}
	TRAVERSE_PREAMBLE_PARAMS(gather_dfunc_return_types,func_names,func_return_types);
}

bool
flow_type_info(ASTNode* node)
{
	string_vec func_names =        VEC_INITIALIZER;
	string_vec func_return_types = VEC_INITIALIZER;
	gather_dfunc_return_types(node,&func_names,&func_return_types);
	const bool res = flow_type_info_base(node,func_names,func_return_types);
	free_str_vec(&func_names);
	free_str_vec(&func_return_types);
	return res;
}

static ASTNode* 
create_func_call(const char* func_name, const node_vec params)
{

	ASTNode* postfix_expression = astnode_create(NODE_UNKNOWN,
			     create_primary_expression(func_name),
			     NULL);
	ASTNode* expression_list = build_list_node(params,",");
	//ASTNode* expression_list = astnode_create(NODE_UNKNOWN,astnode_dup(param,NULL), NULL);
	ASTNode* func_call = astnode_create(NODE_FUNCTION_CALL,postfix_expression,expression_list);
	astnode_set_infix("(",func_call); 
	astnode_set_postfix(")",func_call); 
	return func_call;
}

static ASTNode* 
create_func_call_expr_variadic(const char* func_name, const node_vec params)
{
	ASTNode* func_call = create_func_call(func_name,params);
	ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
	ASTNode* expression         = astnode_create(NODE_EXPRESSION,unary_expression,NULL);
	return expression;
}
static ASTNode* 
create_func_call_expr(const char* func_name, const ASTNode* param)
{
	node_vec params = VEC_INITIALIZER;
	push_node(&params,param);
	ASTNode* func_call = create_func_call(func_name,params);
	free_node_vec(&params);
	ASTNode* unary_expression   = astnode_create(NODE_EXPRESSION,func_call,NULL);
	ASTNode* expression         = astnode_create(NODE_EXPRESSION,unary_expression,NULL);
	return expression;
}

bool
func_params_conversion(ASTNode* node, const ASTNode* root)
{
	bool res = false;
	if(node->lhs)
		res |= func_params_conversion(node->lhs,root);
	if(node->rhs)
		res |= func_params_conversion(node->rhs,root);
	if(!(node->type & NODE_FUNCTION_CALL)) return res;
	const char* func_name = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	if(!check_symbol(NODE_FUNCTION_ID,func_name,0,0)) return res;
	if(str_vec_contains(duplicate_dfuncs.names,func_name)) return res;
	const Symbol* sym = get_symbol(NODE_FUNCTION_ID, func_name, NULL);
	if(!sym) return res;
        if(sym->tspecifier == STENCIL_STR) return res;
	const bool is_intrinsic = str_vec_contains(sym->tqualifiers,INTRINSIC_STR);
	if(!is_intrinsic && !(sym->type & NODE_DFUNCTION_ID)) return res;
	if(is_intrinsic && func_name != intern("min") && func_name != intern("max")) return res;

	func_params_info call_info = get_func_call_params_info(node);
	if(call_info.types.size == 0) return res;
	func_params_info params_info = get_function_params_info(root,func_name);
	if(!is_intrinsic && params_info.types.size != call_info.types.size)
		fatal("number of parameters does not match, expected %zu but got %zu in %s\n",params_info.types.size, call_info.types.size, combine_all_new(node));
	for(size_t i = 0; i < call_info.types.size; ++i)
	{
		if(!call_info.types.data[i]) continue;
		if(!is_intrinsic && !params_info.types.data[i]) continue;
		if(
		      ((is_intrinsic || params_info.types.data[i] == REAL3_STR     ) && call_info.types.data[i] == FIELD3_STR)
		   || ((is_intrinsic || params_info.types.data[i] == REAL_STR      ) && call_info.types.data[i] == FIELD_STR)
		   || ((is_intrinsic || params_info.types.data[i] == REAL_PTR_STR  ) && call_info.types.data[i] == VTXBUF_PTR_STR)
		   || ((is_intrinsic || params_info.types.data[i] == REAL_PTR_STR  ) && call_info.types.data[i] == FIELD_PTR_STR)
		   || ((is_intrinsic || params_info.types.data[i] == REAL3_PTR_STR ) && call_info.types.data[i] == FIELD3_PTR_STR)
		   || ((is_intrinsic || params_info.types.data[i] == REAL_STR      ) && strstr(call_info.types.data[i],"Profile"))
		   || ((is_intrinsic || params_info.types.data[i] == call_info.types.data[i]) && is_output_type(call_info.expr.data[i]))
		  )
		{
			ASTNode* expr = (ASTNode*)call_info.expr_nodes.data[i];
			if(is_output_type(call_info.expr.data[i]) && is_arr_type(call_info.expr.data[i]))
			{
				expr = (ASTNode*)get_parent_node(NODE_ARRAY_ACCESS,expr);
			}
			//expr->expr_type = params_info.types.data[i-offset];
			replace_node(
					expr,
					create_func_call_expr(
						is_output_type(call_info.expr.data[i]) ? OUTPUT_VALUE_STR : VALUE_STR
						,expr)
				     );
		}
		//TP: This translates e.g. any_AC(arr,3) --> any_AC(AC_INTERNAL_d_bool_arrays_arr,3)
		//TP: i.e. translates enums to the appropriate pointers when the function expects to take a pointer
		//TP: This is probably not the best place to place this but works for now
		if((is_intrinsic || strstr(params_info.types.data[i],"*")) && strstr(call_info.types.data[i],"*"))
		{
  			const Symbol* var_sym = get_symbol(NODE_VARIABLE_ID,intern(call_info.expr.data[i]),NULL);
			if(var_sym)
			{
				ASTNode* identifier = get_node_by_token(IDENTIFIER,call_info.expr_nodes.data[i]);
				if(identifier)
				{

					astnode_sprintf(identifier,"%s",get_internal_array_name(var_sym));
				}
			}
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
	//TP: this is done first so that e.g. real3 = real broadcast assignment won't be understood incorrectly
	gen_declared_type_info(root);
	flow_type_info(root);
	while(has_changed)
	{
		has_changed = gen_local_type_info(root);
		has_changed |= flow_type_info(root);
	}
	func_params_conversion(root,root);
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
		my_asprintf(&tmp,"%s_AC_MANGLED_NAME_",dfunc_name);
		for(size_t i = 0; i < types.size; ++i)
			my_asprintf(&tmp,"%s_%s",tmp,
					types.data[i] ? types.data[i] : "Auto");
		if(types.size == 0) my_asprintf(&tmp,"%s_%s",tmp,"Empty");
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
compatible_types(const char* a, const char* b)
{
	if(is_subtype(a,b)) return true;
	if(b && strstr(b,"AcArray") && a && strstr(a,"*"))
	{
		const char* scalar_type = get_array_elem_type(b);
		char* tmp = strdup(a);
		remove_suffix(tmp,"*");
		const char* ptr_scalar_type = intern(tmp);
		free(tmp);
		if(scalar_type == ptr_scalar_type) return true;
	}
	const bool res = !strcmp(a,b) 
	       || 
                  (!strcmp(a,FIELD_STR) && !strcmp(b,"VertexBufferHandle"))  ||
		  (!strcmp(b,FIELD_STR) && !strcmp(a,"VertexBufferHandle"))  ||
                  (!strcmp(a,"Field*") && !strcmp(b,VTXBUF_PTR_STR))  ||
		  (!strcmp(b,"Field*") && !strcmp(a,VTXBUF_PTR_STR))
		  || (a == REAL_STR && b == FIELD_STR)
		  || (a == REAL3_STR && b == FIELD3_STR)
		  || (a == REAL_PTR_STR && b == VTXBUF_PTR_STR)
		  || (a == REAL_PTR_STR && b == FIELD_PTR_STR)
		  || (a == REAL3_PTR_STR && b == FIELD3_PTR_STR)
		  || (a == INT_STR && is_enum_type(b))
		  || (is_enum_type(a) && b == INT_STR)
		  || (is_enum_type(b) && a == INT_STR)
		;
	if(!res)
	{
		if(a == REAL_PTR_STR && get_array_elem_type(b) == REAL_STR)
		{
			return true;
		}
		if(a == REAL3_PTR_STR && get_array_elem_type(b) == REAL3_STR)
		{
			return true;
		}
	}
	return res;
}
typedef struct 
{
	const string_vec* types;
	const char*const * names;
} dfunc_possibilities;
static int_vec
get_possible_dfuncs(const func_params_info call_info, const dfunc_possibilities possibilities, const int dfunc_index, const bool strict, const bool use_auto, const bool allow_unknowns, const char* func_name)
{
	int overload_index = MAX_DFUNCS*dfunc_index-1;
	int_vec possible_indexes = VEC_INITIALIZER;
    	//TP: ugly hack to resolve calls in BoundConds
	while(possibilities.names[++overload_index] == func_name)
	{
		bool possible = true;
		if(call_info.types.size != possibilities.types[overload_index].size) continue;
		bool all_types_specified = true;
		for(size_t i = 0; i < call_info.types.size; ++i)
			all_types_specified &= (possibilities.types[overload_index].data[i] != NULL);
		if(use_auto && all_types_specified) continue;
		for(size_t i = 0; i < call_info.types.size; ++i)
		{
			const char* func_type = possibilities.types[overload_index].data[i];
			const char* call_type = call_info.types.data[i];
			if(strict)
				possible &= !call_type || !func_type || !strcmp(func_type,call_type);
			else
				possible &= !call_type || !func_type || compatible_types(func_type,call_type);
			if(!use_auto) possible &= func_type != NULL;
			if(!allow_unknowns) possible &= call_type != NULL;

		}
		if(possible)
		{
			push_int(&possible_indexes,overload_index);
		}
	}
	return possible_indexes;
}
bool
resolve_overloaded_calls_base(ASTNode* node, const dfunc_possibilities possibilities)
{
	bool res = false;
	if(node->lhs)
		res |= resolve_overloaded_calls_base(node->lhs,possibilities);
	if(node->rhs)
		res |= resolve_overloaded_calls_base(node->rhs,possibilities);
	if(!(node->type & NODE_FUNCTION_CALL))
		return res;
	if(!get_node_by_token(IDENTIFIER,node->lhs))
		return res;
	const int dfunc_index = str_vec_get_index(duplicate_dfuncs.names,get_node_by_token(IDENTIFIER,node->lhs)->buffer);
	if(dfunc_index == -1)
		return res;
	const char* dfunc_name = duplicate_dfuncs.names.data[dfunc_index];
	func_params_info call_info = get_func_call_params_info(node);
	int_vec possible_indexes_strict_no_auto          = get_possible_dfuncs(call_info,possibilities,dfunc_index,true,false ,false,dfunc_name);
	int_vec possible_indexes_conversion_no_auto      = get_possible_dfuncs(call_info,possibilities,dfunc_index,false,false,false,dfunc_name);

	int_vec possible_indexes_strict_unknowns     = get_possible_dfuncs(call_info,possibilities,dfunc_index,true,false, true,dfunc_name);
	int_vec possible_indexes_conversion_unknowns = get_possible_dfuncs(call_info,possibilities,dfunc_index,false,false,true,dfunc_name);

	int_vec possible_indexes_strict                  = get_possible_dfuncs(call_info,possibilities,dfunc_index,true,true, true,dfunc_name);
	int_vec possible_indexes_conversion              = get_possible_dfuncs(call_info,possibilities,dfunc_index,false,true,true,dfunc_name);
	//TP: by default use the strict rules but if there no suitable ones use conversion rules
	//TP: if there are multiple possibilities pick the one with all specified parameters
	const int_vec possible_indexes = 
					possible_indexes_strict_unknowns.size     == 1 ? possible_indexes_strict_unknowns  :
					possible_indexes_conversion_unknowns.size     == 1 ? possible_indexes_conversion_unknowns  :
					possible_indexes_strict_no_auto.size      > 0 ? possible_indexes_strict_no_auto :
					possible_indexes_conversion_no_auto.size  > 0 ? possible_indexes_conversion_no_auto :
					possible_indexes_strict.size              > 0 ? possible_indexes_strict :
					possible_indexes_conversion;
	bool able_to_resolve = possible_indexes.size == 1;
	//if(able_to_resolve) printf("Able to resolve: %s, %s\n",dfunc_name,combine_all_new(node->rhs));
	if(!able_to_resolve) { 
		//if(!strcmp(dfunc_name,"rk3_intermediate"))
		//{
		//	//printf("Not able to resolve: %s\n",combine_all_new(node->rhs)); 
		//	//printf("Not able to resolve: %s,%s,%s,%s\n",call_info.types.data[0],call_info.types.data[1],call_info.types.data[2],call_info.types.data[3]); 
		//	//int overload_index = MAX_DFUNCS*dfunc_index-1;
    		//	////TP: ugly hack to resolve calls in BoundConds
		//	//const int param_offset = (call_info.expr.size > 0 && is_boundary_param(call_info.expr.data[0])) ? 1 : 0;
		//	//while(possibilities.names[++overload_index] == dfunc_name)
		//	//{
		//	//	for(size_t i = 0; i < possibilities.types[overload_index].size; ++i)
		//	//		printf("%s,",possibilities.types[overload_index].data[i]);
		//	//	printf("\n");
		//	//}
		//	//printf("Not able to resolve: %zu\n",possible_indexes.size); 
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
	free_int_vec(&possible_indexes_strict_no_auto);
	free_int_vec(&possible_indexes_conversion_no_auto);
	free_int_vec(&possible_indexes_strict_unknowns);
	free_int_vec(&possible_indexes_conversion_unknowns);
	return true;
}

bool
resolve_overloaded_calls(ASTNode* node, const dfunc_possibilities possibilities)
{
	bool res = false;
	if(node->rhs && node->rhs->type & NODE_FUNCTION) res |= resolve_overloaded_calls_base(node->rhs,possibilities);
	if(node->lhs) res |= resolve_overloaded_calls(node->lhs,possibilities);
	return res;
}
void
transform_array_unary_ops(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_array_unary_ops);
	if(!node_is_unary_expr(node)) return;
	const char* base_type= get_expr_type(node->rhs);
	const char* unary_op = get_node_by_token(UNARY_OP,node->lhs)->buffer;
	if(strcmps(unary_op,PLUS_STR,MINUS_STR)) return;
	if(!base_type) return;
	const bool is_array = strstr(base_type,"*") || strstr(base_type,"AcArray");
	if(!is_array) return;
	if(unary_op == PLUS_STR) return;
	node->lhs = NULL;
	node->rhs = create_func_call_expr(intern("create_neg_arr"),node->rhs);
}
void
transform_array_binary_ops(ASTNode* node)
{
        if(node->lhs)
                transform_array_binary_ops(node->lhs);
        if(node->rhs)
                transform_array_binary_ops(node->rhs);
        if(!node_is_binary_expr(node)) return;

        const char* lhs_expr = get_expr_type(node->lhs);
        const char* rhs_expr = get_expr_type(node->rhs);
        if(!lhs_expr || !rhs_expr) return;
        const char* op = node->rhs->lhs->buffer;
        if(!op) return;
        if(strcmps(op,PLUS_STR,MINUS_STR,DIV_STR,MULT_STR)) return;

        const bool lhs_is_array = strstr(lhs_expr,"*") || strstr(lhs_expr,"AcArray");
        //const bool rhs_is_array = strstr(rhs_expr,"*") || strstr(rhs_expr,"AcArray");
        const bool rhs_is_vec   = rhs_expr == REAL3_STR;

        //TP: these work but generate CUDA code that is quite hard to read
        //instead prefer to add indexes to assignments i.e.
        //arr_c = arr_a + arr_b --> for(i = 0; i < arr_c_dims; ++i) arr_c[i] = arr_a[i] + arr_b[i]
        //Since that is more readable and also generates quite straightforward code that the CUDA/HIP compiler should have an easier time to handle
        //if(lhs_is_array && rhs_is_array)
        //{
        //      node_vec params = VEC_INITIALIZER;
        //      push_node(&params,node->lhs);
        //      push_node(&params,node->rhs->rhs);
        //      const char* func_name =
        //              op == PLUS_STR  ? intern("add_arr") :
        //              op == MINUS_STR ? intern("sub_arr") :
        //              op == MULT_STR  ? intern("mult_arr") :
        //              op == DIV_STR   ? intern("div_arr") :
        //              NULL;
        //      replace_node(node, create_func_call_expr_variadic(func_name,params));
        //      free_node_vec(&params);
        //}
        //if(lhs_is_array && rhs_expr == REAL_STR)
        //{
        //      node_vec params = VEC_INITIALIZER;
        //      push_node(&params,node->lhs);
        //      push_node(&params,node->rhs->rhs);
        //      const char* func_name =
        //              op == PLUS_STR  ? intern("add_arr") :
        //              op == MINUS_STR ? intern("sub_arr") :
        //              op == MULT_STR  ? intern("mult_arr") :
        //              op == DIV_STR   ? intern("div_arr") :
        //              NULL;
        //      replace_node(node, create_func_call_expr_variadic(func_name,params));
        //      free_node_vec(&params);
        //}
        //if(rhs_is_array && lhs_expr == REAL_STR)
        //{
        //      node_vec params = VEC_INITIALIZER;
        //      push_node(&params,node->rhs->rhs);
        //      push_node(&params,node->lhs);
        //      const char* func_name =
        //              op == PLUS_STR  ? intern("add_arr") :
        //              op == MINUS_STR ? intern("sub_arr") :
        //              op == MULT_STR  ? intern("mult_arr") :
        //              op == DIV_STR   ? intern("div_arr") :
        //              NULL;
        //      replace_node(node, create_func_call_expr_variadic(func_name,params));
        //      free_node_vec(&params);
        //}

        ASTNode* identifier = get_node_by_token(IDENTIFIER,node->lhs);
        if(!identifier) return;
        if(lhs_is_array && rhs_is_vec)
        {
                if(op != MULT_STR)
		{
                        fatal("Only mat mul supported for array*vec!: %s\n",combine_all_new_with_whitespace(node));
		}
                if(check_symbol(NODE_VARIABLE_ID,identifier->buffer,0,DCONST_STR))
                {
                        astnode_sprintf(identifier,"AC_INTERNAL_d_real_arrays_%s",identifier->buffer);
                }
                else if(check_symbol(NODE_VARIABLE_ID,identifier->buffer,0,RUN_CONST_STR))
                {
                        astnode_sprintf(identifier,"AC_INTERNAL_run_const_array_here",identifier->buffer);
                }
		else
		{
                        astnode_sprintf(identifier,"AC_INTERNAL_gmem_real_arrays_%s",identifier->buffer);
		}
                node_vec params = VEC_INITIALIZER;
                push_node(&params,node->lhs);
                push_node(&params,node->rhs->rhs);
                const char* func_name = intern("matmul_arr");
                replace_node(node, create_func_call_expr_variadic(func_name,params));
                free_node_vec(&params);
        }
}

void
make_into_reference(ASTNode* node)
{
	TRAVERSE_PREAMBLE(make_into_reference);
	if(!node->expr_type) return;
	node->expr_type = sprintf_intern("%s&",node->expr_type);
}
bool
is_defining_expression(const ASTNode* node)
{
	if(!node) return false;
	return is_first_decl(node) || is_defining_expression(node->lhs) || is_defining_expression(node->rhs);
}
void
turn_array_type_to_scalar_type(ASTNode* node)
{
        const char* type = get_expr_type(node);
	if(strstr(type,"AcArray"))
		 type = get_array_elem_type(type);
	else
	{
		char* new_type = strdup(type);
		remove_suffix(new_type,"*");
		type = intern(new_type);
	}
	node->expr_type = type;
}

void
transform_array_calls_to_scalar_calls(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_array_calls_to_scalar_calls);
	if(!(node->type & NODE_FUNCTION_CALL)) return;
	ASTNode* id = (ASTNode*)get_node_by_token(IDENTIFIER,node);
	if(id->buffer == intern("value_AC_MANGLED_NAME__FieldARRAY"))
	{
		astnode_set_buffer("value_AC_MANGLED_NAME__Field",id);
		turn_array_type_to_scalar_type(node);
	}
	if(id->buffer == intern("value_AC_MANGLED_NAME__Field3ARRAY"))
	{
		astnode_set_buffer("value_AC_MANGLED_NAME__Field3",id);
		turn_array_type_to_scalar_type(node);
	}
}
void
add_index_to_arrays(ASTNode* node, const size_t rank)
{
	if(node->type & NODE_FUNCTION_CALL)
	{
		if(node->expr_type != NULL)
		{
			//TP: if on the rhs one has a function call that returns an array but the loop iterator after the function call and do not add
			//    it to potential parameters of the function call
			if(strstr(node->expr_type,"AcArray") || node->expr_type[strlen(node->expr_type)-1] == '*')
			{
				astnode_sprintf_postfix(node,"%s[AC_INTERNAL_ARRAY_LOOP_INDEX]",node->postfix);
				return;
			}
		}
	}
        TRAVERSE_PREAMBLE_PARAMS(add_index_to_arrays,rank);
        const char* type = get_expr_type(node);
        if(!type) return;
        const bool is_arr = (strstr(type,"*") || strstr(type,"AcArray"));
        if(is_arr && node->token == IDENTIFIER)
        {
                if(rank == 1)
                        astnode_sprintf_postfix(node,"[AC_INTERNAL_ARRAY_LOOP_INDEX]");
                else if(rank == 2)
                        astnode_sprintf_postfix(node,"[AC_INTERNAL_ARRAY_LOOP_INDEX_1][AC_INTERNAL_ARRAY_LOOP_INDEX_2]");
		turn_array_type_to_scalar_type(node);
        }
}

void
transform_array_assignments(ASTNode* node)
{
        TRAVERSE_PREAMBLE(transform_array_assignments);
        if(!(node->type & NODE_ASSIGNMENT)) return;
        const ASTNode* function = get_parent_node(NODE_FUNCTION,node);
        if(!function) return;
        const char* op = node->rhs->lhs->buffer;
        if(op != EQ_STR) return;
        const char* rhs_type = get_expr_type(node->rhs);
        if(!rhs_type) return;
        const bool rhs_is_arr = rhs_type && rhs_type != intern("char*") && (strstr(rhs_type,"*") || strstr(rhs_type,"AcArray"));
        const char* lhs_type = get_expr_type(node->lhs);
        if(!lhs_type) return;
	else if(rhs_is_arr && !get_node(NODE_ARRAY_INITIALIZER,node->rhs->rhs))
        {
                if(is_defining_expression(node->lhs))
                {
                      fatal("NOT ALLOWED! %s,%s\n",combine_all_new(node),rhs_type);
                      replace_node(
                                node->rhs->rhs,
                                create_func_call_expr(intern("dup_arr"),node->rhs->rhs)
                        );
                      make_into_reference(node->lhs);
                }
                else
                {
	  		const bool is_reference = lhs_type[strlen(lhs_type)-1] == '&';
			if(is_reference) 
			{
			}
			else if(strstr(lhs_type,"AcArray"))
                        {
                                string_vec sizes = get_array_elem_size(lhs_type);
                                if(sizes.size == 1)
                                {
                                        astnode_sprintf_prefix(node,"for (int AC_INTERNAL_ARRAY_LOOP_INDEX = 0; AC_INTERNAL_ARRAY_LOOP_INDEX < %s; ++AC_INTERNAL_ARRAY_LOOP_INDEX){",sizes.data[0]);
                                        astnode_sprintf_postfix(node,";}");
                                        add_index_to_arrays(node,sizes.size);
					//transform_array_calls_to_scalar_calls(node);
                                }
                                else if(sizes.size == 2)
                                {
                                        astnode_sprintf_prefix(node,
                                                                "for (int AC_INTERNAL_ARRAY_LOOP_INDEX_1 = 0; AC_INTERNAL_ARRAY_LOOP_INDEX_1 < %s; ++AC_INTERNAL_ARRAY_LOOP_INDEX_1){\n"
                                                                "for (int AC_INTERNAL_ARRAY_LOOP_INDEX_2 = 0; AC_INTERNAL_ARRAY_LOOP_INDEX_2 < %s; ++AC_INTERNAL_ARRAY_LOOP_INDEX_2){\n"
                                                                ,sizes.data[0],sizes.data[1]);
                                        astnode_sprintf_postfix(node,";}\n;}");
                                        add_index_to_arrays(node,sizes.size);
                                }
                        }
			else if(lhs_type == FIELD_PTR_STR)
			{
			    if(!node->prefix || !strstr(node->prefix,"AC_INTERNAL_ARRAY_LOOP"))
			    {
			    	const char* arr = intern(strdup(combine_all_new(node->lhs)));
                            	astnode_sprintf_prefix(node,
                            	                        "for (size_t AC_INTERNAL_ARRAY_LOOP_INDEX = 0; AC_INTERNAL_ARRAY_LOOP_INDEX < AC_get_array_len(%s); ++AC_INTERNAL_ARRAY_LOOP_INDEX){\n"
                            	                        ,arr);
				astnode_sprintf_prefix(node->lhs,"AC_INTERNAL_write_vtxbuf_at_current_point(");
				astnode_set_buffer(",",node->rhs->lhs);
                            	astnode_sprintf_postfix(node,");}\n");
                            	add_index_to_arrays(node,1);
			    }
			}
                        else
                        {
                                node_vec params = VEC_INITIALIZER;
                                push_node(&params,node->lhs);
                                push_node(&params,node->rhs->rhs);
                                replace_node(
                                                node,
                                                create_func_call_expr_variadic(intern("copy_arr"),params)
                                        );
                        }
                }
        }
        const bool lhs_is_arr = (strstr(lhs_type,"*") || strstr(lhs_type,"AcArray"));
        if(lhs_is_arr && (rhs_type == REAL_STR || rhs_type == INT_STR))
        {
                node_vec params = VEC_INITIALIZER;
                push_node(&params,node->lhs);
                push_node(&params,node->rhs->rhs);
                const char* array_elem_type = get_array_elem_type(lhs_type);
                if(array_elem_type && array_elem_type == REAL3_STR && rhs_type == REAL_STR)
                {
                        replace_node(
                                        node,
                                        create_func_call_expr_variadic(intern("broadcast_scalar_to_vec"),params)
                                );
                }
                else
                {
                        string_vec sizes = get_array_elem_size(lhs_type);
                        const char* func_name =
                                sizes.size == 1 ? "broadcast_scalar" :
                                sizes.size == 2 ? "broadcast_scalar_2d" :
                                sizes.size == 3 ? "broadcast_scalar_3d" :
                               "broadcast_scalar_4d";
                        replace_node(
                                        node,
                                        create_func_call_expr_variadic(intern(func_name),params)
                                );
                }
        }
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
	if(!str_vec_contains(sym -> tqualifiers,intern("intrinsic"))) return;
	if(func_name == intern("previous_base")) return;
	if(func_name == intern("ac_get_field_halos")) return;
        func_params_info param_info = get_func_call_params_info(node);
        if(param_info.expr.size == 1 && param_info.types.data[0] == FIELD_STR)
        {
                ASTNode* expression         = create_func_call_expr(VALUE_STR,node->rhs);
                ASTNode* expression_list = astnode_create(NODE_UNKNOWN,expression,NULL);
                node->rhs = expression_list;
        }
        free_func_params_info(&param_info);
} 

void
reset_expr_types(ASTNode* node)
{
	TRAVERSE_PREAMBLE(reset_expr_types);
	node->expr_type = NULL;
}


void
apply_value_to_output_types(ASTNode* node)
{
	if(node->type & NODE_FUNCTION_CALL)
	{
		const char* func_name = get_node_by_token(IDENTIFIER,node)->buffer;
		if(func_name == OUTPUT_VALUE_STR) return;
	}
	TRAVERSE_PREAMBLE(apply_value_to_output_types);
	if(node->token != IDENTIFIER || !node->buffer) return;
	if(!is_output_type(node->buffer)) return;
	if(is_arr_type(node->buffer))
	{
		ASTNode* base = (ASTNode*)get_parent_node(NODE_ARRAY_ACCESS,node);
		replace_node(base,create_func_call_expr(OUTPUT_VALUE_STR,base));
	}
	else
	{
		replace_node(node,create_func_call_expr(OUTPUT_VALUE_STR,node));
	}
}

void
transform_field_unary_ops(ASTNode* node)
{
	TRAVERSE_PREAMBLE(transform_field_unary_ops);
	if(!node_is_unary_expr(node)) return;
	const char* base_type= get_expr_type(node->rhs);
	const char* unary_op = get_node_by_token(UNARY_OP,node->lhs)->buffer;
	if(strcmps(unary_op,PLUS_STR,MINUS_STR)) return;
	if(base_type && is_value_applicable_type(base_type))
	{
		node->rhs = create_func_call_expr(VALUE_STR,node->rhs);
	}
	apply_value_to_output_types(node);

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
		node->lhs = create_func_call_expr(VALUE_STR,node->lhs);
	}
	if(is_value_applicable_type(rhs_expr))
	{

		node->rhs->rhs = create_func_call_expr(VALUE_STR,node->rhs->rhs);
	}

	apply_value_to_output_types(node);
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
resolve_profile_stencils(ASTNode* node)
{
	TRAVERSE_PREAMBLE(resolve_profile_stencils);
	if(!(node->type & NODE_FUNCTION_CALL)) return;
	ASTNode* func_identifier = get_node_by_token(IDENTIFIER,node->lhs);
	const char* func_name = func_identifier->buffer;
	const Symbol* sym = symboltable_lookup(func_name);
	if(!sym) return;
        if(sym->tspecifier != STENCIL_STR) return;
	func_params_info call_info  = get_func_call_params_info(node);
	if(call_info.types.size == 1 && call_info.types.data[0] && strstr(call_info.types.data[0],"Profile")) 
	{
		const char* new_name = sprintf_intern("%s_profile",func_name);
		astnode_set_buffer(new_name, func_identifier);
	}
	free_func_params_info(&call_info);

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
  int overload_counter = 0;
  while(overloaded_something)
  {
	overloaded_something = false;
  	transform_field_intrinsic_func_calls_and_ops(root);
  	gen_type_info(root);

  	transform_array_assignments(root);
  	transform_array_binary_ops(root);

	overloaded_something |= resolve_overloaded_calls(root,overload_possibilities);
	overload_counter++;
  	//for(size_t i = 0; i < duplicate_dfuncs.size; ++i)
  	        //overloaded_something |= resolve_overloaded_calls(root,duplicate_dfuncs.data[i],dfunc_possible_types,i);
  }
  for(size_t i = 0; i < MAX_DFUNCS*duplicate_dfuncs.names.size; ++i)
	  free_str_vec(&dfunc_possible_types[i]);
}



int_vec
get_all_acreal_structs()
{
	int_vec res = VEC_INITIALIZER;
	for(size_t j = 0; j < s_info.user_structs.size; ++j)
	{
		const char* name  = s_info.user_structs.data[j];
		if(all_same_struct(name,REAL_STR) && strstr(name,"AcReal"))
		{
			push_int(&res,j);
		}
	}
	return res;
}

int_vec
get_all_field_structs()
{
	int_vec res = VEC_INITIALIZER;
	for(int j = 0; j < (int)s_info.user_structs.size; ++j)
	{
 		const char* name  = s_info.user_structs.data[j];
 		if(all_same_struct(name,FIELD_STR))
 		{
 			const size_t num_members = s_info.user_struct_field_names[j].size;
 			//TP: do not force the user to qualify the real struct name for elemental purposes
 			const char* qualified_real_name = sprintf_intern("AcReal%zu",num_members);
 			const char* base_real_name = sprintf_intern("real%zu",num_members);
 			if(    str_vec_contains(s_info.user_structs,qualified_real_name) 
 		            || str_vec_contains(s_info.user_structs,base_real_name)
 			  )
			{
				push_int(&res,j);
			}
		}
	}
	return res;
}

int_vec
get_all_same_structs(const char* type)
{
	int_vec res = VEC_INITIALIZER;
	for(size_t j = 0; j < s_info.user_structs.size; ++j)
	{
		const char* name  = s_info.user_structs.data[j];
		if(all_same_struct(name,type))
		{
			push_int(&res,j);
		}
	}
	return res;
}

void
gen_array_elemental(const char* dfunc_name, const char* first_name, const char* second_name, FILE* stream)
{
  		fprintf(stream, "%s(%s[] f_s, %s s_s){ \n",dfunc_name,first_name,second_name);
		fprintf(stream, "for i in 0:size(f_s) {\n");
  		fprintf(stream,"  %s(f_s[i],s_s)\n",dfunc_name);
		fprintf(stream,"}\n");
  		fprintf(stream,"}\n");
}

void
gen_array_elemental_second(const char* dfunc_name, const char* first_name, const char* second_name, FILE* stream)
{
  		fprintf(stream, "%s(%s f_s, %s[] s_s){ \n",dfunc_name,first_name,second_name);
		fprintf(stream, "for i in 0:size(s_s) {\n");
  		fprintf(stream,"  %s(f_s,s_s[i])\n",dfunc_name);
		fprintf(stream,"}\n");
  		fprintf(stream,"}\n");
}

void
gen_three_combinations_scalar_struct_scalar(const char* dfunc_name, const char* first_type, const int_vec second, const char* third_type, FILE* stream)
{
  for(size_t second_index = 0; second_index < second.size; ++second_index)
  {
  		const int f = second.data[second_index];
  		const char* second_name  = s_info.user_structs.data[f];
  		const string_vec second_members = s_info.user_struct_field_names[f];
  		fprintf(stream, "%s(%s s, %s f_s, %s t){ \n",dfunc_name,first_type,second_name,third_type);
  		for(size_t j = 0; j < second_members.size; ++j)
  		{
  			fprintf(stream,"  %s(s,f_s.%s,t)\n",dfunc_name,second_members.data[j]);
  		}
  		fprintf(stream,"}\n");
  }
}

void
gen_two_combinations_scalar_struct(const char* dfunc_name, const char* first_type, const int_vec second, FILE* stream)
{
  for(size_t second_index = 0; second_index < second.size; ++second_index)
  {
  		const int f = second.data[second_index];
  		const char* second_name  = s_info.user_structs.data[f];
  		const string_vec second_members = s_info.user_struct_field_names[f];
  		fprintf(stream, "%s(%s s, %s f_s){ \n",dfunc_name,first_type,second_name);
  		for(size_t j = 0; j < second_members.size; ++j)
  		{
  			fprintf(stream,"  %s(s,f_s.%s)\n",dfunc_name,second_members.data[j]);
  		}
  		fprintf(stream,"}\n");
  }
}

void
gen_two_combinations_struct_scalar(const char* dfunc_name, const int_vec first, const char* second_type, FILE* stream)
{
  for(size_t first_index = 0; first_index < first.size; ++first_index)
  {
  		const int f = first.data[first_index];
  		const char* first_name  = s_info.user_structs.data[f];
  		const string_vec first_members = s_info.user_struct_field_names[f];
  		fprintf(stream, "%s(%s f_s, %s s){ \n",dfunc_name,first_name,second_type);
  		for(size_t j = 0; j < first_members.size; ++j)
  		{
  			fprintf(stream,"  %s(f_s.%s,s)\n",dfunc_name,first_members.data[j]);
  		}
  		fprintf(stream,"}\n");
		gen_array_elemental(dfunc_name,first_name,second_type,stream);
  }
}

void
gen_two_combinations_elementals(const char* dfunc_name, const int_vec first, const int_vec second, FILE* stream)
{
  for(size_t first_index = 0; first_index < first.size; ++first_index)
  {
  	for(size_t second_index = 0; second_index < second.size; ++second_index)
  	{
  		const int f = first.data[first_index];
  		const int r = second.data[second_index];
  		const char* first_name  = s_info.user_structs.data[f];
  		const char* second_name = s_info.user_structs.data[r];
  		const string_vec first_members = s_info.user_struct_field_names[f];
  		const string_vec second_members = s_info.user_struct_field_names[r];
  		if(first_members.size != second_members.size) continue;
  		fprintf(stream, "%s(%s f_s, %s s_s){ \n",dfunc_name,first_name,second_name);
  		for(size_t j = 0; j < first_members.size; ++j)
  		{
  			fprintf(stream,"  %s(f_s.%s,s_s.%s)\n",dfunc_name,first_members.data[j],second_members.data[j]);
  		}
  		fprintf(stream,"}\n");
		gen_array_elemental(dfunc_name,first_name,second_name,stream);

  	}
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
	const bool is_returning = is_returning_function(node,dfunc_name);
	if(info.expr.size == 1 && info.types.data[0] == REAL_STR)
	{
		if(!is_returning) fatal("Only returning real functions covered!\n");
	        int_vec all_real_structs = get_all_acreal_structs();
		for(size_t l = 0; l < all_real_structs.size; ++l)
		{
			const int j = all_real_structs.data[l];
			const char* name  = s_info.user_structs.data[j];
			const size_t num_members = s_info.user_struct_field_names[j].size;
			fprintf(stream, "%s(%s s){ return real%zu(\n",dfunc_name,name,num_members);
			for(size_t f = 0; f < num_members; ++f)
			{
				fprintf(stream,"  %s(s.%s)%s\n",dfunc_name,s_info.user_struct_field_names[j].data[f],f < num_members-1 ? "," : "");
			}
			fprintf(stream,"  )\n}\n");
		}
		fprintf(stream,"inline %s(real[] arr){\nreal res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
		fprintf(stream,"inline %s(real3[] arr){\nreal3 res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
		free_int_vec(&all_real_structs);
	}

	else if(info.expr.size == 1 && info.types.data[0] == FIELD_STR)
	{
		if(intern(node->expr_type) == REAL_STR)
		{
			if(!is_returning) fatal("Only returning Field functions covered!\n");
			fprintf(stream,"inline %s(Field[] arr){\nreal res[size(arr)]\n for i in 0:size(arr)\n   res[i] = %s(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
			int_vec all_field_structs = get_all_field_structs();
			for(size_t l = 0; l < all_field_structs.size; ++l)
			{
				const int j = all_field_structs.data[l];
				const char* name = s_info.user_structs.data[j];
				const size_t num_members = s_info.user_struct_field_names[j].size;
				fprintf(stream, "%s(%s s){ return real%zu(\n",dfunc_name,name,num_members);
				for(size_t f = 0; f < num_members; ++f)
				{
					fprintf(stream,"  %s(s.%s)%s\n",dfunc_name,s_info.user_struct_field_names[j].data[f],f < num_members-1 ? "," : "");
				}
				fprintf(stream,"  )\n}\n");
			}
			fprintf(stream,"inline %s(Field3[] arr){\nreal3 res[size(arr)]\n for i in 0:size(arr)\n  res[i] = %s(arr[i])\nreturn res\n}\n",dfunc_name,dfunc_name);
			free_int_vec(&all_field_structs);
		}
		else if(intern(node->expr_type) == REAL3_STR)
		{
			fprintf(stream,"%s (Field3 v){return Matrix(%s(v.x), %s(v.y), %s(v.z))}\n",dfunc_name,dfunc_name,dfunc_name,dfunc_name);
		}
		else if(!is_returning)
		{
			fprintf(stream,"%s(Field[] arr){for i in 0:size(arr)\n %s(arr[i])\n}\n",dfunc_name,dfunc_name);
			int_vec all_field_structs = get_all_field_structs();
			for(size_t l = 0; l < all_field_structs.size; ++l)
			{
				const int j = all_field_structs.data[l];
				const char* name = s_info.user_structs.data[j];
				const size_t num_members = s_info.user_struct_field_names[j].size;
				fprintf(stream, "%s(%s s){\n",dfunc_name,name);
				for(size_t f = 0; f < num_members; ++f)
				{
					fprintf(stream,"  %s(s.%s)\n",dfunc_name,s_info.user_struct_field_names[j].data[f]);
				}
				fprintf(stream,"}\n");
			}
			fprintf(stream,"%s(Field3[] arr){for i in 0:size(arr)\n  %s(arr[i])\n}\n",dfunc_name,dfunc_name);
			free_int_vec(&all_field_structs);
		}
		else
			fatal("Missing elemental case for func: %s\nReturn type: %s\n",dfunc_name,node->expr_type);
	}
	else if(info.expr.size == 2 && info.types.data[0] == FIELD_STR && info.types.data[1] == REAL_STR)
	{
		if(is_returning) fatal("Only non-returning (Field,real) functions covered!\n");
		int_vec all_field_structs = get_all_field_structs();
		int_vec all_real_structs  = get_all_same_structs(REAL_STR);
		gen_two_combinations_elementals(dfunc_name,all_field_structs, all_real_structs, stream);
		gen_two_combinations_struct_scalar(dfunc_name,all_field_structs, REAL_STR,stream);
		gen_array_elemental(dfunc_name,FIELD_STR,REAL_STR,stream);
		free_int_vec(&all_field_structs);
		free_int_vec(&all_real_structs);

	}
	else if(info.expr.size == 2 && info.types.data[0] == FIELD_STR && info.types.data[1] == INT_STR)
	{
		if(is_returning) fatal("Only non-returning (Field,int) functions covered!\n");
		int_vec all_field_structs = get_all_field_structs();
		int_vec all_int_structs  = get_all_same_structs(INT_STR);
		gen_two_combinations_elementals(dfunc_name,all_field_structs, all_int_structs,stream);
		gen_two_combinations_struct_scalar(dfunc_name,all_field_structs, INT_STR,stream);
		gen_array_elemental(dfunc_name,FIELD_STR,INT_STR,stream);
		free_int_vec(&all_field_structs);
		free_int_vec(&all_int_structs);

	}

	else if(info.expr.size == 2 && info.types.data[0] == intern("AcBoundary") && info.types.data[1] == FIELD_STR)
	{
		int_vec all_field_structs = get_all_field_structs();
		gen_two_combinations_scalar_struct(dfunc_name,intern("AcBoundary"),all_field_structs,stream);
		gen_array_elemental_second(dfunc_name,intern("AcBoundary"),FIELD_STR,stream);
		free_int_vec(&all_field_structs);
	}

	else if(info.expr.size == 3 && info.types.data[0] == intern("AcBoundary") && info.types.data[1] == FIELD_STR && info.types.data[2] == INT_STR)
	{
		int_vec all_field_structs = get_all_field_structs();
		gen_three_combinations_scalar_struct_scalar(dfunc_name,intern("AcBoundary"),all_field_structs,INT_STR,stream);
		free_int_vec(&all_field_structs);
	}
	free_func_params_info(&info);
}

void
mark_first_declarations_in_funcs(ASTNode* node, string_vec* names)
{
	if(node->lhs)
		mark_first_declarations_in_funcs(node->lhs,names);
	if(node->type & NODE_DECLARATION)
	{
		const char* var_name = get_node_by_token(IDENTIFIER,node)->buffer;
		if(!check_symbol(NODE_ANY,var_name,0,0) && !str_vec_contains(*names,var_name))
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

	push(&primitive_datatypes,intern("size_t"));

	VALUE_STR = intern("value");
	OUTPUT_VALUE_STR = intern("output_value");

	COMPLEX_STR= intern("AcComplex");
	REAL3_STR= intern("AcReal3");
	REAL_ARR_STR = intern("AcRealArray");
	INTRINSIC_STR = intern("intrinsic");

	REAL_PTR_STR = intern("AcReal*");
	BOOL_PTR_STR = intern("bool*");
	REAL3_PTR_STR = intern("AcReal3*");
	FIELD3_PTR_STR = intern("Field3*");
	VTXBUF_PTR_STR = intern("VertexBufferHandle*");
	FIELD_PTR_STR = intern("Field*");

	MATRIX_STR = intern("AcMatrix");
	TENSOR_STR = intern("AcTensor");
	INT3_STR = intern("int3");
	EQ_STR = intern("=");

	DOT_STR = intern("dot");

	LESS_STR = intern("<");
	GREATER_STR = intern(">");
	LEQ_STR = intern("<=");
	GEQ_STR = intern(">=");

	MEQ_STR= intern("*=");
	AEQ_STR= intern("+=");
	MODEQ_STR = intern("%=");
	MINUSEQ_STR= intern("-=");
	DEQ_STR= intern("/=");
	PERIODIC = intern("periodic");
	
	CHAR_PTR_STR = intern("char*");

	EMPTY_STR = intern("\0");
	DEAD_STR = intern("dead");
	INLINE_STR = intern("inline");
	UTILITY_STR = intern("utility");
	BOUNDCOND_STR = intern("boundary_condition");
	FIXED_BOUNDARY_STR = intern("fixed_boundary");
	RAYTRACE_STR = intern("Raytrace");
	ELEMENTAL_STR = intern("elemental");
	AUXILIARY_STR = intern("auxiliary");
	COMMUNICATED_STR = intern("communicated");
	DEVICE_ONLY_STR = intern("device_only");
	DIMS_STR = intern("dims");
	HALO_STR = intern("halo");
	FIELD_ORDER_STR = intern("field_order");
	CONST_STR  = intern("const");
	DCONST_STR = intern("dconst");
	CONSTEXPR_STR = intern("constexpr");
	GLOBAL_MEM_STR  = intern("gmem");
	DYNAMIC_STR  = intern("dynamic");
	OUTPUT_STR  = intern("output");
	GLOBAL_STR  = intern("global");
	INPUT_STR  = intern("input");
	RUN_CONST_STR = intern("run_const");
	CONST_DIMS_STR= intern("const_dims");
	FIELD_STR = intern("Field");
	STENCIL_STR = intern("Stencil");
	KERNEL_STR = intern("Kernel");
	FIELD3_STR = intern("Field3");
	FIELD4_STR = intern("Field4");
	PROFILE_STR = intern("Profile");
	COMPLEX_FIELD_STR = intern("ComplexField");

	MULT_STR = intern("*");
	PLUS_STR = intern("+");
	MINUS_STR = intern("-");
	DIV_STR = intern("/");
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
	bool is_output = false;
	if(tquals)
	{
		node_vec tquals_vec = get_nodes_in_list(tquals);
		for(size_t i = 0; i < tquals_vec.size; ++i)
		{
			const ASTNode* tqual = get_node(NODE_TQUAL,tquals_vec.data[i]);
			if(tqual->lhs && tqual->lhs->buffer && tqual->lhs->buffer == intern("output"))
				is_output = true;
		}
		free_node_vec(&tquals_vec);
	}
        if(!is_allocating_type(type) && !is_output) 
	{
		return res;
	}
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
                                        create_type_declaration_with_qualifiers(tquals,
						is_output ? intern(remove_substring(strdup(type),"*")) : type),
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


                ASTNode* type_declaration = create_type_declaration("const",
			is_output ? sprintf_intern("%sOutputParam*", intern(remove_substring(strdup(type),"*"))) : sprintf_intern("%s*",type)
		);

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
                //      node->parent->rhs = res_node;
                //else
                //      node->parent->lhs = res_node;
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
                //new lhs
                astnode_create(NODE_DECLARATION | NODE_GLOBAL, astnode_dup(node->lhs,NULL),astnode_dup(elems,NULL));
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
void gen_boundcond_kernels(const ASTNode* root_in, FILE* stream)
{
    ASTNode* root = astnode_dup(root_in,NULL);
          symboltable_reset();
        traverse(root, 0, NULL);
    s_info = read_user_structs(root);
    e_info = read_user_enums(root);
    expand_allocating_types(root);
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
	char* op = strdup(node->rhs->lhs->buffer);
	if(strcmps(op,MEQ_STR,MINUSEQ_STR,AEQ_STR,DEQ_STR,MODEQ_STR))   return;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return;
	ASTNode* assign_expression = node->rhs->rhs->lhs;
	remove_substring(op,EQ_STR);
	ASTNode* binary_expression = create_binary_expression(node->lhs, assign_expression, op);
	ASTNode* assignment        = create_assignment(node->lhs, binary_expression, EQ_STR); 
	assignment->parent = node->parent;
	node->parent->lhs = assignment;
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
	const char* op = node->rhs->lhs->buffer;
	if(op != EQ_STR) return;
	if(count_num_of_nodes_in_list(node->rhs->rhs) != 1)   return;
	const char* lhs_type = get_expr_type(node->lhs);
	const char* rhs_type = get_expr_type(node->rhs);
	if(!lhs_type || !rhs_type) return;
	//TP: expression like complex c = 1.0 means the real component is 1.0 and the imaginary component is 0.0
	if(lhs_type == COMPLEX_STR) return;
	if(all_same_struct(lhs_type,REAL_STR) && rhs_type == REAL_STR)
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
void
get_used_vars_base(const ASTNode* node, string_vec* dst, bool skip, const char* assigned_var)
{
        if(node->token == VARIABLE_DECLARATION) return;
        if(node->type & NODE_ASSIGNMENT)
        {
                const char* var = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
                assigned_var = var;
        }
        if(node->lhs)
        {
                get_used_vars_base(node->lhs,dst,skip | (node->type & NODE_ASSIGNMENT),assigned_var);
        }
        if(node->rhs)
        {
                get_used_vars_base(node->rhs,dst,skip && !(node->type & NODE_ARRAY_ACCESS),assigned_var);
        }
        if(skip) return;
        if(node->token != IDENTIFIER) return;
        if(node->buffer == assigned_var) return;
        push(dst,node->buffer);
}
void
get_used_vars(const ASTNode* node, string_vec* dst)
{
        get_used_vars_base(node,dst,false,NULL);
}

void
remove_dead_assignments(ASTNode* node, const string_vec vars_used)
{
	TRAVERSE_PREAMBLE_PARAMS(remove_dead_assignments,vars_used);
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const char* expr_type = get_expr_type(node);
	if(!expr_type) return;
	if(expr_type == FIELD_STR) return;
	const char* var = get_node_by_token(IDENTIFIER,node->lhs)->buffer;
	if(check_symbol(NODE_ANY,var,0,DYNAMIC_STR)) return;
	if(strstr(expr_type,"*")) return;
	if(!str_vec_contains(vars_used,var))
	{
		node->lhs = NULL;
		node->rhs = NULL;
		node->parent->postfix = NULL;
	}
}
void
remove_dead_declarations(ASTNode* node, const string_vec vars_used)
{
	TRAVERSE_PREAMBLE_PARAMS(remove_dead_declarations,vars_used);
	if(!(node->type & NODE_DECLARATION)) return;
	if(node->parent->token != VARIABLE_DECLARATION) return;
	const char* var = get_node_by_token(IDENTIFIER,node)->buffer;
	if(!str_vec_contains(vars_used,var))
	{
		node->lhs = NULL;
		node->rhs = NULL;
		node->parent->postfix = NULL;
	}
}
void
remove_dead_writes(ASTNode* node)
{
	TRAVERSE_PREAMBLE(remove_dead_writes);
	if(!(node->type & NODE_FUNCTION)) return;
	if((node->type & NODE_DFUNCTION)) return;
	const ASTNode* statements_node = node->rhs->rhs->lhs;
	if(!statements_node) return;
	node_vec statements = get_nodes_in_list(statements_node);
	string_vec vars_used = VEC_INITIALIZER;
	for(int i = (int)statements.size-1; i >= 0; --i)
	{
		get_used_vars(statements.data[i],&vars_used);
		remove_dead_assignments((ASTNode*)statements.data[i],vars_used);
	}
	remove_dead_declarations(node,vars_used);
	free_str_vec(&vars_used);
	free_node_vec(&statements);
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
check_for_illegal_writes_in_func(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(check_for_illegal_writes_in_func);
	if(!(node->type & NODE_ASSIGNMENT)) return;
	const ASTNode* id = get_node_by_token(IDENTIFIER,node);
	if(!id) return;
	if(check_symbol(NODE_ANY,id->buffer,0,DCONST_STR))
		fatal("Write to dconst variable: %s\n",combine_all_new(node));
	if(check_symbol(NODE_ANY,id->buffer,0,CONST_STR))
	{
		if(check_symbol(NODE_ANY,id->buffer,FIELD_STR,CONST_STR)) return;
		if(check_symbol(NODE_ANY,id->buffer,FIELD3_STR,CONST_STR)) return;
		if(check_symbol(NODE_ANY,id->buffer,FIELD4_STR,CONST_STR)) return;
		if(check_symbol(NODE_ANY,id->buffer,FIELD_PTR_STR,CONST_STR)) return;
		fatal("Write to const variable: %s\n",combine_all_new(node));
	}
	if(check_symbol(NODE_ANY,id->buffer,0,RUN_CONST_STR))
		fatal("Write to run_const variable: %s\n",combine_all_new(node));
	if(check_symbol(NODE_FUNCTION_ID,id->buffer,0,NULL))
		fatal("Write to function: %s\n",combine_all_new(node));
}

void
check_for_illegal_writes(const ASTNode* node)
{
	TRAVERSE_PREAMBLE(check_for_illegal_writes);
	if(!(node->type & NODE_FUNCTION)) return;
	check_for_illegal_writes_in_func(node);
}
void
check_for_illegal_func_calls_in_func(const ASTNode* node, const char* name)
{
	TRAVERSE_PREAMBLE_PARAMS(check_for_illegal_func_calls_in_func,name);
	if(!(node->type & NODE_FUNCTION_CALL)) return;
	const char* func_name = get_node_by_token(IDENTIFIER,node)->buffer;
        if(check_symbol(NODE_FUNCTION_ID,func_name,KERNEL_STR,0))
	{
		fatal("Can not call Kernel in DSL; Incorrect call in %s: %s\n",name,combine_all_new(node));
	}
}
void
check_for_illegal_func_calls(const ASTNode* node)
{
	if(node->type & NODE_TASKGRAPH_DEF) return;
	if(node->type & NODE_BOUNDCONDS_DEF) return;
	TRAVERSE_PREAMBLE(check_for_illegal_func_calls);
	if(!(node->type & NODE_FUNCTION)) return;
	check_for_illegal_func_calls_in_func(node,get_node_by_token(IDENTIFIER,node)->buffer);
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

const char*
get_monomorphized_name(const char* base_name, const int index)
{
	return sprintf_intern("%s_MONOMORPHIZED_%d",base_name,index);
}
int
gen_monomorphized_kernel(const char* func_name, const string_vec input_params, ASTNode* tail_node)
{
	const int index = str_vec_get_index(kfunc_names,func_name);
	if(index == -1) fatal("No such kernel: %s\n",func_name);
	ASTNode* node = (ASTNode*)kfunc_nodes.data[index];

	func_params_info params_info = get_function_params_info(node, func_name);
	//TP: could but no need to monomorphize kernels with no input params, would simply slow things down
	if(params_info.expr_nodes.size == 0) return -1;

	ASTNode* new_node = astnode_dup(node,NULL);
	make_ids_unique(new_node);
	ASTNode* function_id = (ASTNode*) get_node(NODE_FUNCTION_ID,new_node->lhs);
	astnode_sprintf(function_id,get_monomorphized_name(function_id->buffer,monomorphization_index));

	if(input_params.size != params_info.types.size)
		fatal("Number of inputs (%zu) for %s does not match the number of input params (%zu)\n",input_params.size,func_name,params_info.types.size);
	for(size_t i = 0; i < params_info.expr.size; ++i)
		rename_variables(new_node,input_params.data[i],params_info.types.data[i],params_info.expr.data[i]);

	//TP: atm monomorphized kernels don't have input parameters
	new_node->rhs->lhs = NULL;

	ASTNode* head = astnode_create(NODE_UNKNOWN,NULL,new_node);
	append_to_tail_node(tail_node, head);
	//replace_node((ASTNode*)node,head);
	++monomorphization_index;
	//TP: returns the index that was used for the current monomorphization
	free_func_params_info(&params_info);
	return monomorphization_index-1;
}

void
monomorphize_kernel_calls(ASTNode* node, ASTNode* tail_node)
{
	TRAVERSE_PREAMBLE_PARAMS(monomorphize_kernel_calls,tail_node);
	if(node->type != NODE_TASKGRAPH_DEF)
		return;
	node_vec kernel_call_nodes = get_nodes_in_list(node->rhs);
	for(size_t i = 0; i < kernel_call_nodes.size; ++i)
	{
		ASTNode* func_call = (ASTNode*)kernel_call_nodes.data[i];
		const char* func_name = get_node_by_token(IDENTIFIER,func_call)->buffer;
		func_params_info params_info =  get_func_call_params_info(func_call);
		bool can_monomorphize = true;
		for(size_t j = 0; j < params_info.expr.size; ++j)
		{
			const char* type = get_expr_type((ASTNode*)params_info.expr_nodes.data[j]);
			can_monomorphize &= all_identifiers_are_constexpr(params_info.expr_nodes.data[j]) || type == FIELD_STR;
		}
		if(can_monomorphize)
		{
			const int res_index = gen_monomorphized_kernel(func_name,params_info.expr,tail_node);
			if(res_index == -1) continue;
			ASTNode* fn_identifier = (ASTNode*)get_node_by_token(IDENTIFIER,func_call);
			astnode_sprintf(fn_identifier,get_monomorphized_name(fn_identifier->buffer, res_index));
			//TP: all input params are monomorphized, so don't pass anything
			func_call->rhs = NULL;
		}
		free_func_params_info(&params_info);
	}
	free_node_vec(&kernel_call_nodes);
}
const char*
get_fused_res_name(const node_vec kernels)
{
	const char* res = get_node_by_token(IDENTIFIER,kernels.data[0])->buffer;
	for(size_t i = 1; i < kernels.size; ++i)
	{
		const char* name = get_node_by_token(IDENTIFIER,kernels.data[i])->buffer;
		res = sprintf_intern("%s_FUSED_%s",res,name);
	}
	return res;
}
bool can_fuse_kernels(const node_vec kernels)
{
	bool updated_fields[num_fields];
	bool stencil_used[num_fields];
	bool updated_profiles[num_profiles];
	memset(updated_fields,false,num_fields*sizeof(bool));
	memset(updated_profiles,false,num_profiles*sizeof(bool));
	memset(stencil_used,false,num_profiles*sizeof(bool));

	for(size_t i = 0; i < kernels.size; ++i)
	{
		const char* func_name = get_node_by_token(IDENTIFIER,kernels.data[i])->buffer;
		const int kernel_index = get_symbol_index(NODE_FUNCTION_ID, func_name, KERNEL_STR);
		if(kernel_index == -1) fatal("Did not find kernel: %s\n",func_name);

		//TP: can't fuse if Field F is updated before and then used in a succeeding stencil call
		for(size_t field_index = 0; field_index < num_fields; ++field_index)
			if(field_has_stencil_op[field_index + num_fields*kernel_index] && updated_fields[field_index]) return false;

		//TP: For now skip fusing kernels with previous called, since this in general quite unsafe
		for(size_t field_index = 0; field_index < num_fields; ++field_index)
			if(field_has_previous_call[field_index + num_fields*kernel_index]) return false;

		//TP: can't fuse if Profile P is updated before and then used in succeeding call
		for(size_t profile_index = 0; profile_index < num_profiles; ++profile_index)
			if(read_profiles[profile_index + num_profiles*kernel_index] && updated_profiles[profile_index]) return false;

		for(size_t field_index = 0; field_index < num_fields; ++field_index)
			updated_fields[field_index] |= written_fields[field_index + num_fields*kernel_index];

		for(size_t field_index = 0; field_index < num_fields; ++field_index)
			stencil_used[field_index] |= field_has_stencil_op[field_index + num_fields*kernel_index];

		for(size_t profile_index = 0; profile_index < num_profiles; ++profile_index)
			updated_profiles[profile_index] |= reduced_profiles[profile_index + num_profiles*kernel_index];
	}
	return true;
}
bool should_fuse_kernels(const node_vec kernels)
{
	bool fields_in_working_memory[num_fields];
	bool fields_written[num_fields];
	bool stencil_computed[num_fields];

	memset(fields_in_working_memory,false,num_fields*sizeof(bool));
	memset(fields_written,false,num_fields*sizeof(bool));
	memset(stencil_computed,false,num_fields*sizeof(bool));

	for(size_t i = 0; i < kernels.size; ++i)
	{
		const char* func_name = get_node_by_token(IDENTIFIER,kernels.data[i])->buffer;
		const int kernel_index = get_symbol_index(NODE_FUNCTION_ID, func_name, KERNEL_STR);
		if(kernel_index == -1) fatal("Did not find kernel: %s\n",func_name);

		for(size_t field_index = 0; field_index < num_fields; ++field_index)
		{
			//TP: we save gmem accesses since can skip reading the field from gmem
			if(fields_in_working_memory[field_index] && read_fields[field_index + num_fields*kernel_index]) return true;
			//TP: we save gmem accesses since can combine some output writes together
			if(fields_written[field_index] && written_fields[field_index + num_fields*kernel_index]) return true;
		}

		for(size_t field_index = 0; field_index < num_fields; ++field_index)
		{
			fields_in_working_memory[field_index] |= written_fields[field_index + num_fields*kernel_index];
			fields_in_working_memory[field_index] |= read_fields[field_index + num_fields*kernel_index];
			//TP: it is not required to check the stencil offsets since even if for example is reading one to the left
			//and later reads at the current point there is reuse since the current point of the left vertex is the left of the current vertex thus is likely to be in the cache
			fields_in_working_memory[field_index] |= field_has_stencil_op[field_index + num_fields*kernel_index];

			fields_written[field_index] |= written_fields[field_index + num_fields*kernel_index];
		}
	}
	return true;
}

ASTNode*
fuse_kernels(const node_vec kernels)
{
	if(!has_optimization_info()) return NULL;
	node_vec bodies = VEC_INITIALIZER;
	//TP: have to be in reverse order that bodies before come before
	for(int i = kernels.size-1; i >= 0; --i)
		push_node(&bodies,kernels.data[i]->rhs->rhs->lhs);

	ASTNode* combined_body = build_list_node(bodies,"");

	ASTNode* compound_statement = astnode_create(NODE_BEGIN_SCOPE,combined_body,NULL);
	astnode_set_prefix("{",compound_statement);
	astnode_set_postfix("}",compound_statement);

	const char* res_name = get_fused_res_name(kernels);
	ASTNode* decl = create_declaration(res_name,KERNEL_STR,NULL);
        set_identifier_type(NODE_FUNCTION_ID, decl);

	ASTNode* function_body = astnode_create(NODE_BEGIN_SCOPE,NULL,compound_statement);
	astnode_set_prefix("(",function_body);
	astnode_set_infix(")",function_body);
	ASTNode* func_def = astnode_create(NODE_KFUNCTION,decl,function_body);
	func_def->type |= NODE_KFUNCTION;
        const char* default_param_list=  "(const int3 start, const int3 end, DeviceVertexBufferArray vba";
        astnode_set_prefix(default_param_list, func_def->rhs);
	astnode_set_prefix("__global__ void \n#if MAX_THREADS_PER_BLOCK\n__launch_bounds__(MAX_THREADS_PER_BLOCK)\n#endif\n",func_def);
	return func_def;
}
void
fuse_kernel_calls(const node_vec calls, ASTNode* tail_node, string_vec* generated_names)
{
	if(calls.size <= 1) return;
	node_vec kernels = VEC_INITIALIZER;
	for(size_t i = 0; i< calls.size; ++i)
	{
		const ASTNode* call = calls.data[i];
		const char* func_name = get_node_by_token(IDENTIFIER,call)->buffer;
		const int index = str_vec_get_index(kfunc_names,func_name);
		push_node(&kernels,kfunc_nodes.data[index]);
		
	}
	//TP: it can be possible that in different combination branches the same kernel is trying to be generated, simply skip if already generated
	const char* res_name = get_fused_res_name(kernels);
	if(str_vec_contains(*generated_names,res_name)) return;
	push(generated_names,res_name);

	const bool fuse = can_fuse_kernels(kernels) && should_fuse_kernels(kernels);
	if(!fuse)
	{
		free_node_vec(&kernels);
		return;
	}
	ASTNode* fused_kernel = fuse_kernels(kernels);

	ASTNode* new_node = astnode_create(NODE_UNKNOWN,NULL,fused_kernel);
	append_to_tail_node(tail_node,new_node);

	free_node_vec(&kernels);
}
void
gen_all_fused_combinations(const node_vec input, const int index, const node_vec prev, ASTNode* tail_node, string_vec* generated_names)
{
	if(index == (int)input.size)
	{
		fuse_kernel_calls(prev,tail_node,generated_names);
		return;
	}
	node_vec lhs = VEC_INITIALIZER;
	node_vec rhs = VEC_INITIALIZER;
	for(size_t i = 0; i < prev.size; ++i)
	{
		push_node(&lhs,prev.data[i]);
		push_node(&rhs,prev.data[i]);
	}	
	push_node(&rhs,input.data[index]);

	gen_all_fused_combinations(input,index+1,lhs,tail_node,generated_names);
	gen_all_fused_combinations(input,index+1,rhs,tail_node,generated_names);

	free_node_vec(&lhs);
	free_node_vec(&rhs);
}
void
fuse_computesteps_calls(ASTNode* node, ASTNode* tail_node)
{
	TRAVERSE_PREAMBLE_PARAMS(fuse_computesteps_calls,tail_node);
	if(node->type != NODE_TASKGRAPH_DEF) return;
	node_vec kernel_call_nodes = get_nodes_in_list(node->rhs);
	node_vec no_param_calls = VEC_INITIALIZER;
	for(size_t i= 0; i < kernel_call_nodes.size; ++i)
	{
		func_params_info params_info =  get_func_call_params_info(kernel_call_nodes.data[i]);
		free_func_params_info(&params_info);
		if(params_info.expr.size == 0) push_node(&no_param_calls,kernel_call_nodes.data[i]);
	}
	free_node_vec(&kernel_call_nodes);
	node_vec empty_node = VEC_INITIALIZER;
	string_vec generated_names = VEC_INITIALIZER;
	gen_all_fused_combinations(no_param_calls,0,empty_node,tail_node,&generated_names);
}
void
gen_fused_kernels(ASTNode* root)
{
	if(!has_optimization_info() && kfunc_nodes.size == 0) return;
	fuse_computesteps_calls(root, get_tail_node(root));
}

void
preprocess(ASTNode* root, const bool optimize_input_params)
{
  replace_const_ints(root,const_int_values,const_ints);
  memset(&kfunc_nodes,0,sizeof(kfunc_nodes));
  memset(&kfunc_names,0,sizeof(kfunc_names));
  get_nodes(root,&kfunc_nodes,&kfunc_names,NODE_KFUNCTION);

  process_overrides(root);
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  expand_allocating_types(root);

  remove_extra_braces_in_arr_initializers(root);
  symboltable_reset();
  rename_scoped_variables(root,NULL,NULL);
  symboltable_reset();
  free_node_vec(&dfunc_nodes);
  free_str_vec(&dfunc_names);
  canonalize(root);


  mark_kernel_inputs(root);

  traverse_base_params params;
  memset(&params,0,sizeof(params));
  params.do_checks = true;
  traverse_base(root, 0, NULL, params);

  check_for_illegal_writes(root);
  check_for_illegal_func_calls(root);
  //We use duplicate dfuncs from gen_boundcond_kernels
  //duplicate_dfuncs = get_duplicate_dfuncs(root);
  mark_first_declarations(root);

  gen_overloads(root);
  //eval_conditionals(root,root);
  transform_broadcast_assignments(root);
  free_structs_info(&s_info);
  symboltable_reset();

  traverse_base(root, 0, NULL, params);

  gen_kernel_ifs(root,optimize_input_params);
  if(monomorphization_index == 0)
  {
  	traverse(root, 0, NULL);
  	gen_kernel_combinatorial_optimizations_and_input(root,optimize_input_params);
  	monomorphize_kernel_calls(root,get_tail_node(root));
	memset(&kfunc_nodes,0,sizeof(kfunc_nodes));
	memset(&kfunc_names,0,sizeof(kfunc_names));
  	get_nodes(root,&kfunc_nodes,&kfunc_names,NODE_KFUNCTION);
	symboltable_reset();
  }

  traverse(root, 0, NULL);
  gen_calling_info(root);
}
void
gen_kfunc_info(const ASTNode* root)
{
  memset(&kfunc_nodes,0,sizeof(kfunc_nodes));
  memset(&kfunc_names,0,sizeof(kfunc_names));
  get_nodes(root,&kfunc_nodes,&kfunc_names,NODE_KFUNCTION);
  gen_calling_info(root);
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

  	process_overrides(root);
	symboltable_reset();
	rename_scoped_variables(root,NULL,NULL);
	symboltable_reset();
	{
  		traverse_base_params params;
  		memset(&params,0,sizeof(params));
  		params.do_checks = true;
  		traverse_base(root, 0, NULL, params);
		gen_ray_names();
	}
	MAX_DFUNCS = count_symbols_type(NODE_DFUNCTION_ID);
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
  //const size_t num_profiles = count_symbols(PROFILE);

  // Device constants
  // gen_dconsts(root, stream);

  // Stencils

  // Stencil generator
  FILE* stencilgen = fopen(STENCILGEN_HEADER, "w");
  assert(stencilgen);

  // Stencil ops

  { // Unary 
    fprintf(stencilgen, "static const char* "
                        "stencil_unary_ops[NUM_STENCILS] = {");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
      const Symbol symbol = symbol_table[i];
       if(symbol.tspecifier == STENCIL_STR) {
	      if(symbol.tqualifiers.size == 1 && symbol.tqualifiers.data[0] == intern("exp_sum"))
	      {
        	fprintf(stencilgen, "\"exp\",");
	      }
	      else
        	fprintf(stencilgen, "\"val\",");
      }
    }
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
		if(symbol.tqualifiers.data[0] == intern("exp_sum"))
		{
        		fprintf(stencilgen, "\"sum\",");
		}
		else
		{
        		fprintf(stencilgen, "\"%s\",",symbol.tqualifiers.data[0]);
		}
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

void
gen_analysis_stencils(FILE* stream)
{
  string_vec stencil_names = get_names(STENCIL_STR);
  for (size_t i = 0; i < stencil_names.size; ++i)
  {
    fprintf(stream,"AcReal %s(const Field& field_in)"
           "{stencils_accessed[field_in][stencil_%s] |= (1 | AC_STENCIL_CALL);return AcReal(1.0);};\n",
           stencil_names.data[i], stencil_names.data[i]);
    fprintf(stream,"AcReal %s_profile(const Profile& profile_in)"
           "{stencils_accessed[NUM_ALL_FIELDS+profile_in][stencil_%s] |= (1 | AC_STENCIL_CALL);return AcReal(1.0);};\n",
           stencil_names.data[i], stencil_names.data[i]);
  }
  free_str_vec(&stencil_names);

  string_vec ray_names = get_names(RAYTRACE_STR);
  for (size_t i = 0; i < ray_names.size; ++i)
  {
    fprintf(stream,"AcReal incoming_%s(const Field& field_in)"
           "{incoming_ray_value_accessed[field_in][ray_%s] |= 1;return AcReal(1.0);};\n",
           ray_names.data[i], ray_names.data[i]);
    fprintf(stream,"AcReal outgoing_%s(const Field& field_in)"
           "{outgoing_ray_value_accessed[field_in][ray_%s] |= 1;return AcReal(1.0);};\n",
           ray_names.data[i], ray_names.data[i]);
  }
  free_str_vec(&ray_names);
}

//These are the same for mem_accesses pass and normal pass
void
gen_output_files(ASTNode* root)
{


  //TP: Get number of run_const variable by skipping overrides
  {
  	traverse_base_params params;
  	memset(&params,0,sizeof(traverse_base_params));
  	params.return_on  = NODE_ASSIGN_LIST;
  	traverse_base(root, NODE_ASSIGN_LIST, NULL, params);
  }
  num_profiles = count_profiles();
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  symboltable_reset();


  traverse(root, 0, NULL);
  process_overrides(root);

  {
  	FILE* fp = fopen("user_typedefs.h","w");
	fprintf(fp,"#include \"func_attributes.h\"\n");
	fclose(fp);
  }
  gen_user_enums();
  gen_user_structs();
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

  FILE* analysis_stencils = fopen("analysis_stencils.h", "w");
  gen_analysis_stencils(analysis_stencils);
  fclose(analysis_stencils);


}
bool
eliminate_conditionals_base(ASTNode* node, const bool gen_mem_accesses)
{
	bool res = false;
	if(node->lhs)
		res |= eliminate_conditionals_base(node->lhs,gen_mem_accesses);
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
			if(!gen_mem_accesses)
			{
				ASTNode* if_scope = (ASTNode*)get_node(NODE_BEGIN_SCOPE,node);
				if_scope->prefix = NULL;
				if_scope->postfix= NULL;
				if_scope->type = NODE_UNKNOWN;
			}
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
				//TP: take out potential else ifs
				statement->rhs = NULL;
				statement->lhs = else_node->rhs;
				else_node->rhs->parent = statement;
				if(!gen_mem_accesses)
				{
					ASTNode* else_scope = (ASTNode*)get_node(NODE_BEGIN_SCOPE,else_node->rhs);
					if(else_scope)
					{
						else_scope->prefix= NULL;
						else_scope->postfix = NULL;
						else_scope->type = NODE_UNKNOWN;
					}
				}
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
		res |= eliminate_conditionals_base(node->rhs,gen_mem_accesses);
	return res;

}
bool
eliminate_conditionals(ASTNode* node, const bool gen_mem_accesses)
{
	bool process = true;
	bool eliminated_something = false;
	while(process)
	{
		const bool eliminated_something_this_round = eliminate_conditionals_base(node,gen_mem_accesses);
		process = eliminated_something_this_round;
		eliminated_something = eliminated_something || eliminated_something_this_round;
	}
	return eliminated_something;
}


void
clean_stream(FILE* stream)
{
	if(freopen(NULL,"w",stream) == NULL) fatal("Was not able to clean stream!\n");
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
print_nested_ints(FILE* fp, size_t x, size_t y, size_t z, size_t dims, const int val)
{
    for (size_t i = 0; i < x; ++i)
    {
      if(dims >= 3) fprintf(fp,"{");
      for (size_t j = 0; j < y; ++j)
      {
        if(dims >= 2) fprintf(fp,"{");
        for (size_t k = 0; k < z; ++k)
          fprintf(fp, "%d,",val);
      	if(dims >= 2) fprintf(fp,"}%s",j+1 != y ? "," : "");
      }
      if(dims >= 3) fprintf(fp,"}%s",i+1 != x ? "," : "");
    }
}

void
print_nested_ones(FILE* fp, size_t x, size_t y, size_t z, size_t dims)
{
	print_nested_ints(fp,x,y,z,dims,1);
}
void
gen_stencils(const bool gen_mem_accesses, FILE* stream)
{
  const int AC_STENCIL_CALL = (1 << 2);
  const size_t num_stencils = count_symbols(STENCIL_STR);
  const size_t num_rays     = count_symbols(RAYTRACE_STR);
  if (gen_mem_accesses || !OPTIMIZE_MEM_ACCESSES) {
    FILE* tmp = fopen("stencil_accesses.h", "w+");
    assert(tmp);

    fprintf(tmp,
            "static int "
            "previous_accessed [NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_fields+num_profiles,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "incoming_ray_value_accessed [NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_RAYS+1] __attribute__((unused)) = {");
    print_nested_ints(tmp,num_kernels,num_fields+num_profiles,num_rays+1,3,0);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "outgoing_ray_value_accessed [NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_RAYS+1] __attribute__((unused)) = {");
    print_nested_ints(tmp,num_kernels,num_fields+num_profiles,num_rays+1,3,0);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "stencils_accessed [NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS] __attribute__((unused)) = {");
    print_nested_ints(tmp,num_kernels,num_fields+num_profiles,num_stencils,3,AC_STENCIL_CALL);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "write_called [NUM_KERNELS][NUM_ALL_FIELDS] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_fields,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "write_complex_called [NUM_KERNELS][NUM_COMPLEX_FIELDS+1] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_complex_fields+1,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "value_complex_called [NUM_KERNELS][NUM_COMPLEX_FIELDS+1] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_complex_fields+1,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "write_called_profile [NUM_KERNELS][NUM_PROFILES+1] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_profiles+1,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "reduced_profiles [NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_profiles,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
            "static int "
            "read_profiles [NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,num_profiles,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
	    "static int "
	    "reduced_reals [NUM_KERNELS][NUM_REAL_OUTPUTS+1] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,count_variables(REAL_STR,OUTPUT_STR)+1,2);
    fprintf(tmp, "};\n");

    fprintf(tmp,
	    "static int "
	    "reduced_ints [NUM_KERNELS][NUM_INT_OUTPUTS+1] __attribute__((unused)) = {");
    print_nested_ones(tmp,1,num_kernels,count_variables(INT_STR,OUTPUT_STR)+1,2);
    fprintf(tmp, "};\n");
    if(AC_DOUBLE_PRECISION)
    {
    	fprintf(tmp,
    	        "static int "
    	        "reduced_floats[NUM_KERNELS][NUM_FLOAT_OUTPUTS+1] __attribute__((unused)) = {");
    	print_nested_ones(tmp,1,num_kernels,count_variables(FLOAT_STR,OUTPUT_STR)+1,2);
    	fprintf(tmp, "};\n");
    }

    fprintf(tmp, "const bool has_mem_access_info __attribute__((unused)) = false;\n");
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
           "gcc -Wfatal-errors -Wall -Wextra -Wdouble-promotion "
           "-DIMPLEMENTATION=%d "
           "-DMAX_THREADS_PER_BLOCK=%d "
           "-Wfloat-conversion -Wshadow -I. %s -lm "
	   "-DAC_USE_HIP=%d "
	   "-DAC_DOUBLE_PRECISION=%d "
	   "-DBUFFERED_REDUCTIONS=%d "
           "-o %s",
           IMPLEMENTATION, MAX_THREADS_PER_BLOCK, STENCILGEN_SRC,HIP_ON,AC_DOUBLE_PRECISION,
	   BUFFERED_REDUCTIONS,
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
get_dfunc_strs(const ASTNode* root, const traverse_base_params params)
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
  		traverse_base(dfunc_heads[i],
  	           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_KFUNCTION |
  	               NODE_HOSTDEFINE | NODE_NO_OUT,
  	           dfunc_fps[i],params);
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
			const char* type = get_expr_type((ASTNode*)cached_calls.data[i]) ? sprintf_intern("%s%s",
					get_expr_type((ASTNode*)cached_calls.data[i]), should_be_reference(cached_calls.data[i]) ? "&" : "")
				: NULL;
			ASTNode* res = create_assignment(
								create_declaration(get_cached_var_name(i),type,CONST_STR),
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
void replace_write_calls(ASTNode* node, const ASTNode* decl_node)
{
	TRAVERSE_PREAMBLE_PARAMS(replace_return_nodes,decl_node);
	if(!is_return_node(node)) return;
	replace_node(node,create_assignment(decl_node,node->rhs,EQ_STR));
}


void
check_uniquenes(const ASTNode* root, const NodeType type, const char* message_name)
{
	node_vec nodes = VEC_INITIALIZER;
	string_vec names = VEC_INITIALIZER;
	get_nodes(root,&nodes,&names,type);
	for(size_t i = 0; i < names.size; ++i)
	{
		for(size_t j = 0; j < names.size; ++j)
		{
			if(i == j) continue;
			if(names.data[i] == names.data[j])
				fatal("multiple definitions of %s: %s\n",message_name,names.data[i]);

		}
	}


}

void
add_tracing_to_conditionals(ASTNode* node)
{
	TRAVERSE_PREAMBLE(add_tracing_to_conditionals);
	if(!(node->type & NODE_IF)) return;
	if(!node->rhs) return;
	if(!node->rhs->lhs) return;
	if(node->rhs->lhs->type & NODE_BEGIN_SCOPE)
		astnode_sprintf_prefix(node->rhs->lhs,"{executed_nodes.push_back(%d);",node->id);
}

void
generate(const ASTNode* root_in, FILE* stream, const bool gen_mem_accesses)
{ 
  symboltable_reset();
  ASTNode* root = astnode_dup(root_in,NULL);
  get_field_order(root);
  check_uniquenes(root,NODE_DFUNCTION,"function");
  check_uniquenes(root,NODE_STENCIL,"stencil");
  s_info = read_user_structs(root);
  e_info = read_user_enums(root);
  gen_type_info(root);

  if(gen_mem_accesses) add_tracing_to_conditionals(root);
  gen_constexpr_info(root,gen_mem_accesses);
  if(gen_mem_accesses)
  {
  	//gen_ssa_in_basic_blocks(root);
	//remove_dead_writes(root);
  }

  {
  	traverse_base_params params;
  	memset(&params,0,sizeof(params));
  	params.do_checks = true;
  	traverse_base(root, 0, NULL, params);
  }
  gen_reduce_info(root);
  gen_kernel_reduce_outputs();

  num_profiles = count_profiles();
  check_global_array_dimensions(root);

  gen_multidimensional_field_accesses_recursive(root,gen_mem_accesses,get_field_dims(root));
  gen_profile_reads(root,gen_mem_accesses);


  // Fill the symbol table
  generate_error_messages();


  // print_symbol_table();

  // Generate user_kernels.h
  fprintf(stream, "#pragma once\n");





  // Device constants
  // gen_dconsts(root, stream);
  traverse(root, NODE_NO_OUT, NULL);
  {
          FILE* fp = fopen("fields_info.h","w");
          gen_field_info(fp);

	  string_vec field_halos = get_field_halos(root);
          fprintf(fp,"static const AcInt3Param vtxbuf_halos[NUM_ALL_FIELDS] = {");
	  for(size_t i = 0; i < field_halos.size; ++i)
	  {
		  fprintf(fp,"%s,",field_halos.data[i]);
	  }
          fprintf(fp,"};");

	  string_vec field_dims = get_field_dims(root);
          fprintf(fp,"static const AcInt3Param vtxbuf_dims[NUM_ALL_FIELDS] = {");
	  for(size_t i = 0; i < field_dims.size; ++i)
	  {
		  fprintf(fp,"%s,",field_dims.data[i]);
	  }
          fprintf(fp,"};");


          fprintf(fp,"static const char* vtxbuf_dims_str[NUM_ALL_FIELDS] __attribute__((unused)) = {");
	  for(size_t i = 0; i < field_dims.size; ++i)
		  fprintf(fp,"\"%s\",",field_dims.data[i]);
          fprintf(fp,"};");

          fclose(fp);

          fp = fopen("device_fields_info.h","w");
          fprintf(fp,"static const __device__ AcInt3Param vtxbuf_device_dims[NUM_ALL_FIELDS] = {");
	  for(size_t i = 0; i < field_dims.size; ++i)
	  {
		  fprintf(fp,"%s,",field_dims.data[i]);
	  }
          fprintf(fp,"};");

          fprintf(fp,"static const __device__ AcInt3Param vtxbuf_device_halos[NUM_ALL_FIELDS] = {");
	  for(size_t i = 0; i < field_halos.size; ++i)
	  {
		  fprintf(fp,"%s,",field_halos.data[i]);
	  }
          fprintf(fp,"};");

	  free_str_vec(&field_halos);
	  free_str_vec(&field_dims);
	  fclose(fp);


	  symboltable_reset();
  	  traverse(root, NODE_NO_OUT, NULL);
	  string_vec datatypes = get_all_datatypes();

  	  FILE* fp_info = fopen("array_info.h","w");
  	  fprintf(fp_info,"\n #ifdef __cplusplus\n");
  	  fprintf(fp_info,"\n#include <array>\n");
  	  fprintf(fp_info,"typedef struct {int base; const char* member; bool from_config;} AcArrayLen;\n");
  	  fprintf(fp_info,"typedef struct { bool is_dconst; int num_dims; std::array<AcArrayLen,%d> dims; const char* name; bool is_alive; bool is_accessed;} array_info;\n",MAX_ARRAY_RANK);
	  gen_array_qualifiers(root);
  	  for (size_t i = 0; i < datatypes.size; ++i)
	  {
  	  	  gen_array_info(fp_info,datatypes.data[i],root);
	  }
  	  fprintf(fp_info,"\n #endif\n");
  	  fclose(fp_info);

	  //TP: !IMPORTANT! gen_array_info will temporarily update the nodes to push DEAD_STR type qualifiers to dead gmem arrays.
	  //This info is used in gen_gmem_array_declarations so they should be called after each other, maybe will simply combine them into a single function
  	  for (size_t i = 0; i < datatypes.size; ++i)
	  	gen_gmem_array_declarations(datatypes.data[i],root);
	  if(gen_mem_accesses)
	  {

		copy_file("gmem_arrays_decl.h","cpu_gmem_arrays_decl.h");
	  }
  }
  gen_array_qualifiers(root);

  traverse(root,NODE_NO_OUT,NULL);
  string_vec* stencils_called = gen_stencil_calling_info(root);
  write_calling_info_for_stencilgen(stencils_called);

  // Stencils

  // Stencil generator

  // Compile
  gen_stencils(gen_mem_accesses,stream);
  check_for_undeclared_functions(root,root);

  traverse(root,NODE_NO_OUT,NULL);
  gen_type_info(root);
  cache_func_calls(root);
  inline_dfuncs(root);
  gen_type_info(root);


  //TP: redo this after inlining and caching
  transform_array_assignments(root);
  transform_array_binary_ops(root);


  gen_matrix_reads(root);
  gen_constexpr_info(root,gen_mem_accesses);

  //TP: do this at the very end to not mess with other passes
  resolve_profile_stencils(root);
  for(size_t i = 0; i  < primitive_datatypes.size; ++i)
  {
	preprocess_array_reads(root,root,primitive_datatypes.data[i],gen_mem_accesses);
  }
  preprocess_array_reads(root,root,REAL3_STR,gen_mem_accesses);
  preprocess_array_reads(root,root,FIELD_STR,gen_mem_accesses);
  for(size_t i = 0; i  < primitive_datatypes.size; ++i)
  {
  	gen_array_reads(root,root,primitive_datatypes.data[i]);
  }

  gen_user_taskgraphs(root);
  combinatorial_params_info info = get_combinatorial_params_info(root);
  gen_kernel_input_params(root,info.params.vals,info.kernels_with_input_params,info.kernel_combinatorial_params,gen_mem_accesses);
  //replace_boolean_dconsts_in_optimized(root,info.params.vals,info.kernels_with_input_params,info.kernel_combinatorial_params);
  free_combinatorial_params_info(&info);

  if(!gen_mem_accesses && executed_nodes.size > 0 && OPTIMIZE_MEM_ACCESSES && ELIMINATE_CONDITIONALS)
  {
	  bool eliminated_something = true;
	  while(eliminated_something)
	  {
	  	eliminated_something = eliminate_conditionals(root,gen_mem_accesses);
		gen_constexpr_info(root,gen_mem_accesses);
	  }
	  remove_dead_writes(root);

	  FILE* fp = fopen("ac_minimized_code.ac.raw","w");
          symboltable_reset();
	  traverse_base_params traverse_params;
	  memset(&traverse_params,0,sizeof(traverse_base_params));
	  traverse_params.to_DSL= true;
          traverse_base(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL, traverse_params);
          char** dfunc_strs = get_dfunc_strs(root,traverse_params);
          for(size_t i = 0; i < num_dfuncs; ++i)
          {
                fprintf(fp,"%s\n",dfunc_strs[i]);
                free(dfunc_strs[i]);
          }
           free(dfunc_strs);

          // Kernels
           symboltable_reset();
          traverse_base(root,
            NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
                NODE_HOSTDEFINE | NODE_NO_OUT,
            fp,traverse_params);
          fclose(fp);
          format_source("ac_minimized_code.ac.raw","ac_minimized_code.ac");
          printf("Wrote minimized code in ac_minimized_code.ac\n");
  }

  //TP: done after code elimination for the code written to ac_minimized to be valid DSL
  traverse(root, NODE_NO_OUT, NULL);
  gen_kernel_postfixes(root,gen_mem_accesses);

  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL | NODE_NO_OUT, NULL);
  char** dfunc_strs = get_dfunc_strs(root,(traverse_base_params){});
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
  if(gen_mem_accesses && ELIMINATE_CONDITIONALS)
  {
	  fflush(stream);
	  //This is used to eliminate known constexpr conditionals
	  //TP: for now set code elimination off
	  //bool eliminated_something = false;
	  bool eliminated_something = true;

	  int round = 0;
  	  gen_constexpr_info(root,gen_mem_accesses);
	  while(eliminated_something)
	  {
		++round;
	  	clean_stream(stream);

		symboltable_reset();
  	  	traverse(root,
           		NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               		NODE_HOSTDEFINE | NODE_NO_OUT,
           	stream);
	  	fflush(stream);
	  	get_executed_nodes(round-1);
	  	eliminated_something = eliminate_conditionals(root,gen_mem_accesses);
		gen_constexpr_info(root,gen_mem_accesses);
	  }
	  remove_dead_writes(root);

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
  copy_file("user_kernels.h","user_kernels_backup.h");
  copy_file("user_kernels.h","user_cpu_kernels.h");
  if(log)
  {
  	printf("Compiling %s...\n", STENCILACC_SRC);
#if AC_USE_HIP
  	printf("--- USE_HIP: `%d`\n", HIP_ON);
#else
  	printf("--- USE_HIP not defined\n");
#endif
  	printf("--- ACC_RUNTIME_API_DIR: `%s`\n", ACC_RUNTIME_API_DIR);
  	printf("--- GPU_API_INCLUDES: `%s`\n", GPU_API_INCLUDES);
  }
  char cmd[4096];
  const char* api_includes = strlen(GPU_API_INCLUDES) > 0 ? " -I " GPU_API_INCLUDES  " " : "";
  sprintf(cmd, "g++ -I. -I " ACC_RUNTIME_API_DIR " %s -DAC_STENCIL_ACCESSES_MAIN=1 -DAC_DOUBLE_PRECISION=%d -DAC_USE_HIP=%d " 
	       STENCILACC_SRC " -lm  -std=c++1z -o " STENCILACC_EXEC" "
  ,api_includes, AC_DOUBLE_PRECISION,HIP_ON 
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
check_status(int status)
{
  if(status != 0)
  {
	if (WIFEXITED(status)) {
    		printf("Stencil accesses exited with status: %d\n", WEXITSTATUS(status));
	}
	else if (WIFSIGNALED(status)) {
    		printf("Stencil accesses killed by signal: %s\n", strsignal(WTERMSIG(status)));
	}
	fatal("Something went wrong during analysis: %d\n",status);
  }
}
void
get_executed_nodes(const int round)
{
	compile_helper(false);
	char dst[4096];
	sprintf(dst,"user_kernels_round_%d.h",round);
  	format_source("user_kernels.h",dst);
  	FILE* proc = popen("./" STENCILACC_EXEC " -C", "r");
  	assert(proc);
	check_status(pclose(proc));

  	free_int_vec(&executed_nodes);
  	FILE* fp = fopen("executed_nodes.bin","rb");
  	int size;
  	int tmp;
  	bool reading_successful = fread_errchk(&size, sizeof(int), 1, fp) == 1;
  	for(int i = 0; i < size; ++i)
  	{
  	      reading_successful &= fread_errchk(&tmp, sizeof(int), 1, fp);
  	      push_int(&executed_nodes,tmp);
  	}
	if(!reading_successful) fatal("Was not able to read executed nodes!\n");
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
  check_status(pclose(proc));

  bool reading_successful = true;
  FILE* fp = fopen("user_written_fields.bin", "rb");
  written_fields = (int*)malloc(num_kernels*num_fields*sizeof(int));
  reading_successful &= fread_errchk(written_fields, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);

  fp = fopen("user_read_fields.bin", "rb");
  read_fields = (int*)malloc(num_kernels*num_fields*sizeof(int));
  reading_successful &= fread_errchk(read_fields, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);

  fp = fopen("user_field_has_stencil_op.bin", "rb");
  field_has_stencil_op = (int*)malloc(num_kernels*num_fields*sizeof(int));
  reading_successful &= fread_errchk(field_has_stencil_op, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);

  fp = fopen("user_field_has_previous_call.bin", "rb");
  field_has_previous_call= (int*)malloc(num_kernels*num_fields*sizeof(int));
  reading_successful &= fread_errchk(field_has_previous_call, sizeof(int), num_kernels*num_fields, fp);
  fclose(fp);

  fp = fopen("user_read_profiles.bin","rb");
  read_profiles     = (int*)malloc(num_kernels*num_profiles*sizeof(int));
  reading_successful &= fread_errchk(read_profiles, sizeof(int), num_kernels*num_profiles, fp);
  fclose(fp);

  fp = fopen("user_reduced_profiles.bin","rb");
  reduced_profiles  = (int*)malloc(num_kernels*num_profiles*sizeof(int));
  reading_successful &= fread_errchk(reduced_profiles, sizeof(int), num_kernels*num_profiles, fp);
  fclose(fp);

  fp = fopen("user_reduced_reals.bin","rb");
  reduced_reals = (int*)malloc(num_kernels*count_variables(REAL_STR, OUTPUT_STR)*sizeof(int));
  reading_successful &= fread_errchk(reduced_reals, sizeof(int), num_kernels*count_variables(REAL_STR, OUTPUT_STR), fp);
  fclose(fp);

  fp = fopen("user_reduced_ints.bin","rb");
  reduced_ints = (int*)malloc(num_kernels*count_variables(INT_STR, OUTPUT_STR)*sizeof(int));
  reading_successful &= fread_errchk(reduced_ints, sizeof(int), num_kernels*count_variables(INT_STR, OUTPUT_STR), fp);
  fclose(fp);

  fp = fopen("user_reduced_floats.bin","rb");
  reduced_floats = (int*)malloc(num_kernels*count_variables(FLOAT_STR, OUTPUT_STR)*sizeof(int));
  reading_successful &= fread_errchk(reduced_floats , sizeof(int), num_kernels*count_variables(FLOAT_STR, OUTPUT_STR), fp);
  fclose(fp);

  if(!reading_successful) fatal("Was not able to read mem accesses output!\n");
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
