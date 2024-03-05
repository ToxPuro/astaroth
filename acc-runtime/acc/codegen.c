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

#include "ast.h"
#include "tab.h"

#define STENCILGEN_HEADER "stencilgen.h"
#define PROFILE_HEADER "profilegen.h"
#define STENCILGEN_SRC ACC_DIR "/stencilgen.c"
#define STENCILGEN_EXEC "stencilgen.out"
#define STENCILACC_SRC ACC_DIR "/stencil_accesses.cpp"
#define STENCILACC_EXEC "stencil_accesses.out"
#define ACC_RUNTIME_API_DIR ACC_DIR "/../api"

// Symbols
#define MAX_ID_LEN (256)
typedef struct {
  NodeType type;
  char tqualifier[MAX_ID_LEN];
  char tspecifier[MAX_ID_LEN];
  char identifier[MAX_ID_LEN];
  } Symbol;


#define SYMBOL_TABLE_SIZE (65536)
static Symbol symbol_table[SYMBOL_TABLE_SIZE] = {};

char* symbol_var_length[SYMBOL_TABLE_SIZE];
#define MAX_NESTS (32)
static size_t num_symbols[MAX_NESTS] = {};
static size_t current_nest           = 0;

//profiles symbol table
#define MAX_NUM_PROFILES (256)
char* profile_names[MAX_NUM_PROFILES];
int profile_read_set_sizes[MAX_NUM_PROFILES];
int* profile_read_set[MAX_NUM_PROFILES];
int num_profiles = 0;

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
add_symbol(const NodeType type, const char* tqualifier, const char* tspecifier,
           const char* id)
{
  assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);

  symbol_table[num_symbols[current_nest]].type          = type;
  symbol_table[num_symbols[current_nest]].tqualifier[0] = '\0';
  symbol_table[num_symbols[current_nest]].tspecifier[0] = '\0';

  if (tqualifier)
    strcpy(symbol_table[num_symbols[current_nest]].tqualifier, tqualifier);
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
  current_nest              = 0;
  num_symbols[current_nest] = 0;

  // Add built-in variables (TODO consider NODE_BUILTIN)
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "print");           // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "threadIdx");       // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "blockIdx");        // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "vertexIdx");       // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "globalVertexIdx"); // TODO REMOVE
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "globalGridN");     // TODO REMOVE

  // add_symbol(NODE_UNKNOWN, NULL, NULL, "true");
  // add_symbol(NODE_UNKNOWN, NULL, NULL, "false");

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "previous");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "vecprevious");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "value");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "vecvalue");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "isnan");  // TODO RECHECK
  //In develop
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "read_w");
  //add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write_w");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "vecwrite");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "Field3"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "dot");    // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "cross");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "len");    // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "uint64_t");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "UINT64_MAX"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "rand_uniform");

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "exp");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "sin");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "cos");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "sqrt");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "fabs");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "pow");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "multm2_sym");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "diagonal");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "sum");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "log");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "abs");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "atan2"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "AC_REAL_PI");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "NUM_FIELDS");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "NUM_VTXBUF_HANDLES");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "NUM_ALL_FIELDS");

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "FIELD_IN");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "FIELD_OUT");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "IDX");

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "true");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "false");

  // Astaroth 2.0 backwards compatibility START
  // (should be actually built-in externs in acc-runtime/api/acc-runtime.h)
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_mx");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_my");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_mz");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nx");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_ny");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nz");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nxgrid");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nygrid");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nzgrid");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nx_min");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_ny_min");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nz_min");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nx_max");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_ny_max");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nz_max");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_mxy");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nxy");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_nxyz");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_xy_plate_bufsize");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_xz_plate_bufsize");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_yz_plate_bufsize");
  add_symbol(NODE_DCONST_ID, NULL, "int3", "AC_domain_decomposition");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_proc_mapping_strategy");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_decompose_strategy");
  

  add_symbol(NODE_DCONST_ID, NULL, "int3", "AC_multigpu_offset");

  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsx");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsy");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsz");

  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsx");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsy");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsz");

  //For special reductions
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_center_x");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_center_y");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_center_z");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_sum_radius");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_window_radius");

  // (BC types do not belong here, BCs not handled with the DSL)
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_bot_x");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_bot_y");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_bot_z");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_top_x");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_top_y");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_top_z");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_init_type");
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

    if (strlen(symbol_table[i].tqualifier) > 0)
      printf("(tqual: %s) ", symbol_table[i].tqualifier);

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
get_parent_node(const NodeType type, const ASTNode* node)
{
  if (node->type & type)
    return node;
  else if (node->parent)
    return get_parent_node(type, node->parent);
  else
    return NULL;
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
void combine(const ASTNode* node, char* res){
  if(node->buffer)
    strcat(res,node->buffer);
  if(node->lhs)
    combine(node->lhs, res);
  if(node->rhs)
    combine(node->rhs, res);
}
bool is_profile_read_root(const ASTNode* node){
  const ASTNode* lhs = node;
  for(int i = 0; i<3; i++){
    if(!lhs->lhs)
      return false;
    lhs = lhs->lhs;
  }
  if(lhs->buffer && node->infix && node->postfix && (strcmp(node->infix, "[") == 0) && (strcmp(node->postfix, "]") == 0))
    for(int i=0;i<num_profiles;i++){
      if((strcmp(lhs->buffer, profile_names[i]) == 0))
        return true;
    }
  return false;
}
bool
is_profile_specifier(const char* tspecifier)
{
  return 
    !strcmp(tspecifier,"Profile_x") |
    !strcmp(tspecifier,"Profile_y") |
    !strcmp(tspecifier,"Profile_z");
}
void add_profile(const char* profile_name){
  profile_names[num_profiles] = strdup(profile_name);
  profile_read_set_sizes[num_profiles] = 0;
  profile_read_set[num_profiles] = (int*)malloc(MAX_NUM_PROFILES * sizeof(int));
  num_profiles++;
}
int get_profile_index(char* profile_name){
  for(int i=0;i<num_profiles;i++){
    if(strcmp(profile_name, profile_names[i]) == 0)
      return i;
  }
  return -1;
}
int add_profile_read_index(int profile_index, int array_index){
  for(int  i=0; i<profile_read_set_sizes[profile_index]; i++){
    if(array_index == profile_read_set[profile_index][i]){
      return i;
    }
  }
  profile_read_set[profile_index][profile_read_set_sizes[profile_index]] = array_index;
  profile_read_set_sizes[profile_index]++;
  return profile_read_set_sizes[profile_index]-1;
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
      char* tqual = NULL;

      const ASTNode* decl = get_parent_node(NODE_DECLARATION, node);
      if (decl) {
        const ASTNode* tspec_node = get_node(NODE_TSPEC, decl);
        const ASTNode* tqual_node = get_node(NODE_TQUAL, decl);

        if (tspec_node && tspec_node->lhs){
          tspec = tspec_node->lhs->buffer;
        }
        if (tqual_node && tqual_node->lhs)
          tqual = tqual_node->lhs->buffer;
      }

      if (stream) {
        const ASTNode* is_dconst = get_parent_node(NODE_DCONST, node);
        if (is_dconst)
          fprintf(stream, "__device__ ");

        if (tqual)
          fprintf(stream, "%s ", tqual);

        if (tspec){
          fprintf(stream, "%s ", tspec);
        }
        else if (!(node->type & NODE_KFUNCTION_ID) &&
                 !get_parent_node(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID) &&
                 !strstr(node->buffer, "ac_input"))
          fprintf(stream, "auto ");
      }
      if (!(node->type & NODE_MEMBER_ID))
      {
        if (tspec != NULL && (is_profile_specifier(tspec) |!strcmp(tspec,"AcReal*")))
        {
          const ASTNode* nd = decl->rhs->lhs->rhs->lhs->lhs->lhs->lhs;
          if(nd)
          {
            if(nd->lhs)
            {
              printf("no left child!\n");
              exit(0);
            }
            if(nd->rhs)
            {
              printf("no right child\n");
              exit(0);
            }
            const int symbol_index = add_symbol(node->type, tqual, tspec, node->buffer);
            symbol_var_length[symbol_index] = nd->buffer;
          }
        }
        else{
          add_symbol(node->type, tqual, tspec, node->buffer);
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
    if (symbol && symbol->type & NODE_DCONST_ID)
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
  traverse(root, NODE_FUNCTION | NODE_VARIABLE | NODE_STENCIL | NODE_HOSTDEFINE,
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
  traverse(root, NODE_DCONST | NODE_VARIABLE | NODE_FUNCTION | NODE_STENCIL, fp);

  symboltable_reset();
  traverse(root, 0, NULL);

  // Stencils
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      fprintf(fp, "stencil_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_STENCILS} Stencil;");

  // Enums
  int num_of_fields=0;
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"Field")){
      fprintf(fp, "%s,", symbol_table[i].identifier);
      num_of_fields++;
    }
//Add Auxiliary fields into Fields after Full fields
//Communicated Auxiliaries come first
  int num_of_auxiliary_fields=0;
  int num_of_communicated_auxiliary_fields = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (!strcmp(symbol_table[i].tspecifier,"AuxiliaryField") && (!strcmp(symbol_table[i].tqualifier, "communicated")))
    {
      printf("Auxilaries are in development\n");
      exit(0);
      fprintf(fp, "%s,", symbol_table[i].identifier);
      num_of_communicated_auxiliary_fields++;
      num_of_auxiliary_fields++;
    }
  }
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
  {
    if (!strcmp(symbol_table[i].tspecifier,"AuxiliaryField") && (strcmp(symbol_table[i].tqualifier, "communicated")))
    {
      printf("Auxilaries are in development\n");
      exit(0);
      fprintf(fp, "%s,", symbol_table[i].identifier);
      num_of_auxiliary_fields++;
    }
  }
  fprintf(fp, "NUM_FIELDS=%d,", num_of_fields+num_of_auxiliary_fields);
  fprintf(fp, "NUM_COMMUNICATED_FIELDS=%d,", num_of_fields+num_of_communicated_auxiliary_fields);
  // fprintf(fp, "NUM_AUXILIARY_FIELDS=%d,", num_of_auxiliary_fields);
  // fprintf(fp, "NUM_ALL_FIELDS=%d,", num_of_auxiliary_fields+num_of_fields);
  fprintf(fp, "} Field;");


  // Enums for profiles
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (is_profile_specifier(symbol_table[i].tspecifier)){
      fprintf(fp, "%s,", symbol_table[i].identifier);
    }
  fprintf(fp, "NUM_PROFILES} Profile;");

  // Enums for work_buffers 
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"WorkBuffer"))
    {
      printf("Workbuffers are under development\n");
      exit(0);
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
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "int"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_INT_PARAMS} AcIntParam;");

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "int3"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_INT3_PARAMS} AcInt3Param;");

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_REAL_PARAMS} AcRealParam;");

  // Enums for arrays
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
       !strcmp(symbol_table[i].tspecifier, "AcReal*"))
       {
        printf("\n\nArrays are under development\n\n");
        exit(0);
        fprintf(fp, "%s,", symbol_table[i].identifier);
       }
  fprintf(fp, "NUM_REAL_ARRAYS} AcRealArrayParam;");

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal3"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_REAL3_PARAMS} AcReal3Param;");

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

  fprintf(fp, "static const bool vtxbuf_is_auxiliary[] __attribute__((unused)) = {");
  for(int i=0;i<num_of_fields;++i)
    fprintf(fp, "%s,", "false");
  for(int i=num_of_fields;i<num_of_fields+num_of_auxiliary_fields;++i)
    fprintf(fp, "%s,", "true");
  fprintf(fp, "};");

  fprintf(fp, "static const char* profile_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (is_profile_specifier(symbol_table[i].tspecifier))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const AcIntParam profile_lengths[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (is_profile_specifier(symbol_table[i].tspecifier))
    {
      fprintf(fp, "%s,", symbol_var_length[i]);
    }
  fprintf(fp, "};");

  fprintf(fp, "static const char* work_buffer_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if(!strcmp(symbol_table[i].tspecifier,"WorkBuffer"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");



  fprintf(fp, "static const int profile_dims[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i){
    if(!strcmp(symbol_table[i].tspecifier,"Profile_x"))
      fprintf(fp, "1,");
    if(!strcmp(symbol_table[i].tspecifier,"Profile_y"))
      fprintf(fp, "2,");
    if(!strcmp(symbol_table[i].tspecifier,"Profile_z"))
      fprintf(fp, "3,");
  }
  fprintf(fp, "};");


  fprintf(fp, "static const char* kernel_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp,
          "static const char* intparam_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "int"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp,
          "static const char* int3param_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "int3"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp,
          "static const char* realparam_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp,
          "static const char* real3param_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal3"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp,
          "static const char* real_array_param_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal*"))
      fprintf(fp, "\"%s\",", symbol_table[i].identifier);
  fprintf(fp, "};");

  fprintf(fp, "static const AcIntParam real_array_lengths[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_VARIABLE_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal*"))
      fprintf(fp, "%s,", symbol_var_length[i]);
  fprintf(fp, "};");

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
  for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
    if (!(symbol_table[i].type & NODE_FUNCTION_ID) &&
        !(symbol_table[i].type & NODE_VARIABLE_ID) &&
        !(symbol_table[i].type & NODE_STENCIL_ID)) {
      fprintf(fp, "// extern __device__ %s %s;\n", symbol_table[i].tspecifier,
              symbol_table[i].identifier);
    }
  }

  // Stencil order
  fprintf(fp, "#ifndef STENCIL_ORDER\n");
  fprintf(fp, "#define STENCIL_ORDER (6)\n");
  fprintf(fp, "#endif\n");
  fprintf(fp, "#define STENCIL_DEPTH (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define STENCIL_HEIGHT (STENCIL_ORDER+1)\n");
  fprintf(fp, "#define STENCIL_WIDTH (STENCIL_ORDER+1)\n");

  fclose(fp);

  symboltable_reset();
}

static void
gen_user_kernels(const ASTNode* root, const char* out, const bool gen_mem_accesses)
{
  symboltable_reset();
  traverse(root, 0, NULL);

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
  // This is not really needed any more, the kernel function pointer is now
  // exposed in the API, so one could use that directly instead of handles.
  fprintf(fp,"#include \"user_kernel_declarations.h\"\n");
  fprintf(fp, "static const Kernel kernels[] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      //TP: this cast is not safe if the user uses kernels with input parameters but this is anyways for backwards compatibility so would not break old code
      // since old code only allows a single signature
      fprintf(fp, "(Kernel)%s,", symbol_table[i].identifier); // Host layer handle
  fprintf(fp, "};");

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
void gen_profile_reads(ASTNode* node){

  if(is_profile_read_root(node)){
    char* profile_name= node->lhs->lhs->lhs->buffer;
    int profile_index = get_profile_index(profile_name);
    char array_index_str[4096];
    strcpy(array_index_str, "");
    combine(node->rhs, array_index_str);
    int array_index = atoi(array_index_str);
    int num_profile_read = add_profile_read_index(profile_index, array_index);
    char builder[4096];
    sprintf(builder, "p_%d_%d", profile_index, num_profile_read);
    node->buffer = strdup(builder);
    node->postfix = node->infix;
    node->lhs=node->rhs = NULL;
  }
  if(node->lhs)
    gen_profile_reads(node->lhs);
  if(node->rhs)
    gen_profile_reads(node->rhs);
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
generate(ASTNode* root, FILE* stream, const bool gen_mem_accesses)
{
  num_profiles = 0;
  assert(root);

  gen_user_defines(root, "user_defines.h");
  gen_user_kernels(root, "user_declarations.h", gen_mem_accesses);

  // Fill the symbol table
  traverse(root, 0, NULL);
  // print_symbol_table();
  num_profiles = 0;

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
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      ++num_kernels;


  for (size_t i = 0; i < num_symbols[current_nest]; ++i)

    if (is_profile_specifier(symbol_table[i].tspecifier))
      add_profile(symbol_table[i].identifier);
  gen_profile_reads(root);

  //generate profile_read_set_sizes and profile_read_set accessible to stencilgen.c
  FILE* profile_file = fopen(PROFILE_HEADER, "w");
  fprintf(profile_file,"int profile_read_set_sizes[%d] = {", num_profiles);
  for(int profile=0;profile<num_profiles;profile++){
    fprintf(profile_file,"%d",profile_read_set_sizes[profile]);
    if(profile<num_profiles-1)
      fprintf(profile_file,",");
  }
  fprintf(profile_file,"};\n");
  fprintf(profile_file,"int profile_read_set[%d][%d] = {", num_profiles, MAX_NUM_PROFILES);
  for(int profile=0;profile<num_profiles;profile++){
    fprintf(profile_file,"{");
    for(int read=0;read<profile_read_set_sizes[profile];read++){
      fprintf(profile_file,"%d",profile_read_set[profile][read]);
      if(read < MAX_NUM_PROFILES)
        fprintf(profile_file,",");
    }
    for(int read=profile_read_set_sizes[profile];read<MAX_NUM_PROFILES;read++){
      fprintf(profile_file,"%d",0);
      if(read < MAX_NUM_PROFILES)
        fprintf(profile_file,",");
    }
    fprintf(profile_file,"}");
    if(profile<num_profiles-1)
      fprintf(profile_file,",");
    fprintf(profile_file,"\n");
  }
  fprintf(profile_file,"};\n");
  fclose(profile_file);

  // Device constants
  // gen_dconsts(root, stream);

  // Stencils

  // Stencil generator
  FILE* stencilgen = fopen(STENCILGEN_HEADER, "w");
  assert(stencilgen);

  // Stencil ops
  symboltable_reset();
  traverse(root, 0, NULL);
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
        fprintf(stencilgen, "\"%s\",",
                strlen(symbol.tqualifier) ? symbol.tqualifier : "sum");
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
               NODE_HOSTDEFINE,
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
               NODE_HOSTDEFINE,
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
               NODE_HOSTDEFINE,
           dfunc_fp);
  fflush(dfunc_fp);

  // Kernels
  symboltable_reset();
  gen_kernels(root, dfunctions, gen_mem_accesses);
  fclose(dfunc_fp); // Frees dfunctions also

  symboltable_reset();
  traverse(root,
           NODE_DCONST | NODE_VARIABLE | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE,
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
  strcat(cmd, STENCILACC_SRC " -lm -o " STENCILACC_EXEC " ");

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
  printf("%s compilation done\n", STENCILACC_SRC);
  if (retval == -1) {
    fprintf(stderr, "Catastrophic error: could not compile the stencil access "
                    "generator.\n");
    assert(retval != -1);
    exit(EXIT_FAILURE);
  }

  // Generate stencil accesses
  FILE* proc = popen("./" STENCILACC_EXEC " stencil_accesses.h", "r");
  assert(proc);
  pclose(proc);
}

