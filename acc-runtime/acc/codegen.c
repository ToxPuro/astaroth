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

static const size_t stencil_order  = 6;
static const size_t stencil_depth  = stencil_order + 1;
static const size_t stencil_width  = stencil_order + 1;
static const size_t stencil_height = stencil_order + 1;

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

#define MAX_NESTS (32)
static size_t num_symbols[MAX_NESTS] = {};
static size_t current_nest           = 0;

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

static void
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
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "Field3"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "dot");    // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "cross");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "len");    // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "exp");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "sin");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "cos");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "sqrt"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "fabs"); // TODO RECHECK

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "AC_REAL_PI");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "NUM_FIELDS");

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

  add_symbol(NODE_DCONST_ID, NULL, "int3", "AC_multigpu_offset");
  add_symbol(NODE_DCONST_ID, NULL, "int3", "AC_global_grid_n");

  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsx");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsy");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsz");

  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsx");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsy");
  // add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsz");

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

        if (tspec_node && tspec_node->lhs)
          tspec = tspec_node->lhs->buffer;

        if (tqual_node && tqual_node->lhs)
          tqual = tqual_node->lhs->buffer;
      }

      if (stream) {
        const ASTNode* is_dconst = get_parent_node(NODE_DCONST, node);
        if (is_dconst)
          fprintf(stream, "__device__ ");

        if (tqual)
          fprintf(stream, "%s ", tqual);

        if (tspec)
          fprintf(stream, "%s ", tspec);
        else if (!(node->type & NODE_KFUNCTION_ID) &&
                 !get_parent_node(NODE_STENCIL, node) &&
                 !(node->type & NODE_MEMBER_ID))
          fprintf(stream, "auto ");
      }
      if (!(node->type & NODE_MEMBER_ID))
        add_symbol(node->type, tqual, tspec, node->buffer);
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
  traverse(root, NODE_FUNCTION | NODE_FIELD | NODE_STENCIL | NODE_HOSTDEFINE,
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

static void
gen_kernels(const ASTNode* node, const char* sdefinitions,
            const char* dfunctions, const char* sfunctions)
{
  assert(node);

  if (node->type & NODE_KFUNCTION) {

    const size_t len = 64 * 1024 * 1024;
    char* prefix     = malloc(len);
    assert(prefix);

    assert(node->rhs);
    assert(node->rhs->rhs);
    ASTNode* compound_statement = node->rhs->rhs;

    strcat(prefix, compound_statement->prefix);
    strcat(prefix, sdefinitions);
    strcat(prefix, sfunctions);
    strcat(prefix, dfunctions);

    astnode_set_prefix(prefix, compound_statement);
    free(prefix);
  }

  if (node->lhs)
    gen_kernels(node->lhs, sdefinitions, dfunctions, sfunctions);

  if (node->rhs)
    gen_kernels(node->rhs, sdefinitions, dfunctions, sfunctions);
}

// Generate User Defines
static void
gen_user_defines(const ASTNode* root, const char* out)
{
  FILE* fp = fopen(out, "w");
  assert(fp);

  fprintf(fp, "#pragma once\n");
  fprintf(fp, "#define STENCIL_ORDER (%lu)\n", stencil_order);
  fprintf(fp, "#define STENCIL_DEPTH (%lu)\n", stencil_depth);
  fprintf(fp, "#define STENCIL_HEIGHT (%lu)\n", stencil_height);
  fprintf(fp, "#define STENCIL_WIDTH (%lu)\n", stencil_width);

  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_FIELD | NODE_FUNCTION | NODE_STENCIL, fp);

  symboltable_reset();
  traverse(root, 0, NULL);

  // Stencils
  fprintf(fp, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      fprintf(fp, "stencil_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_STENCILS} Stencil;");
  /*
  size_t num_stencils = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      ++num_stencils;
  fprintf(fp, "#define NUM_STENCILS (%lu)\n", num_stencils);
  */

  // Enums
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_FIELD_ID)
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_FIELDS} Field;");

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

  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_DCONST_ID &&
        !strcmp(symbol_table[i].tspecifier, "AcReal3"))
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_REAL3_PARAMS} AcReal3Param;");

  // Enum strings (convenience)
  fprintf(fp, "static const char* field_names[] __attribute__((unused)) = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_FIELD_ID)
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
        !(symbol_table[i].type & NODE_FIELD_ID) &&
        !(symbol_table[i].type & NODE_STENCIL_ID)) {
      fprintf(fp, "// extern __device__ %s %s;\n", symbol_table[i].tspecifier,
              symbol_table[i].identifier);
    }
  }

  fclose(fp);

  symboltable_reset();
}

static void
gen_user_kernels(const ASTNode* root, const char* out)
{
  symboltable_reset();
  traverse(root, 0, NULL);

  FILE* fp = fopen(out, "w");
  assert(fp);

  fprintf(fp, "#pragma once\n");

  // Kernels
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp,
              "__global__ void %s(const int3 start, const int3 end, "
              "VertexBufferArray vba);",
              symbol_table[i].identifier);

  // Astaroth 2.0 backwards compatibility START
  // This is not really needed any more, the kernel function pointer is now
  // exposed in the API, so one could use that directly instead of handles.
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "KERNEL_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_KERNELS} AcKernel;");

  fprintf(fp, "static const Kernel kernel_lookup[] = {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "%s,", symbol_table[i].identifier); // Host layer handle
  fprintf(fp, "};");
  // Astaroth 2.0 backwards compatibility END

  fclose(fp);

  symboltable_reset();
}

void
generate(const ASTNode* root, FILE* stream)
{
  assert(root);

  gen_user_defines(root, "user_defines.h");
  gen_user_kernels(root, "user_declarations.h");

  // Fill the symbol table
  traverse(root, 0, NULL);
  // print_symbol_table();

  // Generate kernels.cu
  fprintf(stream, "#pragma once\n");

  size_t num_stencils = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      ++num_stencils;

  size_t num_fields = 0;
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_FIELD_ID)
      ++num_fields;

      // Device constants
      // gen_dconsts(root, stream);

      // Stencils
      /*
      // Defined now in user_defines.h
      fprintf(stream, "typedef enum{");
      for (size_t i = 0; i < num_symbols[current_nest]; ++i)
        if (symbol_table[i].type & NODE_STENCIL_ID)
          fprintf(stream, "stencil_%s,", symbol_table[i].identifier);
      fprintf(stream, "} Stencil;");
      */
      // fprintf(stream, "NUM_STENCILS} Stencil;"); // defined now in
      // user_defines.h

      // Stencil generator
#define STENCILGEN_SRC "stencilgen.c"
#define STENCILGEN_EXEC "stencilgen.out"
  FILE* stencilgen = fopen(STENCILGEN_SRC, "w");
  assert(stencilgen);
  fprintf(stencilgen,
          "#include <stdio.h>\n"
          "#include <stdlib.h>\n"
          "#define STENCIL_ORDER (%lu)\n"
          "#define STENCIL_DEPTH (%lu)\n"
          "#define STENCIL_HEIGHT (%lu)\n"
          "#define STENCIL_WIDTH (%lu)\n"
          "#define NUM_STENCILS (%lu)\n"
          "#define NUM_FIELDS (%lu)\n",
          stencil_order, stencil_depth, stencil_height, stencil_width,
          num_stencils, num_fields);

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
  fprintf(stencilgen, "static char* "
                      "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]["
                      "STENCIL_WIDTH] = {");
  traverse(root,
           NODE_STENCIL_ID | NODE_DCONST | NODE_FIELD | NODE_FUNCTION |
               NODE_HOSTDEFINE,
           stencilgen);
  fprintf(stencilgen, "};");
  // clang-format off
const char* stencilgen_main =
"int main(int argc, char** argv) {\n"
"(void)argv;/*Unused*/"
"if(argc>1){" // Generate stencil definitions
  "printf(\"static __device__ /*const*/ AcReal /*__restrict__*/ stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]={\");"
  "for(int stencil=0;stencil<NUM_STENCILS;++stencil){"
    "printf(\"{\");"
    "for(int depth=0;depth<STENCIL_DEPTH;++depth){"
      "printf(\"{\");"
      "for(int height=0;height<STENCIL_HEIGHT;++height){"
        "printf(\"{\");"
        "for(int width=0;width<STENCIL_WIDTH;++width){"
          "printf(\"%s,\",stencils[stencil][depth][height][width]?stencils[stencil][depth][height][width]:\"0\");"
        "}"
        "printf(\"},\");"
      "}"
      "printf(\"},\");"
    "}"
    "printf(\"},\");"
  "}"
  "printf(\"};\\n\");"
"} else {" // Generate stencil reductions
  "int stencil_initialized[NUM_FIELDS][NUM_STENCILS]={0};"
  "for(int field=0;field<NUM_FIELDS;++field){"
    "printf(\"{const AcReal* __restrict__ in=vba.in[%d];\",field);"
    "for(int depth=0;depth<STENCIL_DEPTH;++depth){"
      "for(int height=0;height<STENCIL_HEIGHT;++height){"
        "for(int width=0;width<STENCIL_WIDTH;++width){"
          "for(int stencil=0;stencil<NUM_STENCILS;++stencil){"
            "if(stencils[stencil][depth][height][width]){"
              "if (!stencil_initialized[field][stencil]) {"
                "printf(\"processed_stencils[%d][%d]=%s(stencils[%d][%d][%d][%d]*in[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d),vertexIdx.z+(%d))]);\","
                         "field,stencil,stencil_unary_ops[stencil],stencil,depth,height,width,-STENCIL_ORDER/2+width,-STENCIL_ORDER/2+height,-STENCIL_ORDER/2+depth);"
                "stencil_initialized[field][stencil] = 1;"
              "} else {"
                "printf(\"processed_stencils[%d][%d]=%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*in[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d),vertexIdx.z+(%d))]));\","
                         "field,stencil,stencil_binary_ops[stencil],field,stencil,stencil_unary_ops[stencil], stencil,depth,height,width,-STENCIL_ORDER/2+width,-STENCIL_ORDER/2+height,-STENCIL_ORDER/2+depth);"
              "}"
            "}"
          "}"
        "}"
      "}"
    "}"
  "printf(\"}\\n\");"
"}"
"}"
"return EXIT_SUCCESS;}";
  // clang-format on
  fprintf(stencilgen, "%s", stencilgen_main);
  fclose(stencilgen);

  // Compile
  const int retval = system("gcc -std=c11 -Wall -Wextra -Wdouble-promotion "
                            "-Wfloat-conversion -Wshadow " STENCILGEN_SRC " "
                            "-o " STENCILGEN_EXEC);
  if (retval == -1) {
    fprintf(stderr,
            "Catastrophic error: could not compile the stencil generator.\n");
    assert(retval != -1);
  }

  // Generate stencil definitions
  FILE* proc = popen("./" STENCILGEN_EXEC " -stencils", "r");
  assert(proc);

  char buf[4096] = {0};
  while (fgets(buf, sizeof(buf), proc))
    fprintf(stream, "%s", buf);

  pclose(proc);

  // Generate stencil FMADs
  proc = popen("./" STENCILGEN_EXEC, "r");
  assert(proc);

  char sdefinitions[1 * 1024 * 1024] = {0};
  while (fgets(buf, sizeof(buf), proc))
    strcat(sdefinitions, buf);

  pclose(proc);

  // Device functions
  symboltable_reset();
  char* dfunctions;
  size_t sizeloc;
  FILE* dfunc_fp = open_memstream(&dfunctions, &sizeloc);
  traverse(root,
           NODE_DCONST | NODE_FIELD | NODE_STENCIL | NODE_KFUNCTION |
               NODE_HOSTDEFINE,
           dfunc_fp);
  fflush(dfunc_fp);

  // Stencil functions
  char sfunctions[1024 * 1024] = {0};
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID) {
      const char* id = symbol_table[i].identifier;
      sprintf(buf,
              "#define %s(field) (processed_stencils[(field)][stencil_%s])\n",
              id, id);
      /*
      sprintf(buf,
              "auto %s = [processed_stencils](Field field) { return "
              "processed_stencils[field][stencil_%s]; };",
              id, id);
              */
      strcat(sfunctions, buf);
    }

  // Kernels
  symboltable_reset();
  gen_kernels(root, sdefinitions, dfunctions, sfunctions);
  fclose(dfunc_fp); // Frees dfunctions also

  symboltable_reset();
  traverse(root,
           NODE_DCONST | NODE_FIELD | NODE_STENCIL | NODE_DFUNCTION |
               NODE_HOSTDEFINE,
           stream);

  // print_symbol_table();
}
