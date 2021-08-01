#include "codegen.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "ast.h"
#include "tab.h"

static const size_t stencil_order = 6;

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
    if (strcmp(identifier, symbol_table[i].identifier) == 0)
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
  // add_symbol(NODE_UNKNOWN, NULL, NULL, "true");
  // add_symbol(NODE_UNKNOWN, NULL, NULL, "false");

  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "previous");
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "write");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "real3");  // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "Field3"); // TODO RECHECK
  // add_symbol(NODE_FUNCTION_ID, NULL, NULL, "Matrix"); // TODO RECHECK
  // add_symbol(NODE_FUNCTION_ID, NULL, NULL, "Matrix"); // TODO RECHECK
  // add_symbol(NODE_FUNCTION_ID, NULL, NULL, "previous"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "dot");   // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "cross"); // TODO RECHECK
  add_symbol(NODE_FUNCTION_ID, NULL, NULL, "exp");   // TODO RECHECK

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

  add_symbol(NODE_DCONST_ID, NULL, "int3", "AC_multigpu_offset");
  add_symbol(NODE_DCONST_ID, NULL, "int3", "AC_global_grid_n");

  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsx");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsy");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_dsz");

  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsx");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsy");
  add_symbol(NODE_DCONST_ID, NULL, "AcReal", "AC_inv_dsz");

  // (BC types do not belong here, BCs not handled with the DSL)
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_bot_x");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_bot_y");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_bot_z");

  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_top_x");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_top_y");
  add_symbol(NODE_DCONST_ID, NULL, "int", "AC_bc_type_top_z");
  // Astaroth 2.0 backwards compatibility END
}

static inline void
print_symbol_table(void)
{
  printf("\n---\n");
  printf("Symbol table:\n");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
    printf("%lu: ", i);
    printf("%s ", symbol_table[i].identifier);

    if (strlen(symbol_table[i].tspecifier) > 0)
      printf("(%s) ", symbol_table[i].tspecifier);
    else
      printf("(auto) ");

    if (symbol_table[i].type & NODE_FUNCTION_ID)
      printf("(%s function)",
             symbol_table[i].type & NODE_KFUNCTION_ID ? "kernel" : "device");

    if (symbol_table[i].type & NODE_DCONST_ID)
      printf("(dconst)");

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

  // Infix translation
  if (stream)
    if (node->infix)
      fprintf(stream, "%s", node->infix);

  if (node->buffer) {
    if (node->token == IDENTIFIER) {

      const Symbol* symbol = symboltable_lookup(node->buffer);
      // TODO REMOVE BELOW--------------------------------
      // Though note that symbol table search must be reversed!!
      if (symbol && node->type & NODE_FUNCTION_PARAM) {
        fprintf(stderr,
                "Error! Symbol '%s' already present in symbol table. Shadowing "
                "is not allowed.\n",
                node->buffer);
        assert(0);
      }
      // TODO REMOVE ABOVE--------------------------------------
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
    /*
    if (stream)
      fprintf(stream, "%s", node->buffer);
    */
    // Astaroth 2.0 backwards compatibility START
    if (stream) {
      const Symbol* symbol = symboltable_lookup(node->buffer);
      if (symbol && symbol->type & NODE_DCONST_ID)
        fprintf(stream, "DCONST(%s)", node->buffer);
      else
        fprintf(stream, "%s", node->buffer);
    }
    // Astaroth 2.0 backwards compatibility END
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

static void
gen_dconsts(const ASTNode* root, FILE* stream)
{
  fprintf(stream, "/*"); // Astaroth 2.0 backwards compatibility
  symboltable_reset();
  traverse(root, NODE_FUNCTION | NODE_FIELD | NODE_STENCIL | NODE_HOSTDEFINE,
           stream);
  fprintf(stream, "*/"); // Astaroth 2.0 backwards compatibility

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

  symboltable_reset();
  traverse(root, NODE_DCONST | NODE_FIELD | NODE_FUNCTION | NODE_STENCIL, fp);

  symboltable_reset();
  traverse(root, 0, NULL);

  // Enums
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_FIELD_ID)
      fprintf(fp, "%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_FIELDS} Field;");

  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK START---------------------------
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

  // This is not really needed any more, the kernel function pointer is now
  // exposed in the API, so one could use that directly instead of handles.
  fprintf(fp, "typedef enum {");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_KFUNCTION_ID)
      fprintf(fp, "KERNEL_%s,", symbol_table[i].identifier);
  fprintf(fp, "NUM_KERNELS} AcKernel;");
  // ASTAROTH 2.0 BACKWARDS COMPATIBILITY BLOCK END-----------------------------

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
  gen_dconsts(root, stream);

  // Stencils
  fprintf(stream, "typedef enum{");
  for (size_t i = 0; i < num_symbols[current_nest]; ++i)
    if (symbol_table[i].type & NODE_STENCIL_ID)
      fprintf(stream, "stencil_%s,", symbol_table[i].identifier);
  fprintf(stream, "NUM_STENCILS} Stencil;");

  // Stencil generator
  symboltable_reset();
#define STENCILGEN_SRC "stencilgen.c"
#define STENCILGEN_EXEC "stencilgen.out"
  FILE* stencilgen = fopen(STENCILGEN_SRC, "w");
  assert(stencilgen);
  fprintf(stencilgen,
          "#include <stdio.h>\n"
          "#include <stdlib.h>\n"
          "#define STENCIL_ORDER (%lu)\n"
          "#define NN (STENCIL_ORDER+1)\n"
          "#define STENCIL_DEPTH (NN)\n"
          "#define STENCIL_HEIGHT (NN)\n"
          "#define STENCIL_WIDTH (NN)\n"
          "#define NUM_STENCILS (%lu)\n"
          "#define NUM_FIELDS (%lu)\n",
          stencil_order, num_stencils, num_fields);
  fprintf(stencilgen, "static char* "
                      "stencils[][NN][NN][NN] = {");
  traverse(root,
           NODE_STENCIL_ID | NODE_DCONST | NODE_FIELD | NODE_FUNCTION |
               NODE_HOSTDEFINE,
           stencilgen);
  fprintf(stencilgen, "};");
  const char* stencilgen_main = R"(
int main(void) {
  for (int field = 0; field < NUM_FIELDS; ++field) {
      printf("{\n\tconst AcReal* __restrict__ in = vba.in[%%d];\n", field);
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
          for (int height = 0; height < STENCIL_HEIGHT; ++height) {
              for (int width = 0; width < STENCIL_WIDTH; ++width) {
                  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                      if (stencils[stencil][depth][height][width] != 0) {
                          printf("\tprocessed_stencils[%%d][%%d] += %%s * in[IDX(vertexIdx.x + (%%d), vertexIdx.y + (%%d), vertexIdx.z + (%%d))];\n",
                                  field, stencil, stencils[stencil][depth][height][width],
                                  -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                                  -STENCIL_ORDER / 2 + depth);
                      }
                  }
              }
          }
      }
      printf("}\n");
  }
}
                      )";
  fprintf(stencilgen, stencilgen_main);
  fclose(stencilgen);

  // Compile
  system("gcc -std=c11 -Wall -Wextra -Wdouble-promotion "
         "-Wfloat-conversion -Wshadow " STENCILGEN_SRC " "
         "-o " STENCILGEN_EXEC);

  // Generate stencils
  FILE* proc = popen("./" STENCILGEN_EXEC, "r");
  assert(proc);

  char sdefinitions[1 * 1024 * 1024];
  char buf[4096];
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
  char sfunctions[1024 * 1024];
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