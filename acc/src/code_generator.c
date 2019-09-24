/*
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acc.tab.h"
#include "ast.h"

ASTNode* root = NULL;

static const char inout_name_prefix[] = "handle_";
typedef enum { STENCIL_ASSEMBLY, STENCIL_PROCESS, STENCIL_HEADER } CompilationType;
static CompilationType compilation_type;

/*
 * =============================================================================
 * Translation
 * =============================================================================
 */
#define TRANSLATION_TABLE_SIZE (1024)
static const char* translation_table[TRANSLATION_TABLE_SIZE] = {
    [0] = NULL,
    // Control flow
    [IF]    = "if",
    [ELSE]  = "else",
    [ELIF]  = "else if",
    [WHILE] = "while",
    [FOR]   = "for",
    // Type specifiers
    [VOID]        = "void",
    [INT]         = "int",
    [INT3]        = "int3",
    [SCALAR]      = "AcReal",
    [VECTOR]      = "AcReal3",
    [MATRIX]      = "AcMatrix",
    [SCALARFIELD] = "AcReal",
    [SCALARARRAY] = "const AcReal* __restrict__",
    [COMPLEX]     = "acComplex",
    // Type qualifiers
    [KERNEL] = "template <int step_number>  static __global__",
    //__launch_bounds__(RK_THREADBLOCK_SIZE,
    // RK_LAUNCH_BOUND_MIN_BLOCKS),
    [PREPROCESSED] = "static __device__ "
                     "__forceinline__",
    [CONSTANT] = "const",
    [IN]       = "in",
    [OUT]      = "out",
    [UNIFORM]  = "uniform",
    // ETC
    [INPLACE_INC] = "++",
    [INPLACE_DEC] = "--",
    // Unary
    [','] = ",",
    [';'] = ";\n",
    ['('] = "(",
    [')'] = ")",
    ['['] = "[",
    [']'] = "]",
    ['{'] = "{\n",
    ['}'] = "}\n",
    ['='] = "=",
    ['+'] = "+",
    ['-'] = "-",
    ['/'] = "/",
    ['*'] = "*",
    ['<'] = "<",
    ['>'] = ">",
    ['!'] = "!",
    ['.'] = "."};

static const char*
translate(const int token)
{
    assert(token >= 0);
    assert(token < TRANSLATION_TABLE_SIZE);
    if (token > 0) {
        if (!translation_table[token])
            printf("ERROR: unidentified token %d\n", token);
        assert(translation_table[token]);
    }

    return translation_table[token];
}

/*
 * =============================================================================
 * Symbols
 * =============================================================================
 */
typedef enum {
    SYMBOLTYPE_FUNCTION,
    SYMBOLTYPE_FUNCTION_PARAMETER,
    SYMBOLTYPE_OTHER,
    NUM_SYMBOLTYPES
} SymbolType;

#define MAX_ID_LEN (128)
typedef struct {
    SymbolType type;
    int type_qualifier;
    int type_specifier;
    char identifier[MAX_ID_LEN];
} Symbol;

#define SYMBOL_TABLE_SIZE (4096)
static Symbol symbol_table[SYMBOL_TABLE_SIZE] = {};
static int num_symbols                        = 0;

static int
symboltable_lookup(const char* identifier)
{
    if (!identifier)
        return -1;

    for (int i = 0; i < num_symbols; ++i)
        if (strcmp(identifier, symbol_table[i].identifier) == 0)
            return i;

    return -1;
}

static void
add_symbol(const SymbolType type, const int tqualifier, const int tspecifier, const char* id)
{
    assert(num_symbols < SYMBOL_TABLE_SIZE);

    symbol_table[num_symbols].type           = type;
    symbol_table[num_symbols].type_qualifier = tqualifier;
    symbol_table[num_symbols].type_specifier = tspecifier;
    strcpy(symbol_table[num_symbols].identifier, id);

    ++num_symbols;
}

static void
rm_symbol(const int handle)
{
    assert(handle >= 0 && handle < num_symbols);
    assert(num_symbols > 0);

    if (&symbol_table[handle] != &symbol_table[num_symbols - 1])
        memcpy(&symbol_table[handle], &symbol_table[num_symbols - 1], sizeof(Symbol));
    --num_symbols;
}

static void
print_symbol(const int handle)
{
    assert(handle < SYMBOL_TABLE_SIZE);

    const char* fields[]    = {translate(symbol_table[handle].type_qualifier),
                            translate(symbol_table[handle].type_specifier),
                            symbol_table[handle].identifier};
    const size_t num_fields = sizeof(fields) / sizeof(fields[0]);

    for (size_t i = 0; i < num_fields; ++i)
        if (fields[i])
            printf("%s ", fields[i]);
}

static void
translate_latest_symbol(void)
{
    const int handle = num_symbols - 1;
    assert(handle < SYMBOL_TABLE_SIZE);

    Symbol* symbol = &symbol_table[handle];

    // FUNCTION
    if (symbol->type == SYMBOLTYPE_FUNCTION) {
        // KERNEL FUNCTION
        if (symbol->type_qualifier == KERNEL) {
            printf("%s %s\n%s", translate(symbol->type_qualifier),
                   translate(symbol->type_specifier), symbol->identifier);
        }
        // PREPROCESSED FUNCTION
        else if (symbol->type_qualifier == PREPROCESSED) {
            printf("%s %s\npreprocessed_%s", translate(symbol->type_qualifier),
                   translate(symbol->type_specifier), symbol->identifier);
        }
        // OTHER FUNCTION
        else {
            const char* regular_function_decorator = "static __device__ "
                                                     "__forceinline__";
            printf("%s %s %s\n%s", regular_function_decorator,
                   translate(symbol->type_qualifier) ? translate(symbol->type_qualifier) : "",
                   translate(symbol->type_specifier), symbol->identifier);
        }
    }
    // FUNCTION PARAMETER
    else if (symbol->type == SYMBOLTYPE_FUNCTION_PARAMETER) {
        if (symbol->type_qualifier == IN || symbol->type_qualifier == OUT) {
            if (compilation_type == STENCIL_ASSEMBLY)
                printf("const __restrict__ %s* %s", translate(symbol->type_specifier),
                       symbol->identifier);
            else if (compilation_type == STENCIL_PROCESS)
                printf("const %sData& %s", translate(symbol->type_specifier), symbol->identifier);
            else
                printf("Invalid compilation type %d, IN and OUT qualifiers not supported\n",
                       compilation_type);
        }
        else {
            print_symbol(handle);
        }
    }
    // UNIFORM
    else if (symbol->type_qualifier == UNIFORM) {
        // if (compilation_type != STENCIL_HEADER) {
        //    printf("ERROR: %s can only be used in stencil headers\n", translation_table[UNIFORM]);
        //}
        /* Do nothing */
    }
    // IN / OUT
    else if (symbol->type != SYMBOLTYPE_FUNCTION_PARAMETER &&
             (symbol->type_qualifier == IN || symbol->type_qualifier == OUT)) {

        printf("static __device__ const %s %s%s",
               symbol->type_specifier == SCALARFIELD ? "int" : "int3", inout_name_prefix,
               symbol_table[handle].identifier);
        if (symbol->type_specifier == VECTOR)
            printf(" = make_int3");
    }
    // OTHER
    else {
        print_symbol(handle);
    }
}

static inline void
print_symbol_table(void)
{
    for (int i = 0; i < num_symbols; ++i) {
        printf("%d: ", i);
        const char* fields[] = {translate(symbol_table[i].type_qualifier),
                                translate(symbol_table[i].type_specifier),
                                symbol_table[i].identifier};

        const size_t num_fields = sizeof(fields) / sizeof(fields[0]);
        for (size_t j = 0; j < num_fields; ++j)
            if (fields[j])
                printf("%s ", fields[j]);

        if (symbol_table[i].type == SYMBOLTYPE_FUNCTION)
            printf("(function)");
        else if (symbol_table[i].type == SYMBOLTYPE_FUNCTION_PARAMETER)
            printf("(function parameter)");
        else
            printf("(other)");
        printf("\n");
    }
}

/*
 * =============================================================================
 * State
 * =============================================================================
 */
static bool inside_declaration                    = false;
static bool inside_function_declaration           = false;
static bool inside_function_parameter_declaration = false;

static bool inside_kernel       = false;
static bool inside_preprocessed = false;

static int scope_start = 0;

/*
 * =============================================================================
 * AST traversal
 * =============================================================================
 */

static int compound_statement_nests = 0;

static void
traverse(const ASTNode* node)
{
    // Prefix logic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (node->type == NODE_FUNCTION_DECLARATION)
        inside_function_declaration = true;
    if (node->type == NODE_FUNCTION_PARAMETER_DECLARATION)
        inside_function_parameter_declaration = true;
    if (node->type == NODE_DECLARATION)
        inside_declaration = true;

    if (!inside_declaration && translate(node->prefix))
        printf("%s", translate(node->prefix));

    if (node->type == NODE_COMPOUND_STATEMENT)
        ++compound_statement_nests;

    // BOILERPLATE START////////////////////////////////////////////////////////
    if (node->type == NODE_TYPE_QUALIFIER && node->token == KERNEL)
        inside_kernel = true;

    // Kernel parameter boilerplate
    const char* kernel_parameter_boilerplate = "GEN_KERNEL_PARAM_BOILERPLATE";
    if (inside_kernel && node->type == NODE_FUNCTION_PARAMETER_DECLARATION) {
        printf("%s", kernel_parameter_boilerplate);

        if (node->lhs != NULL) {
            printf("Compilation error: function parameters for Kernel functions not allowed!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Kernel builtin variables boilerplate (read input/output arrays and setup
    // indices)
    const char* kernel_builtin_variables_boilerplate = "GEN_KERNEL_BUILTIN_VARIABLES_"
                                                       "BOILERPLATE();";
    if (inside_kernel && node->type == NODE_COMPOUND_STATEMENT && compound_statement_nests == 1) {
        printf("%s ", kernel_builtin_variables_boilerplate);

        for (int i = 0; i < num_symbols; ++i) {
            if (symbol_table[i].type_qualifier == IN) {
                printf("const %sData %s = READ(%s%s);\n", translate(symbol_table[i].type_specifier),
                       symbol_table[i].identifier, inout_name_prefix, symbol_table[i].identifier);
            }
            else if (symbol_table[i].type_qualifier == OUT) {
                printf("%s %s = READ_OUT(%s%s);", translate(symbol_table[i].type_specifier),
                       symbol_table[i].identifier, inout_name_prefix, symbol_table[i].identifier);
                // printf("%s %s = buffer.out[%s%s][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)];\n",
                // translate(symbol_table[i].type_specifier), symbol_table[i].identifier,
                // inout_name_prefix, symbol_table[i].identifier);
            }
        }
    }

    // Preprocessed parameter boilerplate
    if (node->type == NODE_TYPE_QUALIFIER && node->token == PREPROCESSED)
        inside_preprocessed = true;
    static const char preprocessed_parameter_boilerplate
        [] = "const int3& vertexIdx, const int3& globalVertexIdx, ";
    if (inside_preprocessed && node->type == NODE_FUNCTION_PARAMETER_DECLARATION)
        printf("%s ", preprocessed_parameter_boilerplate);
    // BOILERPLATE END////////////////////////////////////////////////////////

    // Enter LHS
    if (node->lhs)
        traverse(node->lhs);

    // Infix logic  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (!inside_declaration && translate(node->infix))
        printf("%s ", translate(node->infix));

    if (node->type == NODE_FUNCTION_DECLARATION)
        inside_function_declaration = false;

    // If the node is a subscript expression and the expression list inside it is not empty
    if (node->type == NODE_MULTIDIM_SUBSCRIPT_EXPRESSION && node->rhs)
        printf("IDX(");

    // Do a regular translation
    if (!inside_declaration) {
        const int handle = symboltable_lookup(node->buffer);
        if (handle >= 0) { // The variable exists in the symbol table
            const Symbol* symbol = &symbol_table[handle];

            if (symbol->type_qualifier == UNIFORM) {
                if (inside_kernel && symbol->type_specifier == SCALARARRAY) {
                    printf("buffer.profiles[%s] ", symbol->identifier);
                }
                else {
                    printf("DCONST(%s) ", symbol->identifier);
                }
            }
            else {
                // Do a regular translation
                if (translate(node->token))
                    printf("%s ", translate(node->token));
                if (node->buffer)
                    printf("%s ", node->buffer);
            }
        }
        else {
            // Do a regular translation
            if (translate(node->token))
                printf("%s ", translate(node->token));
            if (node->buffer)
                printf("%s ", node->buffer);
        }
    }

    if (node->type == NODE_FUNCTION_DECLARATION) {
        scope_start = num_symbols;
    }

    // Enter RHS
    if (node->rhs)
        traverse(node->rhs);

    // Postfix logic  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // If the node is a subscript expression and the expression list inside it is not empty
    if (node->type == NODE_MULTIDIM_SUBSCRIPT_EXPRESSION && node->rhs)
        printf(")"); // Closing bracket of IDX()

    // Generate writeback boilerplate for OUT fields
    if (inside_kernel && node->type == NODE_COMPOUND_STATEMENT && compound_statement_nests == 1) {
        for (int i = 0; i < num_symbols; ++i) {
            if (symbol_table[i].type_qualifier == OUT) {
                printf("WRITE_OUT(%s%s, %s);\n", inout_name_prefix, symbol_table[i].identifier,
                       symbol_table[i].identifier);
                // printf("buffer.out[%s%s][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = %s;\n",
                // inout_name_prefix, symbol_table[i].identifier, symbol_table[i].identifier);
            }
        }
    }

    if (!inside_declaration && translate(node->postfix))
        printf("%s", translate(node->postfix));

    if (node->type == NODE_DECLARATION) {
        inside_declaration = false;

        int tqual = 0;
        int tspec = 0;
        if (node->lhs && node->lhs->lhs) {
            if (node->lhs->lhs->type == NODE_TYPE_QUALIFIER)
                tqual = node->lhs->lhs->token;
            else if (node->lhs->lhs->type == NODE_TYPE_SPECIFIER)
                tspec = node->lhs->lhs->token;
        }
        if (node->lhs && node->lhs->rhs) {
            if (node->lhs->rhs->type == NODE_TYPE_SPECIFIER)
                tspec = node->lhs->rhs->token;
        }

        // Determine symbol type
        SymbolType symboltype = SYMBOLTYPE_OTHER;
        if (inside_function_declaration)
            symboltype = SYMBOLTYPE_FUNCTION;
        else if (inside_function_parameter_declaration)
            symboltype = SYMBOLTYPE_FUNCTION_PARAMETER;

        // Determine identifier
        if (node->rhs->type == NODE_IDENTIFIER) {
            add_symbol(symboltype, tqual, tspec, node->rhs->buffer); // Ordinary
            translate_latest_symbol();
        }
        else {
            add_symbol(symboltype, tqual, tspec,
                       node->rhs->lhs->buffer); // Array
            translate_latest_symbol();
            // Traverse the expression once again, this time with
            // "inside_declaration" flag off
            printf("%s ", translate(node->rhs->infix));
            if (node->rhs->rhs)
                traverse(node->rhs->rhs);
            printf("%s ", translate(node->rhs->postfix));
        }
    }

    if (node->type == NODE_COMPOUND_STATEMENT)
        --compound_statement_nests;

    if (node->type == NODE_FUNCTION_PARAMETER_DECLARATION)
        inside_function_parameter_declaration = false;

    if (node->type == NODE_FUNCTION_DEFINITION) {
        while (num_symbols > scope_start)
            rm_symbol(num_symbols - 1);

        inside_kernel       = false;
        inside_preprocessed = false;
    }
}

// TODO: these should use the generic type names SCALAR and VECTOR
static void
generate_preprocessed_structures(void)
{
    // PREPROCESSED DATA STRUCT
    printf("\n");
    printf("typedef struct {\n");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            printf("%s %s;\n", translate(symbol_table[i].type_specifier),
                   symbol_table[i].identifier);
    }
    printf("} %sData;\n", translate(SCALAR));

    // FILLING THE DATA STRUCT
    printf("static __device__ __forceinline__ AcRealData\
            read_data(const int3& vertexIdx,\
                const int3& globalVertexIdx,\
            AcReal* __restrict__ buf[], const int handle)\
            {\n\
                %sData data;\n",
           translate(SCALAR));

    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            printf("data.%s = preprocessed_%s(vertexIdx, globalVertexIdx, buf[handle]);\n",
                   symbol_table[i].identifier, symbol_table[i].identifier);
    }
    printf("return data;\n");
    printf("}\n");

    // FUNCTIONS FOR ACCESSING MEMBERS OF THE PREPROCESSED STRUCT
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            printf("static __device__ __forceinline__ %s\
                    %s(const AcRealData& data)\
                    {\n\
                        return data.%s;\
                    }\n",
                   translate(symbol_table[i].type_specifier), symbol_table[i].identifier,
                   symbol_table[i].identifier);
    }

    // Syntactic sugar: generate also a Vector data struct
    printf("\
        typedef struct {\
            AcRealData x;\
            AcRealData y;\
            AcRealData z;\
        } AcReal3Data;\
        \
        static __device__ __forceinline__ AcReal3Data\
        read_data(const int3& vertexIdx,\
                  const int3& globalVertexIdx,\
                  AcReal* __restrict__ buf[], const int3& handle)\
        {\
            AcReal3Data data;\
        \
            data.x = read_data(vertexIdx, globalVertexIdx, buf, handle.x);\
            data.y = read_data(vertexIdx, globalVertexIdx, buf, handle.y);\
            data.z = read_data(vertexIdx, globalVertexIdx, buf, handle.z);\
        \
            return data;\
        }\
    ");
}

static void
generate_header(void)
{
    printf("\n#pragma once\n");

    // Int params
    printf("#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_specifier == INT) {
            printf("\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    printf("\n\n");

    // Int3 params
    printf("#define AC_FOR_USER_INT3_PARAM_TYPES(FUNC)");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_specifier == INT3) {
            printf("\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    printf("\n\n");

    // Scalar params
    printf("#define AC_FOR_USER_REAL_PARAM_TYPES(FUNC)");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_specifier == SCALAR) {
            printf("\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    printf("\n\n");

    // Vector params
    printf("#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC)");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_specifier == VECTOR) {
            printf("\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    printf("\n\n");

    // Scalar fields
    printf("#define AC_FOR_VTXBUF_HANDLES(FUNC)");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_specifier == SCALARFIELD) {
            printf("\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    printf("\n\n");

    // Scalar arrays
    printf("#define AC_FOR_SCALARARRAY_HANDLES(FUNC)");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_specifier == SCALARARRAY) {
            printf("\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    printf("\n\n");

    /*
    printf("\n");
    printf("typedef struct {\n");
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            printf("%s %s;\n", translate(symbol_table[i].type_specifier),
                   symbol_table[i].identifier);
    }
    printf("} %sData;\n", translate(SCALAR));
    */
}

static void
generate_library_hooks(void)
{
    for (int i = 0; i < num_symbols; ++i) {
        if (symbol_table[i].type_qualifier == KERNEL) {
            printf("GEN_DEVICE_FUNC_HOOK(%s)\n", symbol_table[i].identifier);
            // printf("GEN_NODE_FUNC_HOOK(%s)\n", symbol_table[i].identifier);
        }
    }
}

int
main(int argc, char** argv)
{
    if (argc == 2) {
        if (!strcmp(argv[1], "-sas"))
            compilation_type = STENCIL_ASSEMBLY;
        else if (!strcmp(argv[1], "-sps"))
            compilation_type = STENCIL_PROCESS;
        else if (!strcmp(argv[1], "-sdh"))
            compilation_type = STENCIL_HEADER;
        else {
            printf("Unknown flag %s. Generating stencil assembly.\n", argv[1]);
            return EXIT_FAILURE;
        }
    }
    else {
        printf("Usage: ./acc [flags]\n"
               "Flags:\n"
               "\t-sas - Generates code for the stencil assembly stage\n"
               "\t-sps - Generates code for the stencil processing stage\n"
               "\t-hh  - Generates stencil definitions from a header file\n");
        printf("\n");
        return EXIT_FAILURE;
    }

    root = astnode_create(NODE_UNKNOWN, NULL, NULL);

    const int retval = yyparse();
    if (retval) {
        printf("COMPILATION FAILED\n");
        return EXIT_FAILURE;
    }

    // Traverse
    traverse(root);
    if (compilation_type == STENCIL_ASSEMBLY)
        generate_preprocessed_structures();
    else if (compilation_type == STENCIL_HEADER)
        generate_header();
    else if (compilation_type == STENCIL_PROCESS)
        generate_library_hooks();

    // print_symbol_table();

    // Cleanup
    astnode_destroy(root);
    // printf("COMPILATION SUCCESS\n");
    return EXIT_SUCCESS;
}
