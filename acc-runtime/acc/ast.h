/*
    Copyright (C) 2021, Johannes Pekkila.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
/*
 * Abstract Syntax Tree
 */
#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#define BUFFER_SIZE (4096)

typedef enum {
  NODE_UNKNOWN            = 0,
  NODE_PRIMARY_EXPRESSION = (1 << 0),
  NODE_DFUNCTION          = (1 << 1),
  NODE_KFUNCTION          = (1 << 2),
  NODE_FUNCTION_ID        = (1 << 3),
  NODE_FUNCTION           = NODE_DFUNCTION | NODE_KFUNCTION,
  NODE_FUNCTION_PARAM     = (1 << 4),
  NODE_BINARY             = (1 << 5),
  NODE_BEGIN_SCOPE        = (1 << 6),
  NODE_DECLARATION        = (1 << 7),
  NODE_TSPEC              = (1 << 8),
  NODE_TQUAL              = (1 << 9),
  NODE_STENCIL            = (1 << 10),
  NODE_EXPRESSION         = (1 << 11),
  NODE_VARIABLE           = (1 << 12),
  NODE_VARIABLE_ID        = (1 << 13),
  NODE_ARRAY_INITIALIZER  = (1 << 14),
  NODE_DCONST             = (1 << 15),
  NODE_TERNARY            = (1 << 16),
  NODE_MEMBER_ID          = (1 << 17),
  NODE_HOSTDEFINE         = (1 << 18),
  NODE_ASSIGNMENT         = (1 << 19),
  NODE_INPUT              = (1 << 20),
  NODE_DEF                = (1 << 21) | NODE_BEGIN_SCOPE,
  NODE_RETURN             = (1 << 22),
  NODE_ARRAY_ACCESS       = (1 << 23),
  NODE_STRUCT_INITIALIZER = (1 << 24),
  NODE_IF                 = (1 << 25),
  NODE_FUNCTION_CALL      = (1 << 26),
  NODE_DFUNCTION_ID       = (1 << 27),
  NODE_ASSIGN_LIST        = (1 << 28),
  NODE_NO_OUT             = (1 << 29),
  NODE_STRUCT_EXPRESSION  = (1 << 30),
  NODE_ENUM_DEF           = (NODE_DEF + 0 + NODE_NO_OUT),
  NODE_STRUCT_DEF         = (NODE_DEF + 1 + NODE_NO_OUT),
  NODE_TASKGRAPH_DEF      = (NODE_DEF + 2 + NODE_NO_OUT),
  NODE_BOUNDCONDS_DEF     = (NODE_DEF + 3 + NODE_NO_OUT),
  NODE_BINARY_EXPRESSION  = (NODE_BINARY + NODE_EXPRESSION),
  NODE_TERNARY_EXPRESSION = (NODE_TERNARY+ NODE_EXPRESSION),
  NODE_ANY                = ~0,
} NodeType;

typedef struct astnode_s {
  int id;
  struct astnode_s* parent;
  struct astnode_s* lhs;
  struct astnode_s* rhs;
  NodeType type; // Type of the AST node
  char* buffer;  // Indentifiers and other strings (empty by default)

  int token;     // Type of a terminal (that is not a simple char)
  char* prefix;  // Strings. Also makes the grammar since we don't have
  char* infix;   // to divide it into max two-child rules
  char* postfix; // (which makes it much harder to read)
  bool is_constexpr; //Whether the node represents information known at compile time
  char* expr_type; //The type of the expr the node represent
  bool no_auto;
} ASTNode;


static inline ASTNode*
astnode_dup(const ASTNode* node, ASTNode* parent)
{
	ASTNode* res = (ASTNode*)calloc(1,sizeof(node[0]));
	res->id = node->id;
	res->type = node->type;
	res->parent = parent;
	res -> token = node->token;
	res -> is_constexpr = node->is_constexpr;
	res -> expr_type = node->expr_type;
	res -> no_auto = node->no_auto;

	if(node->buffer)
		res->buffer = strdup(node->buffer);
	if(node->prefix)
		res->prefix= strdup(node->prefix);
	if(node->infix)
		res->infix = strdup(node->infix);
	if(node->postfix)
		res->postfix = strdup(node->postfix);
	if(node->lhs)
		res->lhs= astnode_dup(node->lhs,res);
	if(node->rhs)
		res->rhs= astnode_dup(node->rhs,res);
	return res;
}



static inline ASTNode*
astnode_create(const NodeType type, ASTNode* lhs, ASTNode* rhs)
{
  static int id_counter = 0;
  ASTNode* node = (ASTNode*)calloc(1, sizeof(node[0]));

  node->id              = id_counter++;
  node->type            = type;
  node->lhs             = lhs;
  node->rhs             = rhs;
  node->buffer          = NULL;
  node->is_constexpr    = false;
  node->no_auto         = false;
  node->expr_type            = NULL;

  node->token  = 0;
  node->prefix = node->infix = node->postfix = NULL;

  if (lhs)
    node->lhs->parent = node;

  if (rhs)
    node->rhs->parent = node;

  return node;
}

static inline void
astnode_destroy(ASTNode* node)
{
  if (node->lhs)
    astnode_destroy(node->lhs);
  if (node->rhs)
    astnode_destroy(node->rhs);
  if (node->buffer)
    free(node->buffer);
  if (node->prefix)
    free(node->prefix);
  if (node->infix)
    free(node->infix);
  if (node->postfix)
    free(node->postfix);
  free(node);
}

static inline void
astnode_set_buffer(const char* buffer, ASTNode* node)
{
  if (node->buffer)
    free(node->buffer);
  node->buffer = strdup(buffer);
}

static inline void
astnode_set_prefix(const char* prefix, ASTNode* node)
{
  if (node->prefix)
    free(node->prefix);
  node->prefix = strdup(prefix);
}

static inline void
astnode_set_infix(const char* infix, ASTNode* node)
{
  if (node->infix)
    free(node->infix);
  node->infix = strdup(infix);
}

static inline void
astnode_set_postfix(const char* postfix, ASTNode* node)
{
  if (node->postfix)
    free(node->postfix);
  node->postfix = strdup(postfix);
}

static inline void
astnode_print(const ASTNode* node)
{
  printf("%u (%p)\n", node->type, node);
  printf("\tid:      %d\n", node->id);
  printf("\tparent:  %p\n", node->parent);
  printf("\tlhs:     %p\n", node->lhs);
  printf("\trhs:     %p\n", node->rhs);
  printf("\tbuffer:  %s\n", node->buffer);
  printf("\ttoken:   %d\n", node->token);
  printf("\tprefix:  %p (\"%s\")\n", node->prefix, node->prefix);
  printf("\tinfix:   %p (\"%s\")\n", node->infix, node->infix);
  printf("\tpostfix: %p (\"%s\")\n", node->postfix, node->postfix);
}
#include "vecs.h"


static inline  void combine_buffers_recursive(const ASTNode* node, char* res){
  if(node->buffer)
    strcat(res,node->buffer);
  if(node->lhs)
    combine_buffers_recursive(node->lhs, res);
  if(node->rhs)
    combine_buffers_recursive(node->rhs, res);
}
static inline  void combine_buffers(const ASTNode* node, char* res){
  res[0] = '\0';	
  combine_buffers_recursive(node,res);
}
static inline void combine_all_recursive(const ASTNode* node, char* res){
  if(node->prefix)
    strcat(res,node->prefix);
  if(node->lhs)
    combine_all_recursive(node->lhs, res);
  if(node->buffer)
    strcat(res,node->buffer);
  if(node->infix)
    strcat(res,node->infix);
  if(node->rhs)
    combine_all_recursive(node->rhs, res);
  if(node->postfix)
    strcat(res,node->postfix);
}
static inline void 
strip_whitespace(char *str) {
    char *dest = str; // Destination pointer to overwrite the original string
    char *src = str;  // Source pointer to traverse the original string

    // Skip leading whitespace
    while (isspace((unsigned char)(*src))) {
        src++;
    }

    // Copy non-whitespace characters to the destination
    while (*src) {
        if (!isspace((unsigned char)(*src))) {
            *dest++ = *src;
        }
        src++;
    }

    // Null-terminate the destination string
    *dest = '\0';
}
static inline void combine_all(const ASTNode* node, char* res){
  res[0] = '\0';	
  combine_all_recursive(node,res);
  strip_whitespace(res);
}

static inline ASTNode*
get_node_by_token(const int token, const ASTNode* node)
{
  assert(node);

  if (node->token == token)
    return (ASTNode*) node;
  else if (node->lhs && get_node_by_token(token, node->lhs))
    return get_node_by_token(token, node->lhs);
  else if (node->rhs && get_node_by_token(token, node->rhs))
    return get_node_by_token(token, node->rhs);
  else
    return NULL;
}
static inline ASTNode*
get_node_by_buffer(const char* test, const ASTNode* node)
{
  assert(node);

  ASTNode* res = NULL;
  if (node->buffer && !strcmp(test,node->buffer))
    res = (ASTNode*) node;
  if (node->lhs && !res)
    res = get_node_by_buffer(test, node->lhs);
  if (node->rhs && !res)
    res = get_node_by_buffer(test, node->rhs);
  return res;
}
static inline ASTNode*
get_node_by_buffer_and_type(const char* test, const NodeType type, const ASTNode* node)
{
  assert(node);

  ASTNode* res = NULL;
  if (node->buffer && !strcmp(test,node->buffer) && node->type & type)
    res =  (ASTNode*) node;
  if (node->lhs && !res)
    res = get_node_by_buffer_and_type(test, type, node->lhs);
  if (node->rhs && !res)
    res = get_node_by_buffer_and_type(test, type, node->rhs);
  return res;
}

static inline ASTNode*
get_node_by_buffer_and_token(const char* test, const int token, const ASTNode* node)
{
  assert(node);

  ASTNode* res = NULL;
  if (node->buffer && !strcmp(test,node->buffer) && node->token == token)
    res =  (ASTNode*) node;
  if (node->lhs && !res)
    res = get_node_by_buffer_and_token(test, token, node->lhs);
  if (node->rhs && !res)
    res = get_node_by_buffer_and_token(test, token, node->rhs);
  return res;
}
static inline ASTNode*
get_node_by_id(const int id, const ASTNode* node)
{
  assert(node);

  ASTNode* res = NULL;
  if (node->id == id)
    res = (ASTNode*) node;
  if (node->lhs && !res)
    res = get_node_by_id(id, node->lhs);
  if (node->rhs && !res)
    res = get_node_by_id(id, node->rhs);
  return res;
}
static const inline ASTNode*
get_parent_node(const NodeType type, const ASTNode* node)
{
  if(!node->parent)
    return NULL;
  if (node->parent->type & type)
    return node->parent;
  return get_parent_node(type, node->parent);
}
static const inline ASTNode*
get_parent_node_exclusive(const NodeType type, const ASTNode* node)
{
  if(!node->parent)
    return NULL;
  if (node->parent->type == type)
    return node->parent;
  return get_parent_node(type, node->parent);
}
typedef struct CodeGenInput
{
	string_vec const_ints;
	string_vec const_int_values;
} CodeGenInput;

static inline char* itoa(const int x)
{
	char* tmp = (char*)malloc(100*sizeof(char));
	sprintf(tmp,"%d",x);
	const int n = strlen(tmp);
	char* res = (char*)malloc(n*sizeof(char));
	sprintf(res,"%d",x);
	free(tmp);
	return res;
}
static inline void
set_buffers_empty(ASTNode* node)
{
	node->buffer = NULL;
	node->prefix = NULL;
	node->infix  = NULL;
	node->postfix = NULL;
	if(node->lhs)
		set_buffers_empty(node->lhs);
	if(node->rhs)
		set_buffers_empty(node->rhs);
}
static inline void
add_node_type(const NodeType type, ASTNode* node, const char* str_to_check)
{
	if(node->lhs)
		add_node_type(type,node->lhs,str_to_check);
	if(node->rhs)
		add_node_type(type,node->rhs,str_to_check);
	if(!str_to_check || (node->buffer && !strcmp(node->buffer,str_to_check)))
		node->type |= type;
}
static inline void
add_no_auto(ASTNode* node, const char* str_to_check)
{
	if(node->lhs)
		add_no_auto(node->lhs,str_to_check);
	if(node->rhs)
		add_no_auto(node->rhs,str_to_check);
	if(!str_to_check || (node->buffer && !strcmp(node->buffer,str_to_check)))
		node->no_auto = true;
}
static inline void strprepend(char* dst, const char* src)
{	
    memmove(dst + strlen(src), dst, strlen(dst)+ 1); // Move existing data including null terminator
    memcpy(dst, src, strlen(src)); // Copy src to the beginning of dst
}
static inline char* readFile(const char *filename) {
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
static inline void
file_prepend(const char* filename, const char* str_to_prepend)
{
	const char* file_tmp = readFile(filename);
	FILE* fp = fopen(filename,"w");
	fprintf(fp,"%s%s",str_to_prepend,file_tmp);
	fclose(fp);
	free((void*)file_tmp);
}

static inline void
file_append(const char* filename, const char* str_to_append)
{
	FILE* fp = fopen(filename,"a");
	fprintf(fp,"%s",str_to_append);
	fclose(fp);
}

static void
remove_substrings(ASTNode* node, const char* sub)
{
	if(node->lhs)
		remove_substrings(node->lhs,sub);
	if(node->rhs)
		remove_substrings(node->rhs,sub);
	if(node->prefix)  remove_substring(node->prefix,sub);
	if(node->postfix) remove_substring(node->postfix,sub);
	if(node->infix)   remove_substring(node->infix,sub);
	if(node->buffer)  remove_substring(node->buffer,sub);
}

static inline bool
is_number(const char* str)
{
	const size_t n = strlen(str);
	bool res = true;
	for(size_t i = 0; i < n; ++i)
		res &= (isdigit(str[i]) > 0);
	return res;
}
static inline bool
is_real(const char* str)
{
	char* tmp = strdup(str);
	remove_substring(tmp,".");
	remove_substring(tmp,"(");
	remove_substring(tmp,")");
	remove_substring(tmp,"AcReal");
	const bool res = is_number(tmp);
	free(tmp);
	return res;
}

static int
count_num_of_nodes_in_list(const ASTNode* list_head)
{
	int res = 0;
	while(list_head->rhs)
	{
		list_head = list_head->lhs;
		++res;
	}
	res += (list_head->lhs != NULL);
	return res;
}
static bool has_qualifier(const ASTNode* node, const char* qualifier)
{
	bool res = false;
	if(node->lhs)
		res |= has_qualifier(node->lhs,qualifier);
	if(node->rhs)
		res |= has_qualifier(node->rhs,qualifier);
	if(node->type & NODE_TQUAL)
		res |= !strcmp(node->lhs->buffer,qualifier);
	return res;
}
