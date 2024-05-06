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
  NODE_UNKNOWN        = 0,
  NODE_FUNCTION       = (1 << 0),
  NODE_DFUNCTION      = (1 << 1),
  NODE_KFUNCTION      = (1 << 2),
  NODE_FUNCTION_ID    = (1 << 3),
  NODE_FUNCTION_PARAM = (1 << 4),
  NODE_RANGE          = (1 << 5),
  NODE_BEGIN_SCOPE    = (1 << 6),
  NODE_DECLARATION    = (1 << 7),
  NODE_TSPEC          = (1 << 8),
  NODE_TQUAL          = (1 << 9),
  NODE_STENCIL        = (1 << 10),
  NODE_STENCIL_ID     = (1 << 11),
  NODE_VARIABLE       = (1 << 12),
  NODE_VARIABLE_ID    = (1 << 13),
  NODE_KFUNCTION_ID   = (1 << 14),
  NODE_DCONST         = (1 << 15),
  NODE_DCONST_ID      = (1 << 16),
  NODE_MEMBER_ID      = (1 << 17),
  NODE_HOSTDEFINE     = (1 << 18),
  NODE_ASSIGNMENT     = (1 << 19),
  NODE_INPUT          = (1 << 20),
  NODE_CODEGEN_INPUT  = (1 << 21),
  NODE_ENUM_DEF       = (1 << 22),
  NODE_ENUM           = (1 << 23),
  NODE_SELECTION_STATEMENT = (1 << 24),
  NODE_STRUCT_DEF     = (1 << 25),
  NODE_IF             = (1 << 26),
  NODE_FUNCTION_CALL  = (1 << 27),
  NODE_DFUNCTION_ID   = (1 << 28),
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
} ASTNode;


static inline ASTNode*
astnode_dup(const ASTNode* node, ASTNode* parent)
{
	ASTNode* res = (ASTNode*)calloc(1,sizeof(node[0]));
	res->id = node->id;
	res->type = node->type;
	res->parent = parent;
	if(node->buffer)
		res->buffer = strdup(node->buffer);
	if(node->prefix)
		res->prefix= strdup(node->prefix);
	if(node->infix)
		res->infix = strdup(node->infix);
	if(node->postfix)
		res->postfix = strdup(node->postfix);
	res -> token = node->token;
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

typedef enum ReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} ReduceOp;

typedef struct string_vec
{
	//char* data[256];
	char** data;
	size_t size;
	int capacity;

} string_vec;
typedef struct int_vec 
{
	int* data;
	size_t size;
	int capacity;

} int_vec;

typedef struct op_vec 
{
	ReduceOp* data;
	size_t size;
	int capacity;

} op_vec;

static inline void
init_str_vec(string_vec* vec)
{
	//for(size_t i = 0; i < vec->size; ++i)
	//	free(vec->data[i]);
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 100;
	vec -> data = malloc(sizeof(char*)*vec ->capacity);
}

static inline void
free_str_vec(string_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 0;
	vec -> data = NULL;
	//vec -> data = malloc(sizeof(char*)*vec ->capacity);
}
static inline void
init_int_vec(int_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 1;
	vec -> data = malloc(sizeof(int)*vec ->capacity);
}
static inline void
init_op_vec(op_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 1;
	vec -> data = malloc(sizeof(ReduceOp)*vec ->capacity);
}
static inline int
str_vec_get_index(string_vec vec, const char* str)
{
	for(size_t i = 0; i <  vec.size; ++i)
		if(!strcmp(vec.data[i],str)) return i;
	return -1;
}
static inline bool
str_vec_contains(string_vec vec, const char* str)
{
	return str_vec_get_index(vec,str) >= 0;
}
static inline int 
push(string_vec* dst, const char* src)
{
	dst->data[dst->size] = strdup(src);
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		char** tmp = malloc(sizeof(char*)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}
static inline int 
push_int(int_vec* dst, int src)
{
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		int* tmp = malloc(sizeof(int)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}
static inline string_vec
str_vec_copy(string_vec vec)
{
        string_vec copy;
	init_str_vec(&copy);
        for(size_t i = 0; i<vec.size; ++i)
                push(&copy,vec.data[i]);
        return copy;
}

static inline int
push_op(op_vec* dst, ReduceOp src)
{
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		ReduceOp* tmp = malloc(sizeof(ReduceOp)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}

static inline  void combine_buffers_recursive(const ASTNode* node, char* res){
  if(node->buffer && !(node->type & NODE_CODEGEN_INPUT))
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
static inline void combine_all(const ASTNode* node, char* res){
  res[0] = '\0';	
  combine_all_recursive(node,res);
}

static inline ASTNode*
get_node_by_token(const int token, ASTNode* node)
{
  assert(node);

  if (node->token == token)
    return node;
  else if (node->lhs && get_node_by_token(token, node->lhs))
    return get_node_by_token(token, node->lhs);
  else if (node->rhs && get_node_by_token(token, node->rhs))
    return get_node_by_token(token, node->rhs);
  else
    return NULL;
}
static inline ASTNode*
get_node_by_buffer(const char* test, ASTNode* node)
{
  assert(node);

  if (node->buffer && !strcmp(test,node->buffer))
    return node;
  else if (node->lhs && get_node_by_buffer(test, node->lhs))
    return get_node_by_buffer(test, node->lhs);
  else if (node->rhs && get_node_by_buffer(test, node->rhs))
    return get_node_by_buffer(test, node->rhs);
  else
    return NULL;
}
static const inline ASTNode*
get_parent_node(const NodeType type, const ASTNode* node)
{
  if (node->type & type)
    return node;
  else if (node->parent)
    return get_parent_node(type, node->parent);
  else
    return NULL;
}
typedef struct CodeGenInput
{
	string_vec const_ints;
	string_vec const_int_values;
} CodeGenInput;

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
