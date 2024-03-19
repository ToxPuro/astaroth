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

typedef struct string_vec
{
	char* data[256];
	size_t size;

} string_vec;

static inline bool
str_vec_contains(string_vec vec, const char* str)
{
	for(size_t i = 0; i <  vec.size; ++i)
		if(!strcmp(vec.data[i],str)) return true;
	return false;
}
static inline void
push(string_vec* dst, char* src)
{
	dst->data[dst->size] = src;
	++(dst->size);
}
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
static inline void combine_all(const ASTNode* node, char* res){
  if(node->prefix)
    strcat(res,node->prefix);
  if(node->lhs)
    combine_all(node->lhs, res);
  if(node->buffer)
    strcat(res,node->buffer);
  if(node->infix)
    strcat(res,node->infix);
  if(node->rhs)
    combine_all(node->rhs, res);
  if(node->postfix)
    strcat(res,node->postfix);
}
