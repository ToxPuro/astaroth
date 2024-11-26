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
#include "vecs.h"
#include <sys/stat.h>

/**
static unsigned long
    hash(char *str)
    {
	if(!str) return -1;
        unsigned long hash = 5381;
        int c;

        while ((c = *str++))
            hash = ((hash << 5) + hash) + c; // hash * 33 + c

        return hash;
    }
**/

#define BUFFER_SIZE (4096)
#define FATAL_ERROR_MESSAGE "\nFATAL AC ERROR: "
static const char*
fatal(const char* format, ...)
{
	va_list args;
	va_start(args,format);
	fprintf(stderr,FATAL_ERROR_MESSAGE);
	vfprintf(stderr,format,args);
	va_end(args);
	exit(EXIT_FAILURE);
}

typedef enum {
  NODE_UNKNOWN             = 0,
  NODE_PRIMARY_EXPRESSION  = (1 << 0),
  NODE_DFUNCTION           = (1 << 1),
  NODE_KFUNCTION           = (1 << 2),
  NODE_FUNCTION_ID         = (1 << 3),
  NODE_GLOBAL              = (1 << 4),
  NODE_BINARY              = (1 << 5),
  NODE_BEGIN_SCOPE         = (1 << 6),
  NODE_DECLARATION         = (1 << 7),
  NODE_TSPEC               = (1 << 8),
  NODE_TQUAL               = (1 << 9),
  NODE_STENCIL             = (1 << 10),
  NODE_EXPRESSION          = (1 << 11),
  NODE_VARIABLE            = (1 << 12),
  NODE_VARIABLE_ID         = (1 << 13),
  NODE_ARRAY_INITIALIZER   = (1 << 14),
  NODE_DCONST              = (1 << 15),
  NODE_TERNARY             = (1 << 16),
  NODE_MEMBER_ID           = (1 << 17),
  NODE_HOSTDEFINE          = (1 << 18),
  NODE_ASSIGNMENT          = (1 << 19),
  NODE_INPUT               = (1 << 20),
  NODE_DEF                 = (1 << 21) | NODE_BEGIN_SCOPE,
  NODE_STRUCT_INITIALIZER  = (1 << 22),
  NODE_ARRAY_ACCESS        = (1 << 23),
  NODE_STATEMENT_LIST_HEAD = (1 << 24),
  NODE_IF                  = (1 << 25),
  NODE_FUNCTION_CALL       = (1 << 26),
  NODE_DFUNCTION_ID        = (1 << 27),
  NODE_ASSIGN_LIST         = (1 << 28 | NODE_GLOBAL), 
  NODE_NO_OUT              = (1 << 29),
  NODE_STRUCT_EXPRESSION   = (1 << 30),
  NODE_FUNCTION            = NODE_DFUNCTION | NODE_KFUNCTION,
  NODE_ENUM_DEF            = (NODE_DEF + 0 + NODE_NO_OUT),
  NODE_STRUCT_DEF          = (NODE_DEF + 1 + NODE_NO_OUT),
  NODE_TASKGRAPH_DEF       = (NODE_DEF + 2 + NODE_NO_OUT),
  NODE_BOUNDCONDS_DEF      = (NODE_DEF + 3 + NODE_NO_OUT),
  NODE_BINARY_EXPRESSION   = (NODE_BINARY + NODE_EXPRESSION),
  NODE_TERNARY_EXPRESSION  = (NODE_TERNARY+ NODE_EXPRESSION),
  NODE_ANY                 = ~0,
} NodeType;

typedef struct astnode_s {
  int id;
  struct astnode_s* parent;
  struct astnode_s* lhs;
  struct astnode_s* rhs;
  NodeType type; // Type of the AST node
  const char* buffer;  // Indentifiers and other strings (empty by default)
  const char* buffer_token;  // Indentifiers and other strings (empty by default)


  int token;     // Type of a terminal (that is not a simple char)
  const char* prefix;  // Strings. Also makes the grammar since we don't have
  const char* infix;   // to divide it into max two-child rules
  const char* postfix; // (which makes it much harder to read)
  bool is_constexpr; //Whether the node represents information known at compile time
  const char* expr_type; //The type of the expr the node represent
  bool no_auto;
} ASTNode;


static inline ASTNode*
astnode_dup(const ASTNode* node, ASTNode* parent)
{
	if(!node) return NULL;
	ASTNode* res = (ASTNode*)calloc(1,sizeof(node[0]));
	res->id = node->id;
	res->type = node->type;
	res->parent = parent;
	res -> token = node->token;
	res -> is_constexpr = node->is_constexpr;
	res -> no_auto = node->no_auto;
	res -> expr_type = node->expr_type;
	res->buffer = node->buffer;
	res->prefix=  node->prefix;
	res->infix =  node->infix;
	res->postfix= node->postfix;
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
  node->expr_type       = NULL;

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
  node->buffer  = NULL;
  node->prefix  = NULL;
  node->infix   = NULL;
  node->postfix = NULL;
  free(node);
}
static inline void
astnode_free(ASTNode* node)
{
	node->lhs = NULL;
	node->rhs = NULL;
	node->buffer = NULL;
	node->prefix= NULL;
	node->infix= NULL;
	node->postfix= NULL;
}

static inline void
astnode_set_buffer(const char* buffer, ASTNode* node)
{
  node->buffer = intern(buffer);
}


static inline void
astnode_set_prefix(const char* prefix, ASTNode* node)
{
  node->prefix= intern(prefix);
}

static inline void
astnode_set_infix(const char* infix, ASTNode* node)
{
  node->infix = intern(infix);
}

static inline void
astnode_set_postfix(const char* postfix, ASTNode* node)
{
  node->postfix = intern(postfix);
}
static const char*
sprintf_intern(const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int ret = vsprintf(buffer, format, args);
	va_end(args);
	return intern(buffer);
}

static inline void
astnode_sprintf(ASTNode* node, const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int ret = vsprintf(buffer, format, args);
	va_end(args);
	astnode_set_buffer(buffer,node);
}


static inline void
astnode_sprintf_postfix(ASTNode* node, const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int ret = vsprintf(buffer, format, args);
	va_end(args);
	astnode_set_postfix(buffer,node);
}

static inline void
astnode_sprintf_infix(ASTNode* node, const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int ret = vsprintf(buffer, format, args);
	va_end(args);
	astnode_set_infix(buffer,node);
}

static inline void
astnode_sprintf_prefix(ASTNode* node, const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int ret = vsprintf(buffer, format, args);
	va_end(args);
	astnode_set_prefix(buffer,node);
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
static const char* combine_buffers_new(const ASTNode* node)
{
	static char res[10000];
	combine_buffers(node,res);
	return res;
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
static inline const char*combine_all_new(const ASTNode* node){
  static char res[10000];
  res[0] = '\0';	
  combine_all_recursive(node,res);
  strip_whitespace(res);
  return res;
}
static inline const char* combine_all_new_with_whitespace(const ASTNode* node){
  static char res[10000];
  res[0] = '\0';	
  combine_all_recursive(node,res);
  return res;
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
  if (node->buffer && node->type & type && !strcmp(test,node->buffer))
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
static inline const ASTNode*
get_parent_node(const NodeType type, const ASTNode* node)
{
  if(!node->parent)
    return NULL;
  if (node->parent->type & type)
    return node->parent;
  return get_parent_node(type, node->parent);
}
static inline const ASTNode*
get_parent_node_inclusive(const NodeType type, const ASTNode* node)
{
  if(node->type & type) return node;
  if(!node->parent)
    return NULL;
  if (node->parent->type & type)
    return node->parent;
  return get_parent_node(type, node->parent);
}

static inline const ASTNode*
get_parent_node_by_token(const int token, const ASTNode* node)
{
  if(!node->parent)
    return NULL;
  if (node->parent->token == token)
    return node->parent;
  return get_parent_node_by_token(token, node->parent);
}

static bool
is_left_child(const NodeType type, const ASTNode* node)
{
	const ASTNode* parent = get_parent_node(type,node);
	if(!parent) return false;
	return get_node_by_id(node->id,parent->lhs) != NULL;
}

static bool
is_right_child(const NodeType type, const ASTNode* node)
{
	const ASTNode* parent = get_parent_node(type,node);
	if(!parent) return false;
	return get_node_by_id(node->id,parent->rhs) != NULL;
}
static inline const ASTNode*
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
	if(!node) return;
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
static inline void
strcatprintf(char* dst, const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int ret = vsprintf(buffer, format, args);
	va_end(args);
	strcat(dst,buffer);
	
}

static inline void
fprintf_filename(const char* filename, const char* format, ...)
{
	FILE* fp = fopen(filename,"a");
	va_list args;
	va_start(args,format);
	int ret = vfprintf(fp, format, args);
	va_end(args);
	fclose(fp);
}
static inline void
fprintf_filename_w(const char* filename, const char* format, ...)
{
	FILE* fp = fopen(filename,"w");
	va_list args;
	va_start(args,format);
	int ret = vfprintf(fp, format, args);
	va_end(args);
	fclose(fp);
}

static inline char*
sprintf_new(const char* format, ...)
{
	static char buffer[10000];
	va_list args;
	va_start(args,format);
	int res = vsprintf(buffer,format,args);
	va_end(args);
	return strdup(buffer);
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
file_append(const char* filename, const char* str_to_append)
{
	FILE* fp = fopen(filename,"a");
	fprintf(fp,"%s",str_to_append);
	fclose(fp);
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
is_number_expression(const char* str)
{
	const size_t n = strlen(str);
	bool res = true;
	for(size_t i = 0; i < n; ++i)
		res &= (
			  isdigit(str[i]) > 0
			|| str[i] == ')'
			|| str[i] == '('
			|| str[i] == '+'
			|| str[i] == '-'
			|| str[i] == '/'
			|| str[i] == '*'
		       );
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
static ASTNode*
get_node_in_list(const ASTNode* list_head, int index)
{
	bool last_elem = count_num_of_nodes_in_list(list_head) == index + 1;
	while(--index)
	{
		list_head = list_head->lhs;
	}
	return last_elem ? list_head->lhs : list_head->rhs;
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

static void
format_source(const char* file_in, const char* file_out)
{
   FILE* in = fopen(file_in, "r");
  assert(in);

  FILE* out = fopen(file_out, "w");
  assert(out);

  while (!feof(in)) {
    const char c = fgetc(in);
    if (c == EOF)
      break;

    fprintf(out, "%c", c);
    if (c == ';' || c == '{')
      fprintf(out, "\n");
  }

  fclose(in);
  fclose(out);
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

typedef struct node_vec
{
	const ASTNode** data;
	size_t size;
	int capacity;

} node_vec;
static inline void
free_node_vec(node_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 0;
	vec -> data = NULL;
}
static inline int
push_node(node_vec* dst, const ASTNode* src)
{
	if(dst->capacity == 0)
	{
		dst->capacity++;
		dst->data = malloc(sizeof(ASTNode*)*dst->capacity);
	}
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		const ASTNode** tmp = malloc(sizeof(ASTNode*)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}

static ASTNode*
build_list_node(const node_vec nodes, const char* separator)
{
	if(nodes.size == 0) return NULL;
	ASTNode* list_head = astnode_create(NODE_UNKNOWN, astnode_dup(nodes.data[0],NULL),NULL);
	for(size_t i = 1; i < nodes.size; ++i)
	{
		list_head = astnode_create(NODE_UNKNOWN,list_head, astnode_dup(nodes.data[i],NULL));
		list_head->buffer = strdup(separator);
	}
	return list_head;
}

static node_vec
get_nodes_in_list(const ASTNode* head)
{
	node_vec res = VEC_INITIALIZER;
	const int num_of_nodes = count_num_of_nodes_in_list(head);
	int counter = num_of_nodes;
	while(--counter)
		head = head -> lhs;
	push_node(&res,head->lhs);
	counter = num_of_nodes;
	while(--counter)
	{
		head = head->parent;
		push_node(&res,head->rhs);
	}
	return res;
}

static const ASTNode*
get_node(const NodeType type, const ASTNode* node)
{
  if(!node) 
  {
  	  assert(node);
	  fatal("WRONG; passed NULL to get_node\n");
  }

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
get_array_access_nodes(const ASTNode* node, node_vec* dst)
{
	if(node->lhs)
		get_array_access_nodes(node->lhs,dst);
	if(node->rhs)
		get_array_access_nodes(node->rhs,dst);
	if(node->type == NODE_ARRAY_ACCESS)
		push_node(dst,node->rhs);
}
static void
replace_node(ASTNode* original, ASTNode* replacement)
{
		if(original->parent->lhs && original->parent->lhs->id == original->id)
			original->parent->lhs = replacement;
		else
			original->parent->rhs = replacement;
		replacement->parent = original->parent;
}
static inline bool
node_vec_contains(const node_vec vec, const char* str)
{
	for(size_t i = 0; i < vec.size; ++i)
		if(!strcmp(combine_all_new(vec.data[i]), str)) return true;
	return false;
}
static void
append_to_file(const char* filename, const char* format, ...)
{
	FILE* fp = fopen(filename,"a");
	va_list args;
	va_start(args,format);
	vfprintf(fp,format,args);
	va_end(args);
	fclose(fp);
}
