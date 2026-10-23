#pragma once

#include "tinyexpr.h"

static void replace_const_ints(ASTNode* node, const string_vec values, const string_vec names)
{
	//TP: do not replace const int on lhs of assignment to avoid replacing the initial assignment
	//if there is an array assignment on the lhs then cannot be the initial assignment
	if(node->lhs && 
			(
				!(node->type & (NODE_DECLARATION | NODE_ASSIGNMENT))
				|| get_node(NODE_ARRAY_ACCESS,node->lhs)
			)
	  )
	{
		replace_const_ints(node->lhs,values,names);
	}
	if(node->rhs)
		replace_const_ints(node->rhs,values,names);
	if(node->token != IDENTIFIER || !node->buffer) return;
	const int index = str_vec_get_index(names,node->buffer);
	if(index == -1) return;
	astnode_set_buffer(values.data[index],node);
	node->type = NODE_UNKNOWN;
	node->token = NUMBER;
}

static void eval_ternaries(ASTNode* node, const string_vec values,
                           const string_vec names, const string_vec run_values,
                           const string_vec run_names, const bool failure_fatal,
                           int* err);
//TP: elements of global const int arrays, stored as "name[i]" --> value, such that e.g. field_order(arr[2]-1) can be evaluated
extern string_vec const_int_arrays;
extern string_vec const_int_array_elems;
extern string_vec const_int_array_elem_values;

static inline int
eval_int(ASTNode* node, const string_vec values, const string_vec names,
         const string_vec run_values, const string_vec run_names,
         const bool failure_fatal, int* error_code);

static void
replace_const_int_array_accesses(ASTNode* node, const string_vec values, const string_vec names,
                                 const string_vec run_values, const string_vec run_names,
                                 const bool failure_fatal)
{
       if(node->lhs)
               replace_const_int_array_accesses(node->lhs, values, names, run_values, run_names, failure_fatal);
       if(node->rhs)
               replace_const_int_array_accesses(node->rhs, values, names, run_values, run_names, failure_fatal);
       //TP: members of const int3s, stored as "name.x" etc.
       if((node->type & NODE_STRUCT_EXPRESSION) && node->lhs && node->rhs)
       {
               if(!(node->lhs->type & NODE_PRIMARY_EXPRESSION)) return;
               const ASTNode* id = get_node_by_token(IDENTIFIER, node->lhs);
               const ASTNode* member = get_node_by_token(IDENTIFIER, node->rhs);
               if(!id || !id->buffer || !member || !member->buffer) return;
               if(str_vec_get_index(const_int_arrays, id->buffer) == -1) return;
               const int elem = str_vec_get_index(const_int_array_elems, sprintf_intern("%s.%s", id->buffer, member->buffer));
               if(elem == -1) return;
               set_buffers_empty(node);
               node->lhs = NULL;
               node->rhs = NULL;
               astnode_set_buffer(const_int_array_elem_values.data[elem], node);
               node->type = NODE_UNKNOWN;
               node->token = NUMBER;
               return;
       }

       if(!(node->type & NODE_ARRAY_ACCESS) || !node->lhs || !node->rhs) return;
       if(!(node->lhs->type & NODE_PRIMARY_EXPRESSION)) return;
       const ASTNode* id = get_node_by_token(IDENTIFIER, node->lhs);
       if(!id || !id->buffer || str_vec_get_index(const_int_arrays, id->buffer) == -1) return;
       int err = 0;
       const int index = eval_int(node->rhs, values, names, run_values, run_names, failure_fatal, &err);
       if(err) return;
       const int elem = str_vec_get_index(const_int_array_elems, sprintf_intern("%s[%d]", id->buffer, index));
       if(elem == -1)
       {
               if(failure_fatal)
               {
                       fprintf(stderr,"Index %d out of bounds for const int array %s\n", index, id->buffer);
                       exit(EXIT_FAILURE);
               }
               return;
       }
       set_buffers_empty(node);
       node->lhs = NULL;
       node->rhs = NULL;
       astnode_set_buffer(const_int_array_elem_values.data[elem], node);
       node->type = NODE_UNKNOWN;
       node->token = NUMBER;
}


static inline int
eval_int(ASTNode* node, const string_vec values, const string_vec names,
         const string_vec run_values, const string_vec run_names,
         const bool failure_fatal, int* error_code)
{
	replace_const_int_array_accesses(node, values, names, run_values, run_names, failure_fatal);
	replace_const_ints(node, values, names);
        replace_const_ints(node, run_values, run_names);
        eval_ternaries(node, values, names, run_values, run_names,
                       failure_fatal, error_code);
        const char* copy = combine_all_new(node);
	if(!strcmp(copy,"INT_MAX"))
		return INT_MAX;
        int err;
        te_expr* expr = te_compile(copy, NULL, 0, &err);
        if(!expr)
        {
		if(failure_fatal)
		{
                	fprintf(stderr,"Parse error at tinyexpr\n");
			fprintf(stderr,"Was not able to parse: %s\n",copy);
                	exit(EXIT_FAILURE);
		}
		*error_code = 1;
		te_free(expr);
		return -1;
        }
        int res = (int) round(te_eval(expr));
        te_free(expr);
        return res;
}

static void
eval_ternaries(ASTNode* node, const string_vec values, const string_vec names,
               const string_vec run_values, const string_vec run_names,
               const bool failure_fatal, int* err)
{
	if(node->lhs)
	{
          eval_ternaries(node->lhs, values, names, run_values, run_names,
                         failure_fatal, err);
	}
        if(node->rhs)
	{
          eval_ternaries(node->rhs, values, names, run_values, run_names,
                         failure_fatal, err);
	}
        if(node->type != NODE_TERNARY_EXPRESSION) return;
	//TP: for now consider only that expr is a < b
	//TP: where a and b are const int expressions
	//printf("ORIG: %s\n",combine_all_new(node));
	const ASTNode* cond = node->lhs->lhs;
	if(!cond) return;
	const char* op = cond->rhs->lhs->buffer;
	if(!op) return;
	const bool less = !strcmp(op,"<");
	const bool more = !strcmp(op,">");
	const bool eq   = !strcmp(op,"==");
	const bool neq   = !strcmp(op,"!=");
	if(!less && !more && !eq && !neq) return;
	const int lhs = eval_int(cond->lhs, values, names, run_values, run_names, failure_fatal,err);
	if(!failure_fatal && *err) return;
	const int rhs = eval_int(cond->rhs->rhs, values, names, run_values, run_names, failure_fatal, err);
	if(!failure_fatal && *err) return;
	const bool pick_left = (less && lhs < rhs) || (more && lhs > rhs) || (eq && lhs == rhs) || (neq && lhs != rhs);
	ASTNode* correct_node = pick_left ? node->rhs->lhs : node->rhs->rhs;
	if(!correct_node) return;
	*node = *correct_node;
	node->prefix = NULL;
}

