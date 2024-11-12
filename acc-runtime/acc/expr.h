#include "tinyexpr.h"
static void replace_const_ints(ASTNode* node, const string_vec values, const string_vec names)
{
	if(node->lhs && !(node->type & (NODE_DECLARATION | NODE_ASSIGNMENT)))
		replace_const_ints(node->lhs,values,names);
	if(node->rhs)
		replace_const_ints(node->rhs,values,names);
	if(node->token != IDENTIFIER || !node->buffer) return;
	const int index = str_vec_get_index(names,node->buffer);
	if(index == -1) return;
	astnode_set_buffer(values.data[index],node);
	node->type = NODE_UNKNOWN;
	node->token = NUMBER;
}
static void eval_ternaries(ASTNode* node, const string_vec values, const string_vec names);
static inline int eval_int(ASTNode* node, const bool failure_fatal, int* error_code)
{
	replace_const_ints(node,const_int_values,const_ints);
	eval_ternaries(node,const_int_values,const_ints);
	const char* copy = combine_all_new(node);
	if(!strcmp(copy,"INT_MAX"))
		return INT_MAX;
        double* vals = malloc(sizeof(double)*const_ints.size);
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

static void eval_ternaries(ASTNode* node, const string_vec values, const string_vec names)
{
	if(node->lhs)
		eval_ternaries(node->lhs,values,names);
	if(node->rhs)
		eval_ternaries(node->rhs,values,names);
	if(node->type != NODE_TERNARY_EXPRESSION) return;
	//TP: for now consider only that expr is a < b
	//TP: where a and b are const int expressions
	//printf("ORIG: %s\n",combine_all_new(node));
	const ASTNode* cond = node->lhs;
	if(cond->type != NODE_BINARY_EXPRESSION) return;
	const char* op = get_node_by_token(BINARY_OP,cond->rhs)->buffer;
	if(!op) return;
	const bool less = !strcmp(op,"<");
	const bool more = !strcmp(op,">");
	if(!less && !more) return;
	const int lhs = eval_int(cond->lhs,true,NULL);
	const int rhs = eval_int(cond->rhs->rhs,true,NULL);
	const bool is_left_child = node->parent->lhs && node->parent->lhs->id == node->id;
	ASTNode* correct_node =  less ? 
					(lhs < rhs) ? node->rhs->lhs : node->rhs->rhs
				:more ? 
					(lhs > rhs) ? node->rhs->lhs : node->rhs->rhs
				     : NULL;
	if(!correct_node) return;
	correct_node->prefix = NULL;
	if(is_left_child)
		node->parent->lhs = correct_node; 
	else
		node->parent->rhs = correct_node; 
	correct_node->parent = node->parent;
}

