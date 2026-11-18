#pragma once
//TP: handling of constant loops at the start of generate (unroll_constant_loops), included in codegen.c after the AST helpers it uses
//    (is_for_iteration_statement, replace_variable_with_constant_int, create_func_call, push_to_statement_list, get_nodes, eval_int, ...).
//    Short loops are unrolled as before, long loops are kept rolled with their iteration-specific stencil calls hoisted (see below).

#ifndef MAX_UNROLL_ITERATIONS
#define MAX_UNROLL_ITERATIONS (4)
#endif
//TP: long constant loops (more than MAX_UNROLL_ITERATIONS iterations), e.g. loops over 32 dust species.
//    Unrolling them makes every iteration-specific stencil call a call on a compile-time known field, so all of them are preloaded
//    at the start of the kernel and the kernels spill heavily to scratch.
//    Instead the loop is kept rolled and the stencil calls that depend only on the loop variable are hoisted to locals at the start of each
//    iteration between ac_dynamic_stencils_begin() and ac_dynamic_stencils_end(). The analysis marks the stencil calls in between
//    with AC_STENCIL_DYNAMIC_CALL and stencilgen computes the stencils that are called only like this on demand instead of preloading them.
//    So the loads of one iteration are still issued together while the working set is that of one iteration.
//    If the arrays indexed in the hoisted calls are used also outside long loops the stencils are preloaded anyways so the loop is unrolled as before.
typedef struct
{
	string_vec stencils;
	string_vec funcs;
	node_vec   func_callees; //TP: data[i] is actually a string_vec* of the callees of funcs[i]
	node_vec   func_stencils; //TP: data[i] is actually a string_vec* of the stencils funcs[i] calls directly or through other functions
	bool*      has_side_effects;
	struct hashmap_s used_outside_long_loops; //TP: keys "array:stencil" of the stencils called on the elements of the array outside long loops
	int        hoist_counter;
} loop_hoist_info;

static bool
is_side_effect_call_name(const char* name)
{
	return !strncmp(name,"write",5) || !strncmp(name,"reduce",6) || !strncmp(name,"print",5) || !strncmp(name,"fprintf",7)
		|| !strncmp(name,"atomic",6) || !strncmp(name,"AC_INTERNAL_write",17) || !strncmp(name,"ac_dummy_write",14);
}
static const char*
get_func_call_name(const ASTNode* node)
{
	if(!(node->type & NODE_FUNCTION_CALL) || !node->lhs) return NULL;
	const ASTNode* id = get_node_by_token(IDENTIFIER,node->lhs);
	return id ? id->buffer : NULL;
}
static void
collect_callee_names(const ASTNode* node, string_vec* dst)
{
	if(node->lhs) collect_callee_names(node->lhs,dst);
	if(node->rhs) collect_callee_names(node->rhs,dst);
	const char* name = get_func_call_name(node);
	if(name && str_vec_get_index(*dst,name) < 0) push(dst,name);
}
static bool
eval_loop_bound(const ASTNode* bound, int* res)
{
	const char* str = intern(combine_buffers_new(bound));
	if(is_number(str))
	{
		*res = atoi(str);
		return true;
	}
	int err = 0;
	ASTNode* copy = astnode_dup(bound,NULL);
	*res = eval_int(copy,const_int_values,const_ints,run_const_int_values,run_const_ints,false,&err);
	return err == 0;
}
//TP: -1 if not a for loop over a range with bounds known at compile time.
//    *literal_bounds tells whether the bounds are int literals: only those loops are unrolled, the others are rolled as before
//    but they can be long loops of which the iteration-specific calls are hoisted, e.g. for k in 1:ndustspec+1
static int
get_const_loop_trip_count_base(const ASTNode* node, bool* literal_bounds, int* start_value)
{
	if(!is_for_iteration_statement(node)) return -1;
	const ASTNode* for_expr = node->lhs->rhs;
	if(!for_expr || !for_expr->rhs || for_expr->rhs->token != RANGE) return -1;
	const ASTNode* start_node = for_expr->rhs->lhs->lhs;
	const ASTNode* end_node   = for_expr->rhs->rhs->lhs;
	if(!start_node || !end_node) return -1;
	*literal_bounds = is_number(intern(combine_buffers_new(start_node))) && is_number(intern(combine_buffers_new(end_node)));
	int start = 0, end = 0;
	if(!eval_loop_bound(start_node,&start) || !eval_loop_bound(end_node,&end)) return -1;
	*start_value = start;
	return end > start ? end-start : 0;
}
static bool
is_long_const_loop(const ASTNode* node)
{
	bool literal_bounds;
	int start;
	return get_const_loop_trip_count_base(node,&literal_bounds,&start) > MAX_UNROLL_ITERATIONS;
}
static const string_vec*
get_callee_stencils(const loop_hoist_info* info, const char* name, string_vec* tmp)
{
	if(str_vec_get_index(info->stencils,name) >= 0)
	{
		push(tmp,name);
		return tmp;
	}
	const int index = str_vec_get_index(info->funcs,name);
	if(index < 0) return tmp;
	return (const string_vec*)info->func_stencils.data[index];
}
static void
mark_used_outside_long_loops(loop_hoist_info* info, const char* array, const char* stencil)
{
	const char* key = sprintf_intern("%s:%s",array,stencil);
	if(!hashmap_get(&info->used_outside_long_loops,key,strlen(key)))
		hashmap_put(&info->used_outside_long_loops,key,strlen(key),(void*)key);
}
static bool
used_outside_long_loops(const loop_hoist_info* info, const char* array, const char* stencil)
{
	const char* key = sprintf_intern("%s:%s",array,stencil);
	return hashmap_get((struct hashmap_s*)&info->used_outside_long_loops,key,strlen(key)) != NULL;
}
static void
collect_identifiers(const ASTNode* node, string_vec* dst)
{
	if(node->token == IDENTIFIER && node->buffer && str_vec_get_index(*dst,node->buffer) < 0) push(dst,node->buffer);
	if(node->lhs) collect_identifiers(node->lhs,dst);
	if(node->rhs) collect_identifiers(node->rhs,dst);
}
//TP: an identifier in the arguments of a call is marked as used with the stencils of the callee,
//    an identifier outside calls, e.g. F_DUST_DENSITY in nd = F_DUST_DENSITY, with the value stencil.
//    Conservative: all identifiers are marked, not only fields
static void
collect_stencil_uses_outside_long_loops(const ASTNode* node, const bool in_function, const bool in_call_args, loop_hoist_info* info)
{
	if(is_long_const_loop(node)) return;
	const bool in_func = in_function || (node->type & NODE_FUNCTION);
	if(in_func && !in_call_args && node->token == IDENTIFIER && node->buffer)
		mark_used_outside_long_loops(info,node->buffer,intern("value_stencil"));
	const char* name = in_func ? get_func_call_name(node) : NULL;
	if(name && node->rhs)
	{
		string_vec tmp = VEC_INITIALIZER;
		const string_vec* stencils = get_callee_stencils(info,name,&tmp);
		if(stencils->size)
		{
			string_vec ids = VEC_INITIALIZER;
			collect_identifiers(node->rhs,&ids);
			for(size_t i = 0; i < ids.size; ++i)
				for(size_t j = 0; j < stencils->size; ++j)
					mark_used_outside_long_loops(info,ids.data[i],stencils->data[j]);
			free_str_vec(&ids);
		}
		free_str_vec(&tmp);
	}
	if(node->lhs) collect_stencil_uses_outside_long_loops(node->lhs,in_func,in_call_args,info);
	if(node->rhs) collect_stencil_uses_outside_long_loops(node->rhs,in_func,in_call_args || name != NULL,info);
}
static loop_hoist_info
get_loop_hoist_info(const ASTNode* root)
{
	loop_hoist_info info;
	memset(&info,0,sizeof(info));
	{
		node_vec nodes = VEC_INITIALIZER;
		get_nodes(root,&nodes,&info.stencils,NODE_STENCIL);
		free_node_vec(&nodes);
	}
	node_vec funcs = VEC_INITIALIZER;
	string_vec func_names = VEC_INITIALIZER;
	get_nodes(root,&funcs,&func_names,NODE_DFUNCTION);
	//TP: overloads share the name so the info of a name is the union over them
	for(size_t i = 0; i < funcs.size; ++i)
	{
		int index = str_vec_get_index(info.funcs,func_names.data[i]);
		if(index < 0)
		{
			push(&info.funcs,func_names.data[i]);
			string_vec* callees = calloc(1,sizeof(string_vec));
			push_node(&info.func_callees,(const ASTNode*)callees);
			index = info.funcs.size-1;
		}
		collect_callee_names(funcs.data[i]->rhs ? funcs.data[i]->rhs : funcs.data[i],(string_vec*)info.func_callees.data[index]);
	}
	free_node_vec(&funcs);
	free_str_vec(&func_names);
	const size_t n = info.funcs.size;
	for(size_t i = 0; i < n; ++i) push_node(&info.func_stencils,(const ASTNode*)calloc(1,sizeof(string_vec)));
	info.has_side_effects = calloc(n+1,sizeof(bool));
	bool changed = true;
	while(changed)
	{
		changed = false;
		for(size_t i = 0; i < n; ++i)
		{
			const string_vec* callees = (const string_vec*)info.func_callees.data[i];
			for(size_t j = 0; j < callees->size; ++j)
			{
				const char* callee = callees->data[j];
				const int index = str_vec_get_index(info.funcs,callee);
				string_vec* own_stencils = (string_vec*)info.func_stencils.data[i];
				if(str_vec_get_index(info.stencils,callee) >= 0 && str_vec_get_index(*own_stencils,callee) < 0) { push(own_stencils,callee); changed = true; }
				if(index >= 0 && index != (int)i)
				{
					const string_vec* callee_stencils = (const string_vec*)info.func_stencils.data[index];
					for(size_t k = 0; k < callee_stencils->size; ++k)
						if(str_vec_get_index(*own_stencils,callee_stencils->data[k]) < 0) { push(own_stencils,callee_stencils->data[k]); changed = true; }
				}
				const bool side_effects = is_side_effect_call_name(callee) || (index >= 0 && info.has_side_effects[index]);
				if(side_effects && !info.has_side_effects[i]) { info.has_side_effects[i] = true; changed = true; }
			}
		}
	}
	hashmap_create(4096,&info.used_outside_long_loops);
	collect_stencil_uses_outside_long_loops(root,false,false,&info);
	return info;
}
static void
free_loop_hoist_info(loop_hoist_info* info)
{
	for(size_t i = 0; i < info->func_callees.size; ++i)
	{
		string_vec* callees = (string_vec*)info->func_callees.data[i];
		free_str_vec(callees);
		free(callees);
	}
	free_node_vec(&info->func_callees);
	for(size_t i = 0; i < info->func_stencils.size; ++i)
	{
		string_vec* stencils = (string_vec*)info->func_stencils.data[i];
		free_str_vec(stencils);
		free(stencils);
	}
	free_node_vec(&info->func_stencils);
	free_str_vec(&info->stencils);
	free_str_vec(&info->funcs);
	hashmap_destroy(&info->used_outside_long_loops);
	free(info->has_side_effects);
}
static bool
subtree_has_identifier(const ASTNode* node, const char* id)
{
	if(node->token == IDENTIFIER && node->buffer == id) return true;
	return (node->lhs && subtree_has_identifier(node->lhs,id)) || (node->rhs && subtree_has_identifier(node->rhs,id));
}
static bool
subtree_has_any_identifier(const ASTNode* node, const string_vec ids)
{
	if(node->token == IDENTIFIER && node->buffer && str_vec_get_index(ids,node->buffer) >= 0) return true;
	return (node->lhs && subtree_has_any_identifier(node->lhs,ids)) || (node->rhs && subtree_has_any_identifier(node->rhs,ids));
}
static void
collect_identifiers_outside_indexes(const ASTNode* node, string_vec* dst)
{
	if(node->token == IDENTIFIER && node->buffer && str_vec_get_index(*dst,node->buffer) < 0) push(dst,node->buffer);
	if(node->lhs) collect_identifiers_outside_indexes(node->lhs,dst);
	if(node->rhs && !(node->type & NODE_ARRAY_ACCESS)) collect_identifiers_outside_indexes(node->rhs,dst);
}
//TP: the variables declared or assigned inside the loop body, a call depending on them can not be moved to the start of the iteration
static void
collect_assigned_identifiers(const ASTNode* node, string_vec* dst)
{
	//TP: the indexes of an assigned array element, e.g. the loop variable in arr[k-1] = ..., are only read
	if((node->type & NODE_ASSIGNMENT) && node->lhs) collect_identifiers_outside_indexes(node->lhs,dst);
	if((node->type & NODE_DECLARATION)  && node->rhs) collect_identifiers_outside_indexes(node->rhs,dst);
	if(node->lhs) collect_assigned_identifiers(node->lhs,dst);
	if(node->rhs) collect_assigned_identifiers(node->rhs,dst);
}
//TP: the bases of the array accesses indexed with the loop variable, e.g. F_DUST_VELOCITY in F_DUST_VELOCITY[k-1]
static void
collect_loop_indexed_arrays(const ASTNode* node, const char* loop_variable, string_vec* dst)
{
	if((node->type & NODE_ARRAY_ACCESS) && node->lhs && node->rhs && subtree_has_identifier(node->rhs,loop_variable))
	{
		const ASTNode* base = get_node_by_token(IDENTIFIER,node->lhs);
		if(base && str_vec_get_index(*dst,base->buffer) < 0) push(dst,base->buffer);
	}
	if(node->lhs) collect_loop_indexed_arrays(node->lhs,loop_variable,dst);
	if(node->rhs) collect_loop_indexed_arrays(node->rhs,loop_variable,dst);
}
static bool
is_loop_statement(const ASTNode* node)
{
	if(is_for_iteration_statement(node)) return true;
	return node->lhs && node->lhs->lhs && node->lhs->lhs->token == WHILE;
}
static bool
is_logical_binary_expression(const ASTNode* node)
{
	if(!node_is_binary_expr(node)) return false;
	const ASTNode* op = get_node_by_token(BINARY_OP,node->rhs->lhs);
	return op && op->buffer && (!strcmp(op->buffer,"&&") || !strcmp(op->buffer,"||"));
}
static bool
is_hoistable_call(const ASTNode* node, const char* loop_variable, const string_vec assigned, const loop_hoist_info* info)
{
	const char* name = get_func_call_name(node);
	if(!name || !node->rhs) return false;
	if(node->parent && node->parent->token == NON_RETURNING_FUNC_CALL) return false;
	const int index = str_vec_get_index(info->funcs,name);
	const bool is_stencil = str_vec_get_index(info->stencils,name) >= 0;
	if(!is_stencil && (index < 0 || ((const string_vec*)info->func_stencils.data[index])->size == 0 || info->has_side_effects[index])) return false;
	if(!subtree_has_identifier(node->rhs,loop_variable)) return false;
	if(subtree_has_any_identifier(node->rhs,assigned)) return false;
	return true;
}
static bool
subtree_has_func_call(const ASTNode* node)
{
	if(node->type & NODE_FUNCTION_CALL) return true;
	return (node->lhs && subtree_has_func_call(node->lhs)) || (node->rhs && subtree_has_func_call(node->rhs));
}
//TP: a condition that is the same for all iterations and does not read fields, e.g. AC_lpencil__mod__cdata[i_uud-1] in the transpiled PC code.
//    Typically these are eliminated as known at compile time.
static bool
is_uniform_condition(const ASTNode* node, const char* loop_variable, const string_vec assigned)
{
	return !subtree_has_func_call(node) && !subtree_has_identifier(node,loop_variable) && !subtree_has_any_identifier(node,assigned);
}
//TP: goes through the loop body in program order and stops hoisting at the first write since the calls could read the written values.
//    Calls in conditionally executed code are not hoisted since they could be invalid if executed unconditionally,
//    except under uniform conditions (see is_uniform_condition): if such a condition is false the hoisted call is only unneeded work
//    that the compiler removes since the result is not used. The field the call is on could be invalid if the condition is false
//    but on the device the stencil functions return NAN for an unknown field and the analysis skips invalid fields.
static void
find_hoistable_calls(const ASTNode* node, const char* loop_variable, const string_vec assigned, const loop_hoist_info* info, const bool conditional, bool* write_seen, node_vec* dst)
{
	if(*write_seen) return;
	if((node->type & NODE_FUNCTION_CALL) && !conditional && is_hoistable_call(node,loop_variable,assigned,info))
	{
		push_node(dst,node);
		return;
	}
	if(node->type & NODE_IF)
	{
		const bool uniform = node->lhs && is_uniform_condition(node->lhs,loop_variable,assigned);
		if(node->lhs) find_hoistable_calls(node->lhs,loop_variable,assigned,info,conditional,write_seen,dst);
		if(node->rhs) find_hoistable_calls(node->rhs,loop_variable,assigned,info,conditional || !uniform,write_seen,dst);
		return;
	}
	const bool cond = conditional || (node->type & NODE_TERNARY) || is_loop_statement(node);
	if(node->lhs) find_hoistable_calls(node->lhs,loop_variable,assigned,info,cond,write_seen,dst);
	if(node->rhs) find_hoistable_calls(node->rhs,loop_variable,assigned,info,cond || is_logical_binary_expression(node),write_seen,dst);
	const char* name = get_func_call_name(node);
	if(name)
	{
		const int index = str_vec_get_index(info->funcs,name);
		if(is_side_effect_call_name(name) || (index >= 0 && info->has_side_effects[index])) *write_seen = true;
	}
}
static ASTNode*
create_non_returning_call_statement(const char* func_name)
{
	node_vec params = VEC_INITIALIZER;
	ASTNode* call = create_func_call(func_name,params);
	ASTNode* basic_statement = astnode_create(NODE_UNKNOWN,call,NULL);
	astnode_set_postfix(";",basic_statement);
	basic_statement->token = NON_RETURNING_FUNC_CALL;
	return create_basic_statement(basic_statement);
}
typedef enum { LOOP_UNROLL, LOOP_KEEP_ROLLED, LOOP_HOISTED } long_loop_action;
static long_loop_action
hoist_iteration_specific_calls(ASTNode* node, loop_hoist_info* info)
{
	ASTNode* body = node->rhs;
	if(!body || !(body->type & NODE_BEGIN_SCOPE) || !body->lhs || !(body->lhs->type & NODE_STATEMENT_LIST_HEAD)) return LOOP_KEEP_ROLLED;
	const char* loop_variable = node->lhs->rhs->lhs->buffer;
	string_vec assigned = VEC_INITIALIZER;
	collect_assigned_identifiers(body,&assigned);
	node_vec calls = VEC_INITIALIZER;
	bool write_seen = false;
	find_hoistable_calls(body,loop_variable,assigned,info,false,&write_seen,&calls);
	free_str_vec(&assigned);
	if(calls.size == 0)
	{
		free_node_vec(&calls);
		return LOOP_KEEP_ROLLED;
	}
	//TP: unrolled as before if all the stencils of the hoisted calls are called on the same arrays also outside long loops
	bool used_elsewhere = true;
	for(size_t i = 0; i < calls.size && used_elsewhere; ++i)
	{
		string_vec arrays = VEC_INITIALIZER;
		collect_loop_indexed_arrays(calls.data[i]->rhs,loop_variable,&arrays);
		string_vec tmp = VEC_INITIALIZER;
		const string_vec* stencils = get_callee_stencils(info,get_func_call_name(calls.data[i]),&tmp);
		if(arrays.size == 0) used_elsewhere = false;
		for(size_t a = 0; a < arrays.size && used_elsewhere; ++a)
			for(size_t st = 0; st < stencils->size && used_elsewhere; ++st)
				if(!used_outside_long_loops(info,arrays.data[a],stencils->data[st])) used_elsewhere = false;
		free_str_vec(&tmp);
		free_str_vec(&arrays);
	}
	if(used_elsewhere)
	{
		free_node_vec(&calls);
		return LOOP_UNROLL;
	}
	ASTNode* head = body->lhs;
	while(head->lhs && (head->lhs->type & NODE_STATEMENT_LIST_HEAD)) head = head->lhs;
	push_to_statement_list(head,create_non_returning_call_statement("ac_dynamic_stencils_begin"));
	//TP: the same call is hoisted only once
	string_vec hoisted_calls = VEC_INITIALIZER;
	string_vec hoisted_vars  = VEC_INITIALIZER;
	for(size_t i = 0; i < calls.size; ++i)
	{
		ASTNode* call = (ASTNode*)calls.data[i];
		const char* call_str = intern(combine_all_new(call));
		const int index = str_vec_get_index(hoisted_calls,call_str);
		if(index >= 0)
		{
			replace_node(call,create_primary_expression(hoisted_vars.data[index]));
			continue;
		}
		//TP: numbered per function such that the otherwise identical copies of a kernel stay identical (see dedup_optimized_kernels)
		const char* var_name = sprintf_intern("ac_loop_hoisted_%d",info->hoist_counter++);
		//TP: typed such that remove_dead_writes can remove the hoisted calls that are not needed
		const char* type = get_expr_type(call);
		ASTNode* call_copy = astnode_dup(call,NULL);
		replace_node(call,create_primary_expression(var_name));
		push_to_statement_list(head,create_assignment(create_declaration(var_name,type,CONST_STR),call_copy,EQ_STR));
		push(&hoisted_calls,call_str);
		push(&hoisted_vars,var_name);
	}
	free_str_vec(&hoisted_calls);
	free_str_vec(&hoisted_vars);
	push_to_statement_list(head,create_non_returning_call_statement("ac_dynamic_stencils_end"));
	free_node_vec(&calls);
	return LOOP_HOISTED;
}
static void
unroll_constant_loops_recursive(ASTNode* node, loop_hoist_info* info)
{
	if(node->type & NODE_FUNCTION)
	{
		const int counter = info->hoist_counter;
		info->hoist_counter = 0;
		if(node->lhs) unroll_constant_loops_recursive(node->lhs,info);
		if(node->rhs) unroll_constant_loops_recursive(node->rhs,info);
		info->hoist_counter = counter;
		return;
	}
	TRAVERSE_PREAMBLE_PARAMS(unroll_constant_loops_recursive,info);
	bool literal_bounds = false;
	int start_value = 0;
	const int trip_count = get_const_loop_trip_count_base(node,&literal_bounds,&start_value);
	if(trip_count < 0) return;
	if(trip_count > MAX_UNROLL_ITERATIONS && hoist_iteration_specific_calls(node,info) != LOOP_UNROLL) return;
	if(!literal_bounds) return;
	const ASTNode* for_expr = node->lhs->rhs;
	const char* loop_variable = for_expr->lhs->buffer;
	const ASTNode* iterated_body = node->rhs;
	//TP: a compound body has to stay a scope also for the DSL: with only the {} emitted for the C++ compiler
	//the declarations of the later iterations are generated as assignments to variables
	//that the C++ compiler sees declared only inside the {} of the first iteration
	//if(iterated_body->type & NODE_BEGIN_SCOPE && iterated_body->lhs) iterated_body = iterated_body->lhs;

	const int end_value   = start_value + trip_count;
	node_vec nodes = VEC_INITIALIZER;
	for(int i = start_value; i < end_value; ++i)
	{
		const char* replacement = intern(itoa(i));
		ASTNode* res = astnode_dup(iterated_body,NULL);
		replace_variable_with_constant_int(res,loop_variable,replacement);
		push_node(&nodes,res);

	}
	ASTNode* res = build_list_node(nodes,"");
	//res is null if we have 0:0 loop then no need to replace
	//or if we have some other error we should skip unrolling since it is optional and thus return
	if(res == NULL) return;
	free_node_vec(&nodes);
	replace_node(node,res);
	if(node->parent->type == NODE_BEGIN_SCOPE)
		node->parent->type = NODE_UNKNOWN;
}
void
unroll_constant_loops(ASTNode* root)
{
	loop_hoist_info info = get_loop_hoist_info(root);
	unroll_constant_loops_recursive(root,&info);
	free_loop_hoist_info(&info);
}
