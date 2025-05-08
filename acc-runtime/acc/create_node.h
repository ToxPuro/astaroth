static inline ASTNode*
create_identifier_node(const char* identifier)
{	
	ASTNode* identifier_node  = astnode_create(NODE_UNKNOWN, NULL, NULL);  
	if(is_number(identifier))
		identifier_node->token    = NUMBER;
	else
		identifier_node->token    = IDENTIFIER;
	astnode_set_buffer(identifier,identifier_node);
	return identifier_node;
}
static inline ASTNode*
create_type_qualifier(const char* tqual)
{
	if(!tqual) return NULL;
	ASTNode* tqual_identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
	astnode_set_buffer(tqual,tqual_identifier);
	ASTNode* type_qualifier  = astnode_create(NODE_TQUAL,tqual_identifier,NULL);
	return type_qualifier;
}
static inline ASTNode*
create_type_qualifiers(const char* tqual)
{
	ASTNode* type_qualifiers = astnode_create(NODE_UNKNOWN,create_type_qualifier(tqual),NULL);
	return type_qualifiers;
}
static inline ASTNode*
create_tspec(const char* tspec_str)
{
	ASTNode* tspec_identifier  = astnode_create(NODE_UNKNOWN,NULL,NULL);
	astnode_set_buffer(tspec_str,tspec_identifier);
	ASTNode* tspec  = astnode_create(NODE_TSPEC,tspec_identifier,NULL);
	return tspec;
}
static inline ASTNode*
create_type_declaration_with_qualifiers(const ASTNode* qualifiers, const char* tspec)
{

	if(qualifiers && tspec)
		return  astnode_create(NODE_UNKNOWN,astnode_dup(qualifiers,NULL),create_tspec(tspec));
	if(tspec)
		return  astnode_create(NODE_UNKNOWN,create_tspec(tspec),NULL);
	return  astnode_create(NODE_UNKNOWN,astnode_dup(qualifiers,NULL),NULL);

}
static inline ASTNode*
create_type_declaration(const char* tqual, const char* tspec)
{

	if(tqual && tspec)
		return  astnode_create(NODE_UNKNOWN,create_type_qualifiers(tqual),create_tspec(tspec));
	if(tspec)
		return  astnode_create(NODE_UNKNOWN,create_tspec(tspec),NULL);
	return  astnode_create(NODE_UNKNOWN,create_type_qualifiers(tqual),NULL);

}
static inline ASTNode*
create_declaration(const char* identifier, const char* type, const char* tqual)
{
	ASTNode* type_decl   = (type == NULL && tqual == NULL) ? NULL : create_type_declaration(tqual, type);
	ASTNode* decl_vars   = astnode_create(NODE_UNKNOWN,create_identifier_node(identifier),NULL);
	ASTNode* declaration = astnode_create(NODE_DECLARATION,type_decl,decl_vars);
	return declaration;
}
static inline ASTNode*
create_binary_op_expr(const char* op)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,NULL,NULL);
	res->token = BINARY_OP;
	astnode_set_buffer(op,res);
	return res;
}
static inline ASTNode*
create_choose_node(ASTNode* lhs_value, ASTNode* rhs_value)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,lhs_value,rhs_value);	
	astnode_set_prefix("?", res);
	astnode_set_prefix(": ",res->rhs);
	return res;
}
static inline ASTNode*
create_ternary_expr(ASTNode* conditional, ASTNode* lhs_value, ASTNode* rhs_value)
{

	return astnode_create(NODE_TERNARY_EXPRESSION,
			      astnode_dup(conditional,NULL),
			      create_choose_node(astnode_dup(lhs_value,NULL),astnode_dup(rhs_value,NULL))
			);
}
static inline ASTNode*
create_binary_expression(ASTNode* expression , ASTNode* unary_expression, const char* op)
{

	ASTNode* rhs = astnode_create(NODE_UNKNOWN,
					create_binary_op_expr(op),
					unary_expression);
	return 
		astnode_create(NODE_BINARY_EXPRESSION,expression,rhs);
}


static inline ASTNode*
create_assign_op(const char* op)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,NULL,NULL);
	astnode_set_buffer(op,res);
	res->token = ASSIGNOP;
	return res;
}

static inline ASTNode*
create_assignment_body(const ASTNode* assign_expr, const char* op)
{
	ASTNode* expression_list = astnode_create(NODE_UNKNOWN,astnode_dup(assign_expr,NULL),NULL);
	return astnode_create(NODE_UNKNOWN,
				astnode_dup(create_assign_op(op),NULL),
				astnode_dup(expression_list,NULL)
				);
}

static inline ASTNode*
create_basic_statement(const ASTNode* statement)
{
	ASTNode* basic_statement = astnode_create(NODE_UNKNOWN,astnode_dup(statement,NULL),NULL);
	basic_statement ->token = BASIC_STATEMENT;
	ASTNode* res = astnode_create(NODE_UNKNOWN,basic_statement,NULL);
	res->token = STATEMENT;
	return res;
}
static inline ASTNode*
create_variable_definition(const ASTNode* lhs)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,astnode_dup(lhs,NULL),NULL);
	astnode_set_postfix(";",res);
	return res;
}
static inline ASTNode*
create_assignment(const ASTNode* lhs, const ASTNode* assign_expr, const char* op)
{

	return create_basic_statement(create_variable_definition
		(
			astnode_create(NODE_ASSIGNMENT,
			      astnode_dup(lhs,NULL),
			      create_assignment_body(assign_expr,op)
			      )
		));
}
static inline ASTNode* create_arr_initializer(const ASTNode* elems)
{
		ASTNode* arr_initializer = astnode_create(NODE_ARRAY_INITIALIZER,astnode_dup(elems,NULL),NULL);
		astnode_set_prefix("{",arr_initializer);
		astnode_set_postfix("}",arr_initializer);
		return arr_initializer;
}
static inline ASTNode* create_struct_initializer(const ASTNode* elems)
{
		ASTNode* arr_initializer = astnode_create(NODE_STRUCT_INITIALIZER,astnode_dup(elems,NULL),NULL);
		astnode_set_prefix("{",arr_initializer);
		astnode_set_postfix("}",arr_initializer);
		return arr_initializer;
}
static inline ASTNode* create_const_declaration(const ASTNode* rhs, const char* name, const ASTNode* type_declaration)
{
		ASTNode* expression = astnode_create(NODE_UNKNOWN,astnode_dup(rhs,NULL),NULL);
		ASTNode* assign_expression = astnode_create(NODE_UNKNOWN,expression,NULL);
		ASTNode* var_identifier = astnode_create(NODE_VARIABLE_ID,NULL,NULL);
		astnode_sprintf(var_identifier,"%s",name);
		var_identifier->token = IDENTIFIER;
		ASTNode* assign_leaf = astnode_create(NODE_ASSIGNMENT, var_identifier,assign_expression);
		ASTNode* assignment_list = astnode_create(NODE_UNKNOWN, assign_leaf,NULL);
		ASTNode* var_definitions = astnode_create(NODE_ASSIGN_LIST,astnode_dup(type_declaration,NULL),assignment_list);
		var_definitions -> type |= NODE_NO_OUT;
		var_definitions -> type |= NODE_DECLARATION;
		astnode_set_postfix(";",var_definitions);
		return var_definitions;
}
