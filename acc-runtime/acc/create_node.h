static ASTNode*
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
static ASTNode*
create_type_qualifier(const char* tqual, const int token)
{
	if(!tqual) return NULL;
	ASTNode* tqual_identifier = astnode_create(NODE_UNKNOWN,NULL,NULL);
	astnode_set_buffer(tqual,tqual_identifier);
	tqual_identifier -> token = token;
	ASTNode* type_qualifier  = astnode_create(NODE_TQUAL,tqual_identifier,NULL);
	return type_qualifier;
}
static ASTNode*
create_type_qualifiers(const char* tqual, const int token)
{
	ASTNode* type_qualifiers = astnode_create(NODE_UNKNOWN,create_type_qualifier(tqual,token),NULL);
	return type_qualifiers;
}
static ASTNode*
create_tspec(const char* tspec_str)
{
	ASTNode* tspec_identifier  = astnode_create(NODE_UNKNOWN,NULL,NULL);
	astnode_set_buffer(tspec_str,tspec_identifier);
	tspec_identifier -> token = IDENTIFIER;
	ASTNode* tspec  = astnode_create(NODE_TSPEC,tspec_identifier,NULL);
	return tspec;
}
static ASTNode*
create_type_declaration(const char* tqual, const char* tspec, const int token)
{

	if(tqual && tspec)
		return  astnode_create(NODE_UNKNOWN,create_type_qualifiers(tqual,token),create_tspec(tspec));
	if(tspec)
		return  astnode_create(NODE_UNKNOWN,create_tspec(tspec),NULL);
	return  astnode_create(NODE_UNKNOWN,create_type_qualifiers(tqual,token),NULL);

}
static ASTNode*
create_declaration(const char* identifier, const char* type, const char* tqual)
{
	ASTNode* type_decl   = (type == NULL && tqual == NULL) ? NULL : create_type_declaration(tqual, type, 0);
	ASTNode* decl_vars   = astnode_create(NODE_UNKNOWN,create_identifier_node(identifier),NULL);
	ASTNode* declaration = astnode_create(NODE_DECLARATION,type_decl,decl_vars);
	return declaration;
}
static ASTNode*
create_binary_op_expr(const char* op)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,NULL,NULL);
	res->token = BINARY_OP;
	astnode_set_buffer(op,res);
	return res;
}
static ASTNode*
create_choose_node(ASTNode* lhs_value, ASTNode* rhs_value)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,lhs_value,rhs_value);	
	astnode_set_prefix("?", res);
	astnode_set_prefix(": ",res->rhs);
	return res;
}
static ASTNode*
create_ternary_expr(ASTNode* conditional, ASTNode* lhs_value, ASTNode* rhs_value)
{

	return astnode_create(NODE_TERNARY_EXPRESSION,
			      astnode_dup(conditional,NULL),
			      create_choose_node(astnode_dup(lhs_value,NULL),astnode_dup(rhs_value,NULL))
			);
}
static ASTNode*
create_binary_expression(ASTNode* expression , ASTNode* unary_expression, const char* op)
{

	ASTNode* rhs = astnode_create(NODE_UNKNOWN,
					create_binary_op_expr(op),
					unary_expression);
	return 
		astnode_create(NODE_BINARY_EXPRESSION,expression,rhs);
}


static ASTNode*
create_assign_op(const char* op)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,NULL,NULL);
	astnode_set_buffer(op,res);
	res->token = ASSIGNOP;
	return res;
}

static ASTNode*
create_assignment_body(const ASTNode* assign_expr, const char* op)
{
	ASTNode* expression_list = astnode_create(NODE_UNKNOWN,astnode_dup(assign_expr,NULL),NULL);
	return astnode_create(NODE_UNKNOWN,
				astnode_dup(create_assign_op(op),NULL),
				astnode_dup(expression_list,NULL)
				);
}

static ASTNode*
create_basic_statement(const ASTNode* statement)
{
	ASTNode* basic_statement = astnode_create(NODE_UNKNOWN,astnode_dup(statement,NULL),NULL);
	basic_statement ->token = BASIC_STATEMENT;
	ASTNode* res = astnode_create(NODE_UNKNOWN,basic_statement,NULL);
	res->token = STATEMENT;
	return res;
}
static ASTNode*
create_variable_definition(const ASTNode* lhs)
{
	ASTNode* res = astnode_create(NODE_UNKNOWN,astnode_dup(lhs,NULL),NULL);
	astnode_set_postfix(";",res);
	return res;
}
static ASTNode*
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
