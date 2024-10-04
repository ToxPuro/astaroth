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
create_declaration(const char* identifier)
{
	ASTNode* empty       = astnode_create(NODE_UNKNOWN,NULL,NULL);
	ASTNode* decl_vars   = astnode_create(NODE_UNKNOWN,create_identifier_node(identifier),NULL);
	ASTNode* declaration = astnode_create(NODE_DECLARATION,empty,decl_vars);
	return declaration;
}
