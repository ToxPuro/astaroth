#!/usr/bin/env python
# %%
fn_name = "print_type"
fn_params = ["value"]
type_variable = "value"
types = ["int", "int64_t", "size_t", "double"]
specifiers = ["%d", "%lld", "%zu", "%g"]
fn_bodies = [f"printf({specifier}, a);" for specifier in specifiers]


def generate_generics(fn_name, fn_params, type_variable):
    params = ",".join(fn_params)
    declarations = ",".join([f"{type}:{fn_name}_{type}" for type in types])
    print(
        f"#define {fn_name}({params}) _Generic(({type_variable}), {declarations})({params})"
    )


def generate_declarations(fn_name, fn_params):
    fn_declarations = [
        f'void {fn_name}_{type}({",".join([f"const {type} {param}" for param in fn_params])})'
        for type in types
    ]
    print("\n".join([f"{fn_declaration};" for fn_declaration in fn_declarations]))
    print("")
    return fn_declarations


def generate_definitions(fn_declarations, fn_bodies):
    print(
        "\n".join(
            [
                f"{fn_declaration}{{{fn_body}}}"
                for fn_declaration, fn_body in zip(fn_declarations, fn_bodies)
            ]
        )
    )


generate_generics(fn_name, fn_params, type_variable)
fn_declarations = generate_declarations(fn_name, fn_params)
generate_definitions(fn_declarations, fn_bodies)


# %%
fn_name = "print"
fn_params = ["label", "value"]
type_variable = "value"
types = ["int", "int64_t", "size_t", "double"]
fn_body = f'printf("%s: "); print(value); printf("\n");'
generate_generics(fn_name, fn_params, type_variable)
fn_declarations = generate_declarations(fn_name, fn_params)


# %%
fn_name = "print_type"
fn_params = ["const char* label", "const TYPE value"]
fn_param_labels = [for param in fn_params]
return_type = "TYPE"
types = ["int", "int64_t", "size_t", "double"]
specifiers = ["%d", "%lld", "%zu", "%g"]

generic = f'TYPE: {fn_name}_TYPE'
declaration = f'{return_type} {fn_name}_TYPE({",".join(fn_params)})'
definition = f'printf(SPECIFIER, value);'

generics = [generic.replace("TYPE", type) for type in types]
declarations = [declaration.replace("TYPE", type) for type in types]
definitions = [definition.replace("TYPE", type).replace("SPECIFIER", specifier) for type, specifier in zip(types,specifiers)]


print(f'#define {fn_name} _Generic(x)()')

# %%
# GENERIC_ITEMS
# 
fn_name = 'print'
generic_declaration = f'#define {fn_name} _Generic((value), GENERIC_ITEMS)(label, value)'
fn_definition = f'TYPE {fn_name}_TYPE(const char* label, const TYPE value) {{ print(SPECIFIER, value); }}' 

types = ["int", "int64_t", "size_t", "double"]

# Generics
generic_items = [f'TYPE: {fn_name}_TYPE'.replace('TYPE', type) for type in types]
generics = generic_declaration.replace('GENERIC_ITEMS', ','.join(generic_items))
# Declarations
fn_declaration = fn_definition.split('{')[0].strip()
fn_declarations = [fn_declaration.replace('TYPE', type).replace('SPECIFIER', specifier) + ';' for type, specifier in zip(types, specifiers)]
# Definitions
specifiers = ["%d", "%lld", "%zu", "%g"]
fn_definitions = [fn_definition.replace('TYPE', type).replace('SPECIFIER', specifier) for type, specifier in zip(types, specifiers)]


#with open()
print(generics)
print('\n'.join(fn_declarations))
print('\n'.join(fn_definitions))
