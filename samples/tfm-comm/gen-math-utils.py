#!/usr/bin/env python
# %%

types = ['int', 'int64_t', 'size_t', 'float', 'double']

function_name = 'min'
function_params = ['a', 'b']
type_variable = 'a'
function_body = 'return a < b ? a : b;'

# Generics
params = ','.join(function_params)
declarations = ','.join([f'{type}:{function_name}_{type}' for type in types])
print(f'#define {function_name}({params}) _Generic(({type_variable}), {declarations})({params})')

# Function declarations
function_declarations = [f'{type} {function_name}_{type}({",".join([f"const {type} {param}" for param in function_params])})' for type in types]
print('\n'.join([f'{function_declaration};' for function_declaration in function_declarations]))
print('')

# Function definitions
print('\n'.join([f'{function_declaration}{{{function_body}}}' for function_declaration in function_declarations]))

# #for type in types:
# name = 'min'
# params = 'a, b'
# type_variable = 'a'
# declarations = ','.join([f'{type}: {name}_{type}' for type in types])
# print(f'#define {name}({params}) _Generic(({type_variable}), {declarations})({params})')

# function_body = f'return a < b ? a : b;'
# function = f'{type} {name}_{type}({params}) {{{function_body}}}'
# function
