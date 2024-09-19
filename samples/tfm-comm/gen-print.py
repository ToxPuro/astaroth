#!/usr/bin/env python
# %%
from contextlib import redirect_stdout

types = ["size_t", "int64_t", "int", "double"]
specifiers = ["%zu", "%lld", "%d", "%g"]

# print_format_specifier
fn_name = "format_specifier"
generic_declaration = (
    f"#define {fn_name}(value) _Generic((value), GENERIC_ITEMS)(value)"
)
fn_definition = f'const char* {fn_name}_TYPE(const TYPE value) {{ (void)value; static const char specifier[] = "SPECIFIER"; return specifier; }}'

# Generics
generic_items = [f"TYPE: {fn_name}_TYPE".replace("TYPE", type) for type in types]
generics = generic_declaration.replace("GENERIC_ITEMS", ",".join(generic_items))
# Declarations
fn_declaration = fn_definition.split("{")[0].strip()
fn_declarations = [
    fn_declaration.replace("TYPE", type).replace("SPECIFIER", specifier) + ";"
    for type, specifier in zip(types, specifiers)
]
# Definitions
fn_definitions = [
    fn_definition.replace("TYPE", type).replace("SPECIFIER", specifier)
    for type, specifier in zip(types, specifiers)
]


with open("print.h", "w") as file:
    with redirect_stdout(file):
        print("#pragma once")
        print("#include <stddef.h>")
        print("#include <stdint.h>")
        print("")
        print(generics)
        print("")
        print("\n".join(fn_declarations))

with open("print.c", "w") as file:
    with redirect_stdout(file):
        print('#include "print.h"\n')
        print("#include <stdio.h>\n")
        print("\n\n".join(fn_definitions))
print("")

# print_type
fn_name = "print_type"
generic_declaration = (
    f"#define {fn_name}(value) _Generic((value), GENERIC_ITEMS)(value)"
)
fn_definition = f"void {fn_name}_TYPE(const TYPE value) {{ printf(format_specifier(value), value); }}"

# Generics
generic_items = [f"TYPE: {fn_name}_TYPE".replace("TYPE", type) for type in types]
generics = generic_declaration.replace("GENERIC_ITEMS", ",".join(generic_items))
# Declarations
fn_declaration = fn_definition.split("{")[0].strip()
fn_declarations = [
    fn_declaration.replace("TYPE", type).replace("SPECIFIER", specifier) + ";"
    for type, specifier in zip(types, specifiers)
]
# Definitions
fn_definitions = [
    fn_definition.replace("TYPE", type).replace("SPECIFIER", specifier)
    for type, specifier in zip(types, specifiers)
]


with open("print.h", "a") as file:
    with redirect_stdout(file):
        print("")
        print(generics)
        print("")
        print("\n".join(fn_declarations))

with open("print.c", "a") as file:
    with redirect_stdout(file):
        print("\n\n".join(fn_definitions))
print("")

# print
fn_name = "print"
generic_declaration = (
    f"#define {fn_name}(label, value) _Generic((value), GENERIC_ITEMS)(label, value)"
)
fn_definition = f'void {fn_name}_TYPE(const char* label, const TYPE value) {{ printf("%s: ", label); print_type(value); printf("\\n"); }}'

# Generics
generic_items = [f"TYPE: {fn_name}_TYPE".replace("TYPE", type) for type in types]
generics = generic_declaration.replace("GENERIC_ITEMS", ",".join(generic_items))
# Declarations
fn_declaration = fn_definition.split("{")[0].strip()
fn_declarations = [fn_declaration.replace("TYPE", type) + ";" for type in types]
# Definitions
fn_definitions = [fn_definition.replace("TYPE", type) for type in types]


with open("print.h", "a") as file:
    with redirect_stdout(file):
        print("")
        print(generics)
        print("")
        print("\n".join(fn_declarations))

with open("print.c", "a") as file:
    with redirect_stdout(file):
        print("\n\n".join(fn_definitions))
print("")

# print_array
fn_name = "print_array"
generic_declaration = f"#define {fn_name}(label, count, arr) _Generic((arr), GENERIC_ITEMS)(label, count, arr)"
fn_definition = f'void {fn_name}_TYPE(const char* label, const size_t count, const TYPE* arr) {{ printf("%s: (", label); for (size_t i = 0; i < count; ++i) {{print_type(arr[i]); printf("%s", i < count - 1 ? ", " : "");}} printf(")\\n"); }}'

# Generics
generic_items = [f"TYPE*: {fn_name}_TYPE".replace("TYPE", type) for type in types]
generic_items += ["const " + item for item in generic_items]
generics = generic_declaration.replace("GENERIC_ITEMS", ",".join(generic_items))
# Declarations
fn_declaration = fn_definition.split("{")[0].strip()
fn_declarations = [fn_declaration.replace("TYPE", type) + ";" for type in types]
# Definitions
fn_definitions = [fn_definition.replace("TYPE", type) for type in types]


with open("print.h", "a") as file:
    with redirect_stdout(file):
        print("")
        print(generics)
        print("")
        print("\n".join(fn_declarations))

with open("print.c", "a") as file:
    with redirect_stdout(file):
        print("\n\n".join(fn_definitions))
print("")
