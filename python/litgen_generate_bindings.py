#!/usr/bin/env python

import argparse
import re
import tempfile
from pathlib import Path

import litgen
from codemanip import code_utils
from codemanip.amalgamated_header import (
    write_amalgamate_header_file,
    AmalgamationOptions,
)
from codemanip.code_replacements import RegexReplacementList


def get_litgen_options() -> litgen.LitgenOptions:
    def get_function_names_replacements_regex() -> RegexReplacementList:
        regex: RegexReplacementList = RegexReplacementList()
        # All ac-prefixed functions are in an PyAstaroth Python module, so the
        # the prefix is not needed.
        regex.add_replacement(
            "^ac_",
            "",
        )

        return regex

    def get_type_replacements_regex() -> RegexReplacementList:
        regex: RegexReplacementList = RegexReplacementList()

        return regex

    def code_preprocess(code: str) -> str:
        regex: RegexReplacementList = RegexReplacementList()

        #
        # Macros
        #

        regex.add_replacement(
            r"[^ a-z]AC_(BEGIN|END)_C_DECLARATIONS[^;]",
            r"AC_\1_C_DECLARATIONS;",
        )

        #
        # Types
        #

        # "typedef enum { ... } <type>;" into "enum <type> { ... };"
        regex.add_replacement(
            r"typedef\s+enum[\s\w]*\{((?s:.*?))\}\s*(\w+)\s*;",
            r"enum \2 {\1};",
        )

        # "typedef struct { ... } <type>;" into "struct <type> { ... };"
        regex.add_replacement(
            r"typedef\s+struct\s*\w*\s*{((?s:.*?))}\s*(\w+)\s*;",
            r"struct \2 {\1};",
        )

        # "typedef union { ... } <type>;" into "union <type> { ... };"
        regex.add_replacement(
            r"typedef\s+union\s*\w*\s*{((?s:.*?))}\s*(\w+)\s*;",
            r"union \2 {\1};",
        )

        # "typedef <type> <alias>;" into "using <alias> = <type>;"
        regex.add_replacement(
            r"typedef\s([\w\s*]+)\s\b(\w+);",
            r"using \2 = \1;",
        )

        #
        # Functions
        #

        # "OVERLOADED_FUNC_DEFINE(<return_type>, <func_name>, <func_params>)" into "<return_type> <func_name> <func_params>"
        regex.add_replacement(
            r"OVERLOADED_FUNC_DEFINE\s*\(([\w\s\*]+)\s*,\s*([\w]+)\s*,\s*\(([\w\s\*,\[\]\(\)]*)\)\)",
            r"\n",  # FIXME: Ignored for now
        )

        # "FUNC_DEFINE(<return_type>, <func_name>, <func_params>)" into "<return_type> <func_name> <func_params>"
        regex.add_replacement(
            r"FUNC_DEFINE\s*\(([\w\s\*]+)\s*,\s*([\w]+)\s*,\s*\(([\w\s\*,\[\]\(\)]*)\)\)",
            r"\1 \2(\3)",
        )

        regex.add_replacement(
            r"static\s+AcTaskGraph\*\s+acGetOptimizedDSLTaskGraph",
            r"static void *\nacGetOptimizedDSLTaskGraph",
        )

        #
        # Qualifiers
        #

        # "... UNUSED|HOST_INLINE|HOST_DEVICE_INLINE ..." into "... ..."
        # At the same time removes the #define directives which would otherwise
        # cause erros in the parser due to them being empty.
        regex.add_replacement(
            r"(?:#define)* UNUSED|HOST_INLINE|HOST_DEVICE_INLINE",
            r"",
        )

        regex.add_replacement(
            r"__attribute__\(\(unused\)\)",
            r"",
        )

        #
        # Other visual fixes
        #

        # FIXME: Litgen does not properly pick up on the 'const' qualifier
        # after the parameter list which affects the 'this' keyword in a member
        # function.
        # "<return_type> <func_signature> const <func_body>" into "<return_type> <func_signature> <func_body>"
        regex.add_replacement(
            r"\) const {",
            r"\) {",
        )

        for line in regex.apply(code).encode("UTF-7").splitlines():
            print(line)

        return regex.apply(code)

    def fn_exclude_by_name(code) -> bool:
        blacklist = [
            # FIXME: Struggles with templates.
            r"^AS_SIZE_T$",
            r"^ceil",
            r"^acDevice",
            r"^acConstruct.+Param$",
            # FIXME: Cannot use double (or more) pointers in parameters.
            # r"^acMalloc|acLaunchCooperativeKernel",
            # FIXME: Cannot return double (or more) pointers from funcftions.
            # r"^ac_allocate_scratchpad_real|ac_allocate_scratchpad_int|ac_allocate_scratchpad_float$",
            # FIXME: litgen does not properly overload the fun c
            r"^acMemcpy",
            # FIXME: Incorrectly uses BoxedInt in place of a AcReal array
            r"^acKernelFlush",
            # FIXME: litgen does not handle multi-dimensional arrays as func parameters.
            ".*LoadStencil",
            ".*StoreStencil",
            # FIXME: Getting this error from nanobind -> error: invalid use of incomplete type ”struct ompi_communicator_t”
            "acGridMPIComm",
            # FIXME: litgen does not handle double pointers in return values.
            ".*allocate_scratchpad.*",
            # FIXME: no match for call to ....
            "acCompute",
            "acHaloExchange",
            "acBoundaryCondition",
        ]

        for pattern in blacklist:
            if re.match(pattern, code):
                return True

        return False

    def fn_exclude_by_param_type(code) -> bool:
        blacklist = [
            # FIXME: Litgen does not directly support unions, so some more
            # scaffolding will be needed (see https://github.com/pthom/litgen/issues/9).
            r"acKernelInputParams",
            # FIXME: Depends on acKernelInputParams which needs to be first
            # added support for.
            r"DeviceVertexBufferArray",
            # FIXME: Depends on DeviceVertexBufferArray which needs to be first
            # added support for.
            r"VertexBufferArray",
            # FIXME: error: invalid use of incomplete type
            r"AcTaskGraph",
            r"Node",
            r"const Node",
            r"Device",
            # FIXME: Struct/classes with const members do not have a default constructor.
            "ParamLoadingInfo",
        ]

        for pattern in blacklist:
            if re.match(pattern, code):
                return True

        return False

    def fn_return_force_policy_reference_for_pointers(code) -> bool:
        functions = [
            "acGridMPIComm",
            "acGetOptimizedDSLTaskGraph",
        ]

        for func in functions:
            if re.match(func, code):
                return True

        return False

    def fn_force_lambda(code) -> bool:
        functions = [
            "acGetOptimizedDSLTaskGraph",
        ]

        for func in functions:
            if func in code:
                return True

        return False

    def struct_create_default_named_ctor(code: str) -> bool:
        blacklist = [
            # MPI_Comm resolves to an opaque pointer on OpenMPI which causes
            # errors in nanobind due to the type being incomplete.
            "AcCommunicator",
            "AcSubCommunicators",
        ]

        print("NO CTOR!", code)
        for pattern in blacklist:
            if re.match(pattern, code):
                return False

        return True

    def class_exclude_by_name(code: str) -> bool:
        blacklist = [
            "DeviceConfiguration",
            "AcReduceBuffer",
            # FIXME: Getting error from nanobind -> error: invalid use of incomplete type ”struct LoadKernelParamsFunc”
            "AcTaskDefinition",
            # FIXME: Struct/classes with const members do not have a default constructor.
            "AcReduction",
            "ParamLoadingInfo",
            # FIXME: Cannot use double (or more) pointers in parameters.
            "ScalarReduceBuffer",
            # FIXME: Depends on acKernelInputParams which needs to be first
            # added support for.
            "DeviceVertexBufferArray",
            # FIXME: Depends on DeviceVertexBufferArray which needs to be first
            # added support for.
            "VertexBufferArray",
        ]

        for item in blacklist:
            if item in code:
                # print("EXCLUDED", code)
                return True

        return False

    def member_exclude_by_type(code: str) -> bool:
        types = [
            "MPI_Comm",
        ]

        for item in types:
            if item in code:
                return True

        return False

    def header_filter_acceptable(code: str) -> bool:
        _default = r"__cplusplus|_h_$|_h$|_H$|_H_$|hpp$|HPP$|hxx$|HXX$"
        if re.match(_default, code):
            return True

        custom = [
            r"^AC_BEGIN_C_DECLARATIONS$",
            r"^AC_END_C_DECLARATIONS$",
            r"^AC_CPU_BUILD$",
            r"^AC_MPI_ENABLED$",
            r"^AC_RUNTIME_COMPILATION$",
        ]
        for pattern in custom:
            if re.match(pattern, code):
                return True

        return False

    options: litgen.LitgenOptions = litgen.LitgenOptions()

    options.bind_library = litgen.BindLibraryType.nanobind
    options.namespaces_root = ["ac"]
    # options.python_run_black_formatter = True

    # Names translation from C++ to Python
    options.python_convert_to_snake_case: bool = True
    options.function_names_replacements.merge_replacements(
        get_function_names_replacements_regex()
    )
    options.type_replacements.merge_replacements(get_type_replacements_regex())

    # Class, struct, and member adaptations
    options.struct_create_default_named_ctor__regex = struct_create_default_named_ctor
    options.class_exclude_by_name__regex = class_exclude_by_name
    options.member_exclude_by_type__regex = member_exclude_by_type

    options.fn_force_lambda__regex = fn_force_lambda

    # Function and method adaptations
    options.fn_exclude_by_name__regex = fn_exclude_by_name
    options.fn_exclude_by_param_type__regex = fn_exclude_by_param_type

    # Templated functions options
    options.fn_template_options.add_specialization(
        "^TO_VOLUME$", ["dim3", "size3_t"], add_suffix_to_function_name=False
    )
    options.fn_template_options.add_specialization(
        "^max|min$", ["Volume"], add_suffix_to_function_name=False
    )
    options.fn_template_options.add_specialization(
        "^as_int|as_int64_t|AS_SIZE_T$",
        ["size_t", "int"],
        add_suffix_to_function_name=False,
    )

    # Make "immutable python types" modifiable, when passed by pointer or reference
    # options.fn_params_replace_modifiable_immutable_by_boxed__regex = r".*"
    options.fn_params_output_modifiable_immutable_to_return__regex = r".*"

    # Force the function that match those regexes to use `pybind11::return_value_policy::reference`
    #
    # Note:
    #    you can also write "// py::return_value_policy::reference" as an end of line comment after the function.
    #    See packages/litgen/integration_tests/mylib/include/mylib/return_value_policy_test.h as an example
    options.fn_return_force_policy_reference_for_pointers__regex = (
        fn_return_force_policy_reference_for_pointers
    )

    # Force using a lambda for functions that matches these regexes
    # (useful when pybind11 is confused and gives error like
    #     error: no matching function for call to object of type 'const detail::overload_cast_impl<...>'
    options.fn_force_lambda__regex = fn_force_lambda

    # Adapt class members
    options.member_numeric_c_array_types = code_utils.join_string_by_pipe_char(
        [
            options.member_numeric_c_array_types,
            "size_t",
            "AcReal",
            "AcReduceOp",
        ]
    )

    # Custom preprocess of the code
    options.srcmlcpp_options.code_preprocess_function = code_preprocess

    # Exclude certain regions based on preprocessor macros
    options.srcmlcpp_options.header_filter_preprocessor_regions = True
    options.srcmlcpp_options.header_filter_acceptable__regex = header_filter_acceptable

    return options


def get_amalgamation_options(
    input_header: Path,
    output_header: Path,
    base_dir: Path,
    include_directories: list[Path],
) -> AmalgamationOptions:
    include_directories = list(map(lambda x: str(x), include_directories))

    options = AmalgamationOptions()

    options.base_dir = str(base_dir)
    options.include_subdirs = include_directories
    options.main_header_file = str(input_header)
    options.dst_amalgamated_header_file = str(output_header)

    return options


def main() -> None:
    parser = argparse.ArgumentParser(
        __name__,
        description=(
            "Generator of Python bindings for Astaroth. DO NOT use directly"
            " unless you know what you're doing. Otherwise, use CMake with the"
            " -DBUILD_PYTHON_BINDINGS option."
        ),
    )

    parser.add_argument(
        "output_cpp_pydef_file",
        type=Path,
        metavar="OUT_PYDEF",
        help="",
    )
    parser.add_argument(
        "output_stub_pyi_file",
        type=Path,
        metavar="OUT_STUBS_PYI",
        help="",
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        metavar="BASE_DIR",
        help="",
    )
    parser.add_argument(
        "main_header_file",
        type=Path,
        metavar="MAIN_HEADER_FILE",
        help="",
    )
    parser.add_argument(
        "include_directories",
        nargs="+",
        type=str,
        metavar="HEADER_FILES",
        help="",
    )

    args = parser.parse_args()

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        amalgamated_header: Path = Path(temp_dir, args.main_header_file.name)

        write_amalgamate_header_file(
            get_amalgamation_options(
                args.main_header_file,
                amalgamated_header,
                args.base_dir,
                args.include_directories,
            )
        )

        litgen.write_generated_code_for_files(
            options=get_litgen_options(),
            input_cpp_header_files=[str(amalgamated_header)],
            output_cpp_pydef_file=str(args.output_cpp_pydef_file),
            output_stub_pyi_file=str(args.output_stub_pyi_file),
        )


if __name__ == "__main__":
    main()
