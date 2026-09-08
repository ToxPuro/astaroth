#!/usr/bin/env python

import argparse
import itertools
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import litgen
import srcmlcpp
from codemanip import code_utils
from codemanip.amalgamated_header import (
    write_amalgamate_header_file,
    AmalgamationOptions,
)
from codemanip.code_replacements import RegexReplacementList


FUNC_DEFINE_PATTERN: str = (
    r"FUNC_DEFINE\s*\(([\w\s\*]+)\s*,\s*([\w]+)\s*,\s*\(([\w\s\*,\[\]\(\)]*)\)\)"
)
OVERLOADED_FUNC_DEFINE_PATTERN: str = r"OVERLOADED_FUNC_DEFINE\s*\(([\w\s\*]+)\s*,\s*([\w]+)\s*,\s*\(([\w\s\*,\[\]\(\)]*)\)\)"

FUNC_DEFINE_RE = re.compile(FUNC_DEFINE_PATTERN, re.MULTILINE)
OVERLOADED_FUNC_DEFINE_RE = re.compile(OVERLOADED_FUNC_DEFINE_PATTERN, re.MULTILINE)


force_lambda_funcs = [
    "^acGetOptimizedDSLTaskGraph$",
]


def get_litgen_options() -> litgen.LitgenOptions:
    def get_function_names_replacements_regex() -> RegexReplacementList:
        regex: RegexReplacementList = RegexReplacementList()
        # All ac-prefixed functions are in an PyAstaroth Python module, so the
        # the prefix is not needed.
        regex.add_replacement(
            "^ac_",
            "",
        )
        regex.add_replacement(
            "^ac",
            "",
        )

        return regex

    def get_type_replacements_regex() -> RegexReplacementList:
        regex: RegexReplacementList = RegexReplacementList()

        regex.add_replacement(
            "^Ac",
            "",
        )

        return regex

    def code_preprocess(code: str) -> str:
        regex: RegexReplacementList = RegexReplacementList()

        def collect_func_define_functions() -> Iterable[str]:
            return map(
                lambda m: rf"{m.group(2)}",
                itertools.chain(
                    re.finditer(FUNC_DEFINE_RE, code),
                    re.finditer(OVERLOADED_FUNC_DEFINE_RE, code),
                ),
            )

        force_lambda_funcs.extend(collect_func_define_functions())

        #
        # Macros
        #

        regex.add_replacement(
            r"[^ a-z]AC_(BEGIN|END)_C_DECLARATIONS[^;]",
            r"AC_\1_C_DECLARATIONS;",
        )

        # Add extra blank lines around #ifdef __cplusplus
        # regex.add_replacement(
        #     r"#ifdef __cplusplus",
        #     "\n#ifdef __cplusplus\n",
        # )
        #
        # regex.add_replacement(
        #     r"#endif",
        #     "\n#endif\n",
        # )

        #
        # Types
        #

        # "typedef enum { ... } <type>;"
        #   into
        # "enum <type> { ... };"
        regex.add_replacement(
            r"typedef\s+enum[\s\w]*\{((?s:.*?))\}\s*(\w+)\s*;",
            r"enum \2 {\1};\n",
        )

        # "typedef struct { ... } <type>;"
        #   into
        # "struct <type> { ... };"
        regex.add_replacement(
            r"typedef\s+struct\s*\w*\s*{((?s:.*?))}\s*(\w+)\s*;",
            r"struct \2 {\1};;\n",
        )

        # "typedef union { ... } <type>;"
        #   into
        # "union <type> { ... };"
        regex.add_replacement(
            r"typedef\s+union\s*\w*\s*{((?s:.*?))}\s*(\w+)\s*;",
            r"union \2 {\1};\n",
        )

        # "typedef <type> <alias>;"
        #   into
        # "using <alias> = <type>;"
        regex.add_replacement(
            r"typedef\s([\w\s*]+)\s\b(\w+);",
            r"using \2 = \1;\n",
        )

        # Opaque structs (forward declarations)
        # "typedef struct <struct_name> <alias>
        #   into
        # "struct <struct_name>"
        # regex.add_replacement(
        #     r"typedef\s+struct\s+(\w*)\s+\w+\s*;",
        #     r"struct \1;",
        # )

        #
        # Functions
        #

        # "OVERLOADED_FUNC_DEFINE(<return_type>, <func_name>, <func_params>)"
        #   into
        # "<return_type> <func_name> <func_params>"
        regex.add_replacement(
            OVERLOADED_FUNC_DEFINE_PATTERN,
            r"\1 \2_BASE(\3)",
        )

        # "FUNC_DEFINE(<return_type>, <func_name>, <func_params>)"
        #   into
        # "<return_type> <func_name> <func_params>"
        regex.add_replacement(
            FUNC_DEFINE_PATTERN,
            r"\1 \2(\3)",
        )

        # FIXME: This is only temporary to make the binding work
        # regex.add_replacement(
        #     r"static\s+AcTaskGraph\*\s+acGetOptimizedDSLTaskGraph",
        #     r"static void *\nacGetOptimizedDSLTaskGraph",
        # )

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
            r"__attribute__\(\(unused\)\)|\[\[maybe_unused\]\]",
            r"",
        )

        #
        # Other visual fixes
        #

        # FIXME: Litgen does not properly pick up on the 'const' qualifier
        # after the parameter list which affects the 'this' keyword in a member
        # function.
        # "<return_type> <func_signature> const <func_body>"
        #   into
        # "<return_type> <func_signature> <func_body>"
        regex.add_replacement(
            r"\) const {",
            r"\) {",
        )

        # for line in regex.apply(code).encode("UTF-7").splitlines():
        #     print(line)

        return regex.apply(code)

    def fn_exclude_by_name(code) -> bool:
        blacklist = [
            # FIXME: Struggles with templates.
            r"^AS_SIZE_T$",
            r"^ceil",
            # r"^acDevice", # Re-enabled!
            r"^acConstruct.+Param$",
            # FIXME: Cannot use double (or more) pointers in parameters.
            r"^acMalloc|acLaunchCooperativeKernel|acHostMeshDestroyVertexBuffer",
            r"acDeviceGetVertexBufferPtrs",
            # FIXME: Cannot return double (or more) pointers from functions.
            r"^ac_allocate_scratchpad_real|ac_allocate_scratchpad_int|ac_allocate_scratchpad_float$",
            ".*allocate_scratchpad.*",
            # FIXME: litgen does not properly overload the fun c
            r"^acMemcpy",
            # FIXME: Incorrectly uses BoxedInt in place of a AcReal array
            r"^acKernelFlush",
            # FIXME: litgen does not handle multi-dimensional arrays as func parameters.
            ".*LoadStencil",
            ".*StoreStencil",
            # FIXME: Getting this error from nanobind -> error: invalid use of incomplete type ”struct ompi_communicator_t”
            # "acGridMPIComm",
            # FIXME: no match for call to ....
            "acCompute",
            "acHaloExchange",
            "acBoundaryCondition",
            # Not needed to be exposed through the bindings.
            "ac_library_not_yet_loaded",
            "acLoadRunTime",
            # FIXME For some reason these struggles with overloads when RUNTIME_COMPILATION=ON
            "acScan",
            "acRayUpdate",
            # FIXME: char* parameters (without const) do not get bound
            # correctly. How to handle preallocated strings? Returned items
            # in Python are always allocated on the heap.
            "acDeviceGetPCIBusId",
            # Bound manually
            "acCommunicator.*",
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
            # r"AcTaskGraph",
            # r"Node",
            # r"const Node",
            # r"Device",
            # FIXME: Struct/classes with const members do not have a default constructor.
            "ParamLoadingInfo",
            # asd
            r"\*\*",
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
        for func in force_lambda_funcs:
            if func in code:
                return True

        return False

    def fn_force_overload(code) -> bool:
        functions = [
            # FIXME: For some reason litgen does not detect this function as being overloaded.
            "acRayUpdate",
        ]

        for func in functions:
            if func in code:
                return True

        return False

    def fn_encapsulate_incomplete_types(code) -> bool:
        incomplete_types = [
            r"^AcTaskGraph\s+\*$",
            r"(const)*\s*Device|Device\s+\*$",
            r"(const)*\s*Node|Node\s+\*$",
            r"MPI_Comm",
            # Specific for CUDA-compilation
            # r"(const)*\s*cudaStream_t\s*\**$",
            # r"(const)*\s*cudaEvent_t\s*\**$",
        ]

        for pattern in incomplete_types:
            if re.match(pattern, code):
                return True

        return False

    def struct_create_default_named_ctor(code: str) -> bool:
        blacklist = [
            # MPI_Comm resolves to an opaque pointer on OpenMPI which causes
            # errors in nanobind due to the type being incomplete.
            "AcCommunicator",
            "AcSubCommunicators",
        ]

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

    options: litgen.LitgenOptions = litgen.LitgenOptions()

    options.bind_library = litgen.BindLibraryType.nanobind
    options.namespaces_root = ["ac"]

    # Names translation from C++ to Python
    options.python_convert_to_snake_case = True
    options.function_names_replacements.merge_replacements(
        get_function_names_replacements_regex()
    )
    options.type_replacements.merge_replacements(get_type_replacements_regex())

    # Class, struct, and member adaptations
    options.struct_create_default_named_ctor__regex = struct_create_default_named_ctor
    options.class_exclude_by_name__regex = class_exclude_by_name
    options.member_exclude_by_type__regex = member_exclude_by_type

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
    options.fn_params_output_modifiable_immutable_to_return__regex = r"acDeviceCreate"

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

    # Force using py::overload for functions that matches these regexes
    options.fn_force_overload__regex = fn_force_overload

    options.fn_encapsulate_incomplete_types__regex = fn_encapsulate_incomplete_types

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
    options.srcmlcpp_options.header_filter_acceptable__regex = (
        code_utils.join_string_by_pipe_char(
            [
                str(options.srcmlcpp_options.header_filter_acceptable__regex),
                # Mandatory
                r"AC_BEGIN_C_DECLARATIONS$",
                r"AC_END_C_DECLARATIONS$",
                # Build-specific
                r"AC_CPU_BUILD$",
                r"AC_MPI_ENABLED$",
                r"AC_DOUBLE_PRECISION$",
                r"AC_RUNTIME_COMPILATION$",
            ]
        )
    )

    return options


def get_amalgamation_options(
    input_header: Path,
    output_header: Path,
    base_dir: Path,
    include_directories: list[Path],
) -> AmalgamationOptions:
    include_subdirs: list[str] = list(map(lambda x: str(x), include_directories))

    options = AmalgamationOptions()

    options.base_dir = str(base_dir)
    options.include_subdirs = include_subdirs
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
        "--amalgamate",
        action="store_true",
        help="",
    )

    parser.add_argument(
        "pydef_file_in",
        type=Path,
        metavar="PYDEF_INPUT",
        help="",
    )
    parser.add_argument(
        "stubs_file_in",
        type=Path,
        metavar="STUBS_INPUT",
        help="",
    )
    parser.add_argument(
        "pydef_file_out",
        type=Path,
        metavar="PYDEF_OUTPUT",
        help="",
    )
    parser.add_argument(
        "stubs_file_out",
        type=Path,
        metavar="STUBS_OUTPUT",
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
        nargs="*",
        type=Path,
        metavar="INCLUDE_DIRS",
        help="",
    )
    parser.add_argument(
        "--dump-processed-header",
        action="store_true",
        help="",
    )

    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="litgen", delete=True) as temp_dir:
        temp_pydef_file: Path = Path(temp_dir, args.pydef_file_out.name)
        temp_stubs_file: Path = Path(temp_dir, args.stubs_file_out.name)

        try:
            shutil.copyfile(args.pydef_file_in, temp_pydef_file)
            shutil.copyfile(args.stubs_file_in, temp_stubs_file)
        except Exception as err:
            print(
                f"There was a problem while preparing input files for generating bindings: {err}",
                file=sys.stderr,
            )
            sys.exit(1)

        main_header_file: Path = args.main_header_file
        if args.amalgamate:
            amalgamated_header: Path = Path(temp_dir, args.main_header_file.name)

            write_amalgamate_header_file(
                get_amalgamation_options(
                    args.main_header_file,
                    amalgamated_header,
                    args.base_dir,
                    args.include_directories,
                )
            )

            main_header_file = amalgamated_header

        litgen_options: litgen.LitgenOptions = get_litgen_options()
        if args.dump_processed_header:
            with open(main_header_file, "r") as f:
                cpp_unit = srcmlcpp.code_to_cpp_unit(
                    litgen_options.srcmlcpp_options, f.read()
                )
            temp_processed_header: Path = Path(temp_dir, "processed_header.h")
            with open(temp_processed_header, "w") as f:
                f.write(cpp_unit.str_code())

        litgen.write_generated_code_for_files(
            options=litgen_options,
            input_cpp_header_files=[str(main_header_file)],
            output_cpp_pydef_file=str(temp_pydef_file),
            output_stub_pyi_file=str(temp_stubs_file),
        )

        try:
            shutil.copyfile(temp_pydef_file, args.pydef_file_out)
            shutil.copyfile(temp_stubs_file, args.stubs_file_out)
        except Exception as err:
            print(
                f"There was a problem while copying the generated bindings into the output location: {err}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
