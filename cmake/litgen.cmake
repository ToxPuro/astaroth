# Copyright (c) 2016 The Pybind Development Team, All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.

set(litgen_cmake_help_message "

This Cmake module provides several public functions:

litgen_find_python
******************
litgen_find_python() will find Python >= 3.8 with the interpreter and Development module.
(and working around quirks of CMake 3.18 and below)

litgen_find_nanobind
*********************
litgen_find_nanobind() will find nanobind and Python3

litgen_setup_module(bound_library python_native_module_name python_module_name)
*******************************************************************************
litgen_setup_module is a helper function that will:
* link the python native module (.so) to the bound C++ library (bound_library)
* set the install path of the native module to '.' (so that pip install works)
* set the VERSION_INFO macro to the project version defined in CMakeLists.txt
* when building as a C++ project (i.e. not with skbuild), it will copy the python module
  to the editable_bindings_folder and to the platform dependent installation directory
  (site-packages when using pip install), so that it is used by our next runs of python.


Note: how to specify the version of Python to use
*************************************************
When building via CMake, you may have to specify Python_EXECUTABLE via
     -DPython_EXECUTABLE=/path/to/your/venv/bin/python
")

function(litgen_generate_bindings
         target
         input_pydef
         input_stubs
         header_file
         output_dir
)
    set(options AMALGAMATE)
    set(oneValueArgs BASE_DIR)
    set(multiValueArgs INCLUDE_DIRECTORIES DEPENDENCIES)
    cmake_parse_arguments(PARSE_ARGV 1 arg
        "${options}" "${oneValueArgs}" "${multiValueArgs}"
    )

    set(litgen_generator "${CMAKE_SOURCE_DIR}/python/litgen_generate_bindings.py")
    set(litgen_target_msg "Generating C++ bindings and Python stubs using litgen for target ${target}")
    list(JOIN arg_INCLUDE_DIRECTORIES " " INCLUDE_DIRECTORIES)

    cmake_path(GET input_pydef STEM module_name)

    set(${target}_pydef "${output_dir}/pydef/${module_name}.cc")
    set(${target}_stubs "${output_dir}/stubs/${module_name}.pyi")

    set(output_pydef_in "${${target}_pydef}.in")
    set(output_stubs_in "${${target}_stubs}.in")

    configure_file("${input_pydef}" "${output_pydef_in}" COPYONLY)
    configure_file("${input_stubs}" "${output_stubs_in}" COPYONLY)

    if(arg_AMALGAMATE)
        add_custom_command(
            OUTPUT
                ${${target}_pydef}
                ${${target}_stubs}
            COMMAND
                ${Python3_Executable}
                ${litgen_generator}
                --amalgamate
                ${output_pydef_in}
                ${output_stubs_in}
                ${${target}_pydef}
                ${${target}_stubs}
                ${arg_BASE_DIR}
                ${header_file}
                ${INCLUDE_DIRECTORIES}
            WORKING_DIRECTORY
                ${arg_BASE_DIR}
            COMMENT
                ${litgen_target_msg}
            DEPENDS
                ${arg_DEPENDENCIES}
                ${litgen_generator}
        )
    else()
        add_custom_command(
            OUTPUT
                ${${target}_pydef}
                ${${target}_stubs}
            COMMAND
                ${Python3_Executable}
                ${litgen_generator}
                ${output_pydef_in}
                ${output_stubs_in}
                ${${target}_pydef}
                ${${target}_stubs}
                ${arg_BASE_DIR}
                ${header_file}
            WORKING_DIRECTORY
                ${arg_BASE_DIR}
            COMMENT
                ${litgen_target_msg}
            DEPENDS
                ${arg_DEPENDENCIES}
                ${litgen_generator}
        )
    endif()

    foreach(dir ${INCLUDE_DIRECTORIES})
        file(
            GLOB
            _globbed_headers
            LIST_DIRECTORIES false
            CONFIGURE_DEPENDS
            "${dir}/*.h")

        foreach(header ${_globbed_headers})
            add_custom_command(
                OUTPUT
                    ${${target}_pydef}
                    ${${target}_stubs}
                APPEND
                DEPENDS
                    ${header}
                    ${litgen_generator}
            )
        endforeach()
    endforeach()

    add_custom_target(
        ${target}
        ALL
        DEPENDS
            ${${target}_pydef}
            ${${target}_stubs}
    )

    set(${target}_pydef "${${target}_pydef}" PARENT_SCOPE)
    set(${target}_stubs "${${target}_stubs}" PARENT_SCOPE)
endfunction()
