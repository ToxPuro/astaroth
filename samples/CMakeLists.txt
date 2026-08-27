# If the option is prefixed with "group:" we want to build a group of related
# samples that share their DSL code and thus can be compiled using the same
# build configuration.
string(FIND ${BUILD_SAMPLE} "group:" build_sample_has_group_prefix)
string(FIND ${BUILD_SAMPLE} "python:" build_sample_has_python_prefix)

if (NOT build_sample_has_group_prefix EQUAL -1)
    string(REPLACE "group:" "" build_sample_group ${BUILD_SAMPLE})

    if (build_sample_group STREQUAL "mhd-modular")
        # add_subdirectory(standalone)
        # add_subdirectory(standalone_mpi)
        add_subdirectory(mpitest)
        add_subdirectory(devicetest)
        #add_subdirectory(stencil-loader)
    endif()
elseif (NOT build_sample_has_python_prefix EQUAL -1)
    string(REPLACE "python:" "" python_sample ${BUILD_SAMPLE})

    add_subdirectory("python/${python_sample}")
else()
    add_subdirectory(${BUILD_SAMPLE})
endif()
