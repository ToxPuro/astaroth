set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)
set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/blur CACHE BOOL "" FORCE)

add_executable(blur blur/blur.c)
target_link_libraries(blur astaroth_core astaroth_utils)

#add_executable(hydro hydro/hydro.c)
#target_link_libraries(hydro astaroth_core astaroth_utils)

#add_executable(hydro-sgs hydro/sgs.c)
#target_link_libraries(hydro-sgs astaroth_core astaroth_utils)