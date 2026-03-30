add_executable(pc-varfile-import main.c)
set_property(TARGET pc-varfile-import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pc-varfile-import astaroth_core astaroth_utils m)
