file(GLOB kernels "${SCAN_DIR}/*.cu")
target_sources(${OBJECT_LIB} PRIVATE ${kernels})
