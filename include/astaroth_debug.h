#if AC_MPI_ENABLED
/** */
void acGraphPrintDependencies(const AcTaskGraph* graph);

/** */
void acGraphWriteDependencies(const char* path, const AcTaskGraph* graph);
#endif
