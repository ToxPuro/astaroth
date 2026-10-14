#include "acc_runtime.h"
#include "astaroth_runtime_compilation.h"
#include "host_datatypes.h"

AcResult
acCompile(const char*, const char*, AcMeshInfo)
{
	return AC_FAILURE;
}

void
acLoadRunConsts(AcMeshInfo){}

#include "astaroth_lib.h"

extern "C"
{

AcLibHandle
acLibLoadLibrary(const AcMeshInfo* info, const char* lib_path, const char* lib_filename,
                 int* counter)
{
	return NULL;
}

AcResult 
acLoadLibrary(const AcMeshInfo info)
{
    return AC_FAILURE;
}

AcResult 
acCloseLibrary()
{
    return AC_FAILURE;
}

}
