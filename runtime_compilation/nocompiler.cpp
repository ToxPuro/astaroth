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

#include "config_helpers.h"

extern "C" void
acStoreConfig(const AcMeshInfo info, const char* filename)
{
        ac_unset_floating_point_exceptions();
	FILE* fp =  filename == NULL ? stdout : fopen(filename,"w");
	AcScalarTypes::run<load_scalars>(info, fp, "", false);
	AcArrayTypes::run<load_arrays>(info,fp, "", false);

	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp, "", false);
	AcArrayCompTypes::run<load_comp_arrays>(info,    fp, "", false);
	if(filename != NULL) fclose(fp);
        ac_restore_floating_point_exceptions();
}

