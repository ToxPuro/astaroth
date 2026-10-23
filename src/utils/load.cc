#include "astaroth.h"
#include "astaroth_utils.h"

#if AC_RUNTIME_COMPILATION
#include "astaroth_lib.h"

AcResult 
acLoadUtils(const AcMeshInfo info)
{
	static int counter = 0;
	AcLibHandle handle = acLibLoadLibrary(&info, "src/utils", "libastaroth_utils", &counter);

	LOAD_DSYM(acHostVertexBufferSet);
	LOAD_DSYM(acHostMeshSet);
	LOAD_DSYM(acHostMeshApplyPeriodicBounds);
	LOAD_DSYM(acHostMeshApplyConstantBounds);
	LOAD_DSYM(acHostMeshClear);
	LOAD_DSYM(acHostReduceScal);
	LOAD_DSYM(acHostReduceVec);
	LOAD_DSYM(acHostReduceVecScal);
	LOAD_DSYM(acEvalError);
	LOAD_DSYM(acEvalErrorWithMaximumError);
	LOAD_DSYM(acVerifyMesh);
	LOAD_DSYM(acVerifyMeshWithMaximumError);
	LOAD_DSYM(acMeshDiffWriteSliceZ);
	LOAD_DSYM(acMeshDiffWrite);
	LOAD_DSYM(acHostMeshWriteToFile);
	LOAD_DSYM(acHostMeshReadFromFile);
	LOAD_DSYM(acGetError);
	LOAD_DSYM(acHostIntegrateStep);
	LOAD_DSYM(acHostWriteProfileToFile);
	LOAD_DSYM(acHostInitProfileToCosineWave);
	LOAD_DSYM(acHostInitProfileToSineWave);
	LOAD_DSYM(acHostInitProfileToValue);

	return AC_SUCCESS;
}
#else
AcResult acLoadUtils(const AcMeshInfo info)
{
	return AC_FAILURE;
}
#endif
