#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
AcResult
acCompile(const char* compilation_string, const char* target, AcMeshInfo info);
void
acLoadRunConsts(AcMeshInfo info);


#ifdef __cplusplus
}
#endif
#ifdef __cplusplus

static UNUSED AcResult
acCompile(const char* compilation_string, AcMeshInfo info)
{
	return acCompile(compilation_string,"",info);
}

#endif
