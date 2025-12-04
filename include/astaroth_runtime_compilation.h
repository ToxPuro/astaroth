#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
void
acCompile(const char* compilation_string, const char* target, AcMeshInfo info);
void
acLoadRunConsts(AcMeshInfo info);


#ifdef __cplusplus
}
#endif
#ifdef __cplusplus

static UNUSED void
acCompile(const char* compilation_string, AcMeshInfo info)
{
	acCompile(compilation_string,"",info);
}

#endif
