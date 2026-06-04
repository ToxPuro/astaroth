#pragma once

#include "func_define.h"

AC_BEGIN_C_DECLARATIONS

AcResult
acCompile(const char* compilation_string, const char* target, AcMeshInfo info);
void
acLoadRunConsts(AcMeshInfo info);

AC_END_C_DECLARATIONS

#ifdef __cplusplus

static UNUSED AcResult
acCompile(const char* compilation_string, AcMeshInfo info)
{
	return acCompile(compilation_string,"",info);
}

#endif
