#pragma once
#include "func_define.h"
#include "acreal.h"
#include "acc_runtime.h"

AC_BEGIN_C_DECLARATIONS

AcReal
get_reduce_state_flush_var_real(const AcReduceOp state);

int
get_reduce_state_flush_var_int(const AcReduceOp state);

#if AC_DOUBLE_PRECISION
float
get_reduce_state_flush_var_float(const AcReduceOp state);
#endif

AC_END_C_DECLARATIONS
