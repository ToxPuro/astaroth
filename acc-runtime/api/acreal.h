#pragma once

#if AC_DOUBLE_PRECISION
typedef double AcReal;
#define AC_REAL_MAX (DBL_MAX)
#define AC_REAL_MIN (DBL_MIN)
#define AC_REAL_EPSILON (DBL_EPSILON)
#define AC_REAL_MPI_TYPE (MPI_DOUBLE)
#define AC_REAL_INVALID_VALUE (DBL_MAX)
#else
typedef float AcReal;
#define AC_REAL_MAX (FLT_MAX)
#define AC_REAL_MIN (FLT_MIN)
#define AC_REAL_EPSILON (FLT_EPSILON)
#define AC_REAL_MPI_TYPE (MPI_FLOAT)
#define AC_REAL_INVALID_VALUE (FLT_MAX)
#endif


#define AC_REAL_PI ((AcReal)M_PI)
