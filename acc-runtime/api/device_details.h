#define rocprim__warpSize() rocprim::warp_size()
#define rocprim__warpId()   rocprim::warp_id()
#define rocprim__warp_shuffle_down rocprim::warp_shuffle_down
#define rocprim__warp_shuffle rocprim::warp_shuffle

#if AC_CPU_BUILD

#else

#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels
#include <rocprim/rocprim.hpp>
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#endif
