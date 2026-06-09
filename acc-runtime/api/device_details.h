#include <rocprim/rocprim_version.hpp>

#if ROCPRIM_VERSION >= 400000
#define rocprim__warpSize() rocprim::arch::wavefront::min_size()
#else
#define rocprim__warpSize() rocprim::device_warp_size()
#endif
#define rocprim__warpId()   rocprim::warp_id()
#define rocprim__warp_shuffle_down rocprim::warp_shuffle_down
#define rocprim__warp_shuffle rocprim::warp_shuffle

#if AC_CPU_BUILD

#else

#if AC_USE_HIP
#include <string.h> // OM: At least with ROCm 7.2.1 and rocPRIM 7.2.0 on Fedora 44 rocPRIM sources
                    // fail to compile due to missing declaration of memset. The error is quite
                    // curious because the header does correctly include <cstring> but seems like
                    // including the C header is necessary.
#include <hip/hip_runtime.h> // Needed in files that include kernels
#include <rocprim/rocprim.hpp>
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#endif
