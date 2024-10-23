#pragma once
#ifndef UNUSED

#define UNUSED __attribute__((unused))


#endif
#ifndef HOST_DEVICE
#if (defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)) && !defined(AC_CPU_CODE)
#define HOST_DEVICE __host__ __device__ UNUSED
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__ constexpr UNUSED
#define HOST_INLINE __host__  __forceinline__ UNUSED
#else
#define HOST_DEVICE UNUSED
#define HOST_DEVICE_INLINE inline constexpr UNUSED
#define HOST_INLINE  __forceinline__ UNUSED
#endif // __CUDACC__ || __HIPCC__
#endif //HOST_DEVICE
