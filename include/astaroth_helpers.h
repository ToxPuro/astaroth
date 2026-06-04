#pragma once

#ifdef __cplusplus
#include "user_builtin_non_scalar_constants.h"
#endif

AC_BEGIN_C_DECLARATIONS

int3 acConstructInt3Param(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                          const AcMeshInfo info);

static inline AcReal3
acConstructReal3Param(const AcRealParam a, const AcRealParam b, const AcRealParam c,
                     const AcMeshInfo info)
{
    return (AcReal3){
        info.real_params[a],
        info.real_params[b],
        info.real_params[c],
    };
}

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */

FUNC_DEFINE(Volume, acGetLocalNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetLocalMM, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetGridNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetGridMM, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetMinNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetMaxNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetGridMaxNN, (const AcMeshInfo info));
FUNC_DEFINE(AcReal3, acGetLengths, (const AcMeshInfo info));

static inline size_t
acVertexBufferSize(const AcMeshInfo info)
{
    const Volume mm = acGetLocalMM(info);
    return mm.x*mm.y*mm.z;
}

static inline size_t
acVertexBufferVariableSize(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    const Volume mm =
        (Volume)
        {
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].x,
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].y,
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].z
        };
    return mm.x*mm.y*mm.z;
}

static inline size_t
acGridVertexBufferSize(const AcMeshInfo info)
{
    const Volume mm = acGetGridMM(info);
    return mm.x*mm.y*mm.z;
}

static inline Volume 
acVertexBufferDims(const AcMeshInfo info)
{
    return acGetLocalMM(info);
}

static inline Volume 
acVertexBufferDimsVariable(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    return
        (Volume)
        {
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].x,
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].x,
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].z
        };
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferSize(info);
}

static inline size_t
acVertexBufferSizeBytesVariable(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    return sizeof(AcReal) * acVertexBufferVariableSize(info,vtxbuf);
}

static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info)
{
    const Volume nn = acGetLocalNN(info);
    return nn.x*nn.y*nn.z;
}

static inline size_t
acVertexBufferCompdomainSizeVariable(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    const Volume nmin = acGetMinNN(info);
    const Volume nn = 
        (Volume)
        {
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].x -2*nmin.x,
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].y -2*nmin.y,
            (size_t)info.int3_params[vtxbuf_dims[vtxbuf]].z -2*nmin.z
        };
    return nn.x*nn.y*nn.z;
}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSize(info);
}

static inline size_t
acVertexBufferCompdomainSizeBytesVariable(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSizeVariable(info,vtxbuf);
}

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info)
{
   const Volume n0 = acGetMinNN(info);
   const Volume n1 = acGetMaxNN(info);
   const Volume m0 = (Volume){0, 0, 0};
   const Volume m1 = acGetLocalMM(info);
   const Volume nn = acGetLocalNN(info);
   const Volume reduction_tile = (Volume)
   {
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].z)
   };

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
       .reduction_tile = reduction_tile,
   };
}

static inline AcMeshDims
acGetGridMeshDims(const AcMeshInfo info)
{
   const Volume n0 = acGetMinNN(info);
   const Volume n1 = acGetGridMaxNN(info);
   const Volume m0 = (Volume){0, 0, 0};
   const Volume m1 = acGetGridMM(info);
   const Volume nn = acGetGridNN(info);
   const Volume reduction_tile = (Volume)
   {
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].z)
   };

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
       .reduction_tile = reduction_tile,
   };
}

FUNC_DEFINE(size_t, acGetKernelId,(const AcKernel kernel));
FUNC_DEFINE(size_t, acGetKernelIdByName,(const char* name));

FUNC_DEFINE(AcResult, acAnalysisGetKernelInfo,(const AcMeshInfo info, KernelAnalysisInfo* dst));
FUNC_DEFINE(KernelAnalysisInfo, acAnalysisGetKernelInfoSingleWithInputParams,(const AcMeshInfo info, const AcKernel kernel, const acKernelInputParams input_params));
FUNC_DEFINE(KernelAnalysisInfo, acAnalysisGetKernelInfoSingle,(const AcMeshInfo info, const AcKernel kernel));
FUNC_DEFINE(AcResult, acAnalysisCheckForDSLErrors,(const AcMeshInfo info));

FUNC_DEFINE(AcMeshInfo, acGridDecomposeMeshInfo,(const AcMeshInfo global_config));

#if AC_RUNTIME_COMPILATION == 0
FUNC_DEFINE(VertexBufferArray, acGridGetVBA,(void));
#endif

FUNC_DEFINE(AcMeshInfo, acGridGetLocalMeshInfo,(void));

FUNC_DEFINE(void, acStoreConfig,(const AcMeshInfo info, const char* filename));

//TP: this is done for perf optim since if acVertexBufferIdx is called often
//Making it an external function call is quite expensive
static inline size_t
acGridVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
#ifdef __cplusplus
    auto mm = info[AC_mgrid];
#else
    const Volume mm = acGetGridMM(info);
#endif
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}

static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
#ifdef __cplusplus
    auto mm = info[AC_mlocal];
#else
    const Volume mm = acGetLocalMM(info);
#endif
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}

static inline size_t
acVertexBufferIdxVariable(const int i, const int j, const int k, const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    const int3 mm = info.int3_params[vtxbuf_dims[vtxbuf]];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}

static inline size_t
acVertexBufferIdxVariable(const int i, const int j, const int k, const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    const int3 mm = info.int3_params[vtxbuf_dims[vtxbuf]];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}

static inline int3
acVertexBufferSpatialIdx(const size_t i, const AcMeshInfo info)
{
    const Volume mm = acGetLocalMM(info);

    return (int3){
        (int)i % (int)mm.x,
        ((int)i % (int)(mm.x * mm.y)) / (int)mm.x,
        (int)i / (int)(mm.x * mm.y),
    };
}

/** Prints all parameters inside AcMeshInfo */
static inline void
acPrintMeshInfo(const AcMeshInfo config)
{
    acStoreConfig(config,NULL);
}

/** Prints a list of initial condition condition types */
void acQueryInitcondtypes(void);

/** Prints a list of int parameters */
void acQueryIntparams(void);

/** Prints a list of int3 parameters */
void acQueryInt3params(void);

/** Prints a list of real parameters */
void acQueryRealparams(void);

/** Prints a list of real3 parameters */
void acQueryReal3params(void);

/** Prints a list of Scalar array handles */
/*
static inline void
acQueryScalarrays(void)
{
    for (int i = 0; i < NUM_REAL_ARRS_1D; ++i)
        printf("%s (%d)\n", realarr1D_names[i], i);
}
*/

/** Prints a list of vertex buffer handles */
static inline void
acQueryVtxbufs(void)
{
    for (int i = 0; i < NUM_ALL_FIELDS; ++i)
        printf("%s (%d)\n", vtxbuf_names[i], i);
}

/** Prints a list of kernels */
void acQueryKernels(void);

static inline void
acPrintIntParam(const AcIntParam a, const AcMeshInfo info)
{
    printf("%s: %d\n", intparam_names[a], info.int_params[a]);
}

void acPrintIntParams(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                      const AcMeshInfo info);

static inline void
acPrintInt3Param(const AcInt3Param a, const AcMeshInfo info)
{
    const int3 vec = info.int3_params[a];
    printf("{%s: (%d, %d, %d)}\n", int3param_names[a], vec.x, vec.y, vec.z);
}

AcResult
acHostUpdateParams(AcMeshInfo* config);

AcResult
acHostUpdateCompParams(AcMeshInfo* config);

OVERLOADED_FUNC_DEFINE(AcReal*, acHostCreateVertexBuffer,(const AcMeshInfo info));
FUNC_DEFINE(AcReal*, acHostCreateVertexBufferVariable,(const AcMeshInfo info, const VertexBufferHandle vtxbuf));
FUNC_DEFINE(AcResult, acHostMeshCreateProfiles,(AcMesh* mesh));
FUNC_DEFINE(AcResult, acHostMeshDestroyVertexBuffer,(AcReal** vtxbuf));
/** Creates a mesh stored in host memory */
FUNC_DEFINE(AcResult, acHostMeshCreate,(const AcMeshInfo mesh_info, AcMesh* mesh));
/** Copies the VertexBuffers from src to dst*/
FUNC_DEFINE(AcResult, acHostMeshCopyVertexBuffers,(const AcMesh src, AcMesh dst));
/** Copies a host mesh to a new host mesh */
FUNC_DEFINE(AcResult, acHostMeshCopy,(const AcMesh src, AcMesh* dst));
/** Creates a mesh stored in host memory (size of the whole grid) */
FUNC_DEFINE(AcResult, acHostGridMeshCreate,(const AcMeshInfo mesh_info, AcMesh* mesh));

/** Checks that the loaded dynamic Astaroth is binary compatible with the loader */
FUNC_DEFINE(AcResult, acVerifyCompatibility, (const size_t mesh_size, const size_t mesh_info_size, const size_t comp_info_size, const int num_reals, const int num_ints, const int num_bools, const int num_real_arrays, const int num_int_arrays, const int num_bool_arrays));

/** Randomizes a host mesh */
FUNC_DEFINE(AcResult, acHostMeshRandomize,(AcMesh* mesh));
/** Randomizes a host mesh (uses n[xyz]grid params)*/
FUNC_DEFINE(AcResult, acHostGridMeshRandomize,(AcMesh* mesh));

/** Destroys a mesh stored in host memory */
FUNC_DEFINE(AcResult, acHostMeshDestroy,(AcMesh* mesh));

/** Sets the dimensions of the computational grid to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */
AcResult acSetGridMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info);

/** Sets the dimensions of the computational subdomain to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */

AcResult acSetLocalMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info);

AC_END_C_DECLARATIONS

#ifdef __cplusplus

static inline size_t
acVertexBufferSize(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    return acVertexBufferVariableSize(info,vtxbuf);
}

#endif
