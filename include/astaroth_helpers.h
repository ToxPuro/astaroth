#pragma once

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

FUNC_DEFINE(void, acStoreConfig,(const AcMeshInfo info, const char* filename));


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

FUNC_DEFINE(AcResult,           acAnalysisCheckForDSLErrors,(const AcMeshInfo info));
FUNC_DEFINE(AcResult,           acAnalysisGetKernelInfo,(const AcMeshInfo info, KernelAnalysisInfo* dst));
FUNC_DEFINE(KernelAnalysisInfo, acAnalysisGetKernelInfoSingle,(const AcMeshInfo info, const AcKernel kernel));
FUNC_DEFINE(KernelAnalysisInfo, acAnalysisGetKernelInfoSingleWithInputParams,(const AcMeshInfo info, const AcKernel kernel, const acKernelInputParams input_params));

FUNC_DEFINE(size_t,             acGetKernelId,(const AcKernel kernel));
FUNC_DEFINE(size_t,             acGetKernelIdByName,(const char* name));

FUNC_DEFINE(AcMeshInfo,         acGridGetLocalMeshInfo,(void));
FUNC_DEFINE(AcMeshInfo,         acGridDecomposeMeshInfo,(const AcMeshInfo global_config));
#if AC_RUNTIME_COMPILATION == 0
FUNC_DEFINE(VertexBufferArray, acGridGetVBA,(void));
#endif

#ifdef __cplusplus
//TP: this is done for perf optim since if acVertexBufferIdx is called often
//Making it an external function call is quite expensive
static inline size_t
acGridVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    auto mm = info[AC_mgrid];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    auto mm = info[AC_mlocal];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
#else
static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    const Volume mm = acGetLocalMM(info);
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
static inline size_t
acGridVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    const Volume mm = acGetGridMM(info);
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
#endif

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
// void acQueryInitcondtypes(void);
//
/** Prints a list of int parameters */
// void acQueryIntparams(void);

/** Prints a list of int3 parameters */
// void acQueryInt3params(void);

/** Prints a list of real parameters */
// void acQueryRealparams(void);

/** Prints a list of real3 parameters */
// void acQueryReal3params(void);

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

#ifdef __cplusplus

FUNC_DEFINE(AcReal*, acHostCreateVertexBufferVariable,(const AcMeshInfo info, const VertexBufferHandle vtxbuf));

static inline size_t
acVertexBufferSize(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acVertexBufferVariableSize(info,vtxbuf);
}
static inline AcReal*
acHostCreateVertexBuffer(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acHostCreateVertexBufferVariable(info,vtxbuf);
}

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
   #include "user_builtin_non_scalar_constants.inc"
   const int3 halos = acGetFieldHalos(info,vtxbuf);
   const Volume n0 = 
          (Volume)
          {
                  as_size_t(halos.x),
                  as_size_t(halos.y),
                  as_size_t(halos.z)
          };
   const Volume m1 = 
	   (Volume){
		as_size_t(info.int3_params[vtxbuf_dims[vtxbuf]].x),
		as_size_t(info.int3_params[vtxbuf_dims[vtxbuf]].y),
		as_size_t(info.int3_params[vtxbuf_dims[vtxbuf]].z)
	   };
   const Volume n1 = 
	   (Volume)
	   {
	   	m1.x-n0.x,
	   	m1.y-n0.y,
	   	m1.z-n0.z,
	   };
   const Volume m0 = (Volume){0, 0, 0};
   const Volume nn = 
	   (Volume)
	   {
	   	m1.x-2*n0.x,
	   	m1.y-2*n0.y,
	   	m1.z-2*n0.z,
	   };
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

static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acVertexBufferIdxVariable(i,j,k,info,vtxbuf);
}

static inline Volume 
acVertexBufferDims(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acVertexBufferDimsVariable(info,vtxbuf);
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    return acVertexBufferSizeBytesVariable(info,vtxbuf);
}

static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info, const VertexBufferHandle vtxbuf) {return acVertexBufferCompdomainSizeVariable(info,vtxbuf);}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info, const VertexBufferHandle vtxbuf) {return  acVertexBufferCompdomainSizeBytesVariable(info,vtxbuf); }

#endif
