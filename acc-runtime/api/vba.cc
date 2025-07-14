#include "acc_runtime.h"
#include "math_utils.h"
#include "astaroth_cuda_wrappers.h"
#include "stencil_accesses.h"

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info)
{
   #include "user_builtin_non_scalar_constants.h"
   const Volume n0 = to_volume(info[AC_nmin]);
   const Volume n1 = to_volume(info[AC_nlocal_max]);
   const Volume m0 = (Volume){0, 0, 0};
   const Volume m1 = to_volume(info[AC_mlocal]);
   const Volume nn = to_volume(info[AC_nlocal]);
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
acGetMeshDims(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
   #include "user_builtin_non_scalar_constants.h"
   const Volume n0 = to_volume(info[AC_nmin]);
   const Volume m1 = to_volume(info[vtxbuf_dims[vtxbuf]]);
   const Volume n1 = m1-n0;
   const Volume m0 = (Volume){0, 0, 0};
   const Volume nn = m1-n0*2;
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

AcResult
acPBAReset(const cudaStream_t stream, ProfileBufferArray* pba, const AcMeshDims* dims)
{
  // Set pba.in data to all-nan and pba.out to 0
  for (int i = 0; i < NUM_PROFILES; ++i) {
    acKernelFlush(stream, pba->in[i],  prof_count(Profile(i),dims[i].m1), (AcReal)AC_REAL_MAX);
    acKernelFlush(stream, pba->out[i], prof_count(Profile(i),dims[i].m1), (AcReal)0);
  }
  return AC_SUCCESS;
}

ProfileBufferArray
acPBACreate(const AcMeshDims* dims)
{
  ProfileBufferArray pba{};
  for (int i = 0; i < NUM_PROFILES; ++i) {
    const size_t bytes = prof_size(Profile(i),dims[i].m1)*sizeof(AcReal);
    acDeviceMalloc((void**)&pba.in[i],  bytes);
    acDeviceMalloc((void**)&pba.out[i], bytes);
    //pba.out[i] = pba.in[i];
  }

  acPBAReset(0, &pba, dims);
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
  return pba;
}

size3_t
acGetProfileReduceScratchPadDims(const int profile, const AcMeshDims dims)
{
	const auto type = prof_types[profile];
    	if(type == PROFILE_YZ || type == PROFILE_ZY)
    		return
    		{
    		    	    dims.reduction_tile.x,
    		    	    dims.m1.y,
    		    	    dims.m1.z
    		};
    	if(type == PROFILE_XZ || type == PROFILE_ZX)
		return
    		{
    		    	    dims.m1.x,
    		    	    dims.reduction_tile.y,
    		    	    dims.m1.z
    		};
    	if(type == PROFILE_YX || type == PROFILE_XY)
		return
    		{
    		    	    dims.m1.x,
    		    	    dims.m1.y,
    		    	    dims.reduction_tile.z
    		};
	if(type == PROFILE_X)
	{
		return
		{
			dims.m1.x,
			dims.reduction_tile.y,
			dims.reduction_tile.z
		};
	}
	if(type == PROFILE_Y)
	{
		return
		{
			dims.reduction_tile.x,
			dims.m1.y,
			dims.reduction_tile.z
		};
	}
	if(type == PROFILE_Z)
	{
		return
		{
			dims.reduction_tile.x,
			dims.reduction_tile.y,
			dims.m1.z
		};
	}
	return dims.m1;
}

static size_t
get_profile_reduce_scratchpad_size(const int profile, const VertexBufferArray vba)
{
	if(!reduced_profiles[profile]) return 0;
	const auto dims = acGetProfileReduceScratchPadDims(profile,vba.profile_dims[profile]);
	return dims.x*dims.y*dims.z*sizeof(AcReal);
}

void
init_scratchpads(VertexBufferArray* vba)
{
    vba->scratchpad_states = (AcScratchpadStates*)malloc(sizeof(AcScratchpadStates));
    memset(vba->scratchpad_states,0,sizeof(AcScratchpadStates));
    // Reductions
    {
	//TP: this is dangerous since it is not always true for DSL reductions but for now keep it
    	for(int i = 0; i < NUM_REAL_SCRATCHPADS; ++i) {
	    const size_t bytes =  
		    		  (i >= NUM_REAL_OUTPUTS) ? get_profile_reduce_scratchpad_size(i-NUM_REAL_OUTPUTS,*vba) :
				  0;
	    AcReal** tmp = ac_allocate_scratchpad_real(i,bytes,vba->scratchpad_states->reals[i]);
	    if(i < NUM_REAL_OUTPUTS)
	    {
	    	vba->reduce_buffer_real[i].src = tmp;
	    	vba->reduce_buffer_real[i].cub_tmp = (AcReal**)malloc(sizeof(AcReal*));
	    	*(vba->reduce_buffer_real[i].cub_tmp) = NULL;
	    	vba->reduce_buffer_real[i].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    	*(vba->reduce_buffer_real[i].cub_tmp_size) = 0;

	    	vba->reduce_buffer_real[i].buffer_size = ac_get_scratchpad_size_real(i);
    		acDeviceMalloc((void**) &vba->reduce_buffer_real[i].res,sizeof(AcReal));
	    }
	    else
	    {
		    const Profile prof = (Profile)(i-NUM_REAL_OUTPUTS);
		    const auto dims = acGetProfileReduceScratchPadDims(prof,vba->profile_dims[prof]);
		    vba->profile_reduce_buffers[prof].src = 
		    {
			    *tmp,
			    dims.x*dims.y*dims.z,
			    true,
			    (AcShape) { dims.x,dims.y,dims.z,1}
		    };
		    vba->profile_reduce_buffers[prof].transposed = acBufferCreateTransposed(
				vba->profile_reduce_buffers[prof].src, 
				acGetMeshOrderForProfile(prof_types[prof])
				  );
		    vba->profile_reduce_buffers[prof].mem_order = acGetMeshOrderForProfile(prof_types[prof]);

	    	    vba->profile_reduce_buffers[prof].cub_tmp = (AcReal**)malloc(sizeof(AcReal*));
	    	    *(vba->profile_reduce_buffers[prof].cub_tmp) = NULL;
	    	    vba->profile_reduce_buffers[prof].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    	    *(vba->profile_reduce_buffers[prof].cub_tmp_size) = 0;
	    }
    	}
    }
    {
    	for(int i = 0; i < NUM_INT_OUTPUTS; ++i) {
	    const size_t bytes = 0;
	    int** tmp = ac_allocate_scratchpad_int(i,bytes,vba->scratchpad_states->ints[i]);

	    vba->reduce_buffer_int[i].src= tmp;
	    vba->reduce_buffer_int[i].cub_tmp = (int**)malloc(sizeof(int*));
	    *(vba->reduce_buffer_int[i].cub_tmp) = NULL;
	    vba->reduce_buffer_int[i].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    *(vba->reduce_buffer_int[i].cub_tmp_size) = 0;
	    vba->reduce_buffer_int[i].buffer_size = ac_get_scratchpad_size_int(i);
    	    acDeviceMalloc((void**) &vba->reduce_buffer_int[i].res,sizeof(int));
    	}

#if AC_DOUBLE_PRECISION
    	for(int i = 0; i < NUM_FLOAT_OUTPUTS; ++i) {
	    const size_t bytes = 0;
	    float** tmp = ac_allocate_scratchpad_float(i,bytes,vba->scratchpad_states->floats[i]);

	    vba->reduce_buffer_float[i].src = tmp;
	    vba->reduce_buffer_float[i].cub_tmp = (float**)malloc(sizeof(float*));
	    *(vba->reduce_buffer_float[i].cub_tmp) = NULL;
	    vba->reduce_buffer_float[i].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    *(vba->reduce_buffer_float[i].cub_tmp_size) = 0;
	    vba->reduce_buffer_float[i].buffer_size = ac_get_scratchpad_size_float(i);
    	    acDeviceMalloc((void**) &vba->reduce_buffer_float[i].res,sizeof(float));
    	}
#endif
    }
}

static AcReal*  vba_in_buff = NULL;
static AcReal*  vba_out_buff = NULL;

AcResult
acVBAReset(const cudaStream_t stream, VertexBufferArray* vba)
{

  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    ERRCHK_ALWAYS(vba->on_device.in[i]);
    ERRCHK_ALWAYS(vba->on_device.out[i]);
    acKernelFlush(stream, vba->on_device.in[i], vba->counts[i], (AcReal)AC_REAL_MAX);
    acKernelFlush(stream, vba->on_device.out[i], vba->counts[i], (AcReal)0.0);
  }

  for(int field = 0; field < NUM_COMPLEX_FIELDS; ++field)
  {
    size_t n = vba->computational_dims.m1.x*vba->computational_dims.m1.y*vba->computational_dims.m1.z;
    ERRCHK_ALWAYS(vba->on_device.complex_in[field]);
    acMultiplyInplaceComplex(AcReal(0.),n,vba->on_device.complex_in[field]);
  }
  memset(&vba->on_device.kernel_input_params,0,sizeof(acKernelInputParams));
  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBAReset(stream, &vba->on_device.profiles, vba->profile_dims);
  return AC_SUCCESS;
}

VertexBufferArray
acVBACreate(const AcMeshInfo config)
{
  #include "user_builtin_non_scalar_constants.h"
  VertexBufferArray vba;
  vba.on_device.block_factor = config[AC_thread_block_loop_factors];

  vba.computational_dims = acGetMeshDims(config);

  size_t in_bytes  = 0;
  size_t out_bytes = 0;
  for(int i = 0; i  < NUM_FIELDS; ++i)
  {
  	vba.dims[i]    = acGetMeshDims(config,Field(i));
  	size_t count = vba.dims[i].m1.x*vba.dims[i].m1.y*vba.dims[i].m1.z;
  	size_t bytes = sizeof(vba.on_device.in[0][0]) * count;
  	vba.counts[i]         = count;
  	vba.bytes[i]          = bytes;
	in_bytes  += vba.bytes[i];
	if(vtxbuf_is_auxiliary[i]) continue;
	out_bytes += vba.bytes[i];
  }
  for(int p = 0; p < NUM_PROFILES; ++p)
  {
	  vba.profile_dims[p] = acGetMeshDims(config);
  	  vba.profile_counts[p] = vba.profile_dims[p].m1.x*vba.profile_dims[p].m1.y*vba.profile_dims[p].m1.z;
  }
  for(int field = 0; field < NUM_COMPLEX_FIELDS; ++field)
  {
  	size_t count = vba.computational_dims.m1.x*vba.computational_dims.m1.y*vba.computational_dims.m1.z;
	acDeviceMalloc((void**)&vba.on_device.complex_in[field],sizeof(AcComplex)*count);
  }

  ERRCHK_ALWAYS(vba_in_buff == NULL);
  ERRCHK_ALWAYS(vba_out_buff == NULL);
  acDeviceMalloc((void**)&vba_in_buff,in_bytes);
  acDeviceMalloc((void**)&vba_out_buff,out_bytes);

  size_t out_offset = 0;
  size_t in_offset = 0;
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    vba.on_device.in[i] = vba_in_buff + in_offset;
    ERRCHK_ALWAYS(vba.on_device.in[i] != NULL);
    in_offset += vba.counts[i];
    if (vtxbuf_is_auxiliary[i])
    {
      vba.on_device.out[i] = vba.on_device.in[i];
      ERRCHK_ALWAYS(vba.on_device.out[i] != NULL);
    }else{
      vba.on_device.out[i] = (vba_out_buff + out_offset);
      out_offset += vba.counts[i];
      if(vba.on_device.out[i] == NULL)
      {
         fprintf(stderr,"In bytes %zu; Out bytes: %zu\n",in_bytes,out_bytes);	
	 fflush(stderr);
       	 ERRCHK_ALWAYS(vba.on_device.out[i] != NULL);
      }
    }
  }

  // Note: should be moved out when refactoring VBA to KernelParameterArray
  vba.on_device.profiles = acPBACreate(vba.profile_dims);
  init_scratchpads(&vba);

  acVBAReset(0, &vba);
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
  return vba;
}


static void
destroy_profiles(VertexBufferArray* vba)
{
    for(int i = 0; i < NUM_PROFILES; ++i)
    {
        //TP: will break if allocated with compressed memory but too lazy to fix now: :(
        acDeviceFree((void**)&(vba->profile_reduce_buffers[i].transposed),0);
        ac_free_scratchpad_real(i+NUM_REAL_OUTPUTS);
    }
}

static void
destroy_real_scratchpads(VertexBufferArray* vba)
{
    for(int j = 0; j < NUM_REAL_OUTPUTS; ++j)
    {
	ac_free_scratchpad_real(j);
	vba->reduce_buffer_real[j].src = NULL;

        ERRCHK_CUDA_ALWAYS(acFree(*vba->reduce_buffer_real[j].cub_tmp));
        ERRCHK_CUDA_ALWAYS(acFree(vba->reduce_buffer_real[j].res));

	free(vba->reduce_buffer_real[j].cub_tmp);
	free(vba->reduce_buffer_real[j].cub_tmp_size);
    }
}

static void
destroy_scratchpads(VertexBufferArray* vba)
{
    destroy_real_scratchpads(vba);

    destroy_profiles(vba);

    for(int j = 0; j < NUM_INT_OUTPUTS; ++j)
    {
	ac_free_scratchpad_int(j);
	vba->reduce_buffer_int[j].src = NULL;

        ERRCHK_CUDA_ALWAYS(acFree(*vba->reduce_buffer_int[j].cub_tmp));
        ERRCHK_CUDA_ALWAYS(acFree(vba->reduce_buffer_int[j].res));

	free(vba->reduce_buffer_int[j].cub_tmp);
	free(vba->reduce_buffer_int[j].cub_tmp_size);
    }
#if AC_DOUBLE_PRECISION
    for(int j = 0; j < NUM_FLOAT_OUTPUTS; ++j)
    {
	ac_free_scratchpad_float(j);
	vba->reduce_buffer_float[j].src = NULL;

        ERRCHK_CUDA_ALWAYS(acFree(*vba->reduce_buffer_float[j].cub_tmp));
        ERRCHK_CUDA_ALWAYS(acFree(vba->reduce_buffer_float[j].res));

	free(vba->reduce_buffer_float[j].cub_tmp);
	free(vba->reduce_buffer_float[j].cub_tmp_size);
    }
#endif
}

void
acPBADestroy(ProfileBufferArray* pba, const AcMeshDims* dims)
{
  for (int i = 0; i < NUM_PROFILES; ++i) {
    const size_t bytes = prof_size(Profile(i),dims[i].m1)*sizeof(AcReal);
    acDeviceFree(&pba->in[i],  bytes);
    acDeviceFree(&pba->out[i], bytes);
    pba->in[i]  = NULL;
    pba->out[i] = NULL;
  }
}

void
acVBADestroy(VertexBufferArray* vba, const AcMeshInfo config)
{
  destroy_scratchpads(vba);
  //TP: does not work for compressible memory TODO: fix it if needed
  acDeviceFree(&(vba_in_buff), 0);
  acDeviceFree(&(vba_out_buff), 0);
  for(int field = 0; field < NUM_COMPLEX_FIELDS; ++field)
  {
  	acDeviceFree(&vba->on_device.complex_in[field], 0);
  }

  acFreeArrays(config);
  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBADestroy(&vba->on_device.profiles,vba->profile_dims);
  memset(vba->profile_dims,0,NUM_PROFILES*sizeof(vba->profile_dims[0]));
  memset(vba->bytes,0,NUM_ALL_FIELDS*sizeof(size_t));
  memset(vba->dims,0,NUM_ALL_FIELDS*sizeof(vba->dims[0]));
}

void
acVBASwapBuffer(const Field field, VertexBufferArray* vba)
{
  AcReal* tmp     = vba->on_device.in[field];
  vba->on_device.in[field]  = vba->on_device.out[field];
  vba->on_device.out[field] = tmp;
}

void
acVBASwapBuffers(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_FIELDS; ++i)
    acVBASwapBuffer((Field)i, vba);
}

void
acPBASwapBuffer(const Profile profile, VertexBufferArray* vba)
{
  AcReal* tmp                = vba->on_device.profiles.in[profile];
  vba->on_device.profiles.in[profile]  = vba->on_device.profiles.out[profile];
  vba->on_device.profiles.out[profile] = tmp;
}

void
acPBASwapBuffers(VertexBufferArray* vba)
{
  for (int i = 0; i < NUM_PROFILES; ++i)
    acPBASwapBuffer((Profile)i, vba);
}


static AcResult
ac_flush_scratchpad(VertexBufferArray vba, const int variable, const AcType type, const AcReduceOp op)
{
	const int n_elems = 
				type == AC_REAL_TYPE ?  NUM_REAL_OUTPUTS :
				type == AC_PROF_TYPE ?  NUM_PROFILES     :
				type == AC_INT_TYPE  ?  NUM_INT_OUTPUTS  :
#if AC_DOUBLE_PRECISION
				type == AC_FLOAT_TYPE  ?  NUM_FLOAT_OUTPUTS  :
#endif
				0;
	ERRCHK_ALWAYS(variable < n_elems);
	const size_t counts = 
			type == AC_INT_TYPE  ? (*vba.reduce_buffer_int[variable].buffer_size)/sizeof(int) :
#if AC_DOUBLE_PRECISION
			type == AC_FLOAT_TYPE  ? (*vba.reduce_buffer_float[variable].buffer_size)/sizeof(float) :
#endif
			type == AC_REAL_TYPE ? (*vba.reduce_buffer_real[variable].buffer_size)/sizeof(AcReal) :
			type == AC_PROF_TYPE ? (acShapeCount(vba.profile_reduce_buffers[variable].src.shape)) :
			0;

	if(type == AC_REAL_TYPE)
	{
		if constexpr (NUM_REAL_OUTPUTS == 0) return AC_FAILURE;
		AcReal* dst = *(vba.reduce_buffer_real[variable].src);
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_real(op));
	}
	else if(type == AC_PROF_TYPE)
	{
		if constexpr(NUM_PROFILES == 0) return AC_FAILURE;
		AcReal* dst = vba.profile_reduce_buffers[variable].src.data;
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_real(op));
	}
#if AC_DOUBLE_PRECISION
	else if(type == AC_FLOAT_TYPE)
	{
		if constexpr(NUM_FLOAT_OUTPUTS  == 0) return AC_FAILURE;
		float* dst = *(vba.reduce_buffer_float[variable].src);
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_float(op));
	}
#endif
	else
	{
		if constexpr (NUM_INT_OUTPUTS == 0) return AC_FAILURE;
		int* dst = *(vba.reduce_buffer_int[variable].src);
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_int(op));
	}
  	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
	return AC_SUCCESS;
}

static AcReduceOp*
get_reduce_buffer_states(const VertexBufferArray vba, const AcType type)
{
	return
#if AC_DOUBLE_PRECISION
			type == AC_FLOAT_TYPE  ? vba.scratchpad_states->floats :
#endif
			type == AC_INT_TYPE    ? vba.scratchpad_states->ints  :
			type == AC_REAL_TYPE   ? vba.scratchpad_states->reals :
			type == AC_PROF_TYPE   ? &vba.scratchpad_states->reals[NUM_REAL_OUTPUTS] :
			NULL;
}

static UNUSED AcReduceOp
get_reduce_buffer_state(const VertexBufferArray vba, const int variable, const AcType type)
{
	return get_reduce_buffer_states(vba,type)[variable];
}

AcResult
acPreprocessScratchPad(VertexBufferArray vba, const int variable, const AcType type,const AcReduceOp op)
{
	AcReduceOp* states = get_reduce_buffer_states(vba,type);
	if(states[variable] == op && !AC_CPU_BUILD) return AC_SUCCESS;
	states[variable] = op;
	return ac_flush_scratchpad(vba,variable,type,op);
}

