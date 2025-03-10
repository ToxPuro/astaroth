#include "astaroth.h"
#include "math_utils.h"
#include "../helpers/ceil_div.h"
#include "user_builtin_non_scalar_constants.h"
#include "is_comptime_param.h"

static cudaDeviceProp
get_device_prop()
{
  cudaDeviceProp props;
  (void)cudaGetDeviceProperties(&props, 0);
  return props;
}

AcResult
acHostUpdateParams(AcMeshInfo* config_ptr)
{
    
    AcMeshInfo& config = *config_ptr; 
    //TP: utility lambdas
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(
			    std::is_same<decltype(param), int>::value     || 
			    std::is_same<decltype(param), AcReal>::value  ||
			    std::is_same<decltype(param), AcReal3>::value ||
			    std::is_same<decltype(param), int3>::value    ||
			    std::is_same<decltype(param), AcBool3>::value
			) return;

    	    //the loaders defined in DSL are supposed to be overwritable i.e. if the user loads some value say for AC_len then that value should be preferred over the default value defined in the DSL
	    if constexpr(IsCompParam(param))
	    {
		    if(config.run_consts.is_loaded[param]) return;
	    }
            else
	    {
		    if(config.params.is_loaded[param]) return;
	    }
	    return acPushToConfig(config,param,val);
    };



    [[maybe_unused]] auto ac_get_dim_products = [&](const auto& dims)
    {
		   return (AcDimProducts){
		    	dims.x*dims.y,	
		    	dims.x*dims.z,	
		    	dims.y*dims.z,	
		    	dims.x*dims.y*dims.z,	
		    };
    };

    [[maybe_unused]] auto ac_get_dim_products_inv = [&](const auto& products)
    {
		   return (AcDimProductsInv){
		    	AcReal(1.0)/products.xy,
		    	AcReal(1.0)/products.xz,
		    	AcReal(1.0)/products.yz,
		    	AcReal(1.0)/products.xyz
		    };
    };

    [[maybe_unused]] auto DCONST = [&](const auto& param)
    {
    	return config[param];
    };

    [[maybe_unused]] auto RCONST = [&](const auto& param)
    {
    	return config[param];
    };

    {
	#include "user_constants.h"
    	#include "user_config_loader.h"
    }

    //TP: for safety have to add the maximum possible tpb dims for the reduce scratchpads
    //If block_factor_z is say 2 can be that some of threads reduce something in the first iteration but not in the second
    //Thus to be on the safe side one cannot use dims.reduction_tile.z but add tpb.z as a safety factor
    const int3 safety_factor =
    {
	    0,
	    config[AC_max_tpb_for_reduce_kernels].y,
	    config[AC_max_tpb_for_reduce_kernels].z
    };
    //TP: because of this safety factor (worse for x since in general x is the biggest tpb) and because x is easy to reduce (can do warp reduce)
    //AC_thread_block_loop_factors.x is for now enforced to be 1
    ERRCHK_ALWAYS(config[AC_thread_block_loop_factors].x == 1);

    int3 tile_dims = ceil_div(config[AC_nlocal],config[AC_thread_block_loop_factors]) + safety_factor;
    tile_dims.x = 
	config[AC_nlocal].x < get_device_prop().warpSize ? config[AC_nlocal].x 
						    : ceil_div(config[AC_nlocal].x,get_device_prop().warpSize);
    push_val(AC_reduction_tile_dimensions,tile_dims);
    printf("HMM reduction tile dimensions: %d,%d,%d\n",tile_dims.x,tile_dims.y,tile_dims.z);

    

    return AC_SUCCESS;
}

AcResult 
acHostUpdateCompParams(AcMeshInfo* config)
{

	AcMeshInfo tmp = acInitInfo();
	tmp.run_consts = config->run_consts;
	auto res = acHostUpdateParams(&tmp);
	config->run_consts = tmp.run_consts;
	return res;
}


AcResult
acSetGridMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* config_ptr)
{

    AcMeshInfo& config = *config_ptr;
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,param,val);
    };
    const int3 ngrid = 
    {
	    (int)nx,
	    (int)ny,
	    (int)nz
    };
#if AC_MPI_ENABLED
    push_val(AC_ngrid,ngrid);
#else
    push_val(AC_nlocal,ngrid);
#endif
    return acHostUpdateParams(&config);
}


AcResult
acSetLocalMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* config_ptr)
{

    AcMeshInfo& config = *config_ptr;
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,param,val);
    };
    const int3 nlocal= 
    {
	    (int)nx,
	    (int)ny,
	    (int)nz
    };
    push_val(AC_nlocal,nlocal);
    return acHostUpdateParams(&config);
}
