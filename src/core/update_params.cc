#include "astaroth.h"
#include "math_utils.h"
#include "../helpers/ceil_div.h"
#include "user_builtin_non_scalar_constants.h"

static cudaDeviceProp
get_device_prop()
{
  cudaDeviceProp props;
  (void)cudaGetDeviceProperties(&props, 0);
  return props;
}

AcResult
acHostUpdateBuiltinParams(AcMeshInfo* config_ptr)
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
			);
	    else
	    	return acPushToConfig(config,param,val);
    };
//TP: for now TWO_D means XY setup
#if TWO_D
    push_val(AC_dimension_inactive,
		    (AcBool3)
		    {
		    	false,
			false,
			true	
		    }
	    );
#endif

    push_val(AC_nmin,
    	(int3){
    	        config[AC_dimension_inactive].x ? 0 : NGHOST,
    	        config[AC_dimension_inactive].y ? 0 : NGHOST,
    	        config[AC_dimension_inactive].z ? 0 : NGHOST
    	}
    );
#if AC_LAGRANGIAN_GRID
    push_val(AC_lagrangian_grid,true);
#endif

    if(!config[AC_dimension_inactive].z)
    	push_val(AC_dsmin,std::min(std::min(config[AC_ds].x,config[AC_ds].y),config[AC_ds].z));
    else
    {
    	push_val(AC_ds,(AcReal3){config[AC_ds].x, config[AC_ds].y, 1.0});
    	push_val(AC_ngrid, (int3){config[AC_ngrid].x, config[AC_ngrid].y, 1});
    	push_val(AC_nlocal, (int3){config[AC_nlocal].x, config[AC_nlocal].y, 1});
    }

    push_val(AC_mlocal,config[AC_nlocal] + 2*config[AC_nmin]);
    push_val(AC_mgrid ,config[AC_ngrid]  + 2*config[AC_nmin]);

    push_val(AC_nlocal_max,config[AC_nlocal] + config[AC_nmin]);
    push_val(AC_ngrid_max,config[AC_ngrid]   + config[AC_nmin]);
    auto get_products = [&](const auto& param)
    {
		   return (AcDimProducts){
		    	config[param].x*config[param].y,	
		    	config[param].x*config[param].z,	
		    	config[param].y*config[param].z,	
		    	config[param].x*config[param].y*config[param].z,	
		    };
    };

    push_val(AC_nlocal_products,get_products(AC_nlocal));
    push_val(AC_mlocal_products,get_products(AC_mlocal));
    push_val(AC_ngrid_products,get_products(AC_ngrid));
    push_val(AC_mgrid_products,get_products(AC_mgrid));
    push_val(AC_len,
    	(AcReal3)
    	{
    		config[AC_ngrid].x*config[AC_ds].x,
    		config[AC_ngrid].y*config[AC_ds].y,
    		config[AC_ngrid].z*config[AC_ds].z
    	}
    );


    const AcReal3 unit = {1.0,1.0,1.0};
    push_val(AC_inv_ds,unit/config[AC_ds]);
    push_val(AC_inv_ds_2,config[AC_inv_ds]*config[AC_inv_ds]);
    push_val(AC_inv_ds_3,config[AC_inv_ds_2]*config[AC_inv_ds]);
    push_val(AC_inv_ds_4,config[AC_inv_ds_2]*config[AC_inv_ds_2]);
    push_val(AC_inv_ds_5,config[AC_inv_ds_3]*config[AC_inv_ds_2]);
    push_val(AC_inv_ds_6,config[AC_inv_ds_3]*config[AC_inv_ds_3]);

    push_val(AC_ds_2,config[AC_ds]*config[AC_ds]);
    push_val(AC_ds_3,config[AC_ds_2]*config[AC_ds]);
    push_val(AC_ds_4,config[AC_ds_2]*config[AC_ds_2]);
    push_val(AC_ds_5,config[AC_ds_3]*config[AC_ds_2]);
    push_val(AC_ds_6,config[AC_ds_3]*config[AC_ds_3]);

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
    

    return AC_SUCCESS;
}

AcResult 
acHostUpdateBuiltinCompParams(AcCompInfo* comp_config)
{

	AcMeshInfo config = acInitInfo();
	config.run_consts = *comp_config;
	auto res = acHostUpdateBuiltinParams(&config);
	*comp_config = config.run_consts;
	return res;
}


AcResult
acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* config_ptr)
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
    push_val(AC_ngrid,ngrid);
    push_val(AC_nlocal,ngrid);
    return acHostUpdateBuiltinParams(&config);
}
