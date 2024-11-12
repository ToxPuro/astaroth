#include "astaroth.h"

AcResult
acHostUpdateBuiltInParamsBase(AcMeshInfo& config)
{
    
    //TP: utility lambdas
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,param,val);
    };

#if TWO_D == 0
    push_val(AC_dsmin,std::min(std::min(config[AC_ds].x,config[AC_ds].y),config[AC_ds].z));
#else
    push_val(AC_ds,(AcReal3){config[AC_ds].x, config[AC_ds].y, 1.0});
    push_val(AC_ngrid, (int3){config[AC_ngrid].x, config[AC_ngrid].y, 1});
    push_val(AC_nlocal, (int3){config[AC_nlocal].x, config[AC_nlocal].y, 1});
#endif
    push_val(AC_mlocal,config[AC_nlocal] + 2*(int3){NGHOST_X,NGHOST_Y,NGHOST_Z});
    push_val(AC_mgrid ,config[AC_ngrid]  + 2*(int3){NGHOST_X,NGHOST_Y,NGHOST_Z});

    push_val(AC_nlocal_max,config[AC_nlocal] + (int3){NGHOST_X,NGHOST_Y,NGHOST_Z});
    push_val(AC_ngrid_max,config[AC_ngrid]   + (int3){NGHOST_X,NGHOST_Y,NGHOST_Z});
    const int3 nmin = (int3)
    {
	    NGHOST_X,
	    NGHOST_Y,
	    NGHOST_Z
    };
    auto get_products = [&](const auto& param)
    {
		   return (AcDimProducts){
		    	config[param].x*config[param].y,	
		    	config[param].x*config[param].z,	
		    	config[param].y*config[param].z,	
		    	config[param].x*config[param].y*config[param].z,	
		    };
    };
    push_val(AC_nmin,nmin);

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

    return AC_SUCCESS;
}

AcResult 
acHostUpdateBuiltinParams(AcMeshInfo* config)
{

	return acHostUpdateBuiltInParamsBase(*config);
}


AcResult 
acHostUpdateBuiltinCompParams(AcCompInfo* comp_config)
{

	AcMeshInfo config = acInitInfo();
	config.run_consts = *comp_config;
	auto res = acHostUpdateBuiltInParamsBase(config);
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
	    nx,
	    ny,
	    nz
    };
    push_val(AC_ngrid,ngrid);
    push_val(AC_nlocal,ngrid);
    
    return acHostUpdateBuiltInParamsBase(config);
}
