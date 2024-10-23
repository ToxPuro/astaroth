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
    auto is_loaded = [&](auto param)
    {
	    return config.run_consts.is_loaded[param];
    };
    auto nx_param = !IsCompParam(AC_nx) ? AC_nx
	    	    : is_loaded(AC_nx) ? AC_nx : AC_nxgrid;
    auto ny_param = !IsCompParam(AC_ny) ? AC_ny
	    	    : is_loaded(AC_ny) ? AC_ny : AC_nygrid;
#if TWO_D == 0
    auto nz_param = !IsCompParam(AC_nz) ? AC_nz
	    	    : is_loaded(AC_nz) ? AC_nz : AC_nzgrid;
#endif

    ERRCHK_ALWAYS(config[nx_param]);
    ERRCHK_ALWAYS(config[ny_param]);
#if TWO_D == 0
    ERRCHK_ALWAYS(config[nz_param]);
#endif

    if(!IsCompParam(AC_nx) && config[AC_nx] <= 0) push_val(AC_nx,config[AC_nxgrid]);
    if(!IsCompParam(AC_ny) && config[AC_ny] <= 0) push_val(AC_ny,config[AC_nygrid]);
#if TWO_D == 0
    if(!IsCompParam(AC_nz) && config[AC_nz] <= 0) push_val(AC_nz,config[AC_nzgrid]);
#endif
    if(IsCompParam(AC_nx) && !is_loaded(AC_nx)) push_val(AC_nx,config[AC_nxgrid]);
    if(IsCompParam(AC_ny) && !is_loaded(AC_ny)) push_val(AC_ny,config[AC_nygrid]);
#if TWO_D == 0
    if(IsCompParam(AC_nz) && !is_loaded(AC_nz)) push_val(AC_nz,config[AC_nzgrid]);
#endif


    push_val(AC_mx,config[nx_param] + STENCIL_ORDER);
    push_val(AC_my,config[ny_param] + STENCIL_ORDER);
#if TWO_D == 0
    push_val(AC_mz,config[nz_param] + STENCIL_ORDER);
#endif

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    push_val(AC_nx_min,STENCIL_ORDER/2);
    push_val(AC_ny_min,STENCIL_ORDER/2);
#if TWO_D == 0
    push_val(AC_nz_min,STENCIL_ORDER/2);
#endif

    push_val(AC_nx_max,config[nx_param] + NGHOST_X); 
    push_val(AC_ny_max,config[ny_param] + NGHOST_Y); 
#if TWO_D == 0
    push_val(AC_nz_max,config[nz_param] + NGHOST_Z); 
#endif

    ERRCHK_ALWAYS(!IsCompParam(AC_nxgrid)  || is_loaded(AC_nxgrid));
    ERRCHK_ALWAYS(!IsCompParam(AC_nygrid)  || is_loaded(AC_nygrid));
#if TWO_D == 0
    ERRCHK_ALWAYS(!IsCompParam(AC_nzgrid)  || is_loaded(AC_nzgrid));
#endif

    push_val(AC_nxgrid_max,config[AC_nxgrid] + NGHOST_X); 
    push_val(AC_nygrid_max,config[AC_nygrid] + NGHOST_Y); 
#if TWO_D == 0
    push_val(AC_nzgrid_max,config[AC_nzgrid] + NGHOST_Z); 
#endif

    push_val(AC_mxgrid,config[AC_nxgrid] + 2*NGHOST_X); 
    push_val(AC_mygrid,config[AC_nygrid] + 2*NGHOST_Y); 
#if TWO_D == 0
    push_val(AC_mzgrid,config[AC_nzgrid] + 2*NGHOST_Z); 
#endif
    /* Additional helper params */
    // Int helpers
    push_val(AC_mxy,config[AC_mx]*config[AC_my]); 
    push_val(AC_nxy,config[AC_nx]*config[AC_ny]); 
    push_val(AC_nxygrid,config[AC_nxgrid]*config[AC_nygrid]); 

    push_val(AC_xlen,config[AC_nxgrid]*config[AC_dsx]); 
    push_val(AC_ylen,config[AC_nygrid]*config[AC_dsy]); 
#if TWO_D == 0
    push_val(AC_mxz,config[AC_mx]*config[AC_mz]); 
    push_val(AC_myz,config[AC_my]*config[AC_mz]); 
    push_val(AC_nxyz,config[AC_nxy]*config[nz_param]); 
    push_val(AC_nxyzgrid,config[AC_nxygrid]*config[AC_nzgrid]); 
    push_val(AC_zlen,config[AC_nzgrid]*config[AC_dsz]); 
#endif

    push_val(AC_inv_dsx,1.0/config[AC_dsx]);
    push_val(AC_inv_dsy,1.0/config[AC_dsy]);
#if TWO_D == 0
    push_val(AC_inv_dsz,1.0/config[AC_dsz]);
#endif

    push_val(AC_inv_dsx_2,config[AC_inv_dsx]*config[AC_inv_dsx]);
    push_val(AC_inv_dsy_2,config[AC_inv_dsy]*config[AC_inv_dsy]);
#if TWO_D == 0
    push_val(AC_inv_dsz_2,config[AC_inv_dsz]*config[AC_inv_dsz]);
#endif

    push_val(AC_inv_dsx_3,config[AC_inv_dsx_2]*config[AC_inv_dsx]);
    push_val(AC_inv_dsy_3,config[AC_inv_dsy_2]*config[AC_inv_dsy]);
#if TWO_D == 0
    push_val(AC_inv_dsz_3,config[AC_inv_dsz_2]*config[AC_inv_dsz]);
#endif

    push_val(AC_inv_dsx_4,config[AC_inv_dsx_2]*config[AC_inv_dsx_2]);
    push_val(AC_inv_dsy_4,config[AC_inv_dsy_2]*config[AC_inv_dsy_2]);
#if TWO_D == 0
    push_val(AC_inv_dsz_4,config[AC_inv_dsz_2]*config[AC_inv_dsz_2]);
#endif

    push_val(AC_inv_dsx_5,config[AC_inv_dsx_3]*config[AC_inv_dsx_2]);
    push_val(AC_inv_dsy_5,config[AC_inv_dsy_3]*config[AC_inv_dsy_2]);
#if TWO_D == 0
    push_val(AC_inv_dsz_5,config[AC_inv_dsz_3]*config[AC_inv_dsz_2]);
#endif

    push_val(AC_inv_dsx_6,config[AC_inv_dsx_3]*config[AC_inv_dsx_3]);
    push_val(AC_inv_dsy_6,config[AC_inv_dsy_3]*config[AC_inv_dsy_3]);
#if TWO_D == 0
    push_val(AC_inv_dsz_6,config[AC_inv_dsz_3]*config[AC_inv_dsz_3]);
#endif
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

	AcMeshInfo config;
	config.run_consts = *comp_config;
	auto res = acHostUpdateBuiltInParamsBase(config);
	*comp_config = config.run_consts;
	return res;
}


#if TWO_D == 0
AcResult
acSetMeshDimsBase(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo& config)
{
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,param,val);
    };
    push_val(AC_nxgrid,nx);
    push_val(AC_nygrid,ny);
    push_val(AC_nzgrid,nz);
    
    //needed to keep since before acGridInit the user can call this arbitary number of times
    push_val(AC_nx,nx);
    push_val(AC_ny,ny);
    push_val(AC_nz,nz);
    
    return acHostUpdateBuiltInParamsBase(config);
}

#else
AcResult
acSetMeshDimsBase(const size_t nx, const size_t ny,AcMeshInfo& config)
{
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,param,val);
    };
    push_val(AC_nxgrid,nx);
    push_val(AC_nygrid,ny);
    
    //needed to keep since before acGridInit the user can call this arbitary number of times
    push_val(AC_nx,nx);
    push_val(AC_ny,ny);
    
    return acHostUpdateBuiltInParamsBase(config,comp_info);
}
#endif
#if TWO_D == 0
AcResult
acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info)
{

	return acSetMeshDimsBase(nx,ny,nz,*info);
}
#else
AcResult
acSetMeshDims(const size_t nx, const size_t ny, AcMeshInfo* info)
{
	return acSetMeshDimsBase(nx,ny,*info);
}
#endif
