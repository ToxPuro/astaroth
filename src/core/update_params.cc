#include "astaroth.h"

AcResult
acHostUpdateBuiltInParamsBase(AcMeshInfo& config, AcCompInfo& comp_info)
{
    
    //TP: utility lambdas
    auto get_val = [&](auto param)
    {
	    return acGetInfoValue(config,comp_info,param);
    };
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,comp_info,param,val);
    };
    auto is_loaded = [&](auto param)
    {
	    return get_is_loaded(param,comp_info.is_loaded);
    };
    auto nx_param = !IsCompParam(AC_nx) ? AC_nx
	    	    : is_loaded(AC_nx) ? AC_nx : AC_nxgrid;
    auto ny_param = !IsCompParam(AC_ny) ? AC_ny
	    	    : is_loaded(AC_ny) ? AC_ny : AC_nygrid;
#if TWO_D == 0
    auto nz_param = !IsCompParam(AC_nz) ? AC_nz
	    	    : is_loaded(AC_nz) ? AC_nz : AC_nzgrid;
#endif

    ERRCHK_ALWAYS(get_val(nx_param));
    ERRCHK_ALWAYS(get_val(ny_param));
#if TWO_D == 0
    ERRCHK_ALWAYS(get_val(nz_param));
#endif

    if(!IsCompParam(AC_nx) && get_val(AC_nx) <= 0) push_val(AC_nx,get_val(AC_nxgrid));
    if(!IsCompParam(AC_ny) && get_val(AC_ny) <= 0) push_val(AC_ny,get_val(AC_nygrid));
#if TWO_D == 0
    if(!IsCompParam(AC_nz) && get_val(AC_nz) <= 0) push_val(AC_nz,get_val(AC_nzgrid));
#endif
    if(IsCompParam(AC_nx) && !is_loaded(AC_nx)) push_val(AC_nx,get_val(AC_nxgrid));
    if(IsCompParam(AC_ny) && !is_loaded(AC_ny)) push_val(AC_ny,get_val(AC_nygrid));
#if TWO_D == 0
    if(IsCompParam(AC_nz) && !is_loaded(AC_nz)) push_val(AC_nz,get_val(AC_nzgrid));
#endif


    push_val(AC_mx,get_val(nx_param) + STENCIL_ORDER);
    push_val(AC_my,get_val(ny_param) + STENCIL_ORDER);
#if TWO_D == 0
    push_val(AC_mz,get_val(nz_param) + STENCIL_ORDER);
#endif

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    push_val(AC_nx_min,STENCIL_ORDER/2);
    push_val(AC_ny_min,STENCIL_ORDER/2);
#if TWO_D == 0
    push_val(AC_nz_min,STENCIL_ORDER/2);
#endif

    push_val(AC_nx_max,get_val(nx_param) + NGHOST_X); 
    push_val(AC_ny_max,get_val(ny_param) + NGHOST_Y); 
#if TWO_D == 0
    push_val(AC_nz_max,get_val(nz_param) + NGHOST_Z); 
#endif

    ERRCHK_ALWAYS(!IsCompParam(AC_nxgrid)  || is_loaded(AC_nxgrid));
    ERRCHK_ALWAYS(!IsCompParam(AC_nygrid)  || is_loaded(AC_nygrid));
#if TWO_D == 0
    ERRCHK_ALWAYS(!IsCompParam(AC_nzgrid)  || is_loaded(AC_nzgrid));
#endif

    push_val(AC_nxgrid_max,get_val(AC_nxgrid) + NGHOST_X); 
    push_val(AC_nygrid_max,get_val(AC_nygrid) + NGHOST_Y); 
#if TWO_D == 0
    push_val(AC_nzgrid_max,get_val(AC_nzgrid) + NGHOST_Z); 
#endif

    push_val(AC_mxgrid,get_val(AC_nxgrid) + 2*NGHOST_X); 
    push_val(AC_mygrid,get_val(AC_nygrid) + 2*NGHOST_Y); 
#if TWO_D == 0
    push_val(AC_mzgrid,get_val(AC_nzgrid) + 2*NGHOST_Z); 
#endif
    /* Additional helper params */
    // Int helpers
    push_val(AC_mxy,get_val(AC_mx)*get_val(AC_my)); 
    push_val(AC_nxy,get_val(AC_nx)*get_val(AC_ny)); 
    push_val(AC_nxygrid,get_val(AC_nxgrid)*get_val(AC_nygrid)); 

    push_val(AC_xlen,get_val(AC_nxgrid)*get_val(AC_dsx)); 
    push_val(AC_ylen,get_val(AC_nygrid)*get_val(AC_dsy)); 
#if TWO_D == 0
    push_val(AC_nxyz,get_val(AC_nxy)*get_val(nz_param)); 
    push_val(AC_nxyzgrid,get_val(AC_nxygrid)*get_val(AC_nzgrid)); 
    push_val(AC_zlen,get_val(AC_nzgrid)*get_val(AC_dsz)); 
#endif

    push_val(AC_inv_dsx,1.0/get_val(AC_dsx));
    push_val(AC_inv_dsy,1.0/get_val(AC_dsy));
#if TWO_D == 0
    push_val(AC_inv_dsz,1.0/get_val(AC_dsz));
#endif

    push_val(AC_inv_dsx_2,get_val(AC_inv_dsx)*get_val(AC_inv_dsx));
    push_val(AC_inv_dsy_2,get_val(AC_inv_dsy)*get_val(AC_inv_dsy));
#if TWO_D == 0
    push_val(AC_inv_dsz_2,get_val(AC_inv_dsz)*get_val(AC_inv_dsz));
#endif

    push_val(AC_inv_dsx_3,get_val(AC_inv_dsx_2)*get_val(AC_inv_dsx));
    push_val(AC_inv_dsy_3,get_val(AC_inv_dsy_2)*get_val(AC_inv_dsy));
#if TWO_D == 0
    push_val(AC_inv_dsz_3,get_val(AC_inv_dsz_2)*get_val(AC_inv_dsz));
#endif

    push_val(AC_inv_dsx_4,get_val(AC_inv_dsx_2)*get_val(AC_inv_dsx_2));
    push_val(AC_inv_dsy_4,get_val(AC_inv_dsy_2)*get_val(AC_inv_dsy_2));
#if TWO_D == 0
    push_val(AC_inv_dsz_4,get_val(AC_inv_dsz_2)*get_val(AC_inv_dsz_2));
#endif

    push_val(AC_inv_dsx_5,get_val(AC_inv_dsx_3)*get_val(AC_inv_dsx_2));
    push_val(AC_inv_dsy_5,get_val(AC_inv_dsy_3)*get_val(AC_inv_dsy_2));
#if TWO_D == 0
    push_val(AC_inv_dsz_5,get_val(AC_inv_dsz_3)*get_val(AC_inv_dsz_2));
#endif

    push_val(AC_inv_dsx_6,get_val(AC_inv_dsx_3)*get_val(AC_inv_dsx_3));
    push_val(AC_inv_dsy_6,get_val(AC_inv_dsy_3)*get_val(AC_inv_dsy_3));
#if TWO_D == 0
    push_val(AC_inv_dsz_6,get_val(AC_inv_dsz_3)*get_val(AC_inv_dsz_3));
#endif
    return AC_SUCCESS;
}

AcResult 
acHostUpdateBuiltinParams(AcMeshInfo* config)
{

	AcCompInfo comp_info = acInitCompInfo();
	return acHostUpdateBuiltInParamsBase(*config, comp_info);
}


AcResult 
acHostUpdateBuiltinCompParams(AcCompInfo* comp_config)
{

	AcMeshInfo config;
	return acHostUpdateBuiltInParamsBase(config, *comp_config);
}

AcResult 
acHostUpdateBuiltinBothParams(AcMeshInfo* config, AcCompInfo* comp_config)
{
	return acHostUpdateBuiltInParamsBase(*config, *comp_config);
}

#if TWO_D == 0
AcResult
acSetMeshDimsBase(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo& config, AcCompInfo& comp_info)
{
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,comp_info,param,val);
    };
    push_val(AC_nxgrid,nx);
    push_val(AC_nygrid,ny);
    push_val(AC_nzgrid,nz);
    
    //needed to keep since before acGridInit the user can call this arbitary number of times
    push_val(AC_nx,nx);
    push_val(AC_ny,ny);
    push_val(AC_nz,nz);
    
    return acHostUpdateBuiltInParamsBase(config,comp_info);
}

#else
AcResult
acSetMeshDimsBase(const size_t nx, const size_t ny,AcMeshInfo& config, AcCompInfo& comp_info)
{
    auto push_val = [&](auto param, auto val)
    {
	    if constexpr(std::is_same<decltype(param), int>::value || std::is_same<decltype(param), AcReal>::value);
	    else
	    	return acPushToConfig(config,comp_info,param,val);
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

	AcCompInfo comp_info = acInitCompInfo();
	return acSetMeshDimsBase(nx,ny,nz,*info,comp_info);
}
#else
AcResult
acSetMeshDims(const size_t nx, const size_t ny, AcMeshInfo* info)
{
	AcCompInfo comp_info = acInitCompInfo();
	return acSetMeshDimsBase(nx,ny,*info,comp_info);
}
#endif
#if TWO_D == 0
AcResult
acSetMeshDimsBoth(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info, AcCompInfo* comp_info)
{

	return acSetMeshDimsBase(nx,ny,nz,*info,*comp_info);
}
#else
AcResult
acSetMeshDimsBoth(const size_t nx, const size_t ny, AcMeshInfo* info, AcCompInfo* comp_info)
{
	return acSetMeshDimsBase(nx,ny,*info,*comp_info);
}
#endif
