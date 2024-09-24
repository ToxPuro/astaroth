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
    auto nz_param = !IsCompParam(AC_nz) ? AC_nz
	    	    : is_loaded(AC_nz) ? AC_nz : AC_nzgrid;

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
    /* Additional helper params */
    // Int helpers
    push_val(AC_mxy,get_val(AC_mx)*get_val(AC_my)); 
    push_val(AC_nxy,get_val(AC_nx)*get_val(AC_ny)); 

    push_val(AC_xlen,get_val(AC_nxgrid)*get_val(AC_dsx)); 
    push_val(AC_ylen,get_val(AC_nygrid)*get_val(AC_dsy)); 
#if TWO_D == 0
    push_val(AC_nxyz,get_val(AC_nxy)*get_val(nz_param)); 
    push_val(AC_zlen,get_val(AC_nzgrid)*get_val(AC_dsz)); 
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
acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info)
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
acSetMeshDimsBoth(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info, AcCompInfo* comp_info)
{
	return acSetMeshDimsBase(nx,ny,*info,*comp_info);
}
#endif
