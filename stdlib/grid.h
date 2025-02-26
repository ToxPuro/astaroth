
int3
ac_global_vertex_idx(const int3 localVertexIdx, const AcMeshInfo config)
{
	return localVertexIdx + config[AC_multigpu_offset];
}
AcReal3
ac_grid_position(const int3 localVertexIdx, const AcMeshInfo config)
{
	//TP: implicitly assumes [0,AC_len] domain
	//    add a new case if you have different grid
	const int3 globalVertexIdx = ac_global_vertex_idx(localVertexIdx,config);
	return (globalVertexIdx - config[AC_nlocal])*config[AC_ds];
}

AcResult
ac_compute_inv_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = malloc(sizeof(AcReal)config[AC_nlocal].x);
	const auto dims = acGetMeshDims(*config);
	for(const auto x = dims.n0.x; x < dims.n1.x; ++x)
	{
		const AcReal3 pos = ac_grid_position((int3){x,0,0},config);
	        if(pos.x == 0)	
			res[x-dims.n0.x] = 0;
		else
			res[x-dims.n0.x] = 1.0/pos.x;
	}
	config[AC_inv_r] = res;
}

//TP: duplicate of ac_compute_inv_r since for now cyl_r is independent of inv_r
//open to changing it in the future but works for now
AcResult
ac_compute_inv_cyl_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = malloc(sizeof(AcReal)config[AC_nlocal].x);
	const auto dims = acGetMeshDims(*config);
	for(const auto x = dims.n0.x; x < dims.n1.x; ++x)
	{
		const AcReal3 pos = ac_grid_position((int3){x,0,0},config);
	        if(pos.x == 0)	
			res[x-dims.n0.x] = 0;
		else
			res[x-dims.n0.x] = 1.0/pos.x;
	}
	config[AC_inv_cyl_r] = res;
}

//  TP: copied from Pencil Code
//  Calculate 1/sin(theta). To avoid the axis we check that sin_theta
//  is always larger than a minmal value, sin_theta_min. The problem occurs
//  on theta=pi, because the theta range is normally only specified
//  with no more than 6 digits, e.g. theta = 0., 3.14159.
//
AcReal
ac_get_inv_sin_theta(const int3 localVertexIdx, const AcMeshInfo config)
{
		constexpr AcReal sin_theta_min = 1e-5;
		const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
		const AcReal sin_theta= sin(pos.y);
		if(sin_theta > sin_theta_min)
			return 1.0/sin_theta;
		else
			return 0.0;
}

AcReal
ac_get_cos_theta(const int3 localVertexIdx, const AcMeshInfo config)
{
		constexpr AcReal sin_theta_min = 1e-5;
		const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
		return cos(pos.y);
}

AcReal
ac_get_cot_theta(const int3 localVertexIdx, const AcMeshInfo config)
{
	return ac_get_cos_theta(localVertexIdx,config)*ac_get_inv_sin_theta(localVertexIdx,config);
}

AcResult
ac_compute_inv_sin_theta(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = malloc(sizeof(AcReal)config[AC_mlocal].y);
	const auto dims = acGetMeshDims(*config);
	for(const auto y = 0; y < dims.m1.y; ++y)
	{
		res[y] = ac_get_inv_sin_theta((int3){0,y,0}, config);
	}
	config[AC_inv_sin_theta] = res;
}

AcResult
ac_compute_cot_theta(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = malloc(sizeof(AcReal)config[AC_mlocal].y);
	const auto dims = acGetMeshDims(*config);
	for(const auto y = 0; y < dims.m1.y; ++y)
	{
		res[y] = ac_get_cot_theta((int3){0,y,0}, config);
	}
	config[AC_cot_theta] = res;
}

