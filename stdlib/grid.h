
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
	return 
		(AcReal3)
		{
			(globalVertexIdx.x -config[AC_nmin].x)*config[AC_ds].x + config[AC_first_gridpoint].x,
			(globalVertexIdx.y -config[AC_nmin].y)*config[AC_ds].y + config[AC_first_gridpoint].y,
			(globalVertexIdx.z -config[AC_nmin].z)*config[AC_ds].z + config[AC_first_gridpoint].z
		};
}

AcResult
ac_compute_inv_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_nlocal].x);
	for(auto x = NGHOST; x < config[AC_nlocal].x+NGHOST; ++x)
	{
		const AcReal3 pos = ac_grid_position((int3){x,0,0},config);
	        if(pos.x == 0)	
			res[x-NGHOST] = 0;
		else
			res[x-NGHOST] = 1.0/pos.x;
	}
	config[AC_inv_r] = res;
	return AC_SUCCESS;
}

//TP: duplicate of ac_compute_inv_r since for now cyl_r is independent of inv_r
//open to changing it in the future but works for now
AcResult
ac_compute_inv_cyl_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_nlocal].x);
	for(auto x = NGHOST; x < config[AC_nlocal].x+NGHOST; ++x)
	{
		const AcReal3 pos = ac_grid_position((int3){x,0,0},config);
	        if(pos.x == 0)	
			res[x-NGHOST] = 0;
		else
			res[x-NGHOST] = 1.0/pos.x;
	}
	config[AC_inv_cyl_r] = res;
	return AC_SUCCESS;
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
		const AcReal3 pos = ac_grid_position((int3){0,localVertexIdx.y,0},config);
		const AcReal sin_theta= sin(pos.y);
		constexpr AcReal sin_theta_min = 1e-5;
		if(sin_theta > sin_theta_min)
			return 1.0/sin_theta;
		else
			return 0.0;
}

AcReal
ac_get_cos_theta(const int3 localVertexIdx, const AcMeshInfo config)
{
		const AcReal3 pos = ac_grid_position((int3){0,localVertexIdx.y,0},config);
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
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].y);
	for(int y = 0; y < config[AC_mlocal].y; ++y)
	{
		res[y] = ac_get_inv_sin_theta((int3){0,y,0}, config);
	}
	config[AC_inv_sin_theta] = res;
	return AC_SUCCESS;
}

AcResult
ac_compute_cot_theta(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].y);
	for(int y = 0; y < config[AC_mlocal].y; ++y)
	{
		res[y] = ac_get_cot_theta((int3){0,y,0}, config);
	}
	config[AC_cot_theta] = res;
	return AC_SUCCESS;
}
AcReal3
ac_get_power_mapping(const AcReal xi_scaled,const AcReal exponent)
{
	return 
	{
		  pow(xi_scaled,1.0/exponent),
		  1.0/exponent * pow(xi_scaled,(1.0/exponent)-1.0),
		  1.0/exponent * ((1.0/exponent)-1.0)*pow(xi_scaled,(1/exponent)-2.0)
	};
}
AcResult
ac_compute_power_law_mapping_x(AcMeshInfo* dst, const AcReal exponent)
{
	  AcMeshInfo& config = *dst;
          const AcReal last_x = ac_grid_position((int3){config[AC_nlocal].x+NGHOST,0,0},config).x;
          const AcReal first_x = ac_grid_position((int3){0,0,0},config).x;
	  const AcReal a = (pow(last_x,exponent)-pow(first_x,exponent))/config[AC_ngrid].x;
	  const AcReal b = 0.5*(config[AC_ngrid].x - (pow(last_x,exponent)+pow(first_x,exponent))/a);

	  std::vector<int> xi{};
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		  xi.push_back(x-NGHOST);
	  }
	  std::vector<AcReal> g{};
	  std::vector<AcReal> gder1{};
	  std::vector<AcReal> gder2{};
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		  const AcReal3 g_res = ac_get_power_mapping(a*(xi[x]-b),exponent);
		  g.push_back(g_res.x);
		  gder1.push_back(g_res.y);
		  gder2.push_back(g_res.z);

	  }
	  const AcReal g1lo = ac_get_power_mapping(a*(0-b),exponent).x;
	  const AcReal g1up = ac_get_power_mapping(a*(config[AC_ngrid].x-b),exponent).x;

	  std::vector<AcReal> x_arr{};
	  std::vector<AcReal> x_prim{};
	  std::vector<AcReal> x_prim2{};
	  FILE* fp = fopen("x.dat","w");
	  fprintf(fp,"x,dx,dx2\n");
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		x_arr.push_back(config[AC_first_gridpoint].x + config[AC_len].x*(g[x]-g1lo)/(g1up-g1lo));
		x_prim.push_back(config[AC_len].x*(gder1[x]*a)/(g1up-g1lo));
		x_prim2.push_back(config[AC_len].x*(gder2[x]*a*a)/(g1up-g1lo));
		if(x >= NGHOST && x < config[AC_mlocal].x-NGHOST)
		{
			fprintf(fp,"%7e,%7e,%7e\n",x_arr[x],x_prim[x],x_prim2[x]);
		}
	  }
	  fclose(fp);
	  AcReal* inv_mapping_der = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	  AcReal* mapping_tilde = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		  inv_mapping_der[x] = 1.0/x_prim[x];
		  mapping_tilde[x] = -x_prim2[x]/(x_prim[x]*x_prim[x]);
	  }
	  config[AC_inv_mapping_func_derivative_x] = inv_mapping_der;
	  config[AC_mapping_func_tilde_x] = mapping_tilde;
	  return AC_SUCCESS;
}

