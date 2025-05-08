
int3
ac_global_vertex_idx(const int3 localVertexIdx, const AcMeshInfo config)
{
	return 
		{
			localVertexIdx.x + config[AC_multigpu_offset].x,
			localVertexIdx.y + config[AC_multigpu_offset].y,
			localVertexIdx.z + config[AC_multigpu_offset].z
		};
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

std::vector<AcReal> ac_get_r_vec(const AcMeshInfo config)
{
	std::vector<AcReal> res{};
	{
		for(int x = 0; x < config[AC_mlocal].x; ++x)
		{
			const AcReal3 pos = ac_grid_position((int3){x,0,0},config);
			res.push_back(pos.x);
		}
	}
	return res;
}

std::vector<AcReal> ac_get_y_vec(const AcMeshInfo config, const std::function<AcReal(const AcReal pos)> map)
{
	std::vector<AcReal> res{};
	{
		for(int y = 0; y < config[AC_mlocal].y; ++y)
		{
			const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
			res.push_back(map(pos.y));
		}
	}
	return res;
}

std::vector<AcReal> ac_get_cos_theta_vec(const AcMeshInfo config)
{
	return ac_get_y_vec(config,[](const AcReal pos){return cos(pos);});
}

std::vector<AcReal> ac_get_sin_theta_vec(const AcMeshInfo config)
{
	return ac_get_y_vec(config,[](const AcReal pos){return sin(pos);});
}

std::vector<AcReal> ac_get_theta_vec(const AcMeshInfo config)
{
	return ac_get_y_vec(config,[](const AcReal pos){return pos;});
}


std::vector<AcReal> ac_get_z_vec(const AcMeshInfo config, const std::function<AcReal(const AcReal pos)> map)
{
	std::vector<AcReal> res{};
	{
		for(int z = 0; z < config[AC_mlocal].z; ++z)
		{
			const AcReal3 pos = ac_grid_position((int3){0,0,z},config);
			res.push_back(map(pos.z));
		}
	}
	return res;
}
std::vector<AcReal> ac_get_phi_vec(const AcMeshInfo config)
{
	return ac_get_z_vec(config,[](const AcReal pos){return pos;});
}

std::vector<AcReal> ac_get_sin_phi_vec(const AcMeshInfo config)
{
	return ac_get_z_vec(config,[](const AcReal pos){return sin(pos);});
}

std::vector<AcReal> ac_get_cos_phi_vec(const AcMeshInfo config)
{
	return ac_get_z_vec(config,[](const AcReal pos){return cos(pos);});
}

AcResult
ac_compute_inv_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_nlocal].x);
	const std::vector R = ac_get_r_vec(config);
	for(auto x = NGHOST; x < config[AC_nlocal].x+NGHOST; ++x)
	{
		if(R[x] == 0.0)
			res[x-NGHOST] = 0.0;
		else
			res[x-NGHOST] = 1.0/R[x];
	}
	config[AC_inv_r] = res;
	return AC_SUCCESS;
}

AcResult
ac_compute_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	const std::vector R = ac_get_r_vec(config);
	for(auto x = 0; x < config[AC_mlocal].x; ++x)
	{
		res[x] = R[x];
	}
	config[AC_r] = res;
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
ac_get_inv_sin_theta(const AcReal sin_theta)
{
		constexpr AcReal sin_theta_min = 1e-5;
		if(sin_theta > sin_theta_min)
			return 1.0/sin_theta;
		else
			return 0.0;
}
AcReal
ac_map_inv_sin_theta(const AcMeshInfo info, const int y)
{
	return ac_get_inv_sin_theta(ac_get_sin_theta_vec(info)[y]);
}

AcReal
ac_map_sin_theta(const AcMeshInfo info, const int y)
{
	return ac_get_sin_theta_vec(info)[y];
}

AcReal
ac_map_theta(const AcMeshInfo info, const int y)
{
	return ac_get_theta_vec(info)[y];
}

AcReal
ac_map_phi(const AcMeshInfo info, const int z)
{
	return ac_get_phi_vec(info)[z];
}

AcReal
ac_map_cos_theta(const AcMeshInfo info, const int y)
{
	return ac_get_cos_theta_vec(info)[y];
}

AcReal
ac_map_sin_phi(const AcMeshInfo info, const int z)
{
	return ac_get_sin_phi_vec(info)[z];
}

AcReal
ac_map_cos_phi(const AcMeshInfo info, const int z)
{
	return ac_get_cos_phi_vec(info)[z];
}

AcReal
ac_get_cos_theta(const int3 localVertexIdx, const AcMeshInfo config)
{
		const AcReal3 pos = ac_grid_position((int3){0,localVertexIdx.y,0},config);
		return cos(pos.y);
}

AcReal
ac_get_cot_theta(const AcReal cos_theta, const AcReal sin_theta)
{
	return cos_theta*ac_get_inv_sin_theta(sin_theta);
}

AcReal
ac_map_cot_theta(const AcMeshInfo info, const int y)
{
	return ac_get_cot_theta(ac_get_cos_theta_vec(info)[y],ac_get_sin_theta_vec(info)[y]);
}

AcResult
ac_compute_y(AcMeshInfo& config, const AcRealArrayParam param, const std::function<AcReal(const AcMeshInfo config, const int y)> map)
{
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].y);
	for(int y = 0; y < config[AC_mlocal].y; ++y)
	{
		res[y] = map(config,y);
	}
	config[param] = res;
	return AC_SUCCESS;
}

AcResult
ac_compute_z(AcMeshInfo& config, const AcRealArrayParam param, const std::function<AcReal(const AcMeshInfo config, const int z)> map)
{
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].z);
	for(int z = 0; z < config[AC_mlocal].z; ++z)
	{
		res[z] = map(config,z);
	}
	config[param] = res;
	return AC_SUCCESS;
}


AcResult
ac_compute_inv_sin_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_inv_sin_theta,ac_map_inv_sin_theta);
}

AcResult
ac_compute_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_theta,ac_map_theta);
}

AcResult
ac_compute_phi(AcMeshInfo* dst)
{
	return ac_compute_z(*dst,AC_phi,ac_map_phi);
}

AcResult
ac_compute_sin_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_sin_theta,ac_map_sin_theta);
}

AcResult
ac_compute_cos_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_cos_theta,ac_map_cos_theta);
}

AcResult
ac_compute_sin_phi(AcMeshInfo* dst)
{
	return ac_compute_z(*dst,AC_sin_phi,ac_map_sin_phi);
}

AcResult
ac_compute_cos_phi(AcMeshInfo* dst)
{
	return ac_compute_z(*dst,AC_cos_phi,ac_map_cos_phi);
}

AcResult
ac_compute_cot_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_cot_theta,ac_map_cot_theta);
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

