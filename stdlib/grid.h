#include <cmath>
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

#if AC_GENERAL_GRID_VARS_INCLUDED
AcReal ac_get_cos_theta(const AcMeshInfo config, const int y)
{
	const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
	return cos(pos.y);
}


AcReal ac_get_sin_theta(const AcMeshInfo config, const int y)
{
	const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
	return sin(pos.y);
}

AcReal ac_get_theta(const AcMeshInfo config, const int y)
{
	const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
	return pos.y;
}



AcResult 
ac_compute_cos_m_phis(AcMeshInfo* dst)
{
#if AC_SPHERICAL_HARMONICS_INCLUDED
	AcMeshInfo& config = *dst;
	const int N = config[AC_n_spherical_harmonics];
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].z*N);
	int offset = 0;
	for(int m = 0; m < N; ++m)
	{
			for(int z = 0; z < config[AC_mlocal].z; ++z)
			{
				const AcReal3 pos = ac_grid_position((int3){0,0,z},config);
				const auto cos_m_phi = cos(m*pos.z);
				res[z + offset] = cos_m_phi;
			}
			offset += config[AC_mlocal].z;
	}
	config[AC_cos_m_phis] = res;
	return AC_SUCCESS;
#else
	return AC_FAILURE;
#endif
}

AcResult 
ac_compute_sin_m_phis(AcMeshInfo* dst)
{
#if AC_SPHERICAL_HARMONICS_INCLUDED
	AcMeshInfo& config = *dst;
	const int N = config[AC_n_spherical_harmonics];
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].z*N);
	int offset = 0;
	for(int m = 0; m < N; ++m)
	{
			for(int z = 0; z < config[AC_mlocal].z; ++z)
			{
				const AcReal3 pos = ac_grid_position((int3){0,0,z},config);
				const auto sin_m_phi = sin(m*pos.z);
				res[z + offset] = sin_m_phi;
			}
			offset += config[AC_mlocal].z;
	}
	config[AC_sin_m_phis] = res;
	return AC_SUCCESS;
#else
	return AC_FAILURE;
#endif
}

int 
factorial(const int n)
{
	if(n < 0)
	{
		fprintf(stderr,"Tried to take negative factorial!\n");
		exit(EXIT_FAILURE);
	}
	if(n == 0) return 1;
	return n*factorial(n-1);
}

AcReal
ac_get_plm(const AcReal x, const unsigned int l, const unsigned int m)
{
	return std::assoc_legendre(l,m,x);
}
AcReal
ac_get_normalized_plm(const AcReal x, const int l, const int m)
{
	const AcReal plm = ac_get_plm(x,l,m);
	const AcReal fact_1 = AcReal(factorial(l-m));
	const AcReal fact_2 = AcReal(factorial(l+m));
	AcReal res = sqrt(((2.0*l+1.0)/(4.0*AC_REAL_PI))*(fact_1/fact_2))*plm;
	if(m != 0) res *= sqrt(2.0);
	//Condon-Shortley phase term which is not included in the stdlib implementation
	if(m != 0 && m %2 == 1) res *= -1;
	return res;
}

AcResult ac_compute_normalized_plms(AcMeshInfo* dst)
{
#if AC_SPHERICAL_HARMONICS_INCLUDED
	AcMeshInfo& config = *dst;
	const int N = config[AC_n_spherical_harmonics];
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].y*N*N);
	for(int l = 0; l < N; ++l)
	{
		for(int m = 0; m <= l; ++m)
		{
			for(int y = 0; y < config[AC_mlocal].y; ++y)
			{
				const AcReal3 pos = ac_grid_position((int3){0,y,0},config);
				const auto plm = ac_get_normalized_plm(cos(pos.y),l,m);
				res[y + config[AC_mlocal].y*(l + N*m)] = plm;
			}
		}
	}
	config[AC_PLM] = res;
	return AC_SUCCESS;
#else
	return AC_FAILURE;
#endif
}
AcResult ac_compute_spherical_harmonics(AcMeshInfo* dst)
{
#if AC_SPHERICAL_HARMONICS_INCLUDED
	ac_compute_normalized_plms(dst);
	ac_compute_cos_m_phis(dst);
	ac_compute_sin_m_phis(dst);
	return AC_SUCCESS;
#else
	return AC_FAILURE;
#endif
}

AcResult
ac_compute_inv_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_nlocal].x);
	for(auto x = NGHOST; x < config[AC_nlocal].x+NGHOST; ++x)
	{
		const AcReal R = ac_grid_position((int3){x,0,0},config).x;
		if(R == 0.0)
			res[x-NGHOST] = 0.0;
		else
			res[x-NGHOST] = 1.0/R;
	}
	config[AC_inv_r] = res;

	res = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_nlocal].x);
        for(int x = NGHOST; x < config[AC_extended_nlocal].x+NGHOST; ++x)
	{
	      const AcReal R = ac_grid_position((int3){x-config[AC_left_extended_halo].x,0,0},config).x;
	      if(R == 0.0)
	      	res[x-NGHOST] = 0.0;
	      else
	      	res[x-NGHOST] = 1.0/R;
	}
	config[AC_inv_r_extended] = res;
	return AC_SUCCESS;
}

AcResult
ac_compute_r(AcMeshInfo* dst)
{
	AcMeshInfo& config = *dst;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	for(auto x = 0; x < config[AC_mlocal].x; ++x)
	{
		const AcReal R = ac_grid_position((int3){x,0,0},config).x;
		res[x] = R;
	}
	config[AC_r] = res;

	res = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_mlocal].x);
        for(int x = 0; x < config[AC_extended_mlocal].x; ++x)
	{
	      const AcReal R = ac_grid_position((int3){x-config[AC_left_extended_halo].x,0,0},config).x;
	      res[x] = R;
	}
	config[AC_r_extended] = res;
	fprintf(stderr,"loaded AC_r_extended: %d, %d!!\n",config[AC_r_extended] == NULL,config[AC_extended_mlocal].x);
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
	return ac_get_inv_sin_theta(ac_get_sin_theta(info,y));
}

AcReal
ac_map_sin_theta(const AcMeshInfo info, const int y)
{
	return ac_get_sin_theta(info,y);
}

AcReal
ac_map_theta(const AcMeshInfo info, const int y)
{
	return ac_get_theta(info,y);
}

AcReal
ac_map_phi(const AcMeshInfo info, const int z)
{
	const AcReal3 pos = ac_grid_position((int3){0,0,z},info);
	return pos.z;
}

AcReal
ac_map_cos_theta(const AcMeshInfo info, const int y)
{
	return ac_get_cos_theta(info,y);
}

AcReal
ac_map_sin_phi(const AcMeshInfo info, const int z)
{
	const AcReal3 pos = ac_grid_position((int3){0,0,z},info);
	return sin(pos.z);
}

AcReal
ac_map_cos_phi(const AcMeshInfo info, const int z)
{
	const AcReal3 pos = ac_grid_position((int3){0,0,z},info);
	return cos(pos.z);
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
	return ac_get_cot_theta(ac_get_cos_theta(info,y),ac_get_sin_theta(info,y));
}

AcResult
ac_compute_y(AcMeshInfo& config, const AcRealArrayParam param, const std::function<AcReal(const AcMeshInfo config, const int y)> map, const int start, const int end)
{
	const int m = end-start;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*m);
	for(int y = start; y < end; ++y)
	{
		res[y-start] = map(config,y);
	}
	config[param] = res;
	return AC_SUCCESS;
}

AcResult
ac_compute_z(AcMeshInfo& config, const AcRealArrayParam param, const std::function<AcReal(const AcMeshInfo config, const int z)> map, const int start, const int end)
{
	const int m = end-start;
	AcReal* res = (AcReal*)malloc(sizeof(AcReal)*m);
	for(int z = start; z < end; ++z)
	{
		res[z-start] = map(config,z);
	}
	config[param] = res;
	return AC_SUCCESS;
}


AcResult
ac_compute_inv_sin_theta(AcMeshInfo* dst)
{
	auto& config = *dst;
	ac_compute_y(*dst,AC_inv_sin_theta,ac_map_inv_sin_theta,0,config[AC_mlocal].y);
	ac_compute_y(*dst,AC_inv_sin_theta_extended,ac_map_inv_sin_theta,-config[AC_left_extended_halo].y,config[AC_mlocal].y + config[AC_right_extended_halo].y);
	return AC_SUCCESS;
}

AcResult
ac_compute_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_theta,ac_map_theta,0,(*dst)[AC_mlocal].y);
}

AcResult
ac_compute_phi(AcMeshInfo* dst)
{
	return ac_compute_z(*dst,AC_phi,ac_map_phi,0,(*dst)[AC_mlocal].z);
}

AcResult
ac_compute_sin_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_sin_theta,ac_map_sin_theta,0,(*dst)[AC_mlocal].y);
}

AcResult
ac_compute_cos_theta(AcMeshInfo* dst)
{
	return ac_compute_y(*dst,AC_cos_theta,ac_map_cos_theta,0,(*dst)[AC_mlocal].y);
}

AcResult
ac_compute_sin_phi(AcMeshInfo* dst)
{
	return ac_compute_z(*dst,AC_sin_phi,ac_map_sin_phi,0,(*dst)[AC_mlocal].z);
}

AcResult
ac_compute_cos_phi(AcMeshInfo* dst)
{
	return ac_compute_z(*dst,AC_cos_phi,ac_map_cos_phi,0,(*dst)[AC_mlocal].z);
}

AcResult
ac_compute_cot_theta(AcMeshInfo* dst)
{
	auto& config = *dst;
	ac_compute_y(*dst,AC_cot_theta,ac_map_cot_theta,0,config[AC_mlocal].y);
	ac_compute_y(*dst,AC_cot_theta_extended,ac_map_cot_theta,-config[AC_left_extended_halo].y,config[AC_mlocal].y + config[AC_right_extended_halo].y);
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

typedef struct
{
	std::vector<AcReal> value;
	std::vector<AcReal> prim;
	std::vector<AcReal> prim2;
} AcGridMappingFunction;

AcGridMappingFunction
ac_compute_power_law_mapping_x( 
				const AcReal exponent,
				const AcReal first_x,
				const AcReal last_x,
				const int ngrid,
				const int n_points,
				const int left_extension,
				const int right_extension,
				const AcReal shift
			      )
{
	  const AcReal a = (pow(last_x,exponent)-pow(first_x,exponent))/ngrid;
	  const AcReal b = 0.5*(ngrid - (pow(last_x,exponent)+pow(first_x,exponent))/a);

	  const AcReal g1lo = ac_get_power_mapping(a*(0-b),exponent).x;
	  const AcReal g1up = ac_get_power_mapping(a*(ngrid-b),exponent).x;

	  std::vector<AcReal> x_arr{};
	  std::vector<AcReal> x_prim{};
	  std::vector<AcReal> x_prim2{};
	  //For visualizing the mapping function and its derivatives
	  /**
	  const char* name = left_extension != 0 ? "x_ext.dat" : "x.dat";
	  FILE* fp = fopen(name,"w");
	  fprintf(fp,"x,dx,dx2\n");
	  **/

	  const AcReal len = last_x - first_x;
          for(int x = -left_extension; x < n_points+right_extension; ++x)
	  {
		const AcReal xi = AcReal(x-NGHOST)+shift;
		const AcReal3 g_res = ac_get_power_mapping(a*(xi-b),exponent);

		const AcReal g     = g_res.x;
		const AcReal gder1 = g_res.y;
		const AcReal gder2 = g_res.z;
		
		const auto x_local       = first_x + len*(g-g1lo)/(g1up-g1lo);
		const auto x_prim_local  = len*(gder1*a)/(g1up-g1lo);
		const auto x_prim2_local = len*(gder2*a*a)/(g1up-g1lo);
		x_arr.push_back(x_local);
		x_prim.push_back(x_prim_local);
		x_prim2.push_back(x_prim2_local);
		//fprintf(fp,"%7e,%7e,%7e\n",x_local,x_prim_local,x_prim2_local);
	  }
	  //fclose(fp);
	  return (AcGridMappingFunction)
	  {
		x_arr,x_prim,x_prim2
	  };
}


AcResult
ac_compute_power_law_mapping_x(AcMeshInfo* dst, const AcReal exponent)
{
	  AcMeshInfo& config = *dst;
	  const auto coordinate = ac_compute_power_law_mapping_x(
			  exponent,
			  ac_grid_position((int3){NGHOST,0,0},config).x,
			  ac_grid_position((int3){config[AC_nlocal].x+NGHOST,0,0},config).x,
			  config[AC_ngrid].x,
			  config[AC_mlocal].x,
			  0,
			  0,
			  0.0
			  );
	  AcReal* inv_r = (AcReal*)malloc(sizeof(AcReal)*config[AC_nlocal].x);
	  AcReal* r = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	  AcReal* inv_mapping_der = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	  AcReal* mapping_der = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	  AcReal* mapping_der2 = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
	  AcReal* mapping_tilde = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		r[x] = coordinate.value[x];
		if(NGHOST <= x && x < config[AC_nlocal].x + NGHOST)
		{
			if(r[x] == 0.0)
				inv_r[x-NGHOST] = 0.0;
			else
				inv_r[x-NGHOST] = 1.0/r[x];
		}
	  }
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		  inv_mapping_der[x] = 1.0/coordinate.prim[x];
		  mapping_der[x] =     coordinate.prim[x];
		  mapping_der2[x] =     coordinate.prim2[x];
		  mapping_tilde[x] =   -coordinate.prim2[x]/(coordinate.prim[x]*coordinate.prim[x]);
	  }
	  config[AC_inv_mapping_func_derivative_x] = inv_mapping_der;
	  config[AC_mapping_func_derivative_x] = mapping_der;
	  config[AC_mapping_func_2nd_derivative_x] = mapping_der2;
	  config[AC_mapping_func_tilde_x] = mapping_tilde;
	  config[AC_r] = r;
	  config[AC_inv_r] = inv_r;

	  const auto coordinate_shifted_by_half = ac_compute_power_law_mapping_x(
			  exponent,
			  ac_grid_position((int3){NGHOST,0,0},config).x,
			  ac_grid_position((int3){config[AC_nlocal].x+NGHOST,0,0},config).x,
			  config[AC_ngrid].x,
			  config[AC_mlocal].x,
			  0,
			  0,
			  0.5
			  );
	  AcReal* x12 = (AcReal*)malloc(sizeof(AcReal)*config[AC_mlocal].x);
          for(int x = 0; x < config[AC_mlocal].x; ++x)
	  {
		x12[x] = coordinate_shifted_by_half.value[x];
	  }
	  config[AC_x12] = x12;

	  const auto extended_coordinate = ac_compute_power_law_mapping_x(
			  exponent,
			  ac_grid_position((int3){NGHOST,0,0},config).x,
			  ac_grid_position((int3){config[AC_nlocal].x+NGHOST,0,0},config).x,
			  config[AC_ngrid].x,
			  config[AC_mlocal].x,
			  config[AC_left_extended_halo].x,
			  config[AC_right_extended_halo].x,
			  0.0
			  );

	  AcReal* inv_r_ext = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_nlocal].x);
	  AcReal* r_ext = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_mlocal].x);
	  AcReal* inv_mapping_der_ext = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_mlocal].x);
	  AcReal* mapping_der_ext = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_mlocal].x);
	  AcReal* mapping_der2_ext = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_mlocal].x);
	  AcReal* mapping_tilde_ext = (AcReal*)malloc(sizeof(AcReal)*config[AC_extended_mlocal].x);
          for(int x = 0; x < config[AC_extended_mlocal].x; ++x)
	  {
		r_ext[x] = extended_coordinate.value[x];
		if(NGHOST <= x && x < config[AC_extended_nlocal].x + NGHOST)
		{
			if(r_ext[x] == 0.0)
				inv_r_ext[x-NGHOST] = 0.0;
			else
				inv_r_ext[x-NGHOST] = 1.0/r_ext[x];
		}
	  }
          for(int x = 0; x < config[AC_extended_mlocal].x; ++x)
	  {
		  inv_mapping_der_ext[x] = 1.0/extended_coordinate.prim[x];
		  mapping_der_ext[x] =     extended_coordinate.prim[x];
		  mapping_der2_ext[x] =     extended_coordinate.prim2[x];
		  mapping_tilde_ext[x] =   -extended_coordinate.prim2[x]/(extended_coordinate.prim[x]*extended_coordinate.prim[x]);
	  }

	  config[AC_inv_mapping_func_derivative_x_extended] = inv_mapping_der_ext;
	  config[AC_mapping_func_2nd_derivative_x_extended] = mapping_der2;
	  config[AC_mapping_func_derivative_x_extended] = mapping_der_ext;
	  config[AC_mapping_func_tilde_x_extended] = mapping_tilde_ext;
	  config[AC_r_extended] = r_ext;
	  config[AC_inv_r_extended] = inv_r_ext;
	  return AC_SUCCESS;
}
#endif
