Raytrace (+0,+0,-1) backwards_ray
Raytrace (+0,+0,+1) forwards_ray
Raytrace (-1,+0,+0) left_ray
Raytrace (+1,+0,+0) right_ray
Raytrace (+0,-1,+0) down_ray
Raytrace (+0,+1,+0) up_ray


//Positive Z direction
Raytrace (+1,+1,+1) ppp_ray
Raytrace (+0,+1,+1) zpp_ray
Raytrace (-1,+1,+1) mpp_ray

Raytrace (+1,+0,+1) pzp_ray
Raytrace (+0,+0,+1) zzp_ray
Raytrace (-1,+0,+1) mzp_ray

Raytrace (+1,-1,+1) pmp_ray
Raytrace (+0,-1,+1) zmp_ray
Raytrace (-1,-1,+1) mmp_ray

//Negative Z direction
Raytrace (+1,+1,-1) ppm_ray
Raytrace (+0,+1,-1) zpm_ray
Raytrace (-1,+1,-1) mpm_ray

Raytrace (+1,+0,-1) pzm_ray
Raytrace (+0,+0,-1) zzm_ray
Raytrace (-1,+0,-1) mzm_ray

Raytrace (+1,-1,-1) pmm_ray
Raytrace (+0,-1,-1) zmm_ray
Raytrace (-1,-1,-1) mmm_ray

//Positive Y direction
Raytrace (+1,+1,+0) ppz_ray
Raytrace (+0,+1,+0) zpz_ray
Raytrace (-1,+1,+0) mpz_ray

//Negative Y direction
Raytrace (+1,-1,+0) pmz_ray
Raytrace (+0,-1,+0) zmz_ray
Raytrace (-1,-1,+0) mmz_ray


//Positive X direction
Raytrace (+1,+0,+0) pzz_ray
//Negative X direction
Raytrace (-1,+0,+0) mzz_ray


outgoing_ray(Field f, int3 direction)
{
        if(direction == (int3){+1,+1,+1}) return outgoing_ppp_ray(f)
        if(direction == (int3){-1,+1,+1}) return outgoing_mpp_ray(f)
        if(direction == (int3){+0,+1,+1}) return outgoing_zpp_ray(f)

        if(direction == (int3){+1,+0,+1}) return outgoing_pzp_ray(f)
        if(direction == (int3){-1,+0,+1}) return outgoing_mzp_ray(f)
        if(direction == (int3){+0,+0,+1}) return outgoing_zzp_ray(f)

        if(direction == (int3){+1,-1,+1}) return outgoing_pmp_ray(f)
        if(direction == (int3){-1,-1,+1}) return outgoing_mmp_ray(f)
        if(direction == (int3){+0,-1,+1}) return outgoing_zmp_ray(f)

        if(direction == (int3){+1,+1,-1}) return outgoing_ppm_ray(f)
        if(direction == (int3){-1,+1,-1}) return outgoing_mpm_ray(f)
        if(direction == (int3){+0,+1,-1}) return outgoing_zpm_ray(f)

        if(direction == (int3){+1,+0,-1}) return outgoing_pzm_ray(f)
        if(direction == (int3){-1,+0,-1}) return outgoing_mzm_ray(f)
        if(direction == (int3){+0,+0,-1}) return outgoing_zzm_ray(f)

        if(direction == (int3){+1,-1,-1}) return outgoing_pmm_ray(f)
        if(direction == (int3){-1,-1,-1}) return outgoing_mmm_ray(f)
        if(direction == (int3){+0,-1,-1}) return outgoing_zmm_ray(f)

        if(direction == (int3){+1,+1,+0}) return outgoing_ppz_ray(f)
        if(direction == (int3){-1,+1,+0}) return outgoing_mpz_ray(f)
        if(direction == (int3){+0,+1,+0}) return outgoing_zpz_ray(f)

        if(direction == (int3){+1,-1,+0}) return outgoing_pmz_ray(f)
        if(direction == (int3){-1,-1,+0}) return outgoing_mmz_ray(f)
        if(direction == (int3){+0,-1,+0}) return outgoing_zmz_ray(f)

        if(direction == (int3){+1,+0,+0}) return outgoing_pzz_ray(f)
        if(direction == (int3){-1,+0,+0}) return outgoing_mzz_ray(f)
        return 0.0;
}

incoming_ray(Field f, int3 direction)
{
        if(direction == (int3){+1,+1,+1}) return incoming_ppp_ray(f)
        if(direction == (int3){-1,+1,+1}) return incoming_mpp_ray(f)
        if(direction == (int3){+0,+1,+1}) return incoming_zpp_ray(f)

        if(direction == (int3){+1,+0,+1}) return incoming_pzp_ray(f)
        if(direction == (int3){-1,+0,+1}) return incoming_mzp_ray(f)
        if(direction == (int3){+0,+0,+1}) return incoming_zzp_ray(f)

        if(direction == (int3){+1,-1,+1}) return incoming_pmp_ray(f)
        if(direction == (int3){-1,-1,+1}) return incoming_mmp_ray(f)
        if(direction == (int3){+0,-1,+1}) return incoming_zmp_ray(f)

        if(direction == (int3){+1,+1,-1}) return incoming_ppm_ray(f)
        if(direction == (int3){-1,+1,-1}) return incoming_mpm_ray(f)
        if(direction == (int3){+0,+1,-1}) return incoming_zpm_ray(f)

        if(direction == (int3){+1,+0,-1}) return incoming_pzm_ray(f)
        if(direction == (int3){-1,+0,-1}) return incoming_mzm_ray(f)
        if(direction == (int3){+0,+0,-1}) return incoming_zzm_ray(f)

        if(direction == (int3){+1,-1,-1}) return incoming_pmm_ray(f)
        if(direction == (int3){-1,-1,-1}) return incoming_mmm_ray(f)
        if(direction == (int3){+0,-1,-1}) return incoming_zmm_ray(f)

        if(direction == (int3){+1,+1,+0}) return incoming_ppz_ray(f)
        if(direction == (int3){-1,+1,+0}) return incoming_mpz_ray(f)
        if(direction == (int3){+0,+1,+0}) return incoming_zpz_ray(f)

        if(direction == (int3){+1,-1,+0}) return incoming_pmz_ray(f)
        if(direction == (int3){-1,-1,+0}) return incoming_mmz_ray(f)
        if(direction == (int3){+0,-1,+0}) return incoming_zmz_ray(f)

        if(direction == (int3){+1,+0,+0}) return incoming_pzz_ray(f)
        if(direction == (int3){-1,+0,+0}) return incoming_mzz_ray(f)
        return 0.0;
}


incoming_ray_length(int3 direction)
{
	return sqrt(
		 	  abs(direction.x)*AC_ds_2.x
			+ abs(direction.y)*AC_ds_2.y
			+ abs(direction.z)*AC_ds_2.z
		   )
}
outgoing_ray_length(int3 direction)
{
	return sqrt(
		 	  abs(direction.x)*AC_ds_2.x
			+ abs(direction.y)*AC_ds_2.y
			+ abs(direction.z)*AC_ds_2.z
		   )
}

run_const real dtau_thresh_min = 1.6*pow(AC_REAL_EPSILON,0.25)
run_const real dtau_thresh_max = -log(AC_REAL_MIN)

ac_ray_func(int3 direction, Field kappa_rho, Field srad, Field Q, Field tau, real in_len, real out_len)
{
	dtau_m = sqrt(incoming_ray(kappa_rho,direction)*kappa_rho)*in_len
	dtau_p = sqrt(outgoing_ray(kappa_rho,direction)*kappa_rho)*out_len

	dtau_m = max(dtau_m,AC_REAL_EPSILON*5)
	dtau_p = max(dtau_p,AC_REAL_EPSILON*5)

	dSdtau_m = (srad - incoming_ray(srad,direction))/dtau_m
	dSdtau_p = (outgoing_ray(srad,direction)- srad)/dtau_p

        Srad1st=(dSdtau_p*dtau_m+dSdtau_m*dtau_p)/(dtau_m+dtau_p)
        Srad2nd=2.0*(dSdtau_p-dSdtau_m)/(dtau_m+dtau_p)
	
	write(tau,incoming_ray(tau,direction)+dtau_m)
        real emdtau	
        real emdtau1
        real emdtau2
	if(dtau_m > dtau_thresh_max)
	{
        	emdtau=0.0
        	emdtau1=1.0
        	emdtau2=-1.0
	}
	else if(dtau_m < dtau_thresh_min)
	{
		emdtau1=dtau_m*(1.0-0.5*dtau_m*(1.0-dtau_m/3.0))
		emdtau=1-emdtau1	
		emdtau2=-(dtau_m*dtau_m)*(0.5-dtau_m/3.0)
	}
	else
	{
        	emdtau=exp(-dtau_m)
        	emdtau1=1.0-emdtau
        	emdtau2=emdtau*(1.0+dtau_m)-1.0
	}
	const Q_res = incoming_ray(Q,direction)*emdtau-Srad1st*emdtau1-Srad2nd*emdtau2
	write(Q,Q_res)
}

ac_ray_func(int3 direction, Field kappa_rho, Field srad, Field Q, Field tau)
{
	ac_ray_func(direction,kappa_rho,srad,Q,tau,incoming_ray_length(direction),outgoing_ray_length(direction))
}

get_incoming_boundary_ray_point(int3 direction)
{
        const int steps_x = (direction.x > 0) ? vertexIdx.x-(NGHOST-1) :
                      (direction.x < 0) ? AC_nlocal_max.x - vertexIdx.x
                                        : INT_MAX
        const int steps_y = (direction.y > 0) ? vertexIdx.y-(NGHOST-1) :
                      (direction.y < 0) ? AC_nlocal_max.y - vertexIdx.y
                                        : INT_MAX
        const int steps_z = (direction.z > 0) ? vertexIdx.z-(NGHOST-1) :
                      (direction.z < 0) ? AC_nlocal_max.z - vertexIdx.z
                                        : INT_MAX
        const int steps = min(steps_x,min(steps_y,steps_z))
        return (int3)
                {
                        vertexIdx.x-steps*direction.x,
                        vertexIdx.y-steps*direction.y,
                        vertexIdx.z-steps*direction.z
                }
}

get_incoming_boundary_ray(Field f, int3 direction)
{
        const int3 boundary_vertex = get_incoming_boundary_ray_point(direction)
        return f[boundary_vertex.x][boundary_vertex.y][boundary_vertex.z]
}

extrinsic_ray_update(int3 direction, Field Q, Field TAU)
{
        incoming_Q = get_incoming_boundary_ray(Q,direction)
        return Q + incoming_Q*exp(-TAU)
}
