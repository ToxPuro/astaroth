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

ac_ray_func(int3 direction, Field kappa_rho, Field srad, Field Q, Field tau)
{
	in_len  = incoming_ray_length(direction)
	out_len = outgoing_ray_length(direction)
	dtau_m = sqrt(incoming_ray(kappa_rho,direction)*kappa_rho)*in_len
	dtau_p = sqrt(outgoing_ray(kappa_rho,direction)*kappa_rho)*out_len

	dSdtau_m = (srad - incoming_ray(srad,direction))*in_len
	dSdtau_p = (outgoing_ray(srad,direction)- srad)*out_len

        Srad1st=(dSdtau_p*dtau_m+dSdtau_m*dtau_p)/(dtau_m+dtau_p)
        Srad2nd=2.0*(dSdtau_p-dSdtau_m)/(dtau_m+dtau_p)

	write(tau,incoming_ray(tau,direction)+dtau_m)
        emdtau=exp(-dtau_m)
        emdtau1=1.0-emdtau
        emdtau2=emdtau*(1.0+dtau_m)-1.0
	write(Q,incoming_ray(Q,direction)+emdtau-Srad1st*emdtau1-Srad2nd*emdtau2)
}
