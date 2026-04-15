#ifdef AC_GENERAL_GRID_VARS_H
global output real AC_central_sink_particle_rhs
real AC_central_sink_particle_mass = 0.0
calculate_sink_particle_rhs(real radial_velocity, real density)
{
	r = AC_r[vertexIdx.x]
	velocity_going_to_center = max(0.,-radial_velocity)
	dtheta = AC_ds.y
	dphi = AC_ds.z
	dA = r*r*AC_sin_theta[vertexIdx.y]*dtheta*dphi
	mass_flux = density*velocity_going_to_center*dA
	mass_going_to_sink_particle  = (globalVertexIdx.x == NGHOST)*mass_flux
	reduce_sum(mass_going_to_sink_particle,AC_central_sink_particle_rhs)
}

force_from_sink_particle(real G)
{
	r = AC_r[vertexIdx.x]
	return -real3(
			(G*AC_central_sink_particle_mass)/(r*r),
			0.0,
			0.0
		    )
}
#else
Sink particle requires general grid vars for sin
#endif
