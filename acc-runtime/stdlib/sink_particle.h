#ifdef AC_GENERAL_GRID_VARS_H
global output real AC_central_sink_particle_rhs
real AC_central_sink_particle_mass = 0.0
calculate_sink_particle_rhs(real radial_velocity, real density)
{
	r = AC_r[vertexIdx.x]
	momentum_flux = density*radial_velocity*r*r*AC_sin_theta[vertexIdx.y]
	//Only flux at the inner radius goes to the central sink particle
	flux_going_to_sink_particle  = (radial_velocity < 0.0 && globalVertexIdx.x == NGHOST)*momentum_flux
	reduce_sum(flux_going_to_sink_particle,AC_central_sink_particle_rhs)
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
