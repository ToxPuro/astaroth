#ifdef AC_GENERAL_GRID_VARS_H
global output real AC_central_sink_particle_momentum
feed_sink_particle(real radial_velocity, real density, real dt)
{
	r = AC_r[vertexIdx.x]
	momentum_flux = density*radial_velocity*r*r*sin(AC_sin_theta[vertexIdx.y])*dt
	//Only flux at the inner radius goes to the central sink particle
	flux_going_to_sink_particle  = (radial_velocity < 0.0 && globalVertexIdx.x == NGHOST)*momentum_flux
	//Hack to increase the sink momentum instead of overwriting it
	resum_particle_momentum = (vertexIdx == (int3){NGHOST,NGHOST,NGHOST})*AC_central_sink_particle_momentum
	reduce_sum(resum_particle_momentum + flux_going_to_sink_particle,AC_central_sink_particle_momentum)
}

force_from_sink_particle(real G)
{
	r = AC_r[vertexIdx.x]*AC_r[vertexIdx.x]
	return -real3(
			(G*AC_central_sink_particle_momentum)/(r*r),
			0.0,
			0.0
		    )
}
initialize_sink_particle(real initial_value)
{
	value_to_sum = (vertexIdx == (int3){NGHOST,NGHOST,NGHOST})*initial_value
	reduce_sum(value_to_sum,AC_central_sink_particle_momentum)
}
#else
Sink particle requires general grid vars for sin
#endif
