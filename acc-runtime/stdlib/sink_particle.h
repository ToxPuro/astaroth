#ifdef AC_GENERAL_GRID_VARS_H
global output real AC_central_sink_particle_momentum
feed_sink_particle(real velocity, real density, real dt)
{
	r = AC_r[vertexIdx.x]
	momentum_flux = density*velocity*r*r*sin(AC_sin_theta[vertexIdx.y])*dt
	//Only flux at the inner radius goes to the central sink particle
	flux_going_to_sink_particle  = (globalVertexIdx.x == NGHOST)*momentum_flux
	//Hack to increase the sink momentum instead of overwriting it
	resum_particle_momentum = AC_ngrid_products_inv.xyz*AC_central_sink_particle_momentum
	reduce_sum(resum_particle_momentum + flux_going_to_sink_particle,AC_central_sink_particle_momentum)
}
#else
Sink particle requires general grid vars for sin
#endif
