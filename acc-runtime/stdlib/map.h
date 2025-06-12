hostdefine AC_MAPPING_FUNCTIONS_INCLUDED (1)
ac_map_inner_product(real3 v)
{
	return AC_dot(v,v)
}
ac_map_norm(real3 v)
{
	return sqrt(AC_dot(v,v));
}
ac_map_get_value(Field src)
{
	return src[vertexIdx.x][vertexIdx.y][vertexIdx.z]
}
ac_map_get_value(Field3 src)
{
	return src[vertexIdx.x][vertexIdx.y][vertexIdx.z]
}
ac_map_get_value(Field4 src)
{
	return src[vertexIdx.x][vertexIdx.y][vertexIdx.z]
}
ac_map_exp(real3 v)
{
	return real3
		(
		 exp(v.x),
		 exp(v.y),
		 exp(v.z)
		)
}
utility Kernel AC_MAP_VTXBUF(Field src, real[] dst)
{
	const int3 launch_dims = end-start;
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = ac_map_get_value(src)
}
utility Kernel AC_MAP_VTXBUF_SQUARE(Field src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src)
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = val*val
}
utility Kernel AC_MAP_VTXBUF_EXP_SQUARE(Field src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src)
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = exp(val)*exp(val)
}
#ifdef AC_INTEGRATION_ENABLED

//TP: sum radius does not seem to be used anymore?
//run_const real AC_sum_radius
run_const real AC_window_radius

radial_window()
{
	position = real3((globalVertexIdx.x - AC_nmin.x) * AC_ds.x,
                         (globalVertexIdx.y - AC_nmin.y) * AC_ds.y,
	                 (globalVertexIdx.z - AC_nmin.z) * AC_ds.z)
	center = real3(((AC_ngrid.x-1) * AC_ds.x)/2.0,
                 ((AC_ngrid.y-1) * AC_ds.y)/2.0,
                 ((AC_ngrid.z-1) * AC_ds.z)/2.0)
	displacement = position-center;
	distance_to_center = sqrt(AC_dot(displacement,displacement))
	return distance_to_center < AC_window_radius ? 1.0 : 0.0;
}
gaussian_window()
{
	position = real3((globalVertexIdx.x - AC_nmin.x) * AC_ds.x,
                         (globalVertexIdx.y - AC_nmin.y) * AC_ds.y,
	                 (globalVertexIdx.z - AC_nmin.z) * AC_ds.z)
	center = real3(((AC_ngrid.x-1) * AC_ds.x)/2.0,
                 ((AC_ngrid.y-1) * AC_ds.y)/2.0,
                 ((AC_ngrid.z-1) * AC_ds.z)/2.0)
	displacement = position-center;
	distance_to_center = sqrt(AC_dot(displacement,displacement))
    	return exp(-(distance_to_center / AC_window_radius) * (distance_to_center / AC_window_radius));
}
utility Kernel AC_MAP_VTXBUF_RADIAL_WINDOW(Field src, real[] dst)
{
	const int3 launch_dims = end-start;
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = radial_window()*ac_map_get_value(src)
}
utility Kernel AC_MAP_VTXBUF_GAUSSIAN_WINDOW(Field src, real[] dst)
{
	const int3 launch_dims = end-start;
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = gaussian_window()*ac_map_get_value(src)
}
#endif
utility Kernel AC_MAP_VTXBUF3_NORM(Field3 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = ac_map_norm(val);
}
utility Kernel AC_MAP_VTXBUF3_SQUARE(Field3 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = ac_map_inner_product(val);
}
utility Kernel AC_MAP_VTXBUF3_EXP_SQUARE(Field3 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_exp(ac_map_get_value(src));
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = ac_map_inner_product(val);
}
#ifdef AC_INTEGRATION_ENABLED
utility Kernel AC_MAP_VTXBUF3_NORM_RADIAL_WINDOW(Field3 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = radial_window()*ac_map_norm(val);
}
utility Kernel AC_MAP_VTXBUF3_NORM_GAUSSIAN_WINDOW(Field3 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = gaussian_window()*ac_map_norm(val);
}
#endif
utility Kernel AC_MAP_VTXBUF4_ALFVEN_NORM(Field4 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	norm_val  = ac_map_norm((real3){val.x,val.y,val.z});
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = norm_val/sqrt(exp(val.w));
}
utility Kernel AC_MAP_VTXBUF4_ALFVEN_SQUARE(Field4 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	inner_product_val = ac_map_inner_product((real3){val.x,val.y,val.z});
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = inner_product_val/exp(val.w);
}
#ifdef AC_INTEGRATION_ENABLED
utility Kernel AC_MAP_VTXBUF4_ALFVEN_NORM_RADIAL_WINDOW(Field4 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	norm_val  = ac_map_norm((real3){val.x,val.y,val.z});
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = radial_window()*(norm_val/sqrt(exp(val.w)));
}
utility Kernel AC_MAP_VTXBUF4_ALFVEN_SQUARE_RADIAL_WINDOW(Field4 src, real[] dst)
{
	const int3 launch_dims = end-start;
	val = ac_map_get_value(src);
	inner_product_val = ac_map_inner_product((real3){val.x,val.y,val.z});
	dst[tid.x + tid.y*launch_dims.x + tid.z*launch_dims.x*launch_dims.y] = radial_window()*(inner_product_val/exp(val.w));
}
#endif
