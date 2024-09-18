//TP: note all the input coordinate and values are in a reference frame where the centre grid point is subtracted from the value
//TP: the motivation is that this takes less compute
TetrahedronVolume(real3 a, real3 b, real3 c)
{
        return dot( cross(a,b), c);
}

GetNormal(real4 a, real4 b, real4 c)
{
        return
        (real4){
                a.y*(b.z*c.w - b.w*c.z)  + a.z*(-b.y*c.w + b.w*c.y)     + a.w*(b.y*c.z + -b.z*c.y),
                a.x*(-b.z*c.w + b.w*c.z) + a.z*(b.x*c.w  - b.w*c.x)     + a.w*(-b.x*c.z + b.z*c.x),
                a.x*(b.y*c.w - b.w*c.y)  + a.y*(-b.x*c.w + b.w*c.x)     + a.w*(b.x*c.y + -b.y*c.x),
                a.x*(b.z*c.y - b.y*c.z)  + a.y*(b.x*c.z - b.z*c.x)      + a.z*(b.y*c.x -b.x*c.y)
	 };
}
PlaneCoefficients(real4 c, real4 b, real4 a)
{
        const real4 normal = GetNormal(a,b,c);
        //const real3 a_coords = {a.x, a.y, a.z};
	const real3 a_coords = {a.x, a.y, a.z};
        const real3 b_coords = {b.x, b.y, b.z};
        const real3 c_coords = {c.x, c.y, c.z};

        //const real inv = AC_dot(a_coords, cross(c_coords,b_coords));
        const real inv = dot(a_coords, cross(c_coords,b_coords));
        return real3(
                -normal.x/inv,
                -normal.y/inv,
                -normal.z/inv
	)
}
PlaneCoefficients_without_inv(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp1 = b.w*c.z-b.z*c.w;
        const real tmp2 = b.y*c.w - b.w*c.y;
        const real tmp3 = b.w*c.x - b.x*c.w;

        const real x = a.y*(tmp1)  + a.z*(tmp2)  + a.w*(precomputed.x);
        const real y = a.x*(-tmp1) + a.z*(tmp3)  + a.w*(precomputed.y);
        const real z = a.x*(-tmp2) + a.y*(-tmp3) + a.w*(precomputed.z);
        return (real3){x, y, z};
}

Plane_x_coefficient(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp1 = b.w*c.z-b.z*c.w;
        const real tmp2 = b.y*c.w - b.w*c.y;
        return a.y*(tmp1)  + a.z*(tmp2)  + a.w*(precomputed.x);
}
Plane_y_coefficient(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp1 = b.w*c.z-b.z*c.w;
        const real tmp3 = b.w*c.x - b.x*c.w;
        return  a.x*(-tmp1) + a.z*(tmp3)  + a.w*(precomputed.y);
}
Plane_z_coefficient(real4 a, real4 b, real4 c, real3 precomputed)
{
        real tmp2 = b.y*c.w - b.w*c.y;
        real tmp3 = b.w*c.x - b.x*c.w;
        return a.x*(-tmp2) + a.y*(-tmp3) + a.w*(precomputed.z);
}

PlaneCoeffients_without_inv(real3 a, real3 b)
{
	return (real2){
		a.z*b.y  - b.z*a.y,
		-(a.z*b.x) + b.z*a.x
	}
}


Plane_x_coefficient(real3 a, real3 b)
{
	return a.z*b.y  - b.z*a.y
}
Plane_y_coefficient(real3 a, real3 b)
{
	return -a.z*b.x + b.z*a.x
}

Stencil diff_up
{
	[1][0]  =  1,
	[0][0]  = -1
}

Stencil diff_down
{
	[-1][0] =  1,
	[0][0]  = -1
}

Stencil diff_right
{
	[0][1]  =  1,
	[0][0]  = -1
}

Stencil diff_left
{
	[0][-1] =  1,
	[0][0]  = -1
}

get_first_order_derivatives(Field coordsx, Field coordsy, Field F)
{
	const real v0x     = diff_left(coordsx)
	const real v1x     = diff_down(coordsx)
	const real v2x     = diff_right(coordsx)
	const real v3x     = diff_up(coordsx)

	const real v0y     = diff_left(coordsy)
	const real v1y     = diff_down(coordsy)
	const real v2y     = diff_right(coordsy)
	const real v3y     = diff_up(coordsy)



	const v0f     =  diff_left(F)
	const v1f     =  diff_down(F)
	const v2f     =  diff_right(F)
	const v3f     =  diff_up(F)

	const p0_coords = (real2){v0x,v0y}
        const p1_coords = (real2){v1x,v1y}
        const p2_coords = (real2){v2x,v2y}
        const p3_coords = (real2){v3x,v3y}

	const real A0  = (p0_coords.x*p1_coords.y - p0_coords.y*p1_coords.x);
	const real A1  = (p1_coords.x*p2_coords.y - p1_coords.y*p2_coords.x);
	const real A2  = (p2_coords.x*p3_coords.y - p2_coords.y*p3_coords.x);
	const real A3  = (p3_coords.x*p0_coords.y - p3_coords.y*p0_coords.x);

        const real totalAreaInv = 1.0/(A0 + A1 + A2 + A3);

	real3 p0 = {v0x,v0y,v0f};
        real3 p1 = {v1x,v1y,v1f};
        real3 p2 = {v2x,v2y,v2f};
        real3 p3 = {v3x,v3y,v3f};


	partials0 = PlaneCoeffients_without_inv(p0,p1);
	partials1 = PlaneCoeffients_without_inv(p1,p2);
	partials2 = PlaneCoeffients_without_inv(p2,p3);
	partials3 = PlaneCoeffients_without_inv(p3,p0);

	const real p0x_local = (partials0.x + partials1.x + partials2.x + partials3.x)*totalAreaInv;
        const real p0y_local = (partials0.y + partials1.y + partials2.y + partials3.y)*totalAreaInv;
	return (real2){p0x_local,p0y_local}
}
