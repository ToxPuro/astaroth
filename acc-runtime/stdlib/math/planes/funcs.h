TetrahedronVolume(real3 a, real3 b, real3 c)
{
        return dot( cross(a,b), c);
}

Get4DNormal(real4 a, real4 b, real4 c)
{
        return
        (real4){
                a.y*(b.z*c.w - b.w*c.z)  + a.z*(-b.y*c.w + b.w*c.y)     + a.w*(b.y*c.z + -b.z*c.y),
                a.x*(-b.z*c.w + b.w*c.z) + a.z*(b.x*c.w  - b.w*c.x)     + a.w*(-b.x*c.z + b.z*c.x),
                a.x*(b.y*c.w - b.w*c.y)  + a.y*(-b.x*c.w + b.w*c.x)     + a.w*(b.x*c.y + -b.y*c.x),
                a.x*(b.z*c.y - b.y*c.z)  + a.y*(b.x*c.z - b.z*c.x)      + a.z*(b.y*c.x -b.x*c.y)
	 };
}
PlaneCoefficients3D(real4 c, real4 b, real4 a)
{
        const real4 normal = Get4DNormal(a,b,c);
        const real3 a_coords = {a.x, a.y, a.z};
        const real3 b_coords = {b.x, b.y, b.z};
        const real3 c_coords = {c.x, c.y, c.z};
        const real inv = dot(a_coords, cross(c_coords,b_coords));
        return real3(
                -normal.x/inv,
                -normal.y/inv,
                -normal.z/inv
	)
}
PlaneCoefficients3D_without_inv(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp1 = b.w*c.z-b.z*c.w;
        const real tmp2 = b.y*c.w - b.w*c.y;
        const real tmp3 = b.w*c.x - b.x*c.w;

        const real x = a.y*(tmp1)  + a.z*(tmp2)  + a.w*(precomputed.x);
        const real y = a.x*(-tmp1) + a.z*(tmp3)  + a.w*(precomputed.y);
        const real z = a.x*(-tmp2) + a.y*(-tmp3) + a.w*(precomputed.z);
        return (real3){x, y, z};
}
Plane3D_x_coefficient(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp1 = b.w*c.z-b.z*c.w;
        const real tmp2 = b.y*c.w - b.w*c.y;
        return a.y*(tmp1)  + a.z*(tmp2)  + a.w*(precomputed.x);
}
Plane3D_y_coefficient(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp1 = b.w*c.z-b.z*c.w;
        const real tmp3 = b.w*c.x - b.x*c.w;
        return  a.x*(-tmp1) + a.z*(tmp3)  + a.w*(precomputed.y);
}
Plane3D_z_coefficient(real4 a, real4 b, real4 c, real3 precomputed)
{
        const real tmp2 = b.y*c.w - b.w*c.y;
        const real tmp3 = b.w*c.x - b.x*c.w;
        return a.x*(-tmp2) + a.y*(-tmp3) + a.w*(precomputed.z);
}

