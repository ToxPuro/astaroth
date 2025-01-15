#define DER1_3 (1. / 60.)
#define DER1_2 (-3. / 20.)
#define DER1_1 (3. / 4.)
#define DER1_0 (0)

#define DER2_3 (1. / 90.)
#define DER2_2 (-3. / 20.)
#define DER2_1 (3. / 2.)
#define DER2_0 (-49. / 18.)


#define DERX_3 (2. / 720.)
#define DERX_2 (-27. / 720.)
#define DERX_1 (270. / 720.)
#define DERX_0 (0)

#define DER6UPWD_3 (  1. / 60.)
#define DER6UPWD_2 ( -6. / 60.)
#define DER6UPWD_1 ( 15. / 60.)
#define DER6UPWD_0 (-20. / 60.)

#define DER6_0 -20.0
#define DER6_1 15.0
#define DER6_2 -6.0
#define DER6_3 1.0

#define DER5_1 2.5
#define DER5_2 2.0
#define DER5_3 0.5

#define DER4_0 (56.0/6.0)
#define DER4_1 (-39.0/6.0)
#define DER4_2 (12.0/6.0)
#define DER4_3 (-1.0)

#define DER4i2j_scaling_factor 1/(6.0*180.0)
#define DER4i2j_first 56.0
#define DER4i2j_second -39.0
#define DER4i2j_third 12.0
#define DER4i2j_fourth -1.0

#define DER4i2j_0 -490.0
#define DER4i2j_1 270.0
#define DER4i2j_2 -27.0
#define DER4i2j_3 2.0


#define DERX_3 (2. / 720.)
#define DERX_2 (-27. / 720.)
#define DERX_1 (270. / 720.)
#define DERX_0 (0)

#define DER6UPWD_3 (  1. / 60.)
#define DER6UPWD_2 ( -6. / 60.)
#define DER6UPWD_1 ( 15. / 60.)
#define DER6UPWD_0 (-20. / 60.)

//Corresponds to der5 in Pencil Code
Stencil der5x {
    [0][0][-3] = -AC_inv_ds_5.x * DER5_3,
    [0][0][-2] = -AC_inv_ds_5.x * DER5_2,
    [0][0][-1] = -AC_inv_ds_5.x * DER5_1,
    [0][0][1]  = AC_inv_ds_5.x * DER5_1,
    [0][0][2]  = AC_inv_ds_5.x * DER5_2,
    [0][0][3]  = AC_inv_ds_5.x * DER5_3
}
Stencil der5y {
    [0][-3][0] = -AC_inv_ds_5.y * DER5_3,
    [0][-2][0] = -AC_inv_ds_5.y * DER5_2,
    [0][-1][0] = -AC_inv_ds_5.y * DER5_1,
    [0][1][0]  = AC_inv_ds_5.y * DER5_1,
    [0][2][0]  = AC_inv_ds_5.y * DER5_2,
    [0][3][0]  = AC_inv_ds_5.y * DER5_3
}
Stencil der5z {
    [-3][0][0] = -AC_inv_ds_5.z * DER5_3,
    [-2][0][0] = -AC_inv_ds_5.z * DER5_2,
    [-1][0][0] = -AC_inv_ds_5.z * DER5_1,
    [1][0][0]  = AC_inv_ds_5.z * DER5_1,
    [2][0][0]  = AC_inv_ds_5.z * DER5_2,
    [3][0][0]  = AC_inv_ds_5.z * DER5_3
}

der5x1y(Field f)
{
	print("NOT implemented der5x1y\n")
	return 0.0
}
der5x1z(Field f)
{
	print("NOT implemented der5x1z\n")
	return 0.0
}
der5y1x(Field f)
{
	print("NOT implemented der5y1x\n")
	return 0.0
}
der5y1z(Field f)
{
	print("NOT implemented der5y1z\n")
	return 0.0
}
der5z1x(Field f)
{
	print("NOT implemented der5z1x\n")
	return 0.0
}
der5z1y(Field f)
{
	print("NOT implemented der5z1y\n")
	return 0.0
}
//TP: corresponds to der4 in Pencil Code
Stencil der4x {
    [0][0][-3] = AC_inv_ds_4.x * DER4_3,
    [0][0][-2] = AC_inv_ds_4.x * DER4_2,
    [0][0][-1] = AC_inv_ds_4.x * DER4_1,
    [0][0][0]  = AC_inv_ds_4.x * DER4_0,
    [0][0][1]  = AC_inv_ds_4.x * DER4_1,
    [0][0][2]  = AC_inv_ds_4.x * DER4_2,
    [0][0][3]  = AC_inv_ds_4.x * DER4_3
}
Stencil der4y {
    [0][-3][0] = AC_inv_ds_4.y * DER4_3,
    [0][-2][0] = AC_inv_ds_4.y * DER4_2,
    [0][-1][0] = AC_inv_ds_4.y * DER4_1,
    [0][0][0]  = AC_inv_ds_4.y * DER4_0,
    [0][1][0]  = AC_inv_ds_4.y * DER4_1,
    [0][2][0]  = AC_inv_ds_4.y * DER4_2,
    [0][3][0]  = AC_inv_ds_4.y * DER4_3
}
Stencil der4z {
    [-3][0][0] = AC_inv_ds_4.z * DER4_3,
    [-2][0][0] = AC_inv_ds_4.z * DER4_2,
    [-1][0][0] = AC_inv_ds_4.z * DER4_1,
    [0][0][0]  = AC_inv_ds_4.z * DER4_0,
    [1][0][0]  = AC_inv_ds_4.z * DER4_1,
    [2][0][0]  = AC_inv_ds_4.z * DER4_2,
    [3][0][0]  = AC_inv_ds_4.z * DER4_3
}
//TP: corresponds to der6_main
Stencil der6x {
    [0][0][-3] = AC_inv_ds_6.x * DER6_3,
    [0][0][-2] = AC_inv_ds_6.x * DER6_2,
    [0][0][-1] = AC_inv_ds_6.x * DER6_1,
    [0][0][0]  = AC_inv_ds_6.x * DER6_0,
    [0][0][1]  = AC_inv_ds_6.x * DER6_1,
    [0][0][2]  = AC_inv_ds_6.x * DER6_2,
    [0][0][3]  = AC_inv_ds_6.x * DER6_3
}
Stencil der6y {
    [0][-3][0] = AC_inv_ds_6.y * DER6_3,
    [0][-2][0] = AC_inv_ds_6.y * DER6_2,
    [0][-1][0] = AC_inv_ds_6.y * DER6_1,
    [0][0][0]  = AC_inv_ds_6.y * DER6_0,
    [0][1][0]  = AC_inv_ds_6.y * DER6_1,
    [0][2][0]  = AC_inv_ds_6.y * DER6_2,
    [0][3][0]  = AC_inv_ds_6.y * DER6_3
}
Stencil der6z {
    [-3][0][0] = AC_inv_ds_6.z * DER6_3,
    [-2][0][0] = AC_inv_ds_6.z * DER6_2,
    [-1][0][0] = AC_inv_ds_6.z * DER6_1,
    [0][0][0]  = AC_inv_ds_6.z * DER6_0,
    [1][0][0]  = AC_inv_ds_6.z * DER6_1,
    [2][0][0]  = AC_inv_ds_6.z * DER6_2,
    [3][0][0]  = AC_inv_ds_6.z * DER6_3
}

//TP: we do it this way since these most probably called less often
//Thus these should be less performant then the normal versions
der6x_ignore_spacing(Field f)
{
	return AC_ds_6.x*der6x(f)
}
der6y_ignore_spacing(Field f)
{
	return AC_ds_6.x*der6y(f)
}
der6z_ignore_spacing(Field f)
{
	return AC_ds_6.z*der6z(f)
}

der4x2y(Field f)
{
	print("NOT implemented der4x2y\n")
	return 0.0
}
der4x2z(Field f)
{
	print("NOT implemented der4x2z\n")
	return 0.0
}
der4y2x(Field f)
{
	print("NOT implemented der4y2x\n")
	return 0.0
}
der4y2z(Field f)
{
	print("NOT implemented der4y2z\n")
	return 0.0
}
der4z2x(Field f)
{
	print("NOT implemented der4z2x\n")
	return 0.0
}
der4z2y(Field f)
{
	print("NOT implemented der4z2y\n")
	return 0.0
}
der2i2j2k(Field f)
{
	print("NOT implemented der2i2j2k\n")
	return 0.0
}
Stencil derx {
    [0][0][-3] = -AC_inv_ds.x * DER1_3,
    [0][0][-2] = -AC_inv_ds.x * DER1_2,
    [0][0][-1] = -AC_inv_ds.x * DER1_1,
    [0][0][1]  = AC_inv_ds.x * DER1_1,
    [0][0][2]  = AC_inv_ds.x * DER1_2,
    [0][0][3]  = AC_inv_ds.x * DER1_3
}

Stencil dery {
    [0][-3][0] = -AC_inv_ds.y * DER1_3,
    [0][-2][0] = -AC_inv_ds.y * DER1_2,
    [0][-1][0] = -AC_inv_ds.y * DER1_1,
    [0][1][0]  = AC_inv_ds.y * DER1_1,
    [0][2][0]  = AC_inv_ds.y * DER1_2,
    [0][3][0]  = AC_inv_ds.y * DER1_3
}

Stencil derz {
    [-3][0][0] = -AC_inv_ds.z * DER1_3,
    [-2][0][0] = -AC_inv_ds.z * DER1_2,
    [-1][0][0] = -AC_inv_ds.z * DER1_1,
    [1][0][0]  = AC_inv_ds.z * DER1_1,
    [2][0][0]  = AC_inv_ds.z * DER1_2,
    [3][0][0]  = AC_inv_ds.z * DER1_3
}

Stencil derxx {
    [0][0][-3] = AC_inv_ds_2.x * DER2_3,
    [0][0][-2] = AC_inv_ds_2.x * DER2_2,
    [0][0][-1] = AC_inv_ds_2.x * DER2_1,
    [0][0][0]  = AC_inv_ds_2.x * DER2_0,
    [0][0][1]  = AC_inv_ds_2.x * DER2_1,
    [0][0][2]  = AC_inv_ds_2.x * DER2_2,
    [0][0][3]  = AC_inv_ds_2.x * DER2_3
}

Stencil deryy {
    [0][-3][0] = AC_inv_ds_2.y * DER2_3,
    [0][-2][0] = AC_inv_ds_2.y * DER2_2,
    [0][-1][0] = AC_inv_ds_2.y * DER2_1,
    [0][0][0]  = AC_inv_ds_2.y * DER2_0,
    [0][1][0]  = AC_inv_ds_2.y * DER2_1,
    [0][2][0]  = AC_inv_ds_2.y * DER2_2,
    [0][3][0]  = AC_inv_ds_2.y * DER2_3
}

Stencil derzz {
    [-3][0][0] = AC_inv_ds_2.z * DER2_3,
    [-2][0][0] = AC_inv_ds_2.z * DER2_2,
    [-1][0][0] = AC_inv_ds_2.z * DER2_1,
    [0][0][0]  = AC_inv_ds_2.z * DER2_0,
    [1][0][0]  = AC_inv_ds_2.z * DER2_1,
    [2][0][0]  = AC_inv_ds_2.z * DER2_2,
    [3][0][0]  = AC_inv_ds_2.z * DER2_3
}

Stencil derxy {
    [0][-3][-3] = AC_inv_ds.x * AC_inv_ds.y * DERX_3,
    [0][-2][-2] = AC_inv_ds.x * AC_inv_ds.y * DERX_2,
    [0][-1][-1] = AC_inv_ds.x * AC_inv_ds.y * DERX_1,
    [0][0][0]  = AC_inv_ds.x * AC_inv_ds.y * DERX_0,
    [0][1][1]  = AC_inv_ds.x * AC_inv_ds.y * DERX_1,
    [0][2][2]  = AC_inv_ds.x * AC_inv_ds.y * DERX_2,
    [0][3][3]  = AC_inv_ds.x * AC_inv_ds.y * DERX_3,
    [0][-3][3] = -AC_inv_ds.x * AC_inv_ds.y * DERX_3,
    [0][-2][2] = -AC_inv_ds.x * AC_inv_ds.y * DERX_2,
    [0][-1][1] = -AC_inv_ds.x * AC_inv_ds.y * DERX_1,
    [0][1][-1] = -AC_inv_ds.x * AC_inv_ds.y * DERX_1,
    [0][2][-2] = -AC_inv_ds.x * AC_inv_ds.y * DERX_2,
    [0][3][-3] = -AC_inv_ds.x * AC_inv_ds.y * DERX_3
}
#define deryx derxy

Stencil derxz {
    [-3][0][-3] = AC_inv_ds.x * AC_inv_ds.z * DERX_3,
    [-2][0][-2] = AC_inv_ds.x * AC_inv_ds.z * DERX_2,
    [-1][0][-1] = AC_inv_ds.x * AC_inv_ds.z * DERX_1,
    [0][0][0]  = AC_inv_ds.x * AC_inv_ds.z * DERX_0,
    [1][0][1]  = AC_inv_ds.x * AC_inv_ds.z * DERX_1,
    [2][0][2]  = AC_inv_ds.x * AC_inv_ds.z * DERX_2,
    [3][0][3]  = AC_inv_ds.x * AC_inv_ds.z * DERX_3,
    [-3][0][3] = -AC_inv_ds.x * AC_inv_ds.z * DERX_3,
    [-2][0][2] = -AC_inv_ds.x * AC_inv_ds.z * DERX_2,
    [-1][0][1] = -AC_inv_ds.x * AC_inv_ds.z * DERX_1,
    [1][0][-1] = -AC_inv_ds.x * AC_inv_ds.z * DERX_1,
    [2][0][-2] = -AC_inv_ds.x * AC_inv_ds.z * DERX_2,
    [3][0][-3] = -AC_inv_ds.x * AC_inv_ds.z * DERX_3
}

#define derzx derxz

Stencil deryz {
    [-3][-3][0] = AC_inv_ds.y * AC_inv_ds.z * DERX_3,
    [-2][-2][0] = AC_inv_ds.y * AC_inv_ds.z * DERX_2,
    [-1][-1][0] = AC_inv_ds.y * AC_inv_ds.z * DERX_1,
    [0][0][0]  = AC_inv_ds.y * AC_inv_ds.z * DERX_0,
    [1][1][0]  = AC_inv_ds.y * AC_inv_ds.z * DERX_1,
    [2][2][0]  = AC_inv_ds.y * AC_inv_ds.z * DERX_2,
    [3][3][0]  = AC_inv_ds.y * AC_inv_ds.z * DERX_3,
    [-3][3][0] = -AC_inv_ds.y * AC_inv_ds.z * DERX_3,
    [-2][2][0] = -AC_inv_ds.y * AC_inv_ds.z * DERX_2,
    [-1][1][0] = -AC_inv_ds.y * AC_inv_ds.z * DERX_1,
    [1][-1][0] = -AC_inv_ds.y * AC_inv_ds.z * DERX_1,
    [2][-2][0] = -AC_inv_ds.y * AC_inv_ds.z * DERX_2,
    [3][-3][0] = -AC_inv_ds.y * AC_inv_ds.z * DERX_3
}

#define derzy deryz

Stencil der6x_upwd {
    [0][0][-3] =  AC_inv_ds.x * DER6UPWD_3,
    [0][0][-2] =  AC_inv_ds.x * DER6UPWD_2,
    [0][0][-1] =  AC_inv_ds.x * DER6UPWD_1,
    [0][0][0]  =  AC_inv_ds.x * DER6UPWD_0,
    [0][0][1]  =  AC_inv_ds.x * DER6UPWD_1,
    [0][0][2]  =  AC_inv_ds.x * DER6UPWD_2,
    [0][0][3]  =  AC_inv_ds.x * DER6UPWD_3
}

Stencil der6y_upwd {
    [0][-3][0] =  AC_inv_ds.y * DER6UPWD_3,
    [0][-2][0] =  AC_inv_ds.y * DER6UPWD_2,
    [0][-1][0] =  AC_inv_ds.y * DER6UPWD_1,
    [0][0][0]  =  AC_inv_ds.y * DER6UPWD_0,
    [0][1][0]  =  AC_inv_ds.y * DER6UPWD_1,
    [0][2][0]  =  AC_inv_ds.y * DER6UPWD_2,
    [0][3][0]  =  AC_inv_ds.y * DER6UPWD_3
}

Stencil der6z_upwd {
    [-3][0][0] =  AC_inv_ds.z * DER6UPWD_3,
    [-2][0][0] =  AC_inv_ds.z * DER6UPWD_2,
    [-1][0][0] =  AC_inv_ds.z * DER6UPWD_1,
    [0][0][0]  =  AC_inv_ds.z * DER6UPWD_0,
    [1][0][0]  =  AC_inv_ds.z * DER6UPWD_1,
    [2][0][0]  =  AC_inv_ds.z * DER6UPWD_2,
    [3][0][0]  =  AC_inv_ds.z * DER6UPWD_3
}


//derx(Field f)
//{
//	res =  DER1_3*-AC_inv_ds.x*f[vertexIdx.x-3][vertexIdx.y][vertexIdx.z];
//	res += DER1_3*+AC_inv_ds.x*f[vertexIdx.x+3][vertexIdx.y][vertexIdx.z];
//	res += DER1_2*-AC_inv_ds.x*f[vertexIdx.x-2][vertexIdx.y][vertexIdx.z];
//	res += DER1_2*+AC_inv_ds.x*f[vertexIdx.x+2][vertexIdx.y][vertexIdx.z];
//	res += DER1_1*-AC_inv_ds.x*f[vertexIdx.x-1][vertexIdx.y][vertexIdx.z];
//	res += DER1_1*+AC_inv_ds.x*f[vertexIdx.x+1][vertexIdx.y][vertexIdx.z];
//	return res;
//}
//
//derxx(Field f)
//{
//	res =  DER2_0*+AC_inv_ds_2.x*f[vertexIdx.x][vertexIdx.y][vertexIdx.z];
//	res += DER2_3*+AC_inv_ds_2.x*f[vertexIdx.x-3][vertexIdx.y][vertexIdx.z];
//	res += DER2_3*+AC_inv_ds_2.x*f[vertexIdx.x+3][vertexIdx.y][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.x*f[vertexIdx.x-2][vertexIdx.y][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.x*f[vertexIdx.x+2][vertexIdx.y][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.x*f[vertexIdx.x-1][vertexIdx.y][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.x*f[vertexIdx.x+1][vertexIdx.y][vertexIdx.z];
//	return res;
//}
//dery(Field f)
//{
//	res =  DER1_3*-AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y-3][vertexIdx.z];
//	res += DER1_3*+AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y+3][vertexIdx.z];
//	res += DER1_2*-AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y-2][vertexIdx.z];
//	res += DER1_2*+AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y+2][vertexIdx.z];
//	res += DER1_1*-AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y-1][vertexIdx.z];
//	res += DER1_1*+AC_inv_ds.y*f[vertexIdx.x][vertexIdx.y+1][vertexIdx.z];
//	return res;
//}
//
//deryy(Field f)
//{
//	res =  DER2_0*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z];
//	res +=  DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y-3][vertexIdx.z];
//	res += DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y+3][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y-2][vertexIdx.z];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y+2][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y-1][vertexIdx.z];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y+1][vertexIdx.z];
//	return res;
//}
//derz(Field f)
//{
//	res =  DER1_3*-AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-3];
//	res += DER1_3*+AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+3];
//	res += DER1_2*-AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-2];
//	res += DER1_2*+AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+2];
//	res += DER1_1*-AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-1];
//	res += DER1_1*+AC_inv_ds.z*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+1];
//	return res;
//}
//derzz(Field f)
//{
//	res =  DER2_0*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z];
//	res +=  DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-3];
//	res += DER2_3*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+3];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-2];
//	res += DER2_2*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+2];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z-1];
//	res += DER2_1*+AC_inv_ds_2.y*f[vertexIdx.x][vertexIdx.y][vertexIdx.z+1];
//	return res;
//}



