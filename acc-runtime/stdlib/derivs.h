#define AC_inv_dsx_2 AC_inv_dsx*AC_inv_dsx
#define AC_inv_dsy_2 AC_inv_dsy*AC_inv_dsy
#define AC_inv_dsz_2 AC_inv_dsz*AC_inv_dsz

//for Pencil Code
#define AC_inv_dsx_6 AC_inv_dsx*AC_inv_dsx*AC_inv_dsx*AC_inv_dsx*AC_inv_dsx*AC_inv_dsx
#define AC_inv_dsy_6 AC_inv_dsy*AC_inv_dsy*AC_inv_dsy*AC_inv_dsy*AC_inv_dsy*AC_inv_dsy
#define AC_inv_dsz_6 AC_inv_dsz*AC_inv_dsz*AC_inv_dsz*AC_inv_dsz*AC_inv_dsz*AC_inv_dsz

#define AC_dsx_6 AC_dsx*AC_dsx*AC_dsx*AC_dsx*AC_dsx*AC_dsx
#define AC_dsy_6 AC_dsy*AC_dsy*AC_dsy*AC_dsy*AC_dsy*AC_dsy
#define AC_dsz_6 AC_dsz*AC_dsz*AC_dsz*AC_dsz*AC_dsz*AC_dsz

#define AC_inv_dsx_5 AC_inv_dsx*AC_inv_dsx*AC_inv_dsx*AC_inv_dsx*AC_inv_dsx
#define AC_inv_dsy_5 AC_inv_dsy*AC_inv_dsy*AC_inv_dsy*AC_inv_dsy*AC_inv_dsy
#define AC_inv_dsz_5 AC_inv_dsz*AC_inv_dsz*AC_inv_dsz*AC_inv_dsz*AC_inv_dsz

#define AC_inv_dsx_4 AC_inv_dsx*AC_inv_dsx*AC_inv_dsx*AC_inv_dsx
#define AC_inv_dsy_4 AC_inv_dsy*AC_inv_dsy*AC_inv_dsy*AC_inv_dsy
#define AC_inv_dsz_4 AC_inv_dsz*AC_inv_dsz*AC_inv_dsz*AC_inv_dsz

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
#if TWO_D == 0
Stencil der5x {
    [0][0][-3] = -AC_inv_dsx_5 * DER5_3,
    [0][0][-2] = -AC_inv_dsx_5 * DER5_2,
    [0][0][-1] = -AC_inv_dsx_5 * DER5_1,
    [0][0][1]  = AC_inv_dsx_5 * DER5_1,
    [0][0][2]  = AC_inv_dsx_5 * DER5_2,
    [0][0][3]  = AC_inv_dsx_5 * DER5_3
}
Stencil der5y {
    [0][-3][0] = -AC_inv_dsy_5 * DER5_3,
    [0][-2][0] = -AC_inv_dsy_5 * DER5_2,
    [0][-1][0] = -AC_inv_dsy_5 * DER5_1,
    [0][1][0]  = AC_inv_dsy_5 * DER5_1,
    [0][2][0]  = AC_inv_dsy_5 * DER5_2,
    [0][3][0]  = AC_inv_dsy_5 * DER5_3
}
Stencil der5z {
    [-3][0][0] = -AC_inv_dsz_5 * DER5_3,
    [-2][0][0] = -AC_inv_dsz_5 * DER5_2,
    [-1][0][0] = -AC_inv_dsz_5 * DER5_1,
    [1][0][0]  = AC_inv_dsz_5 * DER5_1,
    [2][0][0]  = AC_inv_dsz_5 * DER5_2,
    [3][0][0]  = AC_inv_dsz_5 * DER5_3
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
    [0][0][-3] = AC_inv_dsx_4 * DER4_3,
    [0][0][-2] = AC_inv_dsx_4 * DER4_2,
    [0][0][-1] = AC_inv_dsx_4 * DER4_1,
    [0][0][0]  = AC_inv_dsx_4 * DER4_0,
    [0][0][1]  = AC_inv_dsx_4 * DER4_1,
    [0][0][2]  = AC_inv_dsx_4 * DER4_2,
    [0][0][3]  = AC_inv_dsx_4 * DER4_3
}
Stencil der4y {
    [0][-3][0] = AC_inv_dsy_4 * DER4_3,
    [0][-2][0] = AC_inv_dsy_4 * DER4_2,
    [0][-1][0] = AC_inv_dsy_4 * DER4_1,
    [0][0][0]  = AC_inv_dsy_4 * DER4_0,
    [0][1][0]  = AC_inv_dsy_4 * DER4_1,
    [0][2][0]  = AC_inv_dsy_4 * DER4_2,
    [0][3][0]  = AC_inv_dsy_4 * DER4_3
}
Stencil der4z {
    [-3][0][0] = AC_inv_dsz_4 * DER4_3,
    [-2][0][0] = AC_inv_dsz_4 * DER4_2,
    [-1][0][0] = AC_inv_dsz_4 * DER4_1,
    [0][0][0]  = AC_inv_dsz_4 * DER4_0,
    [1][0][0]  = AC_inv_dsz_4 * DER4_1,
    [2][0][0]  = AC_inv_dsz_4 * DER4_2,
    [3][0][0]  = AC_inv_dsz_4 * DER4_3
}
//TP: corresponds to der6_main
Stencil der6x {
    [0][0][-3] = AC_inv_dsx_6 * DER6_3,
    [0][0][-2] = AC_inv_dsx_6 * DER6_2,
    [0][0][-1] = AC_inv_dsx_6 * DER6_1,
    [0][0][0]  = AC_inv_dsx_6 * DER6_0,
    [0][0][1]  = AC_inv_dsx_6 * DER6_1,
    [0][0][2]  = AC_inv_dsx_6 * DER6_2,
    [0][0][3]  = AC_inv_dsx_6 * DER6_3
}
Stencil der6y {
    [0][-3][0] = AC_inv_dsy_6 * DER6_3,
    [0][-2][0] = AC_inv_dsy_6 * DER6_2,
    [0][-1][0] = AC_inv_dsy_6 * DER6_1,
    [0][0][0]  = AC_inv_dsy_6 * DER6_0,
    [0][1][0]  = AC_inv_dsy_6 * DER6_1,
    [0][2][0]  = AC_inv_dsy_6 * DER6_2,
    [0][3][0]  = AC_inv_dsy_6 * DER6_3
}
Stencil der6z {
    [-3][0][0] = AC_inv_dsz_6 * DER6_3,
    [-2][0][0] = AC_inv_dsz_6 * DER6_2,
    [-1][0][0] = AC_inv_dsz_6 * DER6_1,
    [0][0][0]  = AC_inv_dsz_6 * DER6_0,
    [1][0][0]  = AC_inv_dsz_6 * DER6_1,
    [2][0][0]  = AC_inv_dsz_6 * DER6_2,
    [3][0][0]  = AC_inv_dsz_6 * DER6_3
}

//TP: we do it this way since these most probably called less often
//Thus these should be less performant then the normal versions
der6x_ignore_spacing(Field f)
{
	return AC_dsx_6*der6x(f)
}
der6y_ignore_spacing(Field f)
{
	return AC_dsx_6*der6y(f)
}
der6z_ignore_spacing(Field f)
{
	return AC_dsz_6*der6z(f)
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
    [0][0][-3] = -AC_inv_dsx * DER1_3,
    [0][0][-2] = -AC_inv_dsx * DER1_2,
    [0][0][-1] = -AC_inv_dsx * DER1_1,
    [0][0][1]  = AC_inv_dsx * DER1_1,
    [0][0][2]  = AC_inv_dsx * DER1_2,
    [0][0][3]  = AC_inv_dsx * DER1_3
}

Stencil dery {
    [0][-3][0] = -AC_inv_dsy * DER1_3,
    [0][-2][0] = -AC_inv_dsy * DER1_2,
    [0][-1][0] = -AC_inv_dsy * DER1_1,
    [0][1][0]  = AC_inv_dsy * DER1_1,
    [0][2][0]  = AC_inv_dsy * DER1_2,
    [0][3][0]  = AC_inv_dsy * DER1_3
}

Stencil derz {
    [-3][0][0] = -AC_inv_dsz * DER1_3,
    [-2][0][0] = -AC_inv_dsz * DER1_2,
    [-1][0][0] = -AC_inv_dsz * DER1_1,
    [1][0][0]  = AC_inv_dsz * DER1_1,
    [2][0][0]  = AC_inv_dsz * DER1_2,
    [3][0][0]  = AC_inv_dsz * DER1_3
}

Stencil derxx {
    [0][0][-3] = AC_inv_dsx_2 * DER2_3,
    [0][0][-2] = AC_inv_dsx_2 * DER2_2,
    [0][0][-1] = AC_inv_dsx_2 * DER2_1,
    [0][0][0]  = AC_inv_dsx_2 * DER2_0,
    [0][0][1]  = AC_inv_dsx_2 * DER2_1,
    [0][0][2]  = AC_inv_dsx_2 * DER2_2,
    [0][0][3]  = AC_inv_dsx_2 * DER2_3
}

Stencil deryy {
    [0][-3][0] = AC_inv_dsy_2 * DER2_3,
    [0][-2][0] = AC_inv_dsy_2 * DER2_2,
    [0][-1][0] = AC_inv_dsy_2 * DER2_1,
    [0][0][0]  = AC_inv_dsy_2 * DER2_0,
    [0][1][0]  = AC_inv_dsy_2 * DER2_1,
    [0][2][0]  = AC_inv_dsy_2 * DER2_2,
    [0][3][0]  = AC_inv_dsy_2 * DER2_3
}

Stencil derzz {
    [-3][0][0] = AC_inv_dsz_2 * DER2_3,
    [-2][0][0] = AC_inv_dsz_2 * DER2_2,
    [-1][0][0] = AC_inv_dsz_2 * DER2_1,
    [0][0][0]  = AC_inv_dsz_2 * DER2_0,
    [1][0][0]  = AC_inv_dsz_2 * DER2_1,
    [2][0][0]  = AC_inv_dsz_2 * DER2_2,
    [3][0][0]  = AC_inv_dsz_2 * DER2_3
}

Stencil derxy {
    [0][-3][-3] = AC_inv_dsx * AC_inv_dsy * DERX_3,
    [0][-2][-2] = AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][-1][-1] = AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][0][0]  = AC_inv_dsx * AC_inv_dsy * DERX_0,
    [0][1][1]  = AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][2][2]  = AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][3][3]  = AC_inv_dsx * AC_inv_dsy * DERX_3,
    [0][-3][3] = -AC_inv_dsx * AC_inv_dsy * DERX_3,
    [0][-2][2] = -AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][-1][1] = -AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][1][-1] = -AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][2][-2] = -AC_inv_dsx * AC_inv_dsy * DERX_2,
    [0][3][-3] = -AC_inv_dsx * AC_inv_dsy * DERX_3
}

Stencil derxz {
    [-3][0][-3] = AC_inv_dsx * AC_inv_dsz * DERX_3,
    [-2][0][-2] = AC_inv_dsx * AC_inv_dsz * DERX_2,
    [-1][0][-1] = AC_inv_dsx * AC_inv_dsz * DERX_1,
    [0][0][0]  = AC_inv_dsx * AC_inv_dsz * DERX_0,
    [1][0][1]  = AC_inv_dsx * AC_inv_dsz * DERX_1,
    [2][0][2]  = AC_inv_dsx * AC_inv_dsz * DERX_2,
    [3][0][3]  = AC_inv_dsx * AC_inv_dsz * DERX_3,
    [-3][0][3] = -AC_inv_dsx * AC_inv_dsz * DERX_3,
    [-2][0][2] = -AC_inv_dsx * AC_inv_dsz * DERX_2,
    [-1][0][1] = -AC_inv_dsx * AC_inv_dsz * DERX_1,
    [1][0][-1] = -AC_inv_dsx * AC_inv_dsz * DERX_1,
    [2][0][-2] = -AC_inv_dsx * AC_inv_dsz * DERX_2,
    [3][0][-3] = -AC_inv_dsx * AC_inv_dsz * DERX_3
}

Stencil deryz {
    [-3][-3][0] = AC_inv_dsy * AC_inv_dsz * DERX_3,
    [-2][-2][0] = AC_inv_dsy * AC_inv_dsz * DERX_2,
    [-1][-1][0] = AC_inv_dsy * AC_inv_dsz * DERX_1,
    [0][0][0]  = AC_inv_dsy * AC_inv_dsz * DERX_0,
    [1][1][0]  = AC_inv_dsy * AC_inv_dsz * DERX_1,
    [2][2][0]  = AC_inv_dsy * AC_inv_dsz * DERX_2,
    [3][3][0]  = AC_inv_dsy * AC_inv_dsz * DERX_3,
    [-3][3][0] = -AC_inv_dsy * AC_inv_dsz * DERX_3,
    [-2][2][0] = -AC_inv_dsy * AC_inv_dsz * DERX_2,
    [-1][1][0] = -AC_inv_dsy * AC_inv_dsz * DERX_1,
    [1][-1][0] = -AC_inv_dsy * AC_inv_dsz * DERX_1,
    [2][-2][0] = -AC_inv_dsy * AC_inv_dsz * DERX_2,
    [3][-3][0] = -AC_inv_dsy * AC_inv_dsz * DERX_3
}

Stencil der6x_upwd {
    [0][0][-3] =  AC_inv_dsx * DER6UPWD_3,
    [0][0][-2] =  AC_inv_dsx * DER6UPWD_2,
    [0][0][-1] =  AC_inv_dsx * DER6UPWD_1,
    [0][0][0]  =  AC_inv_dsx * DER6UPWD_0,
    [0][0][1]  =  AC_inv_dsx * DER6UPWD_1,
    [0][0][2]  =  AC_inv_dsx * DER6UPWD_2,
    [0][0][3]  =  AC_inv_dsx * DER6UPWD_3
}

Stencil der6y_upwd {
    [0][-3][0] =  AC_inv_dsy * DER6UPWD_3,
    [0][-2][0] =  AC_inv_dsy * DER6UPWD_2,
    [0][-1][0] =  AC_inv_dsy * DER6UPWD_1,
    [0][0][0]  =  AC_inv_dsy * DER6UPWD_0,
    [0][1][0]  =  AC_inv_dsy * DER6UPWD_1,
    [0][2][0]  =  AC_inv_dsy * DER6UPWD_2,
    [0][3][0]  =  AC_inv_dsy * DER6UPWD_3
}

Stencil der6z_upwd {
    [-3][0][0] =  AC_inv_dsz * DER6UPWD_3,
    [-2][0][0] =  AC_inv_dsz * DER6UPWD_2,
    [-1][0][0] =  AC_inv_dsz * DER6UPWD_1,
    [0][0][0]  =  AC_inv_dsz * DER6UPWD_0,
    [1][0][0]  =  AC_inv_dsz * DER6UPWD_1,
    [2][0][0]  =  AC_inv_dsz * DER6UPWD_2,
    [3][0][0]  =  AC_inv_dsz * DER6UPWD_3
}
#else
Stencil der5x {
    [0][-3] = -AC_inv_dsx_5 * DER5_3,
    [0][-2] = -AC_inv_dsx_5 * DER5_2,
    [0][-1] = -AC_inv_dsx_5 * DER5_1,
    [0][1]  = AC_inv_dsx_5 * DER5_1,
    [0][2]  = AC_inv_dsx_5 * DER5_2,
    [0][3]  = AC_inv_dsx_5 * DER5_3
}
Stencil der5y {
    [-3][0] = -AC_inv_dsy_5 * DER5_3,
    [-2][0] = -AC_inv_dsy_5 * DER5_2,
    [-1][0] = -AC_inv_dsy_5 * DER5_1,
    [1][0]  = AC_inv_dsy_5 * DER5_1,
    [2][0]  = AC_inv_dsy_5 * DER5_2,
    [3][0]  = AC_inv_dsy_5 * DER5_3
}
der5z(Field field)
{
	return 0.0
}
//TP: corresponds to der4 in Pencil Code
Stencil der4x {
    [0][-3] = AC_inv_dsx_4 * DER4_3,
    [0][-2] = AC_inv_dsx_4 * DER4_2,
    [0][-1] = AC_inv_dsx_4 * DER4_1,
    [0][0]  = AC_inv_dsx_4 * DER4_0,
    [0][1]  = AC_inv_dsx_4 * DER4_1,
    [0][2]  = AC_inv_dsx_4 * DER4_2,
    [0][3]  = AC_inv_dsx_4 * DER4_3
}
Stencil der4y {
    [-3][0] = AC_inv_dsy_4 * DER4_3,
    [-2][0] = AC_inv_dsy_4 * DER4_2,
    [-1][0] = AC_inv_dsy_4 * DER4_1,
    [0][0]  = AC_inv_dsy_4 * DER4_0,
    [1][0]  = AC_inv_dsy_4 * DER4_1,
    [2][0]  = AC_inv_dsy_4 * DER4_2,
    [3][0]  = AC_inv_dsy_4 * DER4_3
}
der4z(Field field)
{
	return 0.0
}
//TP: corresponds to der6_main
Stencil der6x {
    [0][-3] = AC_inv_dsx_6 * DER6_3,
    [0][-2] = AC_inv_dsx_6 * DER6_2,
    [0][-1] = AC_inv_dsx_6 * DER6_1,
    [0][0]  = AC_inv_dsx_6 * DER6_0,
    [0][1]  = AC_inv_dsx_6 * DER6_1,
    [0][2]  = AC_inv_dsx_6 * DER6_2,
    [0][3]  = AC_inv_dsx_6 * DER6_3
}
Stencil der6y {
    [-3][0] = AC_inv_dsy_6 * DER6_3,
    [-2][0] = AC_inv_dsy_6 * DER6_2,
    [-1][0] = AC_inv_dsy_6 * DER6_1,
    [0][0]  = AC_inv_dsy_6 * DER6_0,
    [1][0]  = AC_inv_dsy_6 * DER6_1,
    [2][0]  = AC_inv_dsy_6 * DER6_2,
    [3][0]  = AC_inv_dsy_6 * DER6_3
}
der6z(Field field)
{
	return 0.0
}

Stencil derx {
    [0][-3] = -AC_inv_dsx * DER1_3,
    [0][-2] = -AC_inv_dsx * DER1_2,
    [0][-1] = -AC_inv_dsx * DER1_1,
    [0][1]  = AC_inv_dsx * DER1_1,
    [0][2]  = AC_inv_dsx * DER1_2,
    [0][3]  = AC_inv_dsx * DER1_3
}

Stencil dery {
    [-3][0] = -AC_inv_dsy * DER1_3,
    [-2][0] = -AC_inv_dsy * DER1_2,
    [-1][0] = -AC_inv_dsy * DER1_1,
    [1][0]  = AC_inv_dsy * DER1_1,
    [2][0]  = AC_inv_dsy * DER1_2,
    [3][0]  = AC_inv_dsy * DER1_3
}

Stencil derxx {
    [0][-3] = AC_inv_dsx_2 * DER2_3,
    [0][-2] = AC_inv_dsx_2 * DER2_2,
    [0][-1] = AC_inv_dsx_2 * DER2_1,
    [0][0]  = AC_inv_dsx_2 * DER2_0,
    [0][1]  = AC_inv_dsx_2 * DER2_1,
    [0][2]  = AC_inv_dsx_2 * DER2_2,
    [0][3]  = AC_inv_dsx_2 * DER2_3
}

Stencil deryy {
    [-3][0] = AC_inv_dsy_2 * DER2_3,
    [-2][0] = AC_inv_dsy_2 * DER2_2,
    [-1][0] = AC_inv_dsy_2 * DER2_1,
    [0][0]  = AC_inv_dsy_2 * DER2_0,
    [1][0]  = AC_inv_dsy_2 * DER2_1,
    [2][0]  = AC_inv_dsy_2 * DER2_2,
    [3][0]  = AC_inv_dsy_2 * DER2_3
}

Stencil derxy {
    [-3][-3] = AC_inv_dsx * AC_inv_dsy * DERX_3,
    [-2][-2] = AC_inv_dsx * AC_inv_dsy * DERX_2,
    [-1][-1] = AC_inv_dsx * AC_inv_dsy * DERX_1,
    [0][0]  = AC_inv_dsx * AC_inv_dsy * DERX_0,
    [1][1]  = AC_inv_dsx * AC_inv_dsy * DERX_1,
    [2][2]  = AC_inv_dsx * AC_inv_dsy * DERX_2,
    [3][3]  = AC_inv_dsx * AC_inv_dsy * DERX_3,
    [-3][3] = -AC_inv_dsx * AC_inv_dsy * DERX_3,
    [-2][2] = -AC_inv_dsx * AC_inv_dsy * DERX_2,
    [-1][1] = -AC_inv_dsx * AC_inv_dsy * DERX_1,
    [1][-1] = -AC_inv_dsx * AC_inv_dsy * DERX_1,
    [2][-2] = -AC_inv_dsx * AC_inv_dsy * DERX_2,
    [3][-3] = -AC_inv_dsx * AC_inv_dsy * DERX_3
}

#define derz(VAL)  (0.0)
#define derzz(VAL) (0.0)
#define derxz(VAL) (0.0)
#define deryz(VAL) (0.0)
#define der6z_upwd(VAL) (0.0)

Stencil der6x_upwd {
    [0][-3] =  AC_inv_dsx * DER6UPWD_3,
    [0][-2] =  AC_inv_dsx * DER6UPWD_2,
    [0][-1] =  AC_inv_dsx * DER6UPWD_1,
    [0][0]  =  AC_inv_dsx * DER6UPWD_0,
    [0][1]  =  AC_inv_dsx * DER6UPWD_1,
    [0][2]  =  AC_inv_dsx * DER6UPWD_2,
    [0][3]  =  AC_inv_dsx * DER6UPWD_3
}

Stencil der6y_upwd {
    [-3][0] =  AC_inv_dsy * DER6UPWD_3,
    [-2][0] =  AC_inv_dsy * DER6UPWD_2,
    [-1][0] =  AC_inv_dsy * DER6UPWD_1,
    [0][0]  =  AC_inv_dsy * DER6UPWD_0,
    [1][0]  =  AC_inv_dsy * DER6UPWD_1,
    [2][0]  =  AC_inv_dsy * DER6UPWD_2,
    [3][0]  =  AC_inv_dsy * DER6UPWD_3
}

#endif




