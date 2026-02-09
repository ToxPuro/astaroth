//Sixth-order values.
//To be redefined by other orders
#define COMPACT_POISSON_CENTRAL (-128.0/30.0)
#define COMPACT_POISSON_FACE (14.0/30.0)
#define COMPACT_POISSON_DIAGONAL (3.0/30.0)
#define COMPACT_POISSON_CUBICAL (1.0/30.0)

#define COMPACT_POISSON_4TH_ORDER_CENTRAL (-24.0/6.0)
#define COMPACT_POISSON_4TH_ORDER_FACE (2.0/6.0)
#define COMPACT_POISSON_4TH_ORDER_DIAGONAL (1.0/6.0)
#define COMPACT_POISSON_4TH_ORDER_CUBICAL (0.0)

#if STENCIL_ORDER == 4
//Redefining the sixth-order coeffs to 4th order
#undef COMPACT_POISSON_CENTRAL
#undef COMPACT_POISSON_FACE
#undef COMPACT_POISSON_DIAGONAL
#undef COMPACT_POISSON_CUBICAL

#define COMPACT_POISSON_CENTRAL COMPACT_POISSON_4TH_ORDER_CENTRAL
#define COMPACT_POISSON_FACE    COMPACT_POISSON_4TH_ORDER_FACE
#define COMPACT_POISSON_DIAGONAL COMPACT_POISSON_4TH_ORDER_DIAGONAL
#define COMPACT_POISSON_CUBICAL COMPACT_POISSON_4TH_ORDER_CUBICAL

#endif

#if STENCIL_ORDER == 2
//Redefining the sixth-order coeffs to 2n order
#undef COMPACT_POISSON_CENTRAL
#undef COMPACT_POISSON_FACE
#undef COMPACT_POISSON_DIAGONAL
#undef COMPACT_POISSON_CUBICAL

#define COMPACT_POISSON_CENTRAL  (-6.0)
#define COMPACT_POISSON_FACE     (1.0)
#define COMPACT_POISSON_DIAGONAL (0.0)
#define COMPACT_POISSON_CUBICAL  (0.0)

#endif

/**
 * Computes the lhs needed when solving the Poisson equation
 * with a compact stencil (e.g. radius 1 stencil for 6th order Laplacian).
 * Requires isotropic spacing.
 * See reference: A High-Order Compact Formulation for the 3D Poisson Equation.
 * For test case see test/compact-poisson-test
 */
Stencil compact_poisson_lhs
{
	[ 0][ 0][ 0]  = (COMPACT_POISSON_CENTRAL)*AC_inv_ds_2.x,
	[ 0][ 0][ 1]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 0][ 0][-1]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 0][ 1][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 0][-1][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 1][ 0][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[-1][ 0][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,

	[-1][-1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][-1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[-1][ 1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,

	[-1][ 0][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 0][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[-1][ 0][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 0][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,

	[ 0][-1][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][ 1][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][-1][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][ 1][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,

	[-1][-1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[-1][-1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[-1][ 1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[-1][ 1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][-1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][-1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][ 1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][ 1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x
}

Stencil compact_poisson_lhs_4th_order
{
	[ 0][ 0][ 0]  = (COMPACT_POISSON_4TH_ORDER_CENTRAL)*AC_inv_ds_2.x,
	[ 0][ 0][ 1]  = (COMPACT_POISSON_4TH_ORDER_FACE)*AC_inv_ds_2.x,
	[ 0][ 0][-1]  = (COMPACT_POISSON_4TH_ORDER_FACE)*AC_inv_ds_2.x,
	[ 0][ 1][ 0]  = (COMPACT_POISSON_4TH_ORDER_FACE)*AC_inv_ds_2.x,
	[ 0][-1][ 0]  = (COMPACT_POISSON_4TH_ORDER_FACE)*AC_inv_ds_2.x,
	[ 1][ 0][ 0]  = (COMPACT_POISSON_4TH_ORDER_FACE)*AC_inv_ds_2.x,
	[-1][ 0][ 0]  = (COMPACT_POISSON_4TH_ORDER_FACE)*AC_inv_ds_2.x,

	[-1][-1][ 0]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][-1][ 0]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[-1][ 1][ 0]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 1][ 0]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,

	[-1][ 0][-1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 0][-1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[-1][ 0][ 1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 0][ 1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,

	[ 0][-1][-1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][ 1][-1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][-1][ 1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][ 1][ 1]  = (COMPACT_POISSON_4TH_ORDER_DIAGONAL)*AC_inv_ds_2.x,

	[-1][-1][-1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[-1][-1][ 1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[-1][ 1][-1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[-1][ 1][ 1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[ 1][-1][-1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[ 1][-1][ 1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[ 1][ 1][-1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x,
	[ 1][ 1][ 1]  = (COMPACT_POISSON_4TH_ORDER_CUBICAL)*AC_inv_ds_2.x
}

/**
 * The same as compact_poisson_lhs
 * (Useful for Jacobi or SOR).
 */
Stencil compact_poisson_lhs_neighbours
{
	[ 0][ 0][ 1]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 0][ 0][-1]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 0][ 1][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 0][-1][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[ 1][ 0][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,
	[-1][ 0][ 0]  = (COMPACT_POISSON_FACE)*AC_inv_ds_2.x,

	[-1][-1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][-1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[-1][ 1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 1][ 0]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,

	[-1][ 0][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 0][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[-1][ 0][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 1][ 0][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,

	[ 0][-1][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][ 1][-1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][-1][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,
	[ 0][ 1][ 1]  = (COMPACT_POISSON_DIAGONAL)*AC_inv_ds_2.x,

	[-1][-1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[-1][-1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[-1][ 1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[-1][ 1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][-1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][-1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][ 1][-1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x,
	[ 1][ 1][ 1]  = (COMPACT_POISSON_CUBICAL)*AC_inv_ds_2.x
}

/**
 * The central coefficient of the compact Poisson matrix
 * used for Jacobi or SOR
 */
compact_poisson_lhs_central_coeff()
{
	return (COMPACT_POISSON_CENTRAL)*AC_inv_ds_2.x
}

/**
 * Computes the rhs needed when solving the Poisson equation
 * with a compact stencil (e.g. radius 1 stencil for 6th order Laplacian).
 * Requires isotropic spacing.
 * See reference: A High-Order Compact Formulation for the 3D Poisson Equation.
 * For test case see test/compact-poisson-test
 */
#if STENCIL_ORDER == 6
compact_poisson_rhs(Field f)
{
	//Assumes equidistant grid
	//Derivative terms of f are not necessarily needed to be computed to h^6
	//accuracy because of the prefactors but for now let us be accurate
	h2 = AC_ds_2.x
	h4 = AC_ds_4.x
	return f + (h2/12.0)*laplace(f) + (h4/360.0)*biharmonic(f)
	       + (h4/180.0)*(der2x2y(f) + der2y2z(f) + der2x2z(f))
}
#else
#if STENCIL_ORDER == 4
compact_poisson_rhs(Field f)
{
	//Assumes equidistant grid
	//Derivative terms of f are not necessarily needed to be computed to h^6
	//accuracy because of the prefactors but for now let us be accurate
	h2 = AC_ds_2.x
	return f + (h2/12.0)*laplace(f)
}
#else
#if STENCIL_ORDER == 2
compact_poisson_rhs(Field f)
{
	return value(f)
}
#else
compact_poisson_rhs(Field f)
{
  	fatal_error_message(true,"Compact stencils for Poisson exist only up to sixth order!\n");
	return 0.0
}
#endif
#endif
#endif
