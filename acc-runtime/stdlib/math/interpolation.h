#ifndef AC_MATH_INTERPOLATION_H
#define AC_MATH_INTERPOLATION_H

#if STENCIL_ORDER == 0
#else
Stencil interpolate_middle_left
{
	[0][0][-1] = 0.5,
	[0][0][0] = 0.5
}

Stencil interpolate_middle_right
{
	[0][0][1] = 0.5,
	[0][0][0] = 0.5
}
Stencil interpolate_middle_down
{
	[0][-1][0] = 0.5,
	[0][0][0] = 0.5
}

Stencil interpolate_middle_up
{
	[0][1][0] = 0.5,
	[0][0][0] = 0.5
}

Stencil interpolate_middle_back
{
	[-1][0][0] = 0.5,
	[0][0][0] = 0.5
}
Stencil interpolate_middle_front
{
	[1][0][0] = 0.5,
	[0][0][0] = 0.5
}
#endif

#endif
