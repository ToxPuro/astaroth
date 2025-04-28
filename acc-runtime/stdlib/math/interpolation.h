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
