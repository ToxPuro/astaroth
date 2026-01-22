#ifndef AC_COLORS_H
#define AC_COLORS_H
/**
 * Functionality required for graph coloring.
 * Used for Red-Black SOR and multi-coloring preconditioners
 */
enum AC_COLOR
{
	AC_COLOR_RED,
	AC_COLOR_BLACK,
	AC_COLOR_GREEN,
	AC_COLOR_YELLOW
}

red_black_is_of_color(int color)
{
	return (globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) % 2 == color
}

is_of_color(int color, int number_of_colors)
{
	return (globalVertexIdx.x + globalVertexIdx.y + globalVertexIdx.z) % number_of_colors == color
}
#endif
