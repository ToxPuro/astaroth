const AcCoordinatesType AC_COORDINATES_TYPE = AcCartesianCoordinates
#if AC_LAGRANGIAN_GRID
const AcGridType        AC_GRID_TYPE        = AcLagrangianGrid
#else
const AcGridType        AC_GRID_TYPE        = AcEulerianGrid
#endif
