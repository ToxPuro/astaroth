struct int3 
{
	int x;
	int y;
	int z;
}

typedef AcReal2 struct real2
{
	real x;
	real y;
}

typedef AcReal3 struct real3
{
	real x;
	real y;
	real z;
}

typedef AcReal4 struct real4
{
	real x;
	real y;
	real z;
	real w;
}

typedef AcReal5 struct real5
{
	real x;
	real y;
	real z;
	real w;
	real v;
}

typedef AcRealSymmetricTensor struct real_symmetric_tensor
{
	real xx;
	real yy;
	real zz;
	real xy;
	real xz;
	real yz;
}

typedef AcComplex struct complex
{
	real x;
	real y;
}
typedef AcBool3 struct bool3
{
	bool x;
	bool y;
	bool z;
}

struct Field2
{
	Field x;
	Field y;
} 
struct Field3
{
	Field x;
	Field y;
	Field z;
} 

struct Field4
{
	Field x;
	Field y;
	Field z;
	Field w;
} 

struct FieldSymmetricTensor
{
	Field xx;
	Field yy;
	Field zz;
	Field xy;
	Field xz;
	Field yz;
}
//TP: do not make xy,xz and yz into long long since that 
//will degrade performance (at least on AMD)
struct AcDimProducts
{
	int xy;
	int xz;
	int yz;
	long long xyz;
};

struct AcDimProductsInv
{
	real xy;
	real xz;
	real yz;
	real xyz;
};

enum AcCoordinateSystem
{
	AC_CARTESIAN_COORDINATES,
	AC_SPHERICAL_COORDINATES,
	AC_CYLINDRICAL_COORDINATES
}

enum AcProcMappingStrategy
{
	AC_PROC_MAPPING_STRATEGY_MORTON,
	AC_PROC_MAPPING_STRATEGY_LINEAR,
	AC_PROC_MAPPING_STRATEGY_HIERARCHICAL
}

enum AcDecomposeStrategy
{
	AC_DECOMPOSE_STRATEGY_MORTON,
	AC_DECOMPOSE_STRATEGY_EXTERNAL,
	AC_DECOMPOSE_STRATEGY_HIERARCHICAL
}

enum AcMPICommStrategy
{
	AC_MPI_COMM_STRATEGY_DUP_WORLD,
	AC_MPI_COMM_STRATEGY_DUP_USER
}

struct VecZProfile
{
	Profile<Z> x
	Profile<Z> y
	Profile<Z> z
}
enum AcBoundary {
    BOUNDARY_NONE  = 0,
    BOUNDARY_X_TOP = 0x01,
    BOUNDARY_X_BOT = 0x02,
    BOUNDARY_X     = BOUNDARY_X_TOP | BOUNDARY_X_BOT,
    BOUNDARY_Y_TOP = 0x04,
    BOUNDARY_Y_BOT = 0x08,
    BOUNDARY_Y     = BOUNDARY_Y_TOP | BOUNDARY_Y_BOT,
    BOUNDARY_Z_TOP = 0x10,
    BOUNDARY_Z_BOT = 0x20,
    BOUNDARY_Z     = BOUNDARY_Z_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_XY    = BOUNDARY_X | BOUNDARY_Y,
    BOUNDARY_XZ    = BOUNDARY_X | BOUNDARY_Z,
    BOUNDARY_YZ    = BOUNDARY_Y | BOUNDARY_Z,
    BOUNDARY_XYZ   = BOUNDARY_X | BOUNDARY_Y | BOUNDARY_Z,
    BOUNDARY_X_BOT_Y_BOT_Z_BOT = BOUNDARY_X_BOT | BOUNDARY_Y_BOT | BOUNDARY_Z_BOT,
    BOUNDARY_X_BOT_Y_BOT_Z_TOP = BOUNDARY_X_BOT | BOUNDARY_Y_BOT | BOUNDARY_Z_TOP,
    BOUNDARY_X_BOT_Y_TOP_Z_BOT = BOUNDARY_X_BOT | BOUNDARY_Y_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_X_BOT_Y_TOP_Z_TOP = BOUNDARY_X_BOT | BOUNDARY_Y_TOP | BOUNDARY_Z_TOP,
    BOUNDARY_X_TOP_Y_BOT_Z_BOT = BOUNDARY_X_TOP | BOUNDARY_Y_BOT | BOUNDARY_Z_BOT,
    BOUNDARY_X_TOP_Y_BOT_Z_TOP = BOUNDARY_X_TOP | BOUNDARY_Y_BOT | BOUNDARY_Z_TOP,
    BOUNDARY_X_TOP_Y_TOP_Z_BOT = BOUNDARY_X_TOP | BOUNDARY_Y_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_X_TOP_Y_TOP_Z_TOP = BOUNDARY_X_TOP | BOUNDARY_Y_TOP | BOUNDARY_Z_TOP,
    BOUNDARY_XY_Z_BOT = BOUNDARY_XY | BOUNDARY_Z_BOT,
    BOUNDARY_XY_Z_TOP = BOUNDARY_XY | BOUNDARY_Z_TOP,

    BOUNDARY_XZ_Y_BOT = BOUNDARY_XZ | BOUNDARY_Y_BOT,
    BOUNDARY_XZ_Y_TOP = BOUNDARY_XZ | BOUNDARY_Y_TOP,

    BOUNDARY_YZ_X_BOT = BOUNDARY_YZ | BOUNDARY_X_BOT,
    BOUNDARY_YZ_X_TOP = BOUNDARY_YZ | BOUNDARY_X_TOP,

    BOUNDARY_X_Z_BOT = BOUNDARY_X | BOUNDARY_Z_BOT,
    BOUNDARY_X_Z_TOP = BOUNDARY_X | BOUNDARY_Z_TOP,
    BOUNDARY_X_Y_BOT = BOUNDARY_X | BOUNDARY_Y_BOT,
    BOUNDARY_X_Y_TOP = BOUNDARY_X | BOUNDARY_Y_TOP,

    BOUNDARY_Y_X_BOT = BOUNDARY_Y | BOUNDARY_X_BOT,
    BOUNDARY_Y_X_TOP = BOUNDARY_Y | BOUNDARY_X_TOP,
    BOUNDARY_Y_Z_BOT = BOUNDARY_Y | BOUNDARY_Z_BOT,
    BOUNDARY_Y_Z_TOP = BOUNDARY_Y | BOUNDARY_Z_TOP,

    BOUNDARY_Z_X_BOT = BOUNDARY_Z | BOUNDARY_X_BOT,
    BOUNDARY_Z_X_TOP = BOUNDARY_Z | BOUNDARY_X_TOP,
    BOUNDARY_Z_Y_BOT = BOUNDARY_Z | BOUNDARY_Y_BOT,
    BOUNDARY_Z_Y_TOP = BOUNDARY_Z | BOUNDARY_Y_TOP
}
enum AcReductionPostProcessingOp {
	AC_NO_REDUCE_POST_PROCESSING,
	AC_RMS,
	AC_POSTPROCESS_SQRT,
	AC_RADIAL_WINDOW_RMS
}

struct Volume {
  size_t x;
  size_t y;
  size_t z;
};
