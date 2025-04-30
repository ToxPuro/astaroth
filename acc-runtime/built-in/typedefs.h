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
