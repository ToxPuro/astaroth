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

struct AcDimProducts
{
	int xy;
	int xz;
	int yz;
	long xyz;
};

enum AC_COORDINATE_SYSTEM
{
	AC_CARTESIAN_COORDINATES,
	AC_SPHERICAL_COORDINATES,
	AC_CYLINDRICAL_COORDINATES
}
