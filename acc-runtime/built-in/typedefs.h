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
//Copy int3's here since they need the struct declaration
int3 AC_domain_decomposition
int3 AC_multigpu_offset
int3 AC_domain_coordinates
