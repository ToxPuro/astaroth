#include <sys/resource.h>

static size_t
get_size_from_dim(const int dim, const Volume dims)
{
    	const auto size   = dim == X_ORDER_INT ? dims.x :
        		    dim == Y_ORDER_INT ? dims.y :
        		    dims.z;
        return size;
}

static Volume
get_volume_from_shape(const AcShape shape)
{
	return {shape.x,shape.y,shape.z};
}

static int memusage()
{
	struct rusage usage;
	int res=getrusage(RUSAGE_SELF,&usage);
	ERRCHK_ALWAYS(res == 0);

	return usage.ru_maxrss;
}
