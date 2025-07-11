#include "astaroth_analysis.h"
typedef struct device_s* Device;
#ifdef __cplusplus
extern "C" 
{
#endif
const char* acLibraryVersion(const char* library, const int counter, const AcMeshInfo info);

size_t acGetSizeFromDim(const int dim, const Volume dims);

Volume acGetVolumeFromShape(const AcShape shape);
int acMemUsage();

size_t
acGetAmountOfDeviceMemoryFree();

#ifdef __cplusplus
}

int3
ceil(AcReal3 a);

size3_t
ceil_div(const size3_t& a, const int3& b);

size3_t
ceil_div(const size3_t& a, const size3_t& b);

int3
ceil_div(const int3& a, const int3& b);

size_t
ceil_div(const size_t& a, const size_t& b); 

void
acDeviceMalloc(void** dst, const size_t bytes);
void
acDeviceMalloc(AcReal** dst, const size_t bytes);

void
acDeviceFree(void** dst, const int bytes);
void
acDeviceFree(AcReal** dst, const int bytes);
void
acDeviceFree(AcComplex** dst, const int bytes);

#endif

