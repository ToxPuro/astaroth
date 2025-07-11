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
#endif
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

