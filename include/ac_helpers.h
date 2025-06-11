#ifdef __cplusplus
extern "C" 
{
#endif
const char* acLibraryVersion(const char* library, const int counter, const AcMeshInfo info);

size_t acGetSizeFromDim(const int dim, const Volume dims);

Volume acGetVolumeFromShape(const AcShape shape);
int acMemUsage();
#ifdef __cplusplus
}
#endif

