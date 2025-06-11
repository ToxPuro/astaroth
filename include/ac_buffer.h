#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
AcBuffer acBufferCreate(const AcShape shape, const bool on_device);
AcBuffer acBufferCreateTransposed(const AcBuffer src, const AcMeshOrder order);
AcBuffer acTransposeBuffer(const AcBuffer src, const AcMeshOrder order, const cudaStream_t stream);

AcShape  acGetTransposeBufferShape(const AcMeshOrder order, const Volume dims);
AcShape  acGetReductionShape(const AcProfileType type, const AcMeshDims dims);
AcResult acReduceProfile(const Profile prof, const AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream);
AcResult acReduceProfileWithBounds(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream, const Volume start, const Volume end, const Volume start_after_transpose, const Volume end_after_transpose);

AcBuffer
acBufferRemoveHalos(const AcBuffer buffer_in, const int3 halo_sizes, const cudaStream_t stream);

void acBufferDestroy(AcBuffer* buffer);

AcResult acBufferMigrate(const AcBuffer in, AcBuffer* out);
AcBuffer acBufferCopy(const AcBuffer in, const bool on_device);
#ifdef __cplusplus
}
#endif

