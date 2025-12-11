#pragma once
#ifdef __cplusplus
extern "C"
{
#endif
AcResult acMultiplyInplace(const AcReal value, const size_t count,
                           AcReal* array);
AcResult
acMultiplyInplaceComplex(const AcReal value, const size_t count, AcComplex* array);

AcResult
acComplexToReal(const AcComplex* src, const size_t count, AcReal* dst);

AcResult
acRealToComplex(const AcReal* src, const size_t count, AcComplex* dst);

AcResult
acPlanarToComplex(const AcReal* real_src, const AcReal* imag_src,const size_t count, AcComplex* dst);

AcResult
acComplexToPlanar(const AcComplex* src,const size_t count,AcReal* real_dst,AcReal* imag_dst);

#ifdef __cplusplus
}
AcResult acKernelVolumeCopy(const cudaStream_t stream,                                    //
                            const AcReal* in, const Volume in_offset, const Volume in_volume, //
                            AcReal* out, const Volume out_offset, const Volume out_volume);
AcResult acKernelVolumeCopyComplex(const cudaStream_t stream,                                    //
                            const AcComplex* in, const Volume in_offset, const Volume in_volume, //
                            AcComplex* out, const Volume out_offset, const Volume out_volume);
AcResult acKernelVolumeCopyRealToComplex(const cudaStream_t stream,                                    //
                            const AcReal* in, const Volume in_offset, const Volume in_volume, const Volume embedded_in_volume,//
                            AcComplex* out, const Volume out_offset, const Volume out_volume, const Volume embedded_out_volume);
AcResult acKernelVolumeCopyComplexToPlanar(const cudaStream_t stream,                                    //
                            const AcComplex* in, const Volume in_offset, const Volume in_volume, const Volume embedded_in_volume,//
                            AcReal* real_out, AcReal* imag_out, const Volume out_offset, const Volume out_volume, const Volume embedded_out_volume);
AcResult acKernelVolumeCopyRealToComplexBatched(const cudaStream_t stream,                                    //
                            const AcReal* in, const Volume in_offset, const Volume in_volume, //
                            AcComplex* out, const Volume out_offset, const Volume out_volume, const int batch_size);
AcResult acKernelVolumeCopyComplexToPlanarBatched(const cudaStream_t stream,                                    //
                            const AcComplex* in, const Volume in_offset, const Volume in_volume, //
                            AcReal* real_out, AcReal* imag_out, const Volume out_offset, const Volume out_volume, const int batch_size);
AcResult acKernelVolumeCopyComplexToReal(const cudaStream_t stream,                                    //
                   const AcComplex* in, const Volume in_offset, const Volume in_volume, const Volume embedded_in_volume,//
                   AcReal* out,const Volume out_offset, const Volume out_volume, const Volume embedded_out_volume);
#endif
