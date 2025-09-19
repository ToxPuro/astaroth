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
#endif
