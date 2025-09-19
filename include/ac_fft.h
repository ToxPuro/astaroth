#pragma once
#ifdef __cplusplus
extern "C" {
#endif
AcResult acFFTBackwardTransformSymmetricC2R(const AcComplex* transformed_in, const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer);
AcResult acFFTForwardTransformSymmetricR2C(const AcReal* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in);
AcResult acFFTForwardTransformR2C(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst);
AcResult acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst);
AcResult acFFTForwardTransformR2Planar(const AcReal* src,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst);
AcResult acFFTBackwardTransformC2R(const AcComplex* transformed_in, const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer);
#ifdef __cplusplus
}
#endif
