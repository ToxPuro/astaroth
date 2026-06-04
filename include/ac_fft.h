#pragma once

#include "func_define.h"
#include "device_headers.h"
#include "host_datatypes.h"

AC_BEGIN_C_DECLARATIONS

AcResult acFFTBackwardTransformSymmetricC2R(const AcComplex* transformed_in, const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer);
AcResult acFFTForwardTransformSymmetricR2C(const AcReal* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in);
AcResult acFFTForwardTransformR2C(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst);
AcResult acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst);
AcResult acFFTBackwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst);
AcResult acFFTBackwardTransformPlanar2R(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* dst);
AcResult acFFTForwardTransformR2Planar(const AcReal* src,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst);
AcResult acFFTForwardTransformR2PlanarBatched(const void* src,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, void* real_dst, void* imag_dst, const int batch_size, const AcPrecision input_precision, const AcPrecision output_precision);
AcResult acFFTForwardTransformR2HermitianPlanarBatched(const AcReal* src,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst, const int batch_size, cudaStream_t stream);
AcResult acFFTBackwardTransformC2R(const AcComplex* transformed_in, const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer);
AcResult acFFTInit(const AcCommunicator* comm, const int* global_offset);
AcResult acFFTQuit();

AC_END_C_DECLARATIONS
