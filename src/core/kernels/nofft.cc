#include "host_datatypes.h"
#include "ac_fft.h"
#include <stdio.h>
#include <cstdlib>

AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTBackwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTBackwardTransformC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTBackwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformR2Planar(const AcReal*, const Volume, const Volume, const Volume, AcReal*, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransformR2Planar!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformPlanar(const AcReal*, const AcReal*,const Volume, const Volume, const Volume, AcReal*, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransformPlanar!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
