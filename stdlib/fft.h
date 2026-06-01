#pragma once

#ifdef AC_MATH_FFT_INCLUDED
AcResult
ac_fft_split_diffusion_update(Field f, const AcReal dt, const AcReal diffusion_coeff, Field real_src, Field imag_src, Field real_dst, Field imag_dst)
{
	Device dev = acGridGetDevice();
	acDeviceFFTR2Planar(dev,f,real_src,imag_src);
	acDeviceSetInput(dev,AC_FFT_REAL_SRC,int(real_src));
	acDeviceSetInput(dev,AC_FFT_IMAG_SRC,int(imag_src));
	acDeviceSetInput(dev,AC_FFT_REAL_DST,int(real_dst));
	acDeviceSetInput(dev,AC_FFT_IMAG_DST,int(imag_dst));
	acDeviceSetInput(dev,AC_FFT_SPLIT_DIFFUSION_UPDATE_DT,dt);
	acDeviceSetInput(dev,AC_FFT_SPLIT_DIFFUSION_UPDATE_DIFFUSION_COEFF,diffusion_coeff);
    	acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(AC_fft_split_diffusion_update_step),1);
	acDeviceFFTBackwardTransformPlanar2R(dev, real_dst, imag_dst, f);
	return AC_SUCCESS;
}
#endif
