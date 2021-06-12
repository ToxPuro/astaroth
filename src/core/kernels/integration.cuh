#pragma once

static __global__ void
dummy_kernel(void)
{
    /*
    // TODO RE-ENABLE WIP
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    */
    acComplex a = exp(AcReal(1) * acComplex(1, 1) * AcReal(1));
    a* a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}

AcResult
acKernelIntegrateSubstep(const cudaStream_t stream, const int step_number, const int3 start,
                         const int3 end, VertexBufferArray vba)
{
    (void)stream;
    (void)step_number;
    (void)start;
    (void)end;
    (void)vba;
    return AC_FAILURE;
}

AcResult
acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba)
{
    (void)start;
    (void)end;
    (void)vba;
    return AC_FAILURE;
}