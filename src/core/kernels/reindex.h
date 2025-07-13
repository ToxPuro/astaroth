#ifdef __cplusplus
extern "C" 
{
#endif
AcResult acReindex(const cudaStream_t stream, //
                   const AcReal* in, const AcIndex in_offset,
                   const AcIndex in_shape, //
                   AcReal* out, const AcIndex out_offset,
                   const AcIndex out_shape, const AcShape block_shape);

/**
AcResult acReindexCross(const cudaStream_t stream, //
                        const VertexBufferArray vba, const AcIndex in_offset,
                        const AcShape in_shape, //
                        AcReal* out, const AcIndex out_offset,
                        const AcShape out_shape, const AcShape block_shape);

**/
#ifdef __cplusplus
}
#endif
