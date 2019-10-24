//  unpacks buffer for outer yz halos in global memory

__global__ void unpackOyzPlates(const AcReal* __restrict__ buffer, VertexBufferArray vba, int3 start, int3 end)
{
    const int y_block_size = end.x-start.x+1,
              z_block_size = (end.y-start.y+1)*y_block_size,
            var_block_size = z_block_size*(end.z-start.z+1);
  
    const int vertexIdx  = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertexIdx >= var_block_size*NUM_VTXBUF_HANDLES)
        return;

    const int vba_handle  = vertexIdx / var_block_size,
              var_blockIdx= vertexIdx % var_block_size;

    const int zIdx = var_blockIdx / z_block_size,
              xyIdx= var_blockIdx % z_block_size,
              yIdx = xyIdx / y_block_size,
              xIdx = xyIdx % y_block_size; 

    const int vba_idx = IDX(start+(int3){xIdx,yIdx,zIdx});
   
    vba.in[vba_handle][vba_idx] = buffer[vertexIdx];
}

