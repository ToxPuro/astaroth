cudaError_t
acStreamSynchronize(cudaStream_t stream);
cudaError_t
acDeviceSynchronize();
cudaError_t
acSetDevice(const int id);
cudaError_t
acGetDeviceCount(int* dst);
cudaError_t
acDeviceSetSharedMemConfig(const cudaSharedMemConfig config);
cudaError_t
acStreamCreateWithPriority(cudaStream_t* dst, int option, int priority);
cudaError_t
acStreamDestroy(cudaStream_t stream);
cudaError_t
acMemcpy(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind);
cudaError_t
acMemcpyAsync(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind, const cudaStream_t stream);
cudaError_t
acMemcpyPeerAsync(AcReal* dst, int dst_id, const AcReal* src, int src_id, const size_t bytes, const cudaStream_t stream);
cudaError_t
acMemGetInfo(size_t* free_mem, size_t* total_mem);
cudaError_t
acStreamQuery(cudaStream_t stream);
const char*
acGetErrorString(cudaError_t err);
cudaError_t
acDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t
acStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
cudaError_t
acMalloc(void** dst, const size_t bytes);
cudaError_t
acFree(void* dst);
cudaError_t
acMallocHost(void** dst, const size_t bytes);
cudaError_t
acGetDevice(int* dst);
cudaError_t
acGetLastError();

