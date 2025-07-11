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
acMemcpy(void* dst, const void* src, const size_t bytes, cudaMemcpyKind kind);
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
const char*
acGetErrorName(cudaError_t err);
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
cudaError_t
acEventCreate(cudaEvent_t* event);
cudaError_t
acEventRecord(cudaEvent_t event);
cudaError_t
acEventSynchronize(cudaEvent_t event);
cudaError_t
acEventElapsedTime(float* time, cudaEvent_t start_event, cudaEvent_t end_event);
cudaError_t
acEventDestroy(cudaEvent_t event);
cudaError_t 
acGetDeviceProperties(cudaDeviceProp* prop, int  device);
cudaError_t
acOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t smemSize);
cudaError_t
acDeviceGetAttribute(int* dst, cudaDeviceAttribute_t attr, int device);
cudaError_t
acLaunchCooperativeKernel(void* func,dim3 bpg,dim3 tpb,void** args,size_t smem,cudaStream_t stream);
cudaError_t
acMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
cudaError_t
acMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t
acDeviceGetPCIBusId(char* pciBusId, int len, int device);
cudaError_t 
acMemcpyFromSymbol( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind);
cudaError_t 
acMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
