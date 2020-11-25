#if AC_MPI_ENABLED
/**
 * Quick overview of tasks
 *
 * Each halo segment is assigned a HaloExchangeTask.
 * A HaloExchangeTask sends local data as a halo to a neighbor
 * and receives halo data from a (possibly different) neighbor.
 *
 * Each shell segment is assigned a ComputationTask.
 * ComputationTasks integrate their segment of the domain.
 *
 * After a task has been completed, its dependent tasks can be started with notify_dependents()
 * E.g. ComputationTasks may depend on HaloExchangeTasks because they're waiting to receive data.
 * Vv. HaloExchangeTasks may depend on ComputationTasks because they're waiting for data to send.
 *
 * This all happens in grid.cc:GridIntegrate
 */

#include "task.h"
#include "astaroth.h"

#include <mpi.h>
#include <vector>

#include "decomposition.h"   //getPid and friends
#include "kernels/kernels.h" //PackedData

#define HALO_TAG_OFFSET (100) //"Namespacing" the MPI tag space to avoid collisions

static PackedData
acCreatePackedData(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data, bytes));
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data.data_pinned, bytes));
    return data;
}

static AcResult
acDestroyPackedData(PackedData* data)
{
    cudaFree(data->data_pinned);

    data->dims = (int3){-1, -1, -1};
    cudaFree(data->data);
    data->data = NULL;

    return AC_SUCCESS;
}

static void
acPinPackedData(const Device device, const cudaStream_t stream, PackedData* ddata)
{
    cudaSetDevice(device->id);
    // TODO sync stream
    ddata->pinned = true;

    const size_t bytes = ddata->dims.x * ddata->dims.y * ddata->dims.z * sizeof(ddata->data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data_pinned, ddata->data, bytes, cudaMemcpyDefault, stream));
}

static void
acUnpinPackedData(const Device device, const cudaStream_t stream, PackedData* ddata)
{
    if (!ddata->pinned) // Unpin iff the data was pinned previously
        return;

    cudaSetDevice(device->id);
    // TODO sync stream
    ddata->pinned = false;

    const size_t bytes = ddata->dims.x * ddata->dims.y * ddata->dims.z * sizeof(ddata->data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data, ddata->data_pinned, bytes, cudaMemcpyDefault, stream));
}

/*   Task interface    */

// Tasks are encapsulated pieces of work that have a clear set of dependencies (and dependents).
void
Task::register_dependent(Task* t)
{
    dependents.push_back(t);
}

void
Task::notify_dependents(int isubstep, AcReal dt)
{
    for (auto& d : dependents) {
        d->notify(isubstep, dt);
    }
}

void
Task::notify(int isubstep, AcReal dt)
{
    active_dependencies--;
    if (active_dependencies == 0 && allowed_triggers > 0) {
        active_dependencies = total_dependencies;
        allowed_triggers--;
        execute(isubstep, dt);
    }
}

void
Task::set_trigger_limit(size_t trigger_limit)
{
    allowed_triggers = trigger_limit;
}

/* Computation */

// Computation task dependencies
#define COMPUTE_FACE_DEPS (1)
#define COMPUTE_EDGE_DEPS (3)

#if MPI_INCL_CORNERS
#define COMPUTE_CORNER_DEPS (7)
#else
#define COMPUTE_CORNER_DEPS (6)
#endif

ComputationTask::ComputationTask(int3 _segment_id, int3 nn, Device _device, Stream _stream)
    : segment_id(_segment_id), device(_device), stream(_stream)
{
    start = (int3){segment_id.x == -1 ? NGHOST : segment_id.x == 1 ? nn.x : NGHOST * 2,
                   segment_id.y == -1 ? NGHOST : segment_id.y == 1 ? nn.y : NGHOST * 2,
                   segment_id.z == -1 ? NGHOST : segment_id.z == 1 ? nn.z : NGHOST * 2};

    dims = (int3){segment_id.x == 0 ? nn.x - NGHOST * 2 : NGHOST,
                  segment_id.y == 0 ? nn.y - NGHOST * 2 : NGHOST,
                  segment_id.z == 0 ? nn.z - NGHOST * 2 : NGHOST};

    int seg_type = segment_type(segment_id);
    // clang-format off
    total_dependencies  = (int[]){0, COMPUTE_FACE_DEPS, COMPUTE_EDGE_DEPS,COMPUTE_CORNER_DEPS}[seg_type];
    active_dependencies = total_dependencies;
}

void
ComputationTask::execute(int isubstep, AcReal dt)
{
    cudaSetDevice(device->id);
    acDeviceIntegrateSubstep(device, stream, isubstep, start, start + dims, dt);
}

/*  Communication   */

// Send task dependencies
#define SEND_FACE_DEPS (9)
#define SEND_EDGE_DEPS (3)
#define SEND_CORNER_DEPS (1)

// HaloMessage contains all information needed to send or receive a single message
// Wraps PackedData. These two structs could be folded together.
HaloMessage::HaloMessage(int3 dims, MPI_Request* _req) : request(_req)
{
    *request = MPI_REQUEST_NULL;
    buffer   = acCreatePackedData(dims);
    length   = dims.x * dims.y * dims.z * NUM_VTXBUF_HANDLES;
}
HaloMessage::~HaloMessage() { acDestroyPackedData(&buffer); }

// MessageBufferSwapChain
MessageBufferSwapChain::MessageBufferSwapChain() : buf_idx(SWAP_CHAIN_LENGTH - 1)
{
    buffers.reserve(SWAP_CHAIN_LENGTH);
}

MessageBufferSwapChain::~MessageBufferSwapChain()
{
    for (auto m : buffers) {
        delete m;
    }
    buffers.clear();
}

void
MessageBufferSwapChain::add_buffer(int3 dims, MPI_Request* req)
{
    buffers.push_back(new HaloMessage(dims, req));
}

HaloMessage*
MessageBufferSwapChain::get_current_buffer()
{
    return buffers[buf_idx];
}

HaloMessage*
MessageBufferSwapChain::get_fresh_buffer()
{
    buf_idx          = (buf_idx + 1) % SWAP_CHAIN_LENGTH;
    MPI_Request* req = buffers[buf_idx]->request;
    if (*req != MPI_REQUEST_NULL) {
        MPI_Wait(req, MPI_STATUS_IGNORE);
    }
    return buffers[buf_idx];
}

// HaloExchangeTask
HaloExchangeTask::HaloExchangeTask(const Device _device, const int _rank, const int3 _segment_id,
                                   const uint3_64 decomposition, const int3 grid_dimensions,
                                   MPI_Request* recv_requests, MPI_Request* send_requests)
    : segment_id(_segment_id), rank(_rank), device(_device)
{

    int seg_type = segment_type(segment_id);
    active       = ((MPI_INCL_CORNERS) || seg_type != 3) ? true : false;

    // total_dependencies = (int[]){0,SEND_FACE_DEPS,SEND_EDGE_DEPS,SEND_CORNER_DEPS}[seg_type];
    // unfulfilled_dependencies = total_dependencies;
    const int3 foreign_segment_id = segment_id;

    // Coordinates
    halo_segment_coordinates  = halo_segment_position(segment_id, grid_dimensions);
    local_segment_coordinates = local_segment_position(-foreign_segment_id, grid_dimensions);

    // MPI
    // domain coordinates are the 3D process coordinates (or id) in the decomposition
    const int3 coarse_grid_coordinates = getPid3D(rank, decomposition);
    const int3 segment_dimensions      = segment_dims(segment_id, grid_dimensions);

    send_rank = getPid(coarse_grid_coordinates + segment_id, decomposition);
    recv_rank = getPid(coarse_grid_coordinates - foreign_segment_id, decomposition);

    send_tag = segment_index(segment_id);
    recv_tag = segment_index(foreign_segment_id);

    // Note: send_tag is also the index in the buffer set
    // That's why both recv buffers and send buffers are indexed by it
    recv_buffers = new MessageBufferSwapChain();
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        recv_buffers->add_buffer(segment_dimensions, &(recv_requests[i * NUM_SEGMENTS + send_tag]));
    }

    send_buffers = new MessageBufferSwapChain();
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        send_buffers->add_buffer(segment_dimensions, &(send_requests[i * NUM_SEGMENTS + send_tag]));
    }
    // CUDA
    cudaSetDevice(device->id);

    int low_prio, high_prio;
    cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);

    // Post receive immediately, this avoids unexpected messages
    if (active) {
        receive();
    }
}

HaloExchangeTask::~HaloExchangeTask()
{
    delete recv_buffers;
    delete send_buffers;
    cudaSetDevice(device->id);
    // dependents.clear();
    cudaStreamDestroy(stream);
}

void
HaloExchangeTask::pack()
{
    auto msg = send_buffers->get_fresh_buffer();
    acKernelPackData(stream, device->vba, local_segment_coordinates, msg->buffer);
}

void
HaloExchangeTask::unpack()
{
    auto msg           = recv_buffers->get_current_buffer();
    msg->buffer.pinned = false;
    acUnpinPackedData(device, stream, &(msg->buffer));
    acKernelUnpackData(stream, msg->buffer, halo_segment_coordinates, device->vba);
}

void
HaloExchangeTask::sync()
{
    cudaStreamSynchronize(stream);
}

void
HaloExchangeTask::wait_recv()
{
    auto msg = recv_buffers->get_current_buffer();
    MPI_Wait(msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::wait_send()
{
    auto msg = send_buffers->get_current_buffer();
    MPI_Wait(msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::receiveDevice()
{
    auto msg = recv_buffers->get_fresh_buffer();
    MPI_Irecv(msg->buffer.data, msg->length, AC_MPI_TYPE, recv_rank, recv_tag + HALO_TAG_OFFSET,
              MPI_COMM_WORLD, msg->request);
    msg->buffer.pinned = false;
}

void
HaloExchangeTask::receiveHost()
{
    if (onTheSameNode(rank, recv_rank)) {
        receiveDevice();
    }
    else {
        auto msg = recv_buffers->get_fresh_buffer();
        MPI_Irecv(msg->buffer.data_pinned, msg->length, AC_MPI_TYPE, recv_rank,
                  recv_tag + HALO_TAG_OFFSET, MPI_COMM_WORLD, msg->request);
        msg->buffer.pinned = true;
    }
}

void
HaloExchangeTask::sendDevice()
{
    auto msg = send_buffers->get_current_buffer();
    sync();
    MPI_Isend(msg->buffer.data, msg->length, AC_MPI_TYPE, send_rank, send_tag + HALO_TAG_OFFSET,
              MPI_COMM_WORLD, msg->request);
}

void
HaloExchangeTask::sendHost()
{
    // POSSIBLE COMPAT ISSUE: is it sensible to always use CUDA memory for node-local exchanges?
    // What if the MPI lib doesn't support CUDA?
    if (onTheSameNode(rank, send_rank)) {
        sendDevice();
    }
    else {
        auto msg = send_buffers->get_current_buffer();
        acPinPackedData(device, stream, &(msg->buffer));
        sync();
        MPI_Isend(msg->buffer.data_pinned, msg->length, AC_MPI_TYPE, send_rank,
                  send_tag + HALO_TAG_OFFSET, MPI_COMM_WORLD, msg->request);
    }
}

void
HaloExchangeTask::exchangeDevice()
{
    // cudaSetDevice(device->id);
    receiveDevice();
    sendDevice();
}

void
HaloExchangeTask::exchangeHost()
{
    // cudaSetDevice(device->id);
    receiveHost();
    sendHost();
}

void
HaloExchangeTask::receive()
{
#if MPI_USE_PINNED == (1)
    receiveHost();
#else
    receiveDevice();
#endif
}

void
HaloExchangeTask::send()
{
#if MPI_USE_PINNED == (1)
    sendHost();
#else
    sendDevice();
#endif
}

void
HaloExchangeTask::exchange()
{
#if MPI_USE_PINNED == (1)
    exchangeHost();
#else
    exchangeDevice();
#endif
}

// TODO: Implement exchanges for HaloExchangeTasks
void
HaloExchangeTask::execute(int isubstep, AcReal dt)
{
    dt = isubstep * dt;
    // swap_buffers();
    // exchange();
}
#endif // AC_MPI_ENABLED
