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
 * After a task has been completed, its dependent tasks can be started with notifyDependents()
 * E.g. ComputationTasks may depend on HaloExchangeTasks because they're waiting to receive data.
 * Vv.  HaloExchangeTasksmay depend on ComputationTasks because they're waiting for data to send.
 *
 * This all happens in grid.cc:GridIntegrate
 */

#include "task.h"
#include "astaroth.h"

#include <mpi.h>
#include <vector>
#include <iostream>

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
Task::logStateChangedEvent(int tag, std::string from, std::string to)
{
    int seg_type = segment_type(index_to_segment_id(tag));
    /*
       std::cout<< "{"
             <<"\"msg_type\":\"state_changed_event\","
             <<"\"rank\":"<<rank
             <<",\"substep\":"<<substep_counter.count
             <<",\"task_type\":\""<<task_type<<"\""
             <<",\"tag\":"<<tag
             <<",\"seg_id\":["
                 <<segment_id.x<<","
                 <<segment_id.y<<","
                 <<segment_id.z<<"],"
             <<"\"seg_type\":"<<seg_type<<","
             <<"\"from\":\""<<from<<"\""<<","
             <<"\"to\":\""<<to<<"\""
             <<"}"<<std::endl;
    */
    
}

void
Task::synchronizeWithDevice()
{
    cudaSetDevice(device->id);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        vba.in[i]  = device->vba.in[i];
        vba.out[i] = device->vba.out[i];
    }
}

//Swap local vba
void
Task::swapVBA()
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        AcReal* tmp        = vba.in[i];
        vba.in[i]  = vba.out[i];
        vba.out[i] = tmp;
    }
}

void
Task::schedule(size_t substeps, AcReal dt_)
{
    substep_counter.start_counting_to(substeps);
    dependencyCounter.extend_to(substeps,0);
    dependencyCounter.reset_counts();
    dt = dt_;
}

bool
Task::isFinished()
{
    return substep_counter.is_finished();
}

void
Task::update()
{
    if (isFinished()) return;
    
    bool ready = (state == iteration_first_step) ? dependencyCounter.is_finished(substep_counter.count) : test();
    if (ready){
        advance();
        if (state == iteration_first_step){
            swapVBA();
            notifyDependents();
            substep_counter.increment();
        }
    }
}

void
Task::registerDependent(Task* t, size_t offset)
{
    dependents.emplace_back(t,offset);
/*
    if (rank == 0){
        std::cout << "{"
        <<"\"msg_type\":\"dependency_registration\","

        <<"\"prerequisite\":{"
            <<"\"seg_id\":["
                <<this->segment_id.x<<","
                <<this->segment_id.y<<","
                <<this->segment_id.z<<"],"
            <<"\"task_type\":"<<"\""<<this->task_type<<"\""
        <<"},"

        <<"\"dependent\":{"
            <<"\"seg_id\":["
                <<t->segment_id.x<<","
                <<t->segment_id.y<<","
                <<t->segment_id.z<<"],"
            <<"\"task_type\":"<<"\""<<t->task_type<<"\""
        <<"},"

        <<"\"offset\":"<<offset
        <<"}"<< std::endl;
    }
*/
    t->dependOn(offset);
}

void
Task::dependOn(size_t offset)
{
    dependencyCounter.increment_target(offset);
}

void
Task::notifyDependents()
{
    for (auto& d : dependents) {
        d.first->notify(substep_counter.count, d.second);
    }
}

void
Task::notify(size_t substep, size_t offset)
{
    dependencyCounter.increment(substep+offset, offset);
}
/* Computation */

ComputationTask::ComputationTask(int3 segment_id_, int3 nn, Device device_, Stream stream_)
    : stream(stream_)
{
    segment_id = segment_id_;
    device = device_;
    task_type = "compute";

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //Copy VBA's CUDA-pointers over
    synchronizeWithDevice();

    start = (int3){segment_id.x == -1 ? NGHOST : segment_id.x == 1 ? nn.x : NGHOST * 2,
                   segment_id.y == -1 ? NGHOST : segment_id.y == 1 ? nn.y : NGHOST * 2,
                   segment_id.z == -1 ? NGHOST : segment_id.z == 1 ? nn.z : NGHOST * 2};

    dims = (int3){segment_id.x == 0 ? nn.x - NGHOST * 2 : NGHOST,
                  segment_id.y == 0 ? nn.y - NGHOST * 2 : NGHOST,
                  segment_id.z == 0 ? nn.z - NGHOST * 2 : NGHOST};
    /*
    if (rank == 0){
        std::cout << "{"
                <<"\"msg_type\":\"task_definition\","
                << "\"task_type\":\"compute\","
                << "\"segment_index\":" << segment_index(segment_id)
                << ","
                << "\"segment_id\":["
                << segment_id.x<<","
                << segment_id.y<<","
                << segment_id.z
                << "],"
                << "\"start\":["
                << start.x<<","
                << start.y<<","
                << start.z
                << "],"
                << "\"dims\":["
                << dims.x<<","
                << dims.y<<","
                << dims.z
                << "]"
                << "}" <<std::endl;
    }
    */
}

void
ComputationTask::compute()
{
    cudaSetDevice(device->id);
    acDeviceLoadScalarUniform(device, stream, AC_dt, dt);
    acKernelIntegrateSubstep(device->streams[stream], substep_counter.count, start, start + dims, vba);
}

bool
ComputationTask::test()
{
    switch (static_cast<ComputeState>(state)){
        case ComputeState::Running:
        {
            cudaError_t err = cudaStreamQuery(device->streams[stream]);
            return err == cudaSuccess;
        }
        default:
            return false;
    }
}

void
ComputationTask::advance()
{
    switch (static_cast<ComputeState>(state)){
        case ComputeState::Waiting_for_halo:
        {
            logStateChangedEvent(segment_index(segment_id),"waiting", "running");
            cudaDeviceSynchronize();
            compute();
            state = static_cast<int>(ComputeState::Running);
            break;
        }
        case ComputeState::Running:
        {
            logStateChangedEvent(segment_index(segment_id),"running", "waiting");
            state = static_cast<int>(ComputeState::Waiting_for_halo);
            break;
        }
    }
}

/*  Communication   */

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
HaloExchangeTask::HaloExchangeTask(const Device device_, const int rank_, const int3 segment_id_,
                                   const uint3_64 decomposition, const int3 grid_dimensions,
                                   MPI_Request* recv_requests, MPI_Request* send_requests)
{
    segment_id = segment_id_;
    rank = rank_;
    device = device_;
    task_type = "halo";

    //Copy VBA's CUDA-pointers over
    synchronizeWithDevice();

    int seg_type = segment_type(segment_id);
    active       = ((MPI_INCL_CORNERS) || seg_type != 3) ? true : false;
    
    const int3 foreign_segment_id = -segment_id;

    // Coordinates
    //These names are bad and confusing?
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
/*
    if (active && rank == 0){
    std::cout   << "{"
                <<"\"msg_type\":\"task_definition\","
                << "\"task_type\":\"halo\","
                <<"\"segment_index\":" << segment_index(segment_id)
                <<","
                << "\"segment_id\":["
                <<segment_id.x<<","
                <<segment_id.y<<","
                <<segment_id.z
                <<"],"
                << "\"foreign_segment_id\":["
                <<foreign_segment_id.x<<","
                <<foreign_segment_id.y<<","
                <<foreign_segment_id.z
                <<"],"
                << "\"halo_segment_coordinates\":["
                <<halo_segment_coordinates.x<<","
                <<halo_segment_coordinates.y<<","
                <<halo_segment_coordinates.z
                <<"],"
                << "\"local_segment_coordinates\":["
                <<local_segment_coordinates.x<<","
                <<local_segment_coordinates.y<<","
                <<local_segment_coordinates.z
                <<"],"
                <<"\"dims\":["
                <<segment_dimensions.x<<","
                <<segment_dimensions.y<<","
                <<segment_dimensions.z
                <<"]"
                << "}" <<std::endl;
    }
*/
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
    acKernelPackData(stream, vba, local_segment_coordinates, msg->buffer);
}

void
HaloExchangeTask::unpack()
{
    
    auto msg           = recv_buffers->get_current_buffer();
    msg->buffer.pinned = false;
    acUnpinPackedData(device, stream, &(msg->buffer));
    acKernelUnpackData(stream, msg->buffer, halo_segment_coordinates, vba);
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

bool
HaloExchangeTask::test()
{
    switch(static_cast<HaloExchangeState>(state)){
        case HaloExchangeState::Packing:
        {
            cudaError_t err = cudaStreamQuery(stream);
            if (err == cudaSuccess){
                return true;
            }
            if (err == cudaErrorNotReady){
                return false;
            }
            return false;
        }
        case HaloExchangeState::Unpacking:
        {
            cudaError_t err = cudaStreamQuery(stream);
            if (err == cudaSuccess){
                //TODO: test if these trigger (the cudaErrorNotReadies)
                return true;
            }
            if (err == cudaErrorNotReady){
                return false;
            }
            return false;
        }
        case HaloExchangeState::Receiving:
        {
            //TODO: see if current buffer is the correct buffer, when do we swap buffers?
            auto msg = recv_buffers->get_current_buffer();
            int request_complete;
            MPI_Test(msg->request, &request_complete, MPI_STATUS_IGNORE);   
            return request_complete?true:false;
        }
        default:
        {
            ERROR("Invalid state for HaloExchangeTask");
            return false;
        }
    }
}

void
HaloExchangeTask::advance()
{
    switch(static_cast<HaloExchangeState>(state)){
        case HaloExchangeState::Waiting_for_compute:
            logStateChangedEvent(send_tag,  "waiting", "packing");
            pack();
            state = static_cast<int>(HaloExchangeState::Packing);
            break;
        case HaloExchangeState::Packing:
            logStateChangedEvent(send_tag, "packing", "receiving");
            sync();
            send();
            state = static_cast<int>(HaloExchangeState::Receiving);
            break;
        case HaloExchangeState::Receiving:
            logStateChangedEvent(send_tag, "receiving", "unpacking");
            sync();
            unpack();
            state = static_cast<int>(HaloExchangeState::Unpacking);
            break;
        case HaloExchangeState::Unpacking:
            logStateChangedEvent(send_tag, "unpacking", "waiting");
            receive();
            sync();
            state = static_cast<int>(HaloExchangeState::Waiting_for_compute);
            break;
        default:
            ERROR("Invalid state for HaloExchangeTask");
    }
}
#endif // AC_MPI_ENABLED
