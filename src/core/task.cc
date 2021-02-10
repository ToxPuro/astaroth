#if AC_MPI_ENABLED
/**
 * Quick overview of tasks
 *
 * Each halo segment is assigned a HaloExchangeTask.
 * A HaloExchangeTask sends local data as a halo to a neighbor
 * and receives halo data from a (possibly different) neighbor.
 *
 * Each shell segment is assigned a ComputeTask.
 * ComputeTasks integrate their segment of the domain.
 *
 * After a task has been completed, its dependent tasks can be started with notifyDependents()
 * E.g. ComputeTasks may depend on HaloExchangeTasks because they're waiting to receive data.
 * Vv.  HaloExchangeTasksmay depend on ComputeTasks because they're waiting for data to send.
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

/* Task interface */
/*
void
Task::logStateChangedEvent(std::string from, std::string to)
{   
    //NOTE: the keys used here don't reflect terminology in Astaroth
    //because the messages are read by a python tool which expects these keys.
    std::cout<< "{"
         <<"\"msg_type\":\"state_changed_event\","
         <<"\"rank\":"<<rank
         <<",\"substep\":"<<loop_cntr.i
         <<",\"task_type\":\""<<task_type<<"\""
         <<",\"tag\":"<<output_region->tag
         <<",\"seg_id\":["
             <<output_region->id.x<<","
             <<output_region->id.y<<","
             <<output_region->id.z<<"],"
         <<"\"seg_type\":"<<output_region->facet_class<<","
         <<"\"from\":\""<<from<<"\""<<","
         <<"\"to\":\""<<to<<"\""
         <<"}"<<std::endl;
}
*/

void
Task::registerDependent(Task* t, size_t offset)
{
    dependents.emplace_back(t,offset);
    t->registerPrerequisite(offset);
}

void
Task::registerPrerequisite(size_t offset)
{
    dep_cntr.max_offset = max(dep_cntr.max_offset, offset);
    dep_cntr.targets.resize(dep_cntr.max_offset+1, 0);
    dep_cntr.targets[offset]++;
}

void
Task::setIterationParams(size_t begin, size_t end)
{
    loop_cntr.i = begin;
    loop_cntr.end = end;
    
    //Ensure dependency counter has enough space to count all iterations
    dep_cntr.num_iters = max(dep_cntr.num_iters, end);
    dep_cntr.counts.resize(dep_cntr.num_iters);

    for (size_t i=0; i<dep_cntr.num_iters; i++){
        size_t num_buckets = max(i, dep_cntr.num_iters) + 1;
        dep_cntr.counts[i].resize(num_buckets,0);
    }

    //Reset counts
    for (auto& count: dep_cntr.counts){
        for (auto& bucket : count){
            bucket = 0;
        }
    }
}

bool
Task::isFinished()
{
    return loop_cntr.i == loop_cntr.end;
}

void
Task::update()
{
    if (isFinished()) return;
        
    bool ready;
    if (state == wait_state) {
        ready = true;
        for (size_t i = 0;i <= loop_cntr.i && i<= dep_cntr.max_offset; i++){
            size_t cnt = dep_cntr.counts[loop_cntr.i][i];
            size_t tgt = dep_cntr.targets[i];
            
            ready &= dep_cntr.counts[loop_cntr.i][i] >= dep_cntr.targets[i];
        }
    } else {
        ready = test();
    }

    if (ready){
        advance();
        if (state == wait_state){
            swapVBA();
            notifyDependents();
            loop_cntr.i++;
        }
    }
}

void
Task::notifyDependents()
{
    for (auto& d : dependents) {
        d.first->satisfyDependency(loop_cntr.i, d.second);
    }
}

void
Task::satisfyDependency(size_t iteration, size_t offset)
{
    if (iteration+offset < dep_cntr.num_iters){
        dep_cntr.counts[iteration+offset][offset]++;
    }
}

void
Task::syncVBA()
{
    cudaSetDevice(device->id);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        vba.in[i]  = device->vba.in[i];
        vba.out[i] = device->vba.out[i];
    }
}

void
Task::swapVBA()
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        AcReal* tmp        = vba.in[i];
        vba.in[i]  = vba.out[i];
        vba.out[i] = tmp;
    }
}

bool
Task::poll_stream()
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

/* Computation */
ComputeTask::ComputeTask(Device device_, int region_tag, int3 nn, Stream stream_id)
{
    //task_type = "compute";
    device = device_;
    stream = device->streams[stream_id];
    syncVBA();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    output_region = new Region(RegionFamily::Compute, region_tag, nn);
}

void
ComputeTask::compute()
{
    size_t substep = loop_cntr.i % 3;
    acKernelIntegrateSubstep(stream, substep, output_region->position, output_region->position + output_region->dims, vba);
}

bool
ComputeTask::test()
{
    switch (static_cast<ComputeState>(state)){
        case ComputeState::Running:
        {
            return poll_stream();
        }
        default:
            ERROR("ComputeTask in an invalid state.");
            return false;
    }
}

void
ComputeTask::advance()
{
    switch (static_cast<ComputeState>(state)){
        case ComputeState::Waiting_for_halo:
        {
            //logStateChangedEvent("waiting", "running"); 
            compute();
            state = static_cast<int>(ComputeState::Running);
            break;
        }
        case ComputeState::Running:
        {
            //logStateChangedEvent("running", "waiting"); 
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
HaloExchangeTask::HaloExchangeTask(const Device device_, const int halo_region_tag,
                                   const int3 nn, const uint3_64 decomp,
                                   MPI_Request* recv_requests, MPI_Request* send_requests)
{
    //task_type = "halo";
    device = device_;
    //Create stream for packing/unpacking
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    syncVBA();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    output_region = new Region(RegionFamily::Incoming, halo_region_tag, nn);
    outgoing_message_region = new Region(RegionFamily::Outgoing, halo_region_tag, nn);
    
    counterpart_rank = getPid(getPid3D(rank,decomp) + output_region->id, decomp);
    send_tag = outgoing_message_region->tag;
    recv_tag = Region::id_to_tag(-output_region->id);

    // Note: send_tag is also the index for both buffer sets.
    recv_buffers = new MessageBufferSwapChain();
    send_buffers = new MessageBufferSwapChain();
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        recv_buffers->add_buffer(output_region->dims, &(recv_requests[i * NUM_SEGMENTS + send_tag]));
        send_buffers->add_buffer(outgoing_message_region->dims, &(send_requests[i * NUM_SEGMENTS + send_tag]));
    }

    // Post receive immediately, this avoids unexpected messages
    active = ((MPI_INCL_CORNERS) || output_region->facet_class != 3) ? true : false;
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
    acKernelPackData(stream, vba, outgoing_message_region->position, msg->buffer);
}

void
HaloExchangeTask::unpack()
{
    
    auto msg           = recv_buffers->get_current_buffer();
    msg->buffer.pinned = false;
    acUnpinPackedData(device, stream, &(msg->buffer));
    acKernelUnpackData(stream, msg->buffer, output_region->position, vba);
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
    MPI_Irecv(msg->buffer.data, msg->length, AC_MPI_TYPE, counterpart_rank, recv_tag + HALO_TAG_OFFSET,
              MPI_COMM_WORLD, msg->request);
    msg->buffer.pinned = false;
}

void
HaloExchangeTask::receiveHost()
{
    if (onTheSameNode(rank, counterpart_rank)) {
        receiveDevice();
    }
    else {
        auto msg = recv_buffers->get_fresh_buffer();
        MPI_Irecv(msg->buffer.data_pinned, msg->length, AC_MPI_TYPE, counterpart_rank,
                  recv_tag + HALO_TAG_OFFSET, MPI_COMM_WORLD, msg->request);
        msg->buffer.pinned = true;
    }
}

void
HaloExchangeTask::sendDevice()
{
    auto msg = send_buffers->get_current_buffer();
    sync();
    MPI_Isend(msg->buffer.data, msg->length, AC_MPI_TYPE, counterpart_rank, send_tag + HALO_TAG_OFFSET,
              MPI_COMM_WORLD, msg->request);
}

void
HaloExchangeTask::sendHost()
{
    // POSSIBLE COMPAT ISSUE: is it sensible to always use CUDA memory for node-local exchanges?
    // What if the MPI lib doesn't support CUDA?
    if (onTheSameNode(rank, counterpart_rank)) {
        sendDevice();
    }
    else {
        auto msg = send_buffers->get_current_buffer();
        acPinPackedData(device, stream, &(msg->buffer));
        sync();
        MPI_Isend(msg->buffer.data_pinned, msg->length, AC_MPI_TYPE, counterpart_rank,
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
            return poll_stream();
        }
        case HaloExchangeState::Unpacking:
        {
            return poll_stream();
        }
        case HaloExchangeState::Exchanging:
        {
            auto msg = recv_buffers->get_current_buffer();
            int request_complete;
            MPI_Test(msg->request, &request_complete, MPI_STATUS_IGNORE);   
            return request_complete?true:false;
        }
        default:
        {
            ERROR("HaloExchangeTask in an invalid state.");
            return false;
        }
    }
}

void
HaloExchangeTask::advance()
{
    switch(static_cast<HaloExchangeState>(state)){
        case HaloExchangeState::Waiting_for_compute:
            //logStateChangedEvent("waiting", "packing");
            pack();
            state = static_cast<int>(HaloExchangeState::Packing);
            break;
        case HaloExchangeState::Packing:
            //logStateChangedEvent("packing", "receiving");
            sync();
            send();
            state = static_cast<int>(HaloExchangeState::Exchanging);
            break;
        case HaloExchangeState::Exchanging:
            //logStateChangedEvent("receiving", "unpacking");
            sync();
            unpack();
            state = static_cast<int>(HaloExchangeState::Unpacking);
            break;
        case HaloExchangeState::Unpacking:
            //logStateChangedEvent("unpacking", "waiting");
            receive();
            sync();
            state = static_cast<int>(HaloExchangeState::Waiting_for_compute);
            break;
        default:
            ERROR("Invalid state for HaloExchangeTask");
    }
}
#endif // AC_MPI_ENABLED
