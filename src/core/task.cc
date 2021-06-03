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
 * Vv.  HaloExchangeTasks may depend on ComputeTasks because they're waiting for data to send.
 *
 * This all happens in grid.cc:GridIntegrate
 */

#include "task.h"
#include "astaroth.h"

#include <memory>
#include <mpi.h>
#include <vector>

#include "decomposition.h"   //getPid and friends
#include "kernels/kernels.h" //AcRealPacked, ComputeKernel

#define HALO_TAG_OFFSET (100) //"Namespacing" the MPI tag space to avoid collisions

TaskDefinition
Compute(const Kernel kernel, VertexBufferHandle variable_scope_arr[], const size_t num_vars)
{
    TaskDefinition task_def;
    task_def.task_type   = TaskType_Compute;
    task_def.kernel      = kernel;
    task_def.variables   = variable_scope_arr;
    task_def.num_vars    = num_vars;
    return task_def;
}

TaskDefinition
HaloExchange(const BoundaryCondition bound_cond, VertexBufferHandle variable_scope_arr[],
             const size_t num_vars)
{
    TaskDefinition task_def;
    task_def.task_type   = TaskType_HaloExchange;
    task_def.bound_cond  = bound_cond;
    task_def.variables   = variable_scope_arr;
    task_def.num_vars    = num_vars;
    return task_def;
}

VariableScope::VariableScope(const VertexBufferHandle h_variables[], const size_t num_vars_)
    : num_vars(num_vars_)
{
    variables         = NULL;
    size_t scope_size = num_vars * sizeof(VertexBufferHandle);
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&variables, scope_size));
    ERRCHK_CUDA_ALWAYS(cudaMemcpyAsync(variables, h_variables, scope_size, cudaMemcpyHostToDevice));
}

VariableScope::~VariableScope()
{
    cudaFree(variables);
    variables     = NULL;
    num_vars = -1;
}

Region::Region(RegionFamily _family, int _tag, int3 nn) : family(_family), tag(_tag)
{
    id          = tag_to_id(tag);
    facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
    ERRCHK_ALWAYS(facet_class <= 3);

    switch (family) {
    case RegionFamily::Compute_output: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? NGHOST : id.x == 1 ? nn.x : NGHOST * 2,
                    id.y == -1  ? NGHOST : id.y == 1 ? nn.y : NGHOST * 2,
                    id.z == -1  ? NGHOST : id.z == 1 ? nn.z : NGHOST * 2};
        // clang-format on
        dims = (int3){id.x == 0 ? nn.x - NGHOST * 2 : NGHOST,
                      id.y == 0 ? nn.y - NGHOST * 2 : NGHOST,
                      id.z == 0 ? nn.z - NGHOST * 2 : NGHOST};
        break;
    }
    case RegionFamily::Compute_input: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? 0 : id.x == 1 ? nn.x - NGHOST : NGHOST ,
                    id.y == -1  ? 0 : id.y == 1 ? nn.y - NGHOST : NGHOST ,
                    id.z == -1  ? 0 : id.z == 1 ? nn.z - NGHOST : NGHOST };
        // clang-format on
        dims = (int3){id.x == 0 ? nn.x : NGHOST * 3, id.y == 0 ? nn.y : NGHOST * 3,
                      id.z == 0 ? nn.z : NGHOST * 3};
        break;
    }
    case RegionFamily::Exchange_output: {
        // clang-format off
        position = (int3){
                    id.x == -1  ? 0 : id.x == 1 ? NGHOST + nn.x : NGHOST,
                    id.y == -1  ? 0 : id.y == 1 ? NGHOST + nn.y : NGHOST,
                    id.z == -1  ? 0 : id.z == 1 ? NGHOST + nn.z : NGHOST};
        // clang-format on
        dims = (int3){id.x == 0 ? nn.x : NGHOST, id.y == 0 ? nn.y : NGHOST,
                      id.z == 0 ? nn.z : NGHOST};
        break;
    }
    case RegionFamily::Exchange_input: {
        position = (int3){id.x == 1 ? nn.x : NGHOST, id.y == 1 ? nn.y : NGHOST,
                          id.z == 1 ? nn.z : NGHOST};
        dims = (int3){id.x == 0 ? nn.x : NGHOST, id.y == 0 ? nn.y : NGHOST,
                      id.z == 0 ? nn.z : NGHOST};
        break;
    }
    default: {
        ERROR("Unknown region family.");
    }
    }
    volume = dims.x * dims.y * dims.z;
}

Region::Region(RegionFamily _family, int3 _id, int3 nn) : Region{_family, id_to_tag(_id), nn}
{
    ERRCHK_ALWAYS(_id.x == id.x && _id.y == id.y && _id.z == id.z);
}

bool
Region::overlaps(const Region* other)
{
    return (this->position.x < other->position.x + other->dims.x) &&
           (other->position.x < this->position.x + this->dims.x) &&
           (this->position.y < other->position.y + other->dims.y) &&
           (other->position.y < this->position.y + this->dims.y) &&
           (this->position.z < other->position.z + other->dims.z) &&
           (other->position.z < this->position.z + this->dims.z);
}

int
Region::id_to_tag(int3 _id)
{
    return ((3 + _id.x) % 3) * 9 + ((3 + _id.y) % 3) * 3 + (3 + _id.z) % 3;
}

int3
Region::tag_to_id(int _tag)
{
    int3 _id = (int3){(_tag) / 9, ((_tag) % 9) / 3, (_tag) % 3};
    _id.x    = _id.x == 2 ? -1 : _id.x;
    _id.y    = _id.y == 2 ? -1 : _id.y;
    _id.z    = _id.z == 2 ? -1 : _id.z;
    ERRCHK_ALWAYS(id_to_tag(_id) == _tag);
    return _id;
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

Task::Task(RegionFamily input_family, RegionFamily output_family, int region_tag, int3 nn)
:state(wait_state), dep_cntr(), loop_cntr(),
 output_region(std::make_unique<Region>(output_family, region_tag, nn)),
 input_region(std::make_unique<Region>(input_family, region_tag, nn))
{

 //output_region = std::make_unique<Region>(output_family, region_tag, nn);
 //input_region = std::make_unique<Region>(input_family, region_tag, nn);
}

void
Task::registerDependent(std::shared_ptr<Task> t, size_t offset)
{
    dependents.emplace_back(t, offset);
    t->registerPrerequisite(offset);
}

void
Task::registerPrerequisite(size_t offset)
{
    // Ensure targets exist
    if (offset >= dep_cntr.targets.size()) {
        size_t initial_val = dep_cntr.targets.empty() ? 0 : dep_cntr.targets.back();
        dep_cntr.targets.resize(offset + 1, initial_val);
    }
    for (; offset < dep_cntr.targets.size(); offset++) {
        dep_cntr.targets[offset]++;
    }
}

bool
Task::isPrerequisiteTo(std::shared_ptr<Task> other)
{
    for (auto dep : dependents) {
        if (dep.first.lock() == other) {
            return true;
        }
    }
    return false;
}

void
Task::setIterationParams(size_t begin, size_t end)
{
    loop_cntr.i   = begin;
    loop_cntr.end = end;

    // Reset dependency counter, and ensure it has enough space
    dep_cntr.counts.resize(0);
    dep_cntr.counts.resize(end, 0);
}

bool
Task::isFinished()
{
    return loop_cntr.i == loop_cntr.end;
}

void
Task::update()
{
    if (isFinished())
        return;

    bool ready;
    if (state == wait_state) {
        size_t offset = min(loop_cntr.i, dep_cntr.targets.size() - 1);
        size_t target = dep_cntr.targets[offset];
        size_t count  = dep_cntr.counts[loop_cntr.i];
        ready         = (count == target);
    }
    else {
        ready = test();
    }

    if (ready) {
        advance();
        if (state == wait_state) {
            swapVBA();
            notifyDependents();
            loop_cntr.i++;
        }
    }
}

void
Task::notifyDependents()
{
    for (auto& dep : dependents) {
        std::shared_ptr<Task> dependent = dep.first.lock();
        dependent->satisfyDependency(loop_cntr.i + dep.second);
    }
}

void
Task::satisfyDependency(size_t iteration)
{
    if (iteration < loop_cntr.end) {
        dep_cntr.counts[iteration]++;
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
        AcReal* tmp = vba.in[i];
        vba.in[i]   = vba.out[i];
        vba.out[i]  = tmp;
    }
}

bool
Task::poll_stream()
{
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess) {
        return true;
    }
    if (err == cudaErrorNotReady) {
        return false;
    }
    return false;
}

/* Computation */
ComputeTask::ComputeTask(ComputeKernel compute_func_,
                         std::shared_ptr<VariableScope> variable_scope_, int order_, int region_tag,
                         int3 nn, Device device_)
:Task(RegionFamily::Compute_input, RegionFamily::Compute_output, region_tag, nn)
{
    device = device_;
    stream = device->streams[STREAM_DEFAULT + region_tag];
    syncVBA();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    active         = true;
    order          = order_;
    compute_func   = compute_func_;
    variable_scope = variable_scope_;

    params = KernelParameters{stream, 0, output_region->position,
                              output_region->position + output_region->dims};
    name   = "Compute(" + std::to_string(output_region->id.x) + "," +
           std::to_string(output_region->id.y) + "," + std::to_string(output_region->id.z) + ")";
    task_type = TaskType_Compute;
}

void
ComputeTask::compute()
{
    // IDEA: we could make loop_cntr.i point at params.step_number
    params.step_number = (int)(loop_cntr.i % 3);
    compute_func(params, vba);
}

bool
ComputeTask::test()
{
    switch (static_cast<ComputeState>(state)) {
    case ComputeState::Running: {
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
    switch (static_cast<ComputeState>(state)) {
    case ComputeState::Waiting_for_halo: {
        // logStateChangedEvent("waiting", "running");
        compute();
        state = static_cast<int>(ComputeState::Running);
        break;
    }
    case ComputeState::Running: {
        // logStateChangedEvent("running", "waiting");
        state = static_cast<int>(ComputeState::Waiting_for_halo);
        break;
    }
    }
}

/*  Communication   */

// HaloMessage contains all information needed to send or receive a single message
// Wraps PackedData. These two structs could be folded together.
HaloMessage::HaloMessage(int3 dims, size_t num_vars)
{
    length       = dims.x * dims.y * dims.z * num_vars;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data, bytes));
#if !(USE_CUDA_AWARE_MPI)
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data_pinned, bytes));
#endif
    request = MPI_REQUEST_NULL;
}

HaloMessage::~HaloMessage()
{
    if (request != MPI_REQUEST_NULL) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    length = -1;
    cudaFree(data);
#if !(USE_CUDA_AWARE_MPI)
    cudaFree(data_pinned);
#endif
    data = NULL;
}

#if !(USE_CUDA_AWARE_MPI)
void
HaloMessage::pin(const Device device, const cudaStream_t stream)
{
    cudaSetDevice(device->id);
    pinned       = true;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA(cudaMemcpyAsync(data_pinned, data, bytes, cudaMemcpyDefault, stream));
}

void
HaloMessage::unpin(const Device device, const cudaStream_t stream)
{
    if (!pinned)
        return;

    cudaSetDevice(device->id);
    pinned       = false;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA(cudaMemcpyAsync(data, data_pinned, bytes, cudaMemcpyDefault, stream));
}
#endif

// HaloMessageSwapChain
HaloMessageSwapChain::HaloMessageSwapChain(){}

HaloMessageSwapChain::HaloMessageSwapChain(int3 dims, size_t num_vars)
: buf_idx(SWAP_CHAIN_LENGTH - 1)
{
    buffers.reserve(SWAP_CHAIN_LENGTH);
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        buffers.emplace_back(dims, num_vars);
    }
}

HaloMessage*
HaloMessageSwapChain::get_current_buffer()
{
    return &buffers[buf_idx];
}

HaloMessage*
HaloMessageSwapChain::get_fresh_buffer()
{
    buf_idx         = (buf_idx + 1) % SWAP_CHAIN_LENGTH;
    MPI_Request req = buffers[buf_idx].request;
    if (req != MPI_REQUEST_NULL) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    return &buffers[buf_idx];
}

// HaloExchangeTask
HaloExchangeTask::HaloExchangeTask(std::shared_ptr<VariableScope> variable_scope_, int order_,
                                   int tag_0, int halo_region_tag, int3 nn, uint3_64 decomp,
                                   Device device_)
:Task(RegionFamily::Exchange_input, RegionFamily::Exchange_output, halo_region_tag, nn),
  recv_buffers(output_region->dims, variable_scope_->num_vars),
  send_buffers(input_region->dims, variable_scope_->num_vars)
{
    device = device_;
    // Create stream for packing/unpacking
    {
        cudaSetDevice(device->id);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    syncVBA();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    order         = order_;

    variable_scope = variable_scope_;

    counterpart_rank = getPid(getPid3D(rank, decomp) + output_region->id, decomp);
    // MPI tags are namespaced to avoid collisions with other MPI tasks
    send_tag = tag_0 + input_region->tag;
    recv_tag = tag_0 + Region::id_to_tag(-output_region->id);

    // Post receive immediately, this avoids unexpected messages
    active = ((MPI_INCL_CORNERS) || output_region->facet_class != 3) ? true : false;
    if (active) {
        receive();
    }
    name = "Halo exchange(" + std::to_string(output_region->id.x) + "," +
           std::to_string(output_region->id.y) + "," + std::to_string(output_region->id.z) + ")";
    task_type = TaskType_HaloExchange;
}

HaloExchangeTask::~HaloExchangeTask()
{
    // Cancel last eager request
    auto msg = recv_buffers.get_current_buffer();
    if (msg->request != MPI_REQUEST_NULL) {
        MPI_Cancel(&msg->request);
    }

    cudaSetDevice(device->id);
    // dependents.clear();
    cudaStreamDestroy(stream);
}

void
HaloExchangeTask::pack()
{
    auto msg = send_buffers.get_fresh_buffer();
    // acKernelPartialPackData(stream, vba, input_region->position, input_region->dims,
    //                 msg->data, pass->variable_scope, pass->variable_scope_length);
    acKernelPackData(stream, vba, input_region->position, input_region->dims, msg->data);
}

void
HaloExchangeTask::unpack()
{

    auto msg = recv_buffers.get_current_buffer();
#if !(USE_CUDA_AWARE_MPI)
    msg->unpin(device, stream);
#endif
    // acKernelPartialUnpackData(stream, msg->data, output_region->position, output_region->dims,
    // vba,
    //                          pass->variable_scope, pass->variable_scope_length);
    acKernelUnpackData(stream, msg->data, output_region->position, output_region->dims, vba);
}

void
HaloExchangeTask::sync()
{
    cudaStreamSynchronize(stream);
}

void
HaloExchangeTask::wait_recv()
{
    auto msg = recv_buffers.get_current_buffer();
    MPI_Wait(&msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::wait_send()
{
    auto msg = send_buffers.get_current_buffer();
    MPI_Wait(&msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::receiveDevice()
{
    auto msg = recv_buffers.get_fresh_buffer();
    MPI_Irecv(msg->data, msg->length, AC_MPI_TYPE, counterpart_rank, recv_tag + HALO_TAG_OFFSET,
              MPI_COMM_WORLD, &msg->request);
}

void
HaloExchangeTask::sendDevice()
{
    auto msg = send_buffers.get_current_buffer();
    sync();
    MPI_Isend(msg->data, msg->length, AC_MPI_TYPE, counterpart_rank, send_tag + HALO_TAG_OFFSET,
              MPI_COMM_WORLD, &msg->request);
}

void
HaloExchangeTask::exchangeDevice()
{
    // cudaSetDevice(device->id);
    receiveDevice();
    sendDevice();
}

#if !(USE_CUDA_AWARE_MPI)
void
HaloExchangeTask::receiveHost()
{
    auto msg = recv_buffers.get_fresh_buffer();
    MPI_Irecv(msg->data_pinned, msg->length, AC_MPI_TYPE, counterpart_rank,
              recv_tag + HALO_TAG_OFFSET, MPI_COMM_WORLD, &msg->request);
    msg->pinned = true;
}

void
HaloExchangeTask::sendHost()
{
    auto msg = send_buffers.get_current_buffer();
    msg->pin(device, stream);
    sync();
    MPI_Isend(msg->data_pinned, msg->length, AC_MPI_TYPE, counterpart_rank,
              send_tag + HALO_TAG_OFFSET, MPI_COMM_WORLD, &msg->request);
}
void
HaloExchangeTask::exchangeHost()
{
    // cudaSetDevice(device->id);
    receiveHost();
    sendHost();
}
#endif

void
HaloExchangeTask::receive()
{
#if USE_CUDA_AWARE_MPI
    receiveDevice();
#else
    receiveHost();
#endif
}

void
HaloExchangeTask::send()
{
#if USE_CUDA_AWARE_MPI
    sendDevice();
#else
    sendHost();
#endif
}

void
HaloExchangeTask::exchange()
{
#if USE_CUDA_AWARE_MPI
    exchangeDevice();
#else
    exchangeHost();
#endif
}

bool
HaloExchangeTask::test()
{
    switch (static_cast<HaloExchangeState>(state)) {
    case HaloExchangeState::Packing: {
        return poll_stream();
    }
    case HaloExchangeState::Unpacking: {
        return poll_stream();
    }
    case HaloExchangeState::Exchanging: {
        auto msg = recv_buffers.get_current_buffer();
        int request_complete;
        MPI_Test(&msg->request, &request_complete, MPI_STATUS_IGNORE);
        return request_complete ? true : false;
    }
    default: {
        ERROR("HaloExchangeTask in an invalid state.");
        return false;
    }
    }
}

void
HaloExchangeTask::advance()
{
    switch (static_cast<HaloExchangeState>(state)) {
    case HaloExchangeState::Waiting_for_compute:
        // logStateChangedEvent("waiting", "packing");
        pack();
        state = static_cast<int>(HaloExchangeState::Packing);
        break;
    case HaloExchangeState::Packing:
        // logStateChangedEvent("packing", "receiving");
        sync();
        send();
        state = static_cast<int>(HaloExchangeState::Exchanging);
        break;
    case HaloExchangeState::Exchanging:
        // logStateChangedEvent("receiving", "unpacking");
        sync();
        unpack();
        state = static_cast<int>(HaloExchangeState::Unpacking);
        break;
    case HaloExchangeState::Unpacking:
        // logStateChangedEvent("unpacking", "waiting");
        receive();
        sync();
        state = static_cast<int>(HaloExchangeState::Waiting_for_compute);
        break;
    default:
        ERROR("Invalid state for HaloExchangeTask");
    }
}
#endif // AC_MPI_ENABLED
