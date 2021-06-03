#pragma once
#include "astaroth.h"

#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include "decomposition.h"   //getPid and friends
#include "kernels/kernels.h" //AcRealPacked, VertexBufferArray
#include "math_utils.h"      //max. Also included in decomposition.h

#define MPI_INCL_CORNERS (0) // Include the 3D corners of subdomains in halo

#define SWAP_CHAIN_LENGTH (2) // Swap chain lengths other than two not supported
static_assert(SWAP_CHAIN_LENGTH == 2);

struct VariableScope {
    VertexBufferHandle* variables;
    size_t num_vars;
    VariableScope(const VertexBufferHandle* variables_, const size_t num_vars_);
    ~VariableScope();
    VariableScope(const VariableScope& other) = delete;
    VariableScope& operator=(const VariableScope& other) = delete;
};

/**
 * Regions
 * -------
 *
 * Regions are identified by a non-zero region id of type {-1,0,1}^3
 * A region's id describes its position the region topology of the subdomain.
 *
 * There are three families of regions: Exchange_output message regions, outgoing
 * message regions, and compute regions. The three families each cover some
 * zone of data, as follows:
 *  - Compute_output: entire inner domain
 *  - Compute_input: the entire extended domain (including the halo)
 *  - Exchange_input: the shell of the inner domain
 *  - Exchange_output: the halo
 * The names of the families have been chosen to represent the constituent
 * regions' purpose in the context of tasks that use them.
 *
 * A triplet in {-1,0,1}^3 identifies each specific region in a family.
 * There is a mapping between integers in {-1,...,25} and the identifiers.
 * The integer form is e.g. used as part of the tag used to identify a region
 * in an MPI message.
 */

enum class RegionFamily { Exchange_output, Exchange_input, Compute_output, Compute_input };

struct Region {
    int3 position;
    int3 dims;
    size_t volume;

    RegionFamily family;
    int3 id;
    size_t facet_class;
    int tag;

    static constexpr int min_halo_tag   = 1;
    static constexpr int max_halo_tag   = 27;
    static constexpr int n_halo_regions = max_halo_tag - min_halo_tag + 1;
    static constexpr int min_comp_tag   = 0;
    static constexpr int max_comp_tag   = 27;
    static constexpr int n_comp_regions = max_comp_tag - min_comp_tag + 1;

    static int id_to_tag(int3 _id);
    static int3 tag_to_id(int _tag);

    Region(RegionFamily _family, int _tag, int3 nn);
    Region(RegionFamily _family, int3 _id, int3 nn);
    bool overlaps(const Region* other);
};

/**
 * Task interface
 * --------------
 *
 * Each task is tied to an output region (and input regions, but they are not explicit members of
 * Task). Tasks may depend on other Tasks. The existence of dependency between two tasks is deduced
 * from their input and output regions. At the moment, this is done explicitly by comparing region
 * ids and happens in grid.cc:GridInit()
 */
typedef class Task {
  protected:
    std::vector<std::pair<std::weak_ptr<Task>, size_t>> dependents;
    Device device;
    cudaStream_t stream;
    VertexBufferArray vba;

    int state;

    struct {
        std::vector<size_t> counts;
        std::vector<size_t> targets;
    } dep_cntr;

    struct {
        size_t i;
        size_t end;
    } loop_cntr;

    bool poll_stream();

  public:
    int rank;
    bool active;
    std::string name;
    TaskType task_type;

    int order; // the ordinal position of the task in a serial execution (within its region)
    std::unique_ptr<Region> output_region;
    std::unique_ptr<Region> input_region;
    std::shared_ptr<VariableScope> variable_scope;

    static const int wait_state = 0;

    Task(RegionFamily input_family, RegionFamily output_family, int region_tag, int3 nn);
    virtual bool test()    = 0;
    virtual void advance() = 0;

    void registerDependent(std::shared_ptr<Task> t, size_t offset);
    void registerPrerequisite(size_t offset);
    bool isPrerequisiteTo(std::shared_ptr<Task> other);

    void setIterationParams(size_t begin, size_t end);
    void update();
    bool isFinished();

    void notifyDependents();
    void satisfyDependency(size_t iteration);

    void syncVBA();
    void swapVBA();

    // void logStateChangedEvent(std::string b, std::string c);
} Task;

// Compute tasks
enum class ComputeState { Waiting_for_halo = Task::wait_state, Running };

typedef class ComputeTask : public Task {
  private:
    ComputeKernel compute_func;
    KernelParameters params;

  public:
    ComputeTask(ComputeKernel compute_func_, std::shared_ptr<VariableScope> variable_scope_,
                int order_, int region_tag, int3 nn, Device device_);
    ComputeTask(const ComputeTask& other) = delete;
    ComputeTask& operator=(const ComputeTask& other) = delete;
    void compute();
    void advance();
    bool test();
} ComputeTask;

// Communication
typedef struct HaloMessage {
    int length;
    AcRealPacked* data;
#if !(USE_CUDA_AWARE_MPI)
    AcRealPacked* data_pinned;
    bool pinned = false; // Set if data was received to pinned memory
#endif
    MPI_Request request;

    HaloMessage(int3 dims, size_t num_vars);
    ~HaloMessage();
#if !(USE_CUDA_AWARE_MPI)
    void pin(const Device device, const cudaStream_t stream);
    void unpin(const Device device, const cudaStream_t stream);
#endif
} HaloMessage;

typedef struct HaloMessageSwapChain {
    int buf_idx;
    std::vector<HaloMessage> buffers;

    HaloMessageSwapChain();
    HaloMessageSwapChain(int3 dims, size_t num_vars);

    HaloMessage* get_current_buffer();
    HaloMessage* get_fresh_buffer();
} HaloMessageSwapChain;

enum class HaloExchangeState {
    Waiting_for_compute = Task::wait_state,
    Packing,
    Exchanging,
    Unpacking
};

typedef class HaloExchangeTask : public Task {
  private:
    int counterpart_rank;
    int send_tag;
    int recv_tag;

    HaloMessageSwapChain recv_buffers;
    HaloMessageSwapChain send_buffers;

  public:
    HaloExchangeTask(std::shared_ptr<VariableScope> variable_scope_, int order_, int tag_0,
                     int halo_region_tag, int3 nn, uint3_64 decomp, Device device_);
    ~HaloExchangeTask();
    HaloExchangeTask(const HaloExchangeTask& other) = delete;
    HaloExchangeTask& operator=(const HaloExchangeTask& other) = delete;

    void sync();
    void wait_send();
    void wait_recv();

    void pack();
    void unpack();

    void send();
    void receive();
    void exchange();

    void sendDevice();
    void receiveDevice();
    void exchangeDevice();

#if !(USE_CUDA_AWARE_MPI)
    void sendHost();
    void receiveHost();
    void exchangeHost();
#endif

    void advance();
    bool test();
} HaloExchangeTask;

struct TaskGraph {
    std::vector<std::shared_ptr<Task>> all_tasks;
    std::vector<std::shared_ptr<ComputeTask>> comp_tasks;
    std::vector<std::shared_ptr<HaloExchangeTask>> halo_tasks;
};
/*
enum class TaskType { Compute, HaloExchange};
enum class Kernel {solve};
enum class BoundaryCondition {Periodic};

typedef struct TaskDefinition {
    TaskType task_type;
    union {
        Kernel kernel;
        BoundaryCondition bound_cond;
    };
    VertexBufferHandle variable_scope[];
    size_t num_vars;
} TaskDefinition;

typedef struct Compute: public TaskDefinition {
    template<size_t n>
    Compute(VertexBufferHandle (&variable_scope_arr)[n], Kernel kernel_)
    {
        task_type = TaskType::Compute;
        variable_scope.insert(variable_scope.end(), &variable_scope_arr[0], &variable_scope_arr[n]);
        kernel = kernel_;
    }
} ComputeTaskDefinition;

typedef struct HaloExchange: public TaskDefinition {
    template<size_t n>
    HaloExchange(VertexBufferHandle (&variable_scope_arr)[n], BoundaryCondition bound_cond_)
    {
        task_type = TaskType::HaloExchange;
        variable_scope.insert(variable_scope.end(), &variable_scope_arr[0], &variable_scope_arr[n]);
        bound_cond = bound_cond_;
    }
} HaloExchangeTaskDefinition;
*/
