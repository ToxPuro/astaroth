/*
    Copyright (C) 2020, Oskar Lappi

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include "astaroth.h"

#include <array>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include "decomposition.h"   //getPid and friends
#include "kernels/kernels.h" //AcRealPacked, VertexBufferArray
#include "math_utils.h"      //max. Also included in decomposition.h
#include "timer_hires.h"


#define SWAP_CHAIN_LENGTH (2) // Swap chain lengths other than two not supported
static_assert(SWAP_CHAIN_LENGTH == 2);

//TP: TODO: move somewhere more appropriate
typedef struct {
    AcKernel kernel_enum;
    cudaStream_t stream;
    int step_number;
    Volume start;
    Volume end;
    #if AC_MPI_ENABLED
    LoadKernelParamsFunc* load_func;
    #endif
} KernelParameters;

struct TraceFile;

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
 * There is a mapping between integers in {0,...,27} and the identifiers.
 * The integer form is e.g. used as part of the tag used to identify a region
 * in an MPI message.
 */

enum class RegionFamily { Exchange_output, Exchange_input, Compute_output, Compute_input, None };

typedef struct
{
	std::vector<Field> fields;
	std::vector<Profile> profiles;
	std::vector<KernelReduceOutput> reduce_outputs;
} RegionMemory;

typedef struct
{
	const Field*   fields;
	const size_t   num_fields;
	const Profile* profiles;
	const size_t   num_profiles;
	const KernelReduceOutput* reduce_outputs;
	const size_t  num_reduce_outputs;
} RegionMemoryInputParams;

struct Region {
    Volume position;
    Volume dims;
    size_t volume;

    RegionFamily family;
    int3 id;
    int tag;

    RegionMemory memory;

    // facet class 0 = inner core
    // facet class 1 = face
    // facet class 2 = edge
    // facet class 3 = corner
    size_t facet_class;

    static constexpr int min_halo_tag   = 1;
    static constexpr int max_halo_tag   = 27;
    static constexpr int n_halo_regions = max_halo_tag - min_halo_tag + 1;
    static constexpr int min_comp_tag   = 0;
    static constexpr int max_comp_tag   = 27;
    static constexpr int n_comp_regions = max_comp_tag - min_comp_tag + 1;

    static int id_to_tag(int3 id);
    static int3 tag_to_id(int tag);

    static AcBoundary boundary(uint3_64 decomp, int pid, int tag, AcProcMappingStrategy proc_mapping_strategy);
    static AcBoundary boundary(uint3_64 decomp, int3 pid3d, int3 id);
    static bool is_on_boundary(uint3_64 decomp, int pid, int tag, AcBoundary boundary, AcProcMappingStrategy proc_mapping_strategy);
    static bool is_on_boundary(uint3_64 decomp, int3 pid3d, int3 id, AcBoundary boundary);

    Region(RegionFamily family_, int tag_, const AcBoundary depends_on_boundary, const AcBoundary computes_on_boundary, Volume nn, const RegionMemoryInputParams);
    Region(RegionFamily family_, int3 id_, Volume nn, const RegionMemoryInputParams);
    Region(Volume position_, Volume dims_, int tag_, const RegionMemory mem_);
    Region(Volume position_, Volume dims_, int tag_, const RegionMemory mem_, RegionFamily family_);

    Region translate(int3 translation);
    bool overlaps(const Region* other) const;
    AcBool3 geometry_overlaps(const Region* other) const;
    bool fields_overlap(const Region* other) const;
    AcBoundary boundary(uint3_64 decomp, int pid, AcProcMappingStrategy proc_mapping_strategy);
    bool is_on_boundary(uint3_64 decomp, int pid, AcBoundary boundary, AcProcMappingStrategy proc_mapping_strategy);
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
    Device device;
    cudaStream_t stream;
    VertexBufferArray vba;
    std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset;

    int state;

  public:
    std::vector<std::pair<std::weak_ptr<Task>, size_t>> dependents;
    struct {
        std::vector<size_t> counts;
        std::vector<size_t> targets;
    } dep_cntr;

    struct {
        size_t i;
        size_t end;
    } loop_cntr;

    int rank;  // MPI rank
    int order; // the ordinal position of the task in a serial execution (within its region)
    bool active;
    std::string name;
    AcTaskType task_type;
    AcBoundary boundary; // non-zero if a boundary condition task, indicating which boundary

    Region input_region;
    Region output_region;

    std::vector<AcRealParam> input_parameters;

    static const int wait_state = 0;

  protected:
    bool poll_stream();

  public:
    Task(int order_, Region input_region_, Region output_region, AcTaskDefinition op,
         Device device_, std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);
    virtual ~Task() {};

    virtual bool test()                               = 0;
    virtual void advance(const TraceFile* trace_file) = 0;

    void registerDependent(std::shared_ptr<Task> t, size_t offset);
    void registerPrerequisite(size_t offset);
    bool isPrerequisiteTo(std::shared_ptr<Task> other);

    void setIterationParams(size_t begin, size_t end);
    void update(std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> vtxbuf_swaps, const TraceFile* trace_file);
    bool isFinished();

    void notifyDependents();
    void satisfyDependency(size_t iteration);

    void syncVBA();
    void swapVBA(std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> vtxbuf_swaps);

    void logStateChangedEvent(const char* from, const char* to);
    virtual bool isComputeTask();
    virtual bool isHaloExchangeTask();
    bool swaps_overlap(const Task* other);
} Task;

// Compute tasks
enum class ComputeState { Waiting = Task::wait_state, Running };

typedef class ComputeTask : public Task {
  private:
    // ComputeKernel compute_func;
    KernelParameters params;

  public:
    ComputeTask(AcTaskDefinition op, int order_, int region_tag, Volume nn, Device device_,
                std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);
    ComputeTask(AcTaskDefinition op, int order_, Region input_region, Region output_region, Device device_,std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);

    ComputeTask(const ComputeTask& other)            = delete;
    ComputeTask& operator=(const ComputeTask& other) = delete;
    void compute();
    void advance(const TraceFile* trace_file);
    bool test();
    bool isComputeTask();
    AcKernel getKernel();
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
    int tag;
    int non_namespaced_tag;

    HaloMessage(Volume dims, size_t num_vars, const int tag0, const int tag);
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
    HaloMessageSwapChain(Volume dims, size_t num_vars, const int tag0, const int tag);

    HaloMessage* get_current_buffer();
    HaloMessage* get_fresh_buffer();
} HaloMessageSwapChain;

enum class HaloExchangeState { Waiting = Task::wait_state, Packing, Exchanging, Unpacking, Moving };

typedef class HaloExchangeTask : public Task {
  private:
    int counterpart_rank;

    HaloMessageSwapChain recv_buffers;
    HaloMessageSwapChain send_buffers;

  public:
    HaloExchangeTask(AcTaskDefinition op, int order_, int tag_0, int halo_region_tag, AcGridInfo grid_info,
                     uint3_64 decomp, Device device_,
                     std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);
    ~HaloExchangeTask();
    HaloExchangeTask(const HaloExchangeTask& other)            = delete;
    HaloExchangeTask& operator=(const HaloExchangeTask& other) = delete;

    void sync();
    void wait_send();
    void wait_recv();

    void pack();
    void unpack();

    void move();

    void send();
    void receive();
    void exchange();

    void sendDevice();
    void receiveDevice();
    void exchangeDevice();
    bool sendingToItself();

#if !(USE_CUDA_AWARE_MPI)
    void sendHost();
    void receiveHost();
    void exchangeHost();
#endif

    void advance(const TraceFile* trace_file);
    bool test();
    bool isHaloExchangeTask();
} HaloExchangeTask;

typedef class SyncTask : public Task {
  public:
    SyncTask(AcTaskDefinition op, int order_, Volume nn, Device device_,
             std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);
    void advance(const TraceFile* trace_file);
    bool test();
} SyncTask;


enum class BoundaryConditionState { Waiting = Task::wait_state, Running };
typedef class BoundaryConditionTask : public Task {
  private:
    KernelParameters params;
    int3 boundary_normal;
    Volume boundary_dims;
    bool fieldwise;

  public:
    BoundaryConditionTask(AcTaskDefinition op, int3 boundary_normal_, int order_,
                                    int region_tag, Volume nn, Device device_,
                                    std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);
    void populate_boundary_region();
    void advance(const TraceFile* trace_file);
    bool test();
} BoundaryConditionTask;

enum class ReduceState { Waiting = Task::wait_state, Reducing, Communicating, Loading };
typedef class ReduceTask : public Task {
  private:
    AcReal local_res_real[NUM_OUTPUTS]{};
    int    local_res_int[NUM_OUTPUTS]{};
#if AC_DOUBLE_PRECISION
    float  local_res_float[NUM_OUTPUTS]{};
#endif
    MPI_Request requests[NUM_OUTPUTS+NUM_PROFILES]{};
    AcProfileType reduces_only_prof{};
  public:
    ReduceTask(AcTaskDefinition op, int order_, int region_tag, Volume nn, Device device_,
                std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_);
    void reduce();
    void communicate();
    void load_outputs();
    void advance(const TraceFile* trace_file);
    bool test();
} ReduceTask;




// A TaskGraph is a graph structure of tasks that will be executed
// The tasks have dependencies, which are defined both within an iteration and between iterations
// This allows the graph to be executed for any number of iterations

struct TraceFile {
    bool enabled;
    std::string filepath;
    FILE* fp;
    Timer timer;
    void trace(const Task* task, const std::string old_state, const std::string new_state) const;
};

struct AcTaskGraph {
    std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> device_swaps;
    std::vector<std::shared_ptr<Task>> all_tasks;
    std::vector<std::shared_ptr<ComputeTask>> comp_tasks;
    std::vector<std::shared_ptr<HaloExchangeTask>> halo_tasks;

    AcBoundary periodic_boundaries;

    TraceFile trace_file;
};

AcBoundary boundary_from_normal(int3 normal);
int3 normal_from_boundary(AcBoundary boundary);
AcTaskDefinition convert_iter_to_normal_compute(AcTaskDefinition op, int step_num);
typedef struct LoadKernelParamsFunc{std::function<void(ParamLoadingInfo)> loader;} LoadKernelParamsFunc;
