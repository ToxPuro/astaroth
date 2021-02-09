#pragma once
#include "astaroth.h"

#include <mpi.h>
#include <vector>
#include <string>

#include "math_utils.h"      //max. Also included in decomposition.h
#include "decomposition.h"   //getPid and friends
#include "kernels/kernels.h" //PackedData, VertexBufferArray

#define MPI_USE_PINNED (0)   // Do inter-node comm with pinned (host) memory
#define MPI_INCL_CORNERS (0) // Include the 3D corners of subdomains in halo

#define NUM_SEGMENTS (26)

// clang-format off
#if MPI_INCL_CORNERS
    #define NUM_ACTIVE_SEGMENTS (26)
#else
    #define NUM_ACTIVE_SEGMENTS (18)
#endif
// clang-format on

#define SWAP_CHAIN_LENGTH (2) // Swap chain lengths other than two not supported
static_assert(SWAP_CHAIN_LENGTH == 2);

/*   Segments   */

/**
 * Segments are identified by a non-zero segment id of type {-1,0,1}^3
 * A segment's id describes its position the segment topology of the subdomain
 *
 * The first two functions are maps to and from {-1,0,1}^3 - space and counting numbers
 * The rest of the functions use the segment id to find properties of the segment
 */

// clang-format off
static inline int
segment_index(const int3 seg_id)
{
    return ((3+seg_id.x)%3)*9 + ((3+seg_id.y)%3)*3 + (3+seg_id.z)%3 - 1;
}

static inline int3
index_to_segment_id(int index)
{
    int3 segment_id = (int3){
                        ( index + 1)      / 9,
                        ((index + 1) % 9) / 3,
                        ( index + 1) % 3  / 1 };
    segment_id.x    = segment_id.x == 2 ? -1 : segment_id.x;
    segment_id.y    = segment_id.y == 2 ? -1 : segment_id.y;
    segment_id.z    = segment_id.z == 2 ? -1 : segment_id.z;
    ERRCHK_ALWAYS(segment_index(segment_id) == index);
    return segment_id;
}

static inline int3
halo_segment_position(const int3 seg_id, const int3 grid_dimensions)
{
    return (int3){
        seg_id.x == -1 ? 0 : seg_id.x == 1 ? NGHOST + grid_dimensions.x : NGHOST,
        seg_id.y == -1 ? 0 : seg_id.y == 1 ? NGHOST + grid_dimensions.y : NGHOST,
        seg_id.z == -1 ? 0 : seg_id.z == 1 ? NGHOST + grid_dimensions.z : NGHOST,
    };
}

static inline int3
local_segment_position(const int3 seg_id, const int3 grid_dimensions)
{
    return (int3){
        seg_id.x == 1 ? grid_dimensions.x : NGHOST,
        seg_id.y == 1 ? grid_dimensions.y : NGHOST,
        seg_id.z == 1 ? grid_dimensions.z : NGHOST,
    };
}

static inline int3
segment_dims(const int3 seg_id, const int3 grid_dimensions)
{
    return (int3){
        seg_id.x == 0 ? grid_dimensions.x : NGHOST,
        seg_id.y == 0 ? grid_dimensions.y : NGHOST,
        seg_id.z == 0 ? grid_dimensions.z : NGHOST,
    };
}

// Corner=3, Edge=2, Face=1, Core=0
static inline int
segment_type(const int3 seg_id)
{
    int seg_type = (seg_id.x == 0 ? 0 : 1) +
                   (seg_id.y == 0 ? 0 : 1) +
                   (seg_id.z == 0 ? 0 : 1);
    ERRCHK_ALWAYS(seg_type <= 3 && seg_type >= 0);
    return seg_type;
}
// clang-format on

/* Task interface */

struct DependencyCounter {
private:
    size_t num_iterations;
    size_t max_offset;
    std::vector<size_t> targets;
    std::vector<std::vector<size_t>> counts;
public:

    void
    extend_to(size_t iterations, size_t offset)
    {
        max_offset = max(offset,max_offset);
        targets.resize(max_offset+1, 0);

        num_iterations = max(iterations,num_iterations);
        counts.resize(num_iterations);

        for (size_t i=0; i<num_iterations; i++){
            size_t num_buckets = max(i, num_iterations) + 1;
            counts[i].resize(num_buckets,0);
        }
    }
    
    void
    increment(size_t iteration, size_t offset)
    {
        if (iteration < num_iterations){
            counts[iteration][offset]++;
        }
    }

    void
    increment_target(size_t offset)
    {
        extend_to(1,offset);
        targets[offset]++;
    }

    void
    reset_counts(){
        for (auto& count: counts){
            for (auto& bucket : count){
                bucket = 0;
            }
        }
    }

    void
    reset_all(){
        for (auto& target : targets){
            target = 0;
        }
        for (auto& count : counts){
            for (auto& bucket : count){
                bucket = 0;
            }
        }
    }
    bool
    iteration_finished(size_t iteration)
    {
        bool ready = true;
        for (size_t i = 0;i <= iteration && i<=max_offset; i++){
            ready &= counts[iteration][i] >= targets[i];
        }
        return ready;
    }
};


/**
 * Tasks may depend on other Tasks.
 *
 * Each task is tied to a segment.
 * The existence of a dependency-relationship between two tasks is deduced from their segment ids.
 * This happens in grid.cc:GridInit()
 */
class Task {
  private:
    std::vector<std::pair<Task*,size_t>> dependents;

  protected:
    VertexBufferArray vba;
    Device device;
    cudaStream_t stream;

    int rank;
    int3 segment_id;

    std::string task_type;

    std::pair<size_t,size_t> loop_counter;
    DependencyCounter dependency_counter;
    int state;

    bool poll_stream();
  public:
    static const int wait_state = 0;

    AcReal dt;

    virtual ~Task()
    {
        //delete dependents;
    }
    virtual bool test() = 0;
    virtual void advance() = 0;

    void registerDependent(Task* t, size_t offset);
    void registerPrerequisite(size_t offset);

    void setIterationParams(size_t begin, size_t end, AcReal dt_);
    void update();
    bool isFinished();

    void notifyDependents();
    void satisfyDependency(size_t iteration, size_t offset);
    
    void syncVBA();
    void swapVBA();

};

// Compute tasks
enum class ComputeState {Waiting_for_halo = Task::wait_state, Running};

typedef class ComputeTask : public Task {
  private:
    int3 start;
    int3 dims;

  public:
    ComputeTask(int3 segment_id_, int3 nn, Device device_, Stream stream_id);

    void compute();
    void advance();
    bool test();
} ComputeTask;

// Communication
typedef struct HaloMessage {
    PackedData buffer;
    MPI_Request* request;
    int length;

    HaloMessage(int3 dims, MPI_Request* req_);
    ~HaloMessage();
    HaloMessage(const HaloMessage& other) = delete;
} HaloMessage;

typedef struct MessageBufferSwapChain {
    int buf_idx;
    std::vector<HaloMessage*> buffers;

    MessageBufferSwapChain();
    ~MessageBufferSwapChain();

    void add_buffer(int3 dims, MPI_Request* req);
    HaloMessage* get_current_buffer();
    HaloMessage* get_fresh_buffer();
} MessageBufferSwapChain;

enum class HaloExchangeState {Waiting_for_compute = Task::wait_state, Packing, Exchanging, Unpacking};

typedef class HaloExchangeTask : public Task {
  private:
    int3 halo_segment_coordinates;
    int3 local_segment_coordinates;

    MessageBufferSwapChain* recv_buffers;
    MessageBufferSwapChain* send_buffers;

    int send_tag;
    int recv_tag;
    int msglen;
    //int rank;
    int send_rank;
    int recv_rank;

    //cudaStream_t stream;

  public:
    bool active;

    HaloExchangeTask(const Device device_, const int rank_, const int3 segment_id_,
                     const uint3_64 decomposition, const int3 grid_dimensions,
                     MPI_Request* recv_requests, MPI_Request* send_requests);
    ~HaloExchangeTask();
    // HaloExchangeTask(const HaloExchangeTask& other) = delete;


    void sync();
    void wait_send();
    void wait_recv();

    void pack();
    void unpack();

    void send();
    void sendDevice();
    void sendHost();

    void receive();
    void receiveDevice();
    void receiveHost();

    void exchange();
    void exchangeDevice();
    void exchangeHost();

    void advance();
    bool test();
} HaloExchangeTask;
