#pragma once
#include "astaroth.h"

#include <mpi.h>
#include <vector>
#include <string>

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

/**
 * Tasks may depend on other Tasks.
 *
 * Each task is tied to a segment.
 * The existence of a dependency-relationship between two tasks is deduced from their segment ids.
 * This happens in grid.cc:GridInit()
 */
class Task {
  private:
    std::vector<Task*> dependents;

  protected:
    size_t total_dependencies;
    size_t active_dependencies;
    size_t allowed_triggers;
    bool started;

  public:
    std::string name;
    virtual ~Task()
    {
        // delete dependents;
    }
    void register_dependent(Task* t);
    void set_trigger_limit(size_t trigger_limit);
    void notify_dependents(int isubstep, AcReal dt);
    void notify(int isubstep, AcReal dt);
    void update(int isubstep, AcReal dt);
    virtual bool test() = 0;
    virtual void execute(int isubstep, AcReal dt) = 0;
};

// Computation
typedef class ComputationTask : public Task {
  private:
    int3 start;
    int3 dims;
    int3 segment_id;

    Device device;
    VertexBufferArray vba;
    Stream stream;

  public:
    ComputationTask(int3 _segment_id, int3 nn, Device _device, Stream _stream);
    void swapBuffers();
    void syncDeviceState();
    void execute(int isubstep, AcReal dt);
    bool test();
} ComputationTask;

// Communication
typedef struct HaloMessage {
    PackedData buffer;
    MPI_Request* request;
    int length;

    HaloMessage(int3 dims, MPI_Request* _req);
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

typedef class HaloExchangeTask : public Task {
  private:
    int3 segment_id;
    int3 halo_segment_coordinates;
    int3 local_segment_coordinates;

    MessageBufferSwapChain* recv_buffers;
    MessageBufferSwapChain* send_buffers;

    int send_tag;
    int recv_tag;
    int msglen;
    int rank;
    int send_rank;
    int recv_rank;

    Device device;
    VertexBufferArray recv_vba;
    VertexBufferArray send_vba;
    cudaStream_t stream;

  public:
    bool active;

    HaloExchangeTask(const Device _device, const int _rank, const int3 _segment_id,
                     const uint3_64 decomposition, const int3 grid_dimensions,
                     MPI_Request* recv_requests, MPI_Request* send_requests);
    ~HaloExchangeTask();
    // HaloExchangeTask(const HaloExchangeTask& other) = delete;

    void pack();
    void unpack();

    void sync();
    void wait_send();
    void wait_recv();

    void receiveDevice();
    void sendDevice();
    void exchangeDevice();

    void receiveHost();
    void sendHost();
    void exchangeHost();

    void receive();
    void send();
    void exchange();

    void swapSendVBA();
    void swapRecvVBA();
    void syncDeviceState();
    void execute(int isubstep, AcReal dt);
    bool test();
} HaloExchangeTask;
