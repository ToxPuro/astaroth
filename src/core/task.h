#pragma once
#include "astaroth.h"

#include <mpi.h>
#include <string>
#include <vector>

#include "decomposition.h"   //getPid and friends
#include "kernels/kernels.h" //PackedData, VertexBufferArray
#include "math_utils.h"      //max. Also included in decomposition.h

#define MPI_USE_PINNED (1)   // Do inter-node comm with pinned (host) memory
#define MPI_INCL_CORNERS (0) // Include the 3D corners of subdomains in halo

#define SWAP_CHAIN_LENGTH (2) // Swap chain lengths other than two not supported
static_assert(SWAP_CHAIN_LENGTH == 2);

#define NUM_SEGMENTS (26)

// clang-format off
#if MPI_INCL_CORNERS
    #define NUM_ACTIVE_SEGMENTS (26)
#else
    #define NUM_ACTIVE_SEGMENTS (18)
#endif
// clang-format on

/**
 * Regions
 * -------
 *
 * Regions are identified by a non-zero region id of type {-1,0,1}^3
 * A region's id describes its position the region topology of the subdomain.
 *
 * There are three families of regions: Incoming message regions, outgoing
 * message regions, and compute regions. The three families each cover some
 * zone of data, as follows:
 *  - Compute: entire inner domain
 *  - Outgoing: the shell of the inner domain
 *  - Incoming: the halo
 * The names of the families have been chosen to represent the constituent
 * regions' purpose in the context of tasks that use them.
 *
 * A triplet in {-1,0,1}^3 identifies each specific region in a family.
 * There is a mapping between integers in {-1,...,25} and the identifiers.
 * The integer form is e.g. used as part of the tag used to identify a region
 * in an MPI message.
 */

enum class RegionFamily { Incoming, Outgoing, Compute };

struct Region {

    int3 position;
    int3 dims;

    RegionFamily family;
    int3 id;
    size_t facet_class;
    int tag;

    static int id_to_tag(int3 _id)
    {
        return ((3 + _id.x) % 3) * 9 + ((3 + _id.y) % 3) * 3 + (3 + _id.z) % 3 - 1;
    }

    static int3 tag_to_id(int _tag)
    {
        int3 _id = (int3){(_tag + 1) / 9, ((_tag + 1) % 9) / 3, (_tag + 1) % 3};
        _id.x    = _id.x == 2 ? -1 : _id.x;
        _id.y    = _id.y == 2 ? -1 : _id.y;
        _id.z    = _id.z == 2 ? -1 : _id.z;
        ERRCHK_ALWAYS(id_to_tag(_id) == _tag);
        return _id;
    }

    Region(RegionFamily _family, int _tag, int3 nn) : family(_family), tag(_tag)
    {
        id          = tag_to_id(tag);
        facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
        ERRCHK_ALWAYS(facet_class <= 3);

        switch (family) {
        case RegionFamily::Compute: {
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
        case RegionFamily::Incoming: {
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
        case RegionFamily::Outgoing: {
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
    }

    Region(RegionFamily _family, int3 _id, int3 nn) : Region{_family, id_to_tag(_id), nn}
    {
        ERRCHK_ALWAYS(_id.x == id.x && _id.y == id.y && _id.z == id.z);
    }
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
class Task {
  private:
    std::vector<std::pair<Task*, size_t>> dependents;

  protected:
    Device device;
    cudaStream_t stream;
    VertexBufferArray vba;
    int rank;

    int state;

    struct {
        size_t num_iters;
        size_t max_offset;
        std::vector<size_t> targets;
        std::vector<std::vector<size_t>> counts;
    } dep_cntr;

    struct {
        size_t i;
        size_t end;
    } loop_cntr;

    bool poll_stream();

  public:
    Region* output_region;
    // std::string task_type;

    static const int wait_state = 0;

    virtual ~Task()
    {
        // delete dependents;
    }
    virtual bool test()    = 0;
    virtual void advance() = 0;

    void registerDependent(Task* t, size_t offset);
    void registerPrerequisite(size_t offset);

    void setIterationParams(size_t begin, size_t end);
    void update();
    bool isFinished();

    void notifyDependents();
    void satisfyDependency(size_t iteration, size_t offset);

    void syncVBA();
    void swapVBA();

    // void logStateChangedEvent(std::string b, std::string c);
};

// Compute tasks
enum class ComputeState { Waiting_for_halo = Task::wait_state, Running };

typedef class ComputeTask : public Task {
  private:
    int3 start;
    int3 dims;

  public:
    ComputeTask(Device device_, int region_tag, int3 nn, Stream stream_id);

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

enum class HaloExchangeState {
    Waiting_for_compute = Task::wait_state,
    Packing,
    Exchanging,
    Unpacking
};

typedef class HaloExchangeTask : public Task {
  private:
    Region* outgoing_message_region;

    MessageBufferSwapChain* recv_buffers;
    MessageBufferSwapChain* send_buffers;

    int counterpart_rank;
    int send_tag;
    int recv_tag;
    int msglen;

  public:
    bool active;

    HaloExchangeTask(const Device device_, const int halo_region_tag, const int3 nn,
                     const uint3_64 decomp, MPI_Request* recv_requests, MPI_Request* send_requests);
    ~HaloExchangeTask();
    // HaloExchangeTask(const HaloExchangeTask& other) = delete;

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
