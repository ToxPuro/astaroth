#include "astaroth.h"
#include "astaroth_decomp.h"
#include "errchk.h"

#include "kernels/kernels.h"

#include "math_utils.h"
#include "rerouted_data.h"


//Below moved here from device.cc
#define MPI_DECOMPOSITION_AXES (3)
uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
    return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

int3
make_int3(const uint3_64 a)
{
    return (int3){(int)a.x, (int)a.y, (int)a.z};
}

uint64_t
mod(const int a, const int b)
{
    const int r = a % b;
    return r < 0 ? r + b : r;
}

uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 3 * bit;
            k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
            j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
            i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
        }
    }
    // Just a quick copy/paste for other decomp dims
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 2 * bit;
            j |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
            k |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 1 * bit;
            k |= ((pid & (mask << 0)) >> 0 * bit) >> 0;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return (uint3_64){i, j, k};
}

uint64_t
morton1D(const uint3_64 pid)
{
    uint64_t i = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 2 * bit;
            i |= ((pid.y & mask) << 1) << 2 * bit;
            i |= ((pid.x & mask) << 2) << 2 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.y & mask) << 0) << 1 * bit;
            i |= ((pid.z & mask) << 1) << 1 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 0 * bit;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return i;
}

uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

uint3_64
wrap(const int3 i, const uint3_64 n)
{
    return (uint3_64){
        mod(i.x, n.x),
        mod(i.y, n.y),
        mod(i.z, n.z),
    };
}

int
getPid(const int3 pid_raw, const uint3_64 decomp)
{
    const uint3_64 pid = wrap(pid_raw, decomp);
    return (int)morton1D(pid);
}

int3
getPid3D(const uint64_t pid, const uint3_64 decomp)
{
    const uint3_64 pid3D = morton3D(pid);
    ERRCHK_ALWAYS(getPid(make_int3(pid3D), decomp) == (int)pid);
    return (int3){(int)pid3D.x, (int)pid3D.y, (int)pid3D.z};
}

static int3
operator*(const halo_direction& a, const int3& b)
{
    return int3{(int)(a)*b.x,(int)(a)*b.y,(int)(a)*b.z};
}

// Assumes that contiguous pids are on the same node and there is one process per GPU.
bool
onTheSameNode(const uint64_t pid_a, const uint64_t pid_b)
{
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    const uint64_t node_a = pid_a / devices_per_node;
    const uint64_t node_b = pid_b / devices_per_node;

    return node_a == node_b;
}

/*
 * Network halo identification
 */

int halo_rerouted_through(
    const int src,
    const int dest,
    const int3 halo_id,
    const halo_direction direction,
    const uint3_64 decomp,
    const int device_per_node)
{
    const int halo_type = abs(halo_id.x) + abs(halo_id.y) + abs(halo_id.z);

    if (halo_type == 1){//i.e. a face       
	return -1;
    }

    const int3 src3d = getPid3D(src, decomp);
    int face_pid = -1;

    const int3 faces[3] = {int3{halo_id.x,0,0},int3{0,halo_id.y,0},int3{0,0,halo_id.z}};
    const int dims[3] = {halo_id.x,halo_id.y,halo_id.z};

    for (int i = 0;i < 3;i++){
	if (dims[i] != 0){
            int candidate_face_pid = getPid(src3d + direction*faces[i] ,decomp);
            if (dest/device_per_node == candidate_face_pid/device_per_node){
                face_pid = candidate_face_pid;
                break;
            }
        }
    }

    return face_pid;
}

void find_rerouted_edges(
    const int3 source_pid3d,
    const int3 face_halo_id,
    const int3 face_halo_dims,
    const uint3_64 decomp,
    const int local_node,
    const int devices_per_node,
    std::vector<ReroutedData>& out_child_segments)
{
    //const int dims[3] = {face_halo_id.x,face_halo_id.y,face_halo_id.z};
    int3 edge_halos[4];
    int3 edge_dims;

    if (face_halo_id.x != 0) {
        //edges
        edge_halos[0] = int3{face_halo_id.x, 1, 0};
        edge_halos[1] = int3{face_halo_id.x,-1, 0};
        edge_halos[2] = int3{face_halo_id.x, 0, 1};
        edge_halos[3] = int3{face_halo_id.x, 0,-1};
    } else if (face_halo_id.y != 0) {
        //edges
        edge_halos[0] = int3{ 1, face_halo_id.y, 0};
        edge_halos[1] = int3{-1, face_halo_id.y, 0};
        edge_halos[2] = int3{ 0, face_halo_id.y, 1};
        edge_halos[3] = int3{ 0, face_halo_id.y,-1};
    } else if (face_halo_id.z != 0) {
        //edges
        edge_halos[0] = int3{ 1, 0,face_halo_id.z};
        edge_halos[1] = int3{-1, 0,face_halo_id.z};
        edge_halos[2] = int3{ 0, 1,face_halo_id.z};
        edge_halos[3] = int3{ 0,-1,face_halo_id.z};
    } else{
        fprintf(stderr,"ERROR when extracting halo: Broken halo data, all zeros");
    }

    for(auto& edge : edge_halos ){
        int halo_target = getPid(source_pid3d + OUTGOING*edge, decomp);
        if (local_node == halo_target/ devices_per_node){
            edge_dims = edge.x == 0 ? int3{face_halo_dims.x, NGHOST, NGHOST} :
                        edge.y == 0 ? int3{NGHOST, face_halo_dims.y , NGHOST} :
                                      int3{NGHOST, NGHOST, face_halo_dims.z};

            const int3 offset = (int3){
                face_halo_id.x != 0 ? 0 : edge.x < 0 ? face_halo_dims.x - NGHOST : 0,
                face_halo_id.y != 0 ? 0 : edge.y < 0 ? face_halo_dims.y - NGHOST : 0,
                face_halo_id.z != 0 ? 0 : edge.z < 0 ? face_halo_dims.z - NGHOST : 0,
            };
            //fprintf(stderr,"Face (%d,%d,%d) contains edge (%d,%d,%d), edge dims = (%d,%d,%d)\n",face_halo_dims.x,face_halo_dims.y,face_halo_dims.z,edge.x,edge.y,edge.z,edge_dims.x,edge_dims.y,edge_dims.z);
            out_child_segments.emplace_back(acCreatePackedData(edge_dims), edge,offset);
        }
    }
}

void find_rerouted_corners(
    const int3 source_pid3d,
    const int3 face_halo_id,
    const int3 face_halo_dims,
    const uint3_64 decomp,
    const int local_node,
    const int devices_per_node,
    std::vector<ReroutedData>& out_child_segments)
{
    int3 corner_halos[4];
    const int3 corner_dims = int3{NGHOST,NGHOST,NGHOST};

    if (face_halo_id.x != 0) {
        corner_halos[0] = int3{face_halo_id.x, 1, 1};
        corner_halos[1] = int3{face_halo_id.x, 1,-1};
        corner_halos[2] = int3{face_halo_id.x,-1, 1};
        corner_halos[3] = int3{face_halo_id.x,-1,-1};
    } else if (face_halo_id.y != 0) {
        corner_halos[0] = int3{ 1,face_halo_id.y, 1};
        corner_halos[1] = int3{ 1,face_halo_id.y,-1};
        corner_halos[2] = int3{-1,face_halo_id.y, 1};
        corner_halos[3] = int3{-1,face_halo_id.y,-1};
    } else if (face_halo_id.z != 0) {
        corner_halos[0] = int3{ 1, 1,face_halo_id.z};
        corner_halos[1] = int3{ 1,-1,face_halo_id.z};
        corner_halos[2] = int3{-1, 1,face_halo_id.z};
        corner_halos[3] = int3{-1,-1,face_halo_id.z};
    } else{
        fprintf(stderr,"ERROR when extracting halo: Broken halo data, all zeros");
    }

    for(auto& corner : corner_halos ){
        int halo_target = getPid(source_pid3d + OUTGOING*corner, decomp);
        if (local_node == halo_target/ devices_per_node){
            const int3 offset = (int3){
                face_halo_id.x != 0 ? 0 : corner.x < 0 ? face_halo_dims.x - NGHOST : 0,
                face_halo_id.y != 0 ? 0 : corner.y < 0 ? face_halo_dims.y - NGHOST : 0,
                face_halo_id.z != 0 ? 0 : corner.z < 0 ? face_halo_dims.z - NGHOST : 0,
            };
            out_child_segments.emplace_back(acCreatePackedData(corner_dims), corner,offset);
        }
    }
}

//Grid not defined here, defined in device.cc
int3 b0_to_halo_id(int3 b0, int3 nn)
{
    return (int3){
        b0.x < NGHOST ? -1 : b0.x >= NGHOST + nn.x ? 1 : 0,
        b0.y < NGHOST ? -1 : b0.y >= NGHOST + nn.y ? 1 : 0,
        b0.z < NGHOST ? -1 : b0.z >= NGHOST + nn.z ? 1 : 0,
    };
}

int calc_neighbor(
    const int source_pid,
    const int3 halo_id)
{
    //todo
    return source_pid + halo_id.x;
}
