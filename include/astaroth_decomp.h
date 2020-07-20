#pragma once

#include <vector>
#include <stdint.h>

struct ReroutedData;

enum halo_direction {INCOMING = 1, OUTGOING = -1};

typedef struct {
        uint64_t x, y, z;
} uint3_64;


int3 make_int3(const uint3_64 a);
uint64_t mod(const int a, const int b);

uint3_64 morton3D(const uint64_t pid);
uint64_t morton1D(const uint3_64 pid);

uint3_64 decompose(const uint64_t target);
uint3_64 wrap(const int3 i, const uint3_64 n);

int getPid(const int3 pid_raw, const uint3_64 decomp);

int3 getPid3D(const uint64_t pid, const uint3_64 decomp);

bool onTheSameNode(const uint64_t pid_a, const uint64_t pid_b);

int halo_rerouted_through(
    const int pid,
    const int dest,
    const int3 halo_id,
    const halo_direction direction,//incoming or outgoing +1 or -1
    const uint3_64 decomp,
    const int devices_per_node
);

void find_rerouted_corners(
    const int3 source_pid3d,
    const int3 face_halo_id,
    const int3 face_halo_dims,
    const uint3_64 decomp,
    const int local_node,
    const int devices_per_node,
    std::vector<struct ReroutedData>& out_child_segments
);

void find_rerouted_edges(
    const int3 source_pid3d,
    const int3 face_halo_id,
    const int3 face_halo_dims,
    const uint3_64 decomp,
    const int local_node,
    const int devices_per_node,
    std::vector<struct ReroutedData>& out_child_segments
);

int3 b0_to_halo_id(int3 b0);

int calc_neighbor(
    const int source_pid,
    const int3 halo_id
);
