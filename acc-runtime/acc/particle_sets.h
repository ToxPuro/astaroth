#pragma once
//Particle sets of the DSL, see particle_sets.c
#include <stdbool.h>
#include <stddef.h>

#define MAX_PARTICLE_SETS    (16)
#define MAX_PARTICLE_MEMBERS (128)
#define MAX_PARTICLE_STR     (256)

typedef struct {
	char name[MAX_PARTICLE_STR];
	char type[MAX_PARTICLE_STR];
	char dims[MAX_PARTICLE_STR]; //leading dimensions including the brackets, e.g. "[6]", or ""
	int rank;                    //number of leading dimensions
	bool evolved;                //has the RK accumulator AC_d<set>_<name>
	bool output;                 //read by the host: kept allocated even if no kernel reads it
	bool packonly;               //in the migration records but not moved or unpacked
} ParticleMember;

typedef struct {
	char name[MAX_PARTICLE_STR];
	char capacity[MAX_PARTICLE_STR];
	char migrated_via[MAX_PARTICLE_STR]; //set holding the migration buffers buf, idx, idx2, or "" if not migrated
	ParticleMember members[MAX_PARTICLE_MEMBERS];
	size_t num_members;
} ParticleSet;

extern ParticleSet particle_sets[MAX_PARTICLE_SETS];
extern size_t num_particle_sets;

//Lowers the particle sets in the preprocessed DSL source filename (declarations, kernels over sets,
//set.member accesses) to plain DSL and writes the host header user_particle_sets.h
void lower_particle_sets(const char* filename);
