/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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
#define GEN_TEST_FILE (0)

/**
    Running: mpirun -np <num processes> <executable>
*/
#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_utils.h"

#include <mpi.h>
#include <unistd.h>
#include <getopt.h>

#include <cstring>
#include <ctime>
#include <cstdarg>

#include "config_loader.h"
#include "errchk.h"
#include "host_forcing.h"
#include "host_memory.h"

#include "math_utils.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

// IO configuration
static const Field io_fields[] = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, VTXBUF_LNRHO, VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ};
static const size_t num_io_fields = ARRAY_SIZE(io_fields);
static const char* snapshot_dir = "output-snapshots";
static const char* slice_dir = "output-slices";

#define fprintf(...)                                                                               \
    {                                                                                              \
        int tmppid;                                                                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &tmppid);                                                    \
        if (!tmppid) {                                                                             \
            fprintf(__VA_ARGS__);                                                                  \
        }                                                                                          \
    }

#define printf(...)                                                                                \
    {                                                                                              \
        int tmppid;                                                                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &tmppid);                                                    \
        if (!tmppid) {                                                                             \
            printf(__VA_ARGS__);                                                                   \
        }                                                                                          \
    }

//TODO: currently, printf & fprintf has been replaced everywhere with one that only prints from MPI rank 0
//This is suboptimal, because now we cannot use printf to print from another rank
//We should rename this rank 0 printf something else, like root_thread::printf
//This is clearer in intent, and allows us to actually use printf elsewhere as well

void
debug_from_root_proc(int pid, const char *msg, ...){
#ifndef NDEBUG
    if (pid == 0){
	std::time_t now = std::time(nullptr);
	char *timestamp = std::ctime(&now);
	size_t stamp_len = strlen(timestamp);
	//Remove trailing newline
	timestamp[stamp_len - 1] = '\0';
	//We know the exact length of the timestamp (26 chars), so we could force this function to take chars with a 26 prefix blank buffer
	fprintf(stderr, "%s : ", timestamp);

	va_list args;
	va_start(args, msg);
        vfprintf(stderr, msg, args);
        fflush(stderr);
	va_end(args);
    }
#endif
}

void
log_from_root_proc(int pid, const char *msg, ...){
    if (pid == 0){
	std::time_t now = std::time(nullptr);
	char *timestamp = std::ctime(&now);
	size_t stamp_len = strlen(timestamp);
	//Remove trailing newline
	timestamp[stamp_len - 1] = '\0';
	//We know the exact length of the timestamp (26 chars), so we could force this function to take chars with a 26 prefix blank buffer
	fprintf(stderr, "%s : ", timestamp);

	va_list args;
	va_start(args, msg);
        vfprintf(stderr, msg, args);
        fflush(stderr);
	va_end(args);
    }
}

/*
void
log_from_root_proc(const char *msg, ...){
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    va_list args;
    va_start(args, msg);
    log_from_root_proc(pid, msg, args);
    va_end(args);
}
*/




// MV: I commented this out because now can be defined in DSL
//// #define LSINK (0)
//// #define LFORCING (0)
//// #define LBFIELD (0)
//// #define LSHOCK (0)

// Write all setting info into a separate ascii file. This is done to guarantee
// that we have the data specifi information in the thing, even though in
// principle these things are in the astaroth.conf.
static inline void
write_info(const AcMeshInfo* config)
{

    FILE* infotxt;

    infotxt = fopen("purge.sh", "w");
    fprintf(infotxt, "#!/bin/bash\n");
    fprintf(infotxt, "rm *.list *.mesh *.csv *.field *.ts purge.sh autotune-result.out\n");
    fprintf(infotxt, "rm -rf *output-s*\n");
    fclose(infotxt);

    infotxt = fopen("mesh_info.list", "w");

    // Determine endianness
    unsigned int EE      = 1;
    char* CC             = (char*)&EE;
    const int endianness = (int)*CC;
    // endianness = 0 -> big endian
    // endianness = 1 -> little endian

    fprintf(infotxt, "size_t %s %lu \n", "AcRealSize", sizeof(AcReal));

    fprintf(infotxt, "int %s %i \n", "endian", endianness);

    // JP: this could be done shorter and with smaller chance for errors with the following
    // (modified from acPrintMeshInfo() in astaroth.cu)
    // MV: Now adapted into working condition. E.g. removed useless / harmful formatting.

    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        fprintf(infotxt, "int %s %d\n", intparam_names[i], config->int_params[i]);

    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        fprintf(infotxt, "int3 %s  %d %d %d\n", int3param_names[i], config->int3_params[i].x,
                config->int3_params[i].y, config->int3_params[i].z);

    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        fprintf(infotxt, "real %s %g\n", realparam_names[i], double(config->real_params[i]));

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        fprintf(infotxt, "real3 %s  %g %g %g\n", real3param_names[i],
                double(config->real3_params[i].x), double(config->real3_params[i].y),
                double(config->real3_params[i].z));

    fclose(infotxt);
}

/*
// This funtion writes a run state into a set of C binaries.
static inline void
save_mesh(const AcMesh& save_mesh, const int step, const AcReal t_step)
{
    FILE* save_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(save_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        strcat(bin_filename, ".mesh");

        printf("Savefile %s \n", bin_filename);

        save_ptr = fopen(bin_filename, "wb");

        // Start file with time stamp
        AcReal write_long_buf = (AcReal)t_step;
        fwrite(&write_long_buf, sizeof(AcReal), 1, save_ptr);
        // Grid data
        for (size_t i = 0; i < n; ++i) {
            const AcReal point_val = save_mesh.vertex_buffer[VertexBufferHandle(w)][i];
            AcReal write_long_buf2 = (AcReal)point_val;
            fwrite(&write_long_buf2, sizeof(AcReal), 1, save_ptr);
        }
        fclose(save_ptr);
    }
}
*/

// This funtion writes a run state into a set of C binaries
// WITH MPI_IO
static inline void
save_mesh_mpi_sync(const AcMeshInfo info, const int pid, const int step, const AcReal t_step)
{
    printf("Saving snapshot at step %i \n", step);

    char cstep[11];
    //char header_filename[80] = "\0";
    sprintf(cstep, "%d", step);

    // Saves a csv file which contains relevant information about the binary
    // snapshot files at the timestep. 
    if (pid == 0) {
        FILE* header_file = fopen("snapshots_info.csv", "a");

        //Header only at the step zero
        if (step == 0) {
            fprintf(header_file, "use_double, mx, my, mz, step_number, t_step \n");
        }

        fprintf(header_file, "%d, %d, %d, %d, %d, %.17e \n", sizeof(AcReal) == 8,
                info.int_params[AC_mx], info.int_params[AC_my], 
                info.int_params[AC_mz], step, t_step);

	    // Writes the header info. Make it into an
	    // appendaple csv table which will be easy to be read into a Pandas
	    // dataframe.
         
        fclose(header_file);
    }


    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const char* buffername = vtxbuf_names[w];
        char bin_filename[80] = "\0";


        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        //strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, snapshot_dir, bin_filename, ACCESS_WRITE);
       
        printf("Savefile %s \n", bin_filename);

        acGridDiskAccessSync();
    }
}

static inline void
save_mesh_mpi_async(const AcMeshInfo info, const char* job_dir, const int pid, const int step, const AcReal t_step)
{
    const int num_snapshots = 2;
    const int modstep = (step/info.int_params[AC_bin_steps]) % num_snapshots;
    printf("Saving snapshot at step %i (%d of %d)\n", step, modstep, num_snapshots);

    // Saves a csv file which contains relevant information about the binary
    // snapshot files at the timestep. 
    if (pid == 0) {
        FILE* header_file = fopen("snapshots_info.csv", "a");

        //Header only at the step zero
        if (step == 0) {
            fprintf(header_file, "use_double, mx, my, mz, step_number, modstep, t_step \n");
        }

        fprintf(header_file, "%d, %d, %d, %d, %d, %d, %.17e \n", sizeof(AcReal) == 8,
                info.int_params[AC_mx], info.int_params[AC_my], 
                info.int_params[AC_mz], step, modstep, t_step);

	    // Writes the header info. Make it into an
	    // appendaple csv table which will be easy to be read into a Pandas
	    // dataframe.
         
        fclose(header_file);
    }

    char cstep[1024];
    sprintf(cstep, "%d", modstep);
    acGridDiskAccessSync();
    acGridWriteMeshToDiskLaunch(job_dir, cstep);
    printf("Write mesh to disk launch %s, %s \n", job_dir, cstep);
    /*
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const char* buffername = vtxbuf_names[w];
        char bin_filename[80] = "\0";


        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        //strcat(bin_filename, ".field");

        // Grid data
        //acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, job_dir, bin_filename, ACCESS_WRITE);
        acGridDiskAccessSync();
        acGridWriteMeshToDiskLaunch(job_dir, step);
        // %JP TODO write async
       
        printf("Savefile %s \n", bin_filename);
    }*/
}

// This funtion reads a run state into a set of C binaries
// WITH MPI_IO
static inline void
read_mesh_mpi(const int pid, const int step, AcReal* t_step)
{
    int stepnumber;
    AcReal time_at_step;
    double time;

    printf("Reading snapshot at step %i \n", step);
    char cstep[11];
    sprintf(cstep, "%d", step);

    if (pid == 0) {


        AcReal element[8];

        // Saves a csv file which contains relevant information about the binary
        // snapshot files at the timestep. 
        FILE* header_file = fopen("snapshots_info.csv", "r");

        //TODO: Loop through the header file to find the step number of snapshots
        //TODO: to be read. And read the relevat other info.

        //Simple cvs file reader. 
        char csv_line[256];
        while (fgets( csv_line, sizeof(csv_line), header_file ) != NULL ) {
            int column_index = 0;
            for (char* csv_loc = strtok( csv_line, ","); csv_loc != NULL; csv_loc = strtok(NULL, ",")) {
                printf("%s, ", csv_loc);
                element[column_index++] = atof(csv_loc);
            }
            printf("\n");
            stepnumber = int(element[4]);
            time_at_step = element[5];
            //printf("stepnumber %i at time_at_step %e \n", stepnumber, time_at_step);

            if ( stepnumber == step) {
                time = double(time_at_step);
            } 
        }

        fclose(header_file);
    }

    MPI_Bcast(&time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    *t_step = time; 

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const char* buffername = vtxbuf_names[w];
        char bin_filename[80] = "\0";


        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        //strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, snapshot_dir, bin_filename, ACCESS_READ);
       
        printf("Read file %s \n", bin_filename);

        acGridDiskAccessSync();
    }
}


// This funtion reads a run state from a set of C binaries.
static inline void
read_mesh(AcMesh& read_mesh, const int step, AcReal* t_step)
{
    FILE* read_ptr;

    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        const size_t n = acVertexBufferSize(read_mesh.info);

        const char* buffername = vtxbuf_names[w];
        char cstep[11];
        char bin_filename[80] = "\0";

        // sprintf(bin_filename, "");

        sprintf(cstep, "%d", step);

        strcat(bin_filename, buffername);
        strcat(bin_filename, "_");
        strcat(bin_filename, cstep);
        strcat(bin_filename, ".mesh");

        printf("Reading savefile %s \n", bin_filename);

        read_ptr = fopen(bin_filename, "rb");

        // Start file with time stamp
        size_t result;
        result = fread(t_step, sizeof(AcReal), 1, read_ptr);
        // Read grid data
        AcReal read_buf;
        for (size_t i = 0; i < n; ++i) {
            result = fread(&read_buf, sizeof(AcReal), 1, read_ptr);
            read_mesh.vertex_buffer[VertexBufferHandle(w)][i] = read_buf;
            if (int(result) != 1) {
                fprintf(stderr, "Reading error in %s, element %i\n", vtxbuf_names[w], int(i));
                fprintf(stderr, "Result = %i,  \n", int(result));
            }
        }
        fclose(read_ptr);
    }
}

static inline void
print_diagnostics_header(FILE *diag_file)
{
    // Generate the file header
    fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
#if LBFIELD
    fprintf(diag_file, "bb_total_min  bb_total_rms  bb_total_max  ");
    fprintf(diag_file, "vA_total_min  vA_total_rms  vA_total_max  ");
#endif
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
                vtxbuf_names[i]);
    }
#if LSINK
    fprintf(diag_file, "sink_mass  accreted_mass  ");
#endif
    fprintf(diag_file, "\n");
}



// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
// JP: MUST BE CALLED FROM PROC 0. Must be rewritten for multiple processes (this implementation
// has write race condition)
static inline void
print_diagnostics_host(const AcMesh mesh, const int step, const AcReal dt, const AcReal t_step,
                       FILE* diag_file, const AcReal sink_mass, const AcReal accreted_mass)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    buf_max = acHostReduceVec(mesh, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_min = acHostReduceVec(mesh, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
    buf_rms = acHostReduceVec(mesh, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

    // MV: The ordering in the earlier version was wrong in terms of variable
    // MV: name and its diagnostics.
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));
#if LBFIELD
    buf_max = acHostReduceVec(mesh, RTYPE_MAX, BFIELDX, BFIELDY, BFIELDZ);
    buf_min = acHostReduceVec(mesh, RTYPE_MIN, BFIELDX, BFIELDY, BFIELDZ);
    buf_rms = acHostReduceVec(mesh, RTYPE_RMS, BFIELDX, BFIELDY, BFIELDZ);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "bb total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

    buf_max = acHostReduceVecScal(mesh, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);
    buf_min = acHostReduceVecScal(mesh, RTYPE_ALFVEN_MIN, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);
    buf_rms = acHostReduceVecScal(mesh, RTYPE_ALFVEN_RMS, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "vA total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
#endif

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        buf_max = acHostReduceScal(mesh, RTYPE_MAX, VertexBufferHandle(i));
        buf_min = acHostReduceScal(mesh, RTYPE_MIN, VertexBufferHandle(i));
        buf_rms = acHostReduceScal(mesh, RTYPE_RMS, VertexBufferHandle(i));

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");
}

// This function prints out the diagnostic values to std.out and also saves and
// appends an ascii file to contain all the result.
// JP: EXECUTES ON MULTIPLE GPUS, MUST BE CALLED FROM ALL PROCS
static inline void
print_diagnostics(const int step, const AcReal dt, const AcReal t_step, FILE* diag_file,
                  const AcReal sink_mass, const AcReal accreted_mass, int* found_nan)
{

    AcReal buf_rms, buf_max, buf_min;
    const int max_name_width = 16;

    // Calculate rms, min and max from the velocity vector field
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_max);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_min);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &buf_rms);

    // MV: The ordering in the earlier version was wrong in terms of variable
    printf("Step %d, t_step %.3e, dt %e s\n", step, double(t_step), double(dt));
    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "uu total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%d %e %e %e %e %e ", step, double(t_step), double(dt), double(buf_min),
            double(buf_rms), double(buf_max));

#if LBFIELD
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, BFIELDX, BFIELDY, BFIELDZ, &buf_max);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, BFIELDX, BFIELDY, BFIELDZ, &buf_min);
    acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, BFIELDX, BFIELDY, BFIELDZ, &buf_rms);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "bb total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &buf_max);
    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_MIN, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &buf_min);
    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_RMS, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &buf_rms);

    printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, "vA total", double(buf_min),
           double(buf_rms), double(buf_max));
    fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));
#endif

    // Calculate rms, min and max from the variables as scalars
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, VertexBufferHandle(i), &buf_max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, VertexBufferHandle(i), &buf_min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, VertexBufferHandle(i), &buf_rms);

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", max_name_width, vtxbuf_names[i],
               double(buf_min), double(buf_rms), double(buf_max));
        fprintf(diag_file, "%e %e %e ", double(buf_min), double(buf_rms), double(buf_max));

        if (isnan(buf_max) || isnan(buf_min) || isnan(buf_rms)) {
            *found_nan = 1;
        }
    }

    if ((sink_mass >= AcReal(0.0)) || (accreted_mass >= AcReal(0.0))) {
        fprintf(diag_file, "%e %e ", double(sink_mass), double(accreted_mass));
    }

    fprintf(diag_file, "\n");

#if LSINK
    printf("sink mass is: %.15e \n", double(sink_mass));
    printf("accreted mass is: %.15e \n", double(accreted_mass));
#endif

    fflush(diag_file);
    fflush(stdout);
}

/*
    MV NOTE: At the moment I have no clear idea how to calculate magnetic
    diagnostic variables from grid. Vector potential measures have a limited
    value. TODO: Smart way to get brms, bmin and bmax.
*/

#include "math_utils.h"
AcReal
calc_timestep(const AcMeshInfo info)
{
    AcReal uumax     = 0.0;
    AcReal vAmax     = 0.0;
    AcReal shock_max = 0.0;
    acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &uumax);
    // TODO ERROR: uumax is currently only seen by the rank 0 process which will
    // lead to asyncronizity in the timestep leading to deadlocks! The resulting
    // dt or uumax and vAmax in rank 0 should be broadcasted to others.
    // SOLUTION: broadcast uumax and vAmax to all ranks
    // NOTE: It would be also possible to edit contents of
    // acGridReduceVecScal(), but for the sake of coherence, with less risk of
    // interfering things elsewhere, I have deiced to try out this approach
    // first, as it is not too complicated anyway.
    //
    // %JP: uumax, vAmax, seems to be OK now with the following bcasts

    // MPI_Bcast to share uumax with all ranks
    MPI_Bcast(&uumax, 1, AC_REAL_MPI_TYPE, 0, MPI_COMM_WORLD);

#if LBFIELD
    acGridReduceVecScal(STREAM_DEFAULT, RTYPE_ALFVEN_MAX, BFIELDX, BFIELDY, BFIELDZ, VTXBUF_LNRHO,
                        &vAmax);

    // MPI_Bcast to share vAmax with all ranks
    MPI_Bcast(&vAmax, 1, AC_REAL_MPI_TYPE, 0, MPI_COMM_WORLD);
#endif

#if LSHOCK
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_SHOCK, &shock_max);

    // MPI_Bcast to share vAmax with all ranks
    MPI_Bcast(&shock_max, 1, AC_REAL_MPI_TYPE, 0, MPI_COMM_WORLD);
#endif

    const long double cdt  = (long double)info.real_params[AC_cdt];
    const long double cdtv = (long double)info.real_params[AC_cdtv];
    // const long double cdts     = (long double)info.real_params[AC_cdts];
    const long double cs2_sound = (long double)info.real_params[AC_cs2_sound];
    const long double nu_visc   = (long double)info.real_params[AC_nu_visc];
    const long double eta       = (long double)info.real_params[AC_eta];
    const long double chi       = 0; // (long double)info.real_params[AC_chi]; // TODO not calculated
    const long double gamma     = (long double)info.real_params[AC_gamma];
    const long double dsmin     = (long double)info.real_params[AC_dsmin];
    const long double nu_shock  = (long double)info.real_params[AC_nu_shock];

    // Old ones from legacy Astaroth
    // const long double uu_dt   = cdt * (dsmin / (uumax + cs_sound));
    // const long double visc_dt = cdtv * dsmin * dsmin / nu_visc;

    // New, closer to the actual Courant timestep
    // See Pencil Code user manual p. 38 (timestep section)
    const long double uu_dt   = cdt * dsmin / (fabsl((long double)uumax) + sqrtl(cs2_sound + (long double)vAmax * (long double)vAmax));
    const long double visc_dt = cdtv * dsmin * dsmin /
                                (max(max(nu_visc, eta), gamma * chi) + nu_shock * (long double)shock_max);

    const long double dt = min(uu_dt, visc_dt);
    ERRCHK_ALWAYS(is_valid((AcReal)dt));
    return AcReal(dt);
}

void dryrun(void)
{
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Scale the fields
    AcReal max, min, sum;
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)0, &max);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)0, &min);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)0, &sum);

    MPI_Barrier(MPI_COMM_WORLD);

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_scaling_factor, (AcReal)2.0);
    AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridLaunchKernel(STREAM_DEFAULT, scale, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    MPI_Barrier(MPI_COMM_WORLD);

    acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)0, &max);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)0, &min);
    acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)0, &sum);
    const AcReal dt = 0.0;
    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridLaunchKernel(STREAM_DEFAULT, reset, dims.n0, dims.n1);
    acGridLaunchKernel(STREAM_DEFAULT, randomize, dims.n0, dims.n1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Reset the fields
    acGridLaunchKernel(STREAM_DEFAULT, reset, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    MPI_Barrier(MPI_COMM_WORLD);

    acGridLaunchKernel(STREAM_DEFAULT, reset, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    MPI_Barrier(MPI_COMM_WORLD);
}

static void read_varfile_to_mesh_and_setup(const AcMeshInfo info, const char *file_path)
{
    // Read PC varfile to Astaroth

    const int3 nn = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 rr = (int3){3, 3, 3};

    log_from_root_proc(0, "nn = (%d, %d, %d)\n", nn.x, nn.y, nn.z);

    acGridReadVarfileToMesh(file_path, io_fields, num_io_fields, nn, rr);

    // Scale the magnetic field
    /*
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_scaling_factor, info.real_params[AC_scaling_factor]);
    AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridLaunchKernel(STREAM_DEFAULT, scale, dims.n0, dims.n1);
    acGridSwapBuffers();
    */
    acGridPeriodicBoundconds(STREAM_DEFAULT);
}

static void read_file_to_mesh_and_setup(void) {

    for (size_t i = 0; i < num_io_fields; ++i){
        const Field field = io_fields[i];
        acGridAccessMeshOnDiskSynchronous(field, snapshot_dir, vtxbuf_names[field], ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT); 
}

static void read_distributed_to_mesh_and_setup(void)
{
    for (size_t i = 0; i < num_io_fields; ++i){
        const Field field = io_fields[i];
        acGridAccessMeshOnDiskSynchronousDistributed(field, snapshot_dir, vtxbuf_names[field], ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT);    
}

static void read_collective_to_mesh_and_setup(void)
{
    for (size_t i = 0; i < num_io_fields; ++i){
        const Field field = io_fields[i];
        acGridAccessMeshOnDiskSynchronousCollective(field, snapshot_dir, vtxbuf_names[field], ACCESS_READ);
    }
    acGridPeriodicBoundconds(STREAM_DEFAULT);    
}

static void create_output_directories(void)
{
    // Improvement suggestion: use create_directory() from <filesystem> (C++17) or mkdir() from <sys/stat.h> (POSIX)
    const size_t cmdlen = 4096;
    char cmd[cmdlen];

    snprintf(cmd, cmdlen, "mkdir -p %s", snapshot_dir);
    system(cmd);

    snprintf(cmd, cmdlen, "mkdir -p %s", slice_dir);
    system(cmd);
}
void
print_usage(const char * name) {
            printf("Usage: ./%s "
		   "[--config <config_path>] "
		   "[ --run-init-kernel | --from-pc-varfile | --from-distributed-snapshot | --from-monolithic-snapshot ]"
		   "\n"
                   "\n"
		   " -c <config_path>\n"
		   " --config <config_path>\n"
		   "\tread config file from directory at config_path, the default config path is: %s\n"
                   "\n"
		   "Mutually exclusive initial mesh load procedures\n"
		   "------------------------------------------------"
            	   "\n"
		   "  the default is --init-kernel\n"
            	   "\n"
		   " -k\n"
		   " --init-kernel\n"
		   "\tRun a kernel to initialize the mesh\n"
		   "\tThe kernel is currently hardcoded\n"
                   "\n"
		   " -p\n"
		   " --from-pc-varfile\n"
		   "\tLoad the mesh from a pc varfile\n"
		   "\tThe path to this file is currently hardcoded\n"
                   "\n"
		   " -d\n --from-distributed-snapshot\n"
		   "\tLoad the mesh from a distributed snapshot (one file per process)\n"
		   "\tThe path to the snapshot is currently hardcoded\n"
                   "\n"
		   " -m\n"
		   " --from-monolithic-snapshot\n"
		   "\tLoad the mesh from a monolithic snapshot (one single file)\n"
		   "\tThe path to the snapshot is currently hardcoded\n",
		   name,
		   AC_DEFAULT_CONFIG
		   );
}

enum class InitialMeshProcedure { Kernel, LoadPC_Varfile, LoadDistributedSnapshot, LoadMonolithicSnapshot };

int
main(int argc, char** argv)
{
    // Use multi-threaded MPI
    int thread_support_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_support_level);
    if (thread_support_level < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE not supported by the MPI implementation\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    /////////////////////////////////
    // Read command line arguments //
    /////////////////////////////////

    //The input mesh files are hardcoded at the moment, but they could easily be passed by parameter
    //Just change no_argument below to required_argument or optional_argument
    //and copy the value of optarg to a filename variable in the switch
    static struct option long_options[] = {{"config", required_argument, 0, 'c'},
					   {"run-init-kernel", no_argument, 0, 'k'},
					   {"from-pc-varfile", required_argument, 0, 'p'},
					   {"from-distributed-snapshot", no_argument, 0, 'd'},
					   {"from-monolithic-snapshot", no_argument, 0, 'm'},
	    				   {"help", no_argument, 0, 'h'}};


    const char *config_path = AC_DEFAULT_CONFIG;
    //Default mesh procedure is kernel randomize
    InitialMeshProcedure initial_mesh_procedure = InitialMeshProcedure::Kernel;
    const char *initial_mesh_procedure_param = nullptr;

    int opt{};
    while ((opt = getopt_long(argc, argv, "c:kpdmh", long_options, nullptr)) != -1) {
        switch (opt) {
	    case 'h':
		print_usage("ac_run_mpi");
		return EXIT_SUCCESS;
	    case 'c':
		config_path = optarg;
                break;
	    case 'k':
                initial_mesh_procedure = InitialMeshProcedure::Kernel;
		break;
	    case 'p':
                initial_mesh_procedure = InitialMeshProcedure::LoadPC_Varfile;
                initial_mesh_procedure_param = optarg;
		break;
	    case 'd':
                initial_mesh_procedure = InitialMeshProcedure::LoadDistributedSnapshot;
		break;
	    case 'm':
                initial_mesh_procedure = InitialMeshProcedure::LoadMonolithicSnapshot;
		break;
	    default:
		print_usage("ac_run_mpi");
		return EXIT_FAILURE;
	}
    }

    //////////////////////
    // Load config file //
    //////////////////////
    
    AcMeshInfo info;
    acLoadConfig(config_path, &info);
    load_config(config_path, &info);
    acHostUpdateBuiltinParams(&info);

    ////////////////////////////////
    // Write the config to a file //
    ////////////////////////////////
    
    if (pid == 0) {
	    // Write purge.sh and meshinfo.list
	    write_info(&info);
	    // Print config to stdout
        acPrintMeshInfo(info);
    }
    ERRCHK_ALWAYS(info.real_params[AC_dsx] == DSX);
    ERRCHK_ALWAYS(info.real_params[AC_dsy] == DSY);
    ERRCHK_ALWAYS(info.real_params[AC_dsz] == DSZ);

    // Ensure all directories exist
    create_output_directories();

    // SMELL: this was set twice, once here and once further down
    // I've commented out the first call to srand, since it would be overridden by the second value
    // TODO: figure out if this is here for a reason, and why was it different?
    // and should this be moved lower
    
    // Set random seed for reproducibility
    //srand(321654987);
    srand(312256655);

#if LSINK
    if (pid == 0) {
	//TODO: can this be set lower down? or is this needed by dryrun?
        acVertexBufferSet(VTXBUF_ACCRETION, 0.0, &mesh);
    }
#endif
    // END SMELL

    ////////////////////////////////////////
    // Initialize internal Astaroth state //
    ////////////////////////////////////////

    acGridInit(info);

    ///////////////////////////////////////////////////
    // Test kernels: scale, solve, reset, randomize. //
    // then reset                                    //
    ///////////////////////////////////////////////////
    
    dryrun();

    // Load input data

    /////////////////////////////////////////////
    // Mesh initialization from file or kernel //
    /////////////////////////////////////////////
    
    log_from_root_proc(pid, "Initializing mesh\n");
    switch (initial_mesh_procedure){
	    case InitialMeshProcedure::Kernel:
	    {
		// Randomize
		log_from_root_proc(pid, "Scrambling mesh with some (low-quality) pseudo-random data\n");
		AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
		acGridLaunchKernel(STREAM_DEFAULT, randomize, dims.n0, dims.n1);
		acGridSwapBuffers();
		log_from_root_proc(pid, "Communicating halos\n");
		acGridPeriodicBoundconds(STREAM_DEFAULT);

	        {
		    // Should some labels be printed here?
		    AcReal max, min, sum;
		    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
		        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)i, &max);
		        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)i, &min);
		        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)i, &sum);
		        if (pid == 0) {
			    printf("max %g, min %g, sum %g\n", (double)max, (double)min, (double)sum);
			}
		    }
	        }
		break;
	    }
	    case InitialMeshProcedure::LoadPC_Varfile:
	    {
		log_from_root_proc(pid, "Reading mesh state from Pencil Code var file %s\n", initial_mesh_procedure_param);
		if (initial_mesh_procedure_param == nullptr){
		    log_from_root_proc(pid, "Error: no file path given");
		    return EXIT_FAILURE;
		}
    		read_varfile_to_mesh_and_setup(info, initial_mesh_procedure_param);
		log_from_root_proc(pid, "Done reading Pencil Code var file\n");
		break;
	    }
	    case InitialMeshProcedure::LoadDistributedSnapshot:
	    {
		log_from_root_proc(pid, "Reading mesh state from distributed snapshot\n");
                read_distributed_to_mesh_and_setup();
		log_from_root_proc(pid, "Done reading distributed snapshot\n");
		break;
            }
	    case InitialMeshProcedure::LoadMonolithicSnapshot:
	    {
		log_from_root_proc(pid, "Reading mesh state monolithic snapshot\n");
    		read_collective_to_mesh_and_setup();
		log_from_root_proc(pid, "Done reading monolithic snapshot\n");
		break;
	    }
	    /*case LoadMeshFile:
	    {
		log_from_root_proc(pid, "Reading mesh file\n");
    		read_file_to_mesh_and_setup();
		log_from_root_proc(pid, "Done reading mesh file\n");
		break;
            }
	    */
    }

    log_from_root_proc(pid, "Initial mesh setup is done\n");



    ////////////////////////////////////////////////////
    // Building the task graph (or using the default) //
    ////////////////////////////////////////////////////
    
    log_from_root_proc(pid, "Defining simulation\n");

    //default simulation is the AcGridIntegrate task graph (MHD)
    AcTaskGraph *simulation_graph = acGridGetDefaultTaskGraph();
    AcTaskGraph *custom_simulation_graph = nullptr;

#if LSHOCK

    // Shock case task graph
    //----------------------
    VertexBufferHandle all_fields[] = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                       VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ, // VTXBUF_ENTROPY,
                                       VTXBUF_SHOCK, BFIELDX,    BFIELDY,    BFIELDZ};

    VertexBufferHandle shock_field[] = {VTXBUF_SHOCK};

    // MV(?): Causes communication related error (unclear comment)
    AcTaskDefinition shock_ops[] =
        {acHaloExchange(all_fields),
         acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
         acCompute(KERNEL_shock_1_divu, shock_field),
         acHaloExchange(shock_field),
         acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
         acCompute(KERNEL_shock_2_max, shock_field),
         acHaloExchange(shock_field),
         acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
         acCompute(KERNEL_shock_3_smooth, shock_field),
         acHaloExchange(shock_field),
         acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
         acCompute(KERNEL_singlepass_solve, all_fields)};

    custom_simulation_graph = acGridBuildTaskGraph(shock_ops);
    simulation_graph = custom_simulation_graph;

    //TODO: Why is this here?
    //acGridSynchronizeStream(STREAM_ALL);
#endif

    ///////////////////////////////////////////////
    // Define variables for main simulation loop //
    ///////////////////////////////////////////////

    // Run-control variables
    // --------------------
    int found_nan  = 0; // Nan or inf finder to give an error signal
    int found_stop = 0;

    const int start_step    = info.int_params[AC_start_step];
    const int max_steps     = info.int_params[AC_max_steps];
    const int save_steps    = info.int_params[AC_save_steps];
    const int bin_steps     = info.int_params[AC_bin_steps];

    const AcReal max_time   = info.real_params[AC_max_time];
    const AcReal bin_save_t = info.real_params[AC_bin_save_t];
    AcReal bin_crit_t       = bin_save_t;
    AcReal t_step           = 0.0;

    // Additional physics variables
    // ----------------------------
    AcReal sink_mass     = -1.0;
    AcReal accreted_mass = -1.0;
    // TODO: hide these in some structure
    
#if LSINK
    sink_mass     = info.real_params[AC_M_sink_init];
    accreted_mass = 0.0;
#endif

    ////////////////////////////////////////////////////////////////////////////
    // Initialize mesh again ? Old initialization code, probably incompatible //
    ////////////////////////////////////////////////////////////////////////////
    
    /*
    if (start_step > 0) {
	//TODO: this part is untested, and may clash with the new initial_mesh_procedure
	//Maybe it can be moved to one of the options
        log_from_root_proc(pid, "Calling read_mesh_mpi\n");
        read_mesh_mpi(pid, start_step, &t_step);
        log_from_root_proc(pid, "Returned from read_mesh_mpi\n");
        bin_crit_t = bin_crit_t + t_step; 
    }
    */

    ///////////////////////////////////////////////////////////
    // Open output files and write the initial state to them //
    ///////////////////////////////////////////////////////////

    FILE* diag_file         = fopen("timeseries.ts", "a");
    ERRCHK_ALWAYS(diag_file);

    if (start_step == 0) {
            //TODO: calculate time step before entering loop, recalculate at end
        if (pid == 0) {
            log_from_root_proc(pid, "Writing out initial state to timeseries.ts\n");
	    print_diagnostics_header(diag_file);
	}

	//TODO: put this and the slice writing thing into a function, to make them the same
	
        print_diagnostics(start_step, 0, t_step, diag_file, sink_mass, accreted_mass, &found_nan);


        char label[4096];
        sprintf(label, "step_%012d", 0);
        log_from_root_proc(pid, "Syncing slice disk access\n");
	acGridDiskAccessSync();
	log_from_root_proc(pid, "Writing slices to %s, timestep = %d\n", slice_dir, start_step);
	//TODO: create new slice_dir for this frame and write to it
	acGridWriteSlicesToDiskLaunch(slice_dir, label);
	log_from_root_proc(pid, "Done writing slices\n");
    } else {
	// add newline to old diag_file from previous run
        if (pid == 0) {
	    fprintf(diag_file, "\n");
        }
    }

    // Save zero state 
    if (start_step <= 0) {
        log_from_root_proc(pid, "Writing out initial state to a mesh file\n");
        log_from_root_proc(pid, "Calling save_mesh_mpi_sync\n");
	save_mesh_mpi_sync(info, pid, 0, 0.0); // // %JP: mesh not available, changed the first param from AcMesh to AcMeshInfo
        log_from_root_proc(pid, "Returned from save_mesh_mpi_sync\n");
    }


    //////////////////////////
    // Main simulation loop //
    //////////////////////////
    
    log_from_root_proc(pid, "Starting simulation\n");

    //This loop has a legacy off-by-one error, if max_steps = 1, we will run 0 steps
    //if max_steps is 100, we will run 99 steps
    int i = start_step + 1;
    for (; i < max_steps; ++i) {

        debug_from_root_proc(pid, "Calculating time step (i=%d)\n", i);
        const AcReal dt = calc_timestep(info);
        debug_from_root_proc(pid, "Done calculating time step (i=%d)\n", i);
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_current_time, t_step);

#if LSINK
        AcReal sum_mass;
        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, VTXBUF_ACCRETION, &sum_mass);
        accreted_mass = accreted_mass + sum_mass;
        sink_mass     = 0.0;
        sink_mass     = info.real_params[AC_M_sink_init] + accreted_mass;
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_M_sink, sink_mass);

        // JP: !!! WARNING !!! acVertexBufferSet operates in host memory. The mesh is
        // never loaded to device memory. Is this intended?
        if (pid == 0){
            acVertexBufferSet(VTXBUF_ACCRETION, 0.0, mesh);
        }

        int on_off_switch;
        if (i < 1) {
            on_off_switch = 0; // accretion is off till certain amount of steps.
        }
        else {
            on_off_switch = 1;
        }
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_switch_accretion, on_off_switch);
#else
        accreted_mass = -1.0;
        sink_mass = -1.0;
#endif

#if LFORCING
        // if (info.int_params[AC_lforcing] == 1) {
        const ForcingParams forcing_params = generateForcingParams(info);
        loadForcingParamsToGrid(forcing_params);
        //}
#endif

        // Set the time delta
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, dt);
        acGridSynchronizeStream(STREAM_DEFAULT);

        // Execute the active task graph for 3 iterations
	// in the case that simulation_graph = acGridGetDefaultTaskGraph(), then this is equivalent to acGridIntegrate(STREAM_DEFAULT, dt)
        acGridExecuteTaskGraph(simulation_graph, 3);
        log_from_root_proc(pid, "acGridExecuteTaskGraph step %d complete\n", i);
        
        t_step += dt;

	//TODO: debug everything in the if below and the exits at the end
	//some process will exit after writing slices with USE_DISTRIBUTED_IO=ON
	
        /* Save the simulation state and print diagnostics */
        if ((i % save_steps) == 0) {

            /*
                print_diagnostics() writes out both std.out printout from the
                results and saves the diagnostics into a table for ascii file
                timeseries.ts.
            */

	    
            print_diagnostics(i, dt, t_step, diag_file, sink_mass, accreted_mass, &found_nan);
            /*
                We would also might want an XY-average calculating funtion,
                which can be very useful when observing behaviour of turbulent
                simulations. (TODO)
            */
                        
            acGridPeriodicBoundconds(STREAM_DEFAULT);

            //////////////////
	    // Write slices //
            //////////////////
	    
            char label[4096];

            sprintf(label, "step_%012d", i);

            log_from_root_proc(pid, "Syncing slice disk access\n");
            acGridDiskAccessSync();
            log_from_root_proc(pid, "Slice disk access synced\n");
	    //TODO: create new slice_dir for this frame and write to it
            log_from_root_proc(pid, "Writing slices to %s, timestep = %d\n", slice_dir, i);
            acGridWriteSlicesToDiskLaunch(slice_dir, label);
	    log_from_root_proc(pid, "Done writing slices\n");

	    //This was here to debug an issue that somehow resolved itself... can't recall what the issue was anymore
	    //Anyway, in some cases
	    //debug_from_root_proc(pid, "Calling post-slice barrier\n");
	    //MPI_Barrier(MPI_COMM_WORLD);
	    //debug_from_root_proc(pid, "Passed post-slice barrier\n");
        }



        /* Save the simulation state and print diagnostics */
        if ((i % bin_steps) == 0 || t_step >= bin_crit_t) {
            /*
                This loop saves the data into simple C binaries which can be
                used for analysing the data snapshots closely.

                The updated mesh will be located on the GPU. Also all calls
                to the astaroth interface (functions beginning with ac*) are
                assumed to be asynchronous, so the meshes must be also synchronized
                before transferring the data to the CPU. Like so:

                acBoundcondStep();
                acStore(mesh);
            */

            // Write snapshots
	    log_from_root_proc(pid, "Writing snapshots to %s, timestep = %d\n", snapshot_dir, i);
            acGridDiskAccessSync();
            save_mesh_mpi_async(info, snapshot_dir, pid, i, t_step);
	    log_from_root_proc(pid, "Done writing snapshots\n");

            bin_crit_t += bin_save_t;
        }

        // Ensures that are known beyond rank 0.
	// TODO: why only found_nan? why not max_time?
        MPI_Bcast(&found_nan, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (access("STOP", F_OK) != -1) {
            found_stop = 1;
        }
        else {
            found_stop = 0;
        }

        // End loop if max time reached.
        if (max_time > AcReal(0.0)) {
            if (t_step >= max_time) {
		printf("Time limit reached! at t = %e \n", double(t_step));
                break;
            }
        }

        // End loop if nan is found
        if (found_nan > 0) {
	    // Only proc 0 will write
            printf("Found nan at t = %e \n", double(t_step));
            break;
        }

        if (found_stop == 1) {
            printf("Found STOP file at t = %e \n", double(t_step));
            break;
        }
    }

    // Save data after the loop ends
    // TODO: do this inside the loop, before the break? then we don't need i as a loop-external variable
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    save_mesh_mpi_sync(info, pid, std::min(max_steps -1, i), t_step);

    // Clean up allocated resources
    if (custom_simulation_graph != nullptr){
        acGridDestroyTaskGraph(custom_simulation_graph);
    }
    acGridQuit();
    fclose(diag_file);
    MPI_Finalize();

    //TODO: might want to change this to EXIT_FAILURE if found_nan
    return EXIT_SUCCESS;
}

#else
#include <cstdio>
#include <cstdlib>

int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
