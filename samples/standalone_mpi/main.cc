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
/**
    Running: mpirun -np <num processes> <executable>
*/
#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_utils.h"

#include <mpi.h>
#include <string.h>

#include <unistd.h>

#include "config_loader.h"
#include "errchk.h"
#include "host_forcing.h"
#include "host_memory.h"

#include "math_utils.h"

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

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

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
    fprintf(infotxt, "rm *.list *.mesh *.csv *.field *.ts purge.sh\n");
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
        strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, ".", bin_filename, ACCESS_WRITE);
       
        printf("Savefile %s \n", bin_filename);

        acGridDiskAccessSync();
    }
}

static inline void
save_mesh_mpi_async(const AcMeshInfo info, const char* job_dir, const int pid, const int step, const AcReal t_step)
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
        strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, job_dir, bin_filename, ACCESS_WRITE);
        // %JP TODO write async
       
        printf("Savefile %s \n", bin_filename);
    }
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
        strcat(bin_filename, ".field");

        // Grid data
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)w, ".", bin_filename, ACCESS_READ);
       
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

int
main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    /////////// Simple example START
    (void)argc; // Unused
    (void)argv; // Unused

    // Set random seed for reproducibility
    srand(321654987);

    // Load config to AcMeshInfo
    AcMeshInfo info;
    if (argc > 1) {
        if (argc == 3 && (!strcmp(argv[1], "-c") || !strcmp(argv[1], "--config"))) {
            acLoadConfig(argv[2], &info);
            load_config(argv[2], &info);
            acHostUpdateBuiltinParams(&info);
        }
        else {
            printf("Usage: ./ac_run\n");
            printf("Usage: ./ac_run -c <config_path>\n");
            printf("Usage: ./ac_run --config <config_path>\n");
            return EXIT_FAILURE;
        }
    }
    else {
        acLoadConfig(AC_DEFAULT_CONFIG, &info);
        load_config(AC_DEFAULT_CONFIG, &info);
        acHostUpdateBuiltinParams(&info);
    }

    const int start_step     = info.int_params[AC_start_step];
    const int max_steps      = info.int_params[AC_max_steps];
    const int save_steps     = info.int_params[AC_save_steps];
    const int bin_save_steps = info.int_params[AC_bin_steps];

    const AcReal max_time   = info.real_params[AC_max_time];
    const AcReal bin_save_t = info.real_params[AC_bin_save_t];
    AcReal bin_crit_t       = bin_save_t;
    AcReal t_step           = 0.0;
    FILE* diag_file         = fopen("timeseries.ts", "a");
    ERRCHK_ALWAYS(diag_file);

    const int init_type = info.int_params[AC_init_type];

    int found_nan  = 0; // Nan or inf finder to give an error signal
    int istep      = 0;
    int found_stop = 0;

    // AcMesh mesh; // %JP: Disabled, large grids will not fit into host memory
    ///////////////////////////////// PROC 0 BLOCK START ///////////////////////////////////////////
    if (pid == 0) {
        //acHostMeshCreate(info, &mesh); // %JP: Disabled, large grids will not fit into host memory
        //acmesh_init_to((InitType)init_type, &mesh); // %JP: Disabled, large grids will not fit into host memory

#if LSINK
        acVertexBufferSet(VTXBUF_ACCRETION, 0.0, &mesh);
#endif
        // Read old binary if we want to continue from an existing snapshot
        // WARNING: Explicit specification of step needed!
        //if (start_step > 0) {
        //    read_mesh(mesh, start_step, &t_step);
        //}

        // Generate the title row.
        if (start_step == 0) {
            fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
#if LBFIELD
            fprintf(diag_file, "bb_total_min  bb_total_rms  bb_total_max  ");
            fprintf(diag_file, "vA_total_min  vA_total_rms  vA_total_max  ");
#endif
            for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
                fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
                        vtxbuf_names[i]);
            }
        }

#if LSINK
        fprintf(diag_file, "sink_mass  accreted_mass  ");
#endif
        fprintf(diag_file, "\n");

        write_info(&info);

        if (start_step == 0) {
#if LSINK
            print_diagnostics_host(mesh, 0, AcReal(.0), t_step, diag_file,
                                   info.real_params[AC_M_sink_init], 0.0);
#else
            // print_diagnostics_host(mesh, 0, AcReal(.0), t_step, diag_file, -1.0, -1.0); // %JP: Disabled, large grids will not fit into host memory
#endif
        }

        //acHostMeshApplyPeriodicBounds(&mesh); // %JP: Disabled, large grids will not fit into host memory
    
    }
    ////////////////////////////////// PROC 0 BLOCK END ////////////////////////////////////////////


    // Init GPU
    acGridInit(info);
    // acGridLoadMesh(STREAM_DEFAULT, mesh); // %JP: Disabled, large grids will not fit into host memory
    
    // %JP start
    // Read PC varfile to Astaroth
    const char* file = "/scratch/project_462000077/mkorpi/forced/mahti_4096/data/allprocs/var.dat";
    const int3 nn = (int3){4096, 4096, 4096};
    //const char* file = "test.dat";
    //const int3 nn = (int3){64, 64, 64};
    const Field fields[]    = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, VTXBUF_LNRHO,
                            VTXBUF_AX,  VTXBUF_AY,  VTXBUF_AZ};
    const size_t num_fields = ARRAY_SIZE(fields);
    const int3 rr = (int3){3, 3, 3};
    acGridReadVarfileToMesh(file, fields, num_fields, nn, rr);
    // %JP end

    // %JP NOTE: need to perform a dryrun (all kernels) if switching
    // to two-pass integration, otherwise the output buffers
    // may be invalid due to automated performance tuning

    /*
    // Scale the fields
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    AcReal max, min, sum;
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)i, &max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)i, &min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)i, &sum);
        if (!pid)
                printf("max %g, min %g, sum %g\n", (double)max, (double)min, (double)sum);
    }

    acGridLoadScalarUniform(STREAM_DEFAULT, AC_scaling_factor, (AcReal)2.0);
    AcMeshDims dims = acGetMeshDims(info);
    acGridLaunchKernel(STREAM_DEFAULT, scale, dims.n0, dims.n1);
    acGridSwapBuffers();
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)i, &max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)i, &min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)i, &sum);
        if (!pid)
                printf("max %g, min %g, sum %g\n", (double)max, (double)min, (double)sum);
    }
    */

    /* initialize random seed: */
    srand(312256655);


#if LSHOCK
    // From taskgraph example
    // First we define what fields we're using.
    // This parameter is a c-style array but only works with c++ at the moment
    //(the interface relies on templates for safety and array type deduction).
    VertexBufferHandle all_fields[] = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                       VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ, // VTXBUF_ENTROPY,
                                       VTXBUF_SHOCK, BFIELDX,    BFIELDY,    BFIELDZ};

    // VertexBufferHandle shock_uu_fields[] = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, VTXBUF_SHOCK};

    VertexBufferHandle shock_field[] = {VTXBUF_SHOCK};
    // VertexBufferHandle shock_field[] = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
    //                                   VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ,
    //                                   BFIELDX, BFIELDY, BFIELDZ};

    // BASIC AcTaskDefinition ALTERNATIVES:

    // This works OK
    // AcTaskDefinition shock_ops[] = {acHaloExchange(all_fields),
    //                                acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,
    //                                all_fields), acCompute(KERNEL_solve, all_fields)};

    // This causes the chess board error
    // AcTaskDefinition shock_ops[] = {acHaloExchange(all_fields),
    //                                acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,
    //                                all_fields), acCompute(KERNEL_shock_1_divu, shock_field),
    //                                acCompute(KERNEL_shock_2_max, shock_field),
    //                                acCompute(KERNEL_shock_3_smooth, shock_field),
    //                                acCompute(KERNEL_solve, all_fields)};


    // Causes communication related error
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

    // AcTaskDefinition shock_ops[] = {acHaloExchange(all_fields),
    //                                acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,
    //                                all_fields), acCompute(KERNEL_shock_1_divu, shock_field),
    //                                acCompute(KERNEL_shock_2_max, shock_field),
    //                                acCompute(KERNEL_solve, all_fields)};

    AcTaskGraph* hc_graph = acGridBuildTaskGraph(shock_ops);

    acGridSynchronizeStream(STREAM_ALL);

    // Build a task graph consisting of:
    // - a halo exchange with periodic boundconds for all fields
    // - a calculation of the solve kernel touching all fields
    //
    // This function call generates tasks for each subregions in the domain
    // and figures out the dependencies between the tasks.
    // AcTaskGraph* hc_graph = acGridBuildTaskGraph(
    //    {acHaloExchange(all_fields),
    //     acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
    // acCompute(KERNEL_shock_1_divu, shock_uu_fields),
    // acHaloExchange(shock_field),
    // acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
    // acCompute(KERNEL_shock_2_max, shock_field),
    // acHaloExchange(shock_field),
    // acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
    // acCompute(KERNEL_shock_3_smooth, shock_field),
    // acHaloExchange(shock_field),
    // acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
    //     acCompute(KERNEL_solve, all_fields)});

    // We can build multiple TaskGraphs, the MPI requests will not collide
    // because MPI tag space has been partitioned into ranges that each HaloExchange step uses.
    /*
    AcTaskGraph* shock_graph = acGridBuildTaskGraph({
        acHaloExchange(all_fields),
        acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, all_fields),
        acCompute(KERNEL_shock1, all_fields),
        acCompute(KERNEL_shock2, shock_fields),
        acCompute(KERNEL_solve, all_fields)
    });
    */
#endif

    if (start_step > 0) {
        read_mesh_mpi(pid, start_step, &t_step);
        bin_crit_t = bin_crit_t + t_step; 
    }

    // Save zero state 
    if (start_step <= 0) save_mesh_mpi_sync(info, pid, 0, 0.0); // // %JP: mesh not available, changed the first param from AcMesh to AcMeshInfo

    /* Step the simulation */
    AcReal accreted_mass = 0.0;
    AcReal sink_mass     = 0.0;
    for (int i = start_step + 1; i < max_steps; ++i) {
        const AcReal dt = calc_timestep(info);
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
        if (pid == 0)
            acVertexBufferSet(VTXBUF_ACCRETION, 0.0, mesh);

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
        // MV 2021-07-13 Code seems fine in terms of functionality
        // MV TODO: Make it possible, using the task system, to run shock viscosity.
        // MV TODO: Make it possible, using the task system, to run nonperiodic boundaty conditions.
        // MV TODO: See if there are other features from the normal standalone which I would like to
        // include.

#if LSHOCK
        // Set the time delta
        acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, dt);
        acGridSynchronizeStream(STREAM_DEFAULT);

        // Execute the task graph for three iterations.
        acGridExecuteTaskGraph(hc_graph, 3);
#else

        acGridIntegrate(STREAM_DEFAULT, dt);
#endif

        t_step += dt;

        /* Save the simulation state and print diagnostics */
        if ((i % save_steps) == 0) {

            /*
                print_diagnostics() writes out both std.out printout from the
                results and saves the diagnostics into a table for ascii file
                timeseries.ts.
            */

            print_diagnostics(i, dt, t_step, diag_file, sink_mass, accreted_mass, &found_nan);
#if LSINK
            printf("sink mass is: %.15e \n", double(sink_mass));
            printf("accreted mass is: %.15e \n", double(accreted_mass));
#endif
            /*
                We would also might want an XY-average calculating funtion,
                which can be very useful when observing behaviour of turbulent
                simulations. (TODO)
            */
        }

        /* Save the simulation state and print diagnostics */
        if ((i % bin_save_steps) == 0 || t_step >= bin_crit_t) {

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
            acGridPeriodicBoundconds(STREAM_DEFAULT);
            //acGridStoreMesh(STREAM_DEFAULT, &mesh);

            // %JP start
            //acGridDiskAccessSync();
            //save_mesh_mpi_async(info, pid, i, t_step); // Snapshots should be written out only rarely, disabled for now

            // Create a tmpdir for output
            const int job_id = 12345;
            char job_dir[4096];
            snprintf(job_dir, 4096, "output-slices-%d", job_id);

            char cmd[4096];
            snprintf(cmd, 4096, "mkdir -p %s", job_dir);
            system(cmd);

            // Write slices
            acGridDiskAccessSync();
            acGridWriteSlicesToDisk(job_dir, i); // %JP: TODO make async
            // %JP end

            bin_crit_t += bin_save_t;
        }

        const int snapshot_save_interval = 100; // %JP: TODO make a config param
        if (!(i % snapshot_save_interval)) {
            // Create a tmpdir for output
            const int job_id = 12345;
            char job_dir[4096];
            snprintf(job_dir, 4096, "output-snapshots-%d", job_id);

            char cmd[4096];
            snprintf(cmd, 4096, "mkdir -p %s", job_dir);
            system(cmd);

            // Write snapshots
            acGridDiskAccessSync();
            save_mesh_mpi_async(info, job_dir, pid, i, t_step);
        }


        istep = i;

        // Ensures that are known beyond rank 0.
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
            printf("Found nan at t = %e \n", double(t_step));
            break;
        }

        if (found_stop == 1) {
            printf("Found STOP file at t = %e \n", double(t_step));
            break;
        }
    }
    // Save data after the loop ends
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    //acGridStoreMesh(STREAM_DEFAULT, &mesh); // %JP: Disabled, large grids will not fit into host memory

    save_mesh_mpi_sync(info, pid, istep, t_step);

#if LSHOCK
    acGridDestroyTaskGraph(hc_graph);
#endif
    acGridQuit();
    //if (pid == 0)
    //    acHostMeshDestroy(&mesh); // %JP: Disabled, large grids will not fit into host memory
    fclose(diag_file);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

#else
#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
