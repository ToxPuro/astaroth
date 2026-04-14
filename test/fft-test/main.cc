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
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"
#include "user_constants.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>


static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}
double
drand()
{
	return (double)(rand()) / (double)(rand());
}
const AcReal3
get_wavevector(const int3 index, const AcMeshInfo info)
{
	const int3 global_idx  = (int3)
	{
		index.x - info[AC_nmin].x,
		index.y - info[AC_nmin].y,
		index.z - info[AC_nmin].z
	};
	const auto k_x = info[AC_frequency_spacing].x*((global_idx.x <= info[AC_ngrid].x/2) ? global_idx.x : global_idx.x - info[AC_ngrid].x);
	const auto k_y = info[AC_frequency_spacing].y*((global_idx.y <= info[AC_ngrid].y/2) ? global_idx.y : global_idx.y - info[AC_ngrid].y);
	const auto k_z = info[AC_frequency_spacing].z*((global_idx.z <= info[AC_ngrid].z/2) ? global_idx.z : global_idx.z - info[AC_ngrid].z);
	return (AcReal3){k_x,k_y,k_z};
}


int
main(void)
{
    atexit(acAbort);

    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("fft.conf", &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
	    (2*AC_REAL_PI)/info[AC_ngrid].x,
	    (2*AC_REAL_PI)/info[AC_ngrid].y,
	    (2*AC_REAL_PI)/info[AC_ngrid].z
    });

    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    info.comm->handle = MPI_COMM_WORLD;

    const int max_devices = 64;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetGridMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    const int3 decomp = acDecompose(nprocs,info);
    acSetLocalMeshDims(info[AC_ngrid].x/decomp.x,info[AC_ngrid].y/decomp.y,info[AC_ngrid].z/decomp.z, &info);

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DFFT_ENABLED=ON -DUSE_HEFFTE=OFF -DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);

    acGridInit(info);
    acGridExecuteTaskGraph(acGetDSLTaskGraph(fft_solve),1);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond),1);
    acDeviceFFTR2C(acGridGetDevice(),HEAT_INIT,HEAT_COMPLEX);
    acDeviceFFTR2Planar(acGridGetDevice(),HEAT_INIT,HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG);
    AcMeshDims comp_dims = acGetMeshDims(acGridGetLocalMeshInfo());
    acGridExecuteTaskGraph(acGetDSLTaskGraph(fft_solve),1);
    acGridExecuteTaskGraph(acGetDSLTaskGraph(fft_planar_solve),1);
    acDeviceFFTC2R(acGridGetDevice(),HEAT_COMPLEX_SOLUTION,HEAT_SOLUTION);
    acDeviceFFTC2R(acGridGetDevice(),HEAT_COMPLEX_PLANAR_SOLUTION,HEAT_PLANAR_SOLUTION);
    acDeviceFFTC2R(acGridGetDevice(),HEAT_COMPLEX,HEAT_FORWARD_AND_BACK);
    acGridWriteSlicesToDiskCollectiveSynchronous("slices", 0, 0.0);
    acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &model);
    acGridSynchronizeStream(STREAM_ALL);
    auto IDX = [](const int x, const int y, const int z)
    {
	return acVertexBufferIdx(x,y,z,acGridGetLocalMeshInfo());
    };

    AcReal epsilon  =  8*pow(10.0,-11.0);
    auto relative_diff = [](const auto a, const auto b)
    {
            const auto abs_diff = fabs(a-b);
            return  abs_diff/a;
    };
    auto in_eps_threshold = [&](const auto a, const auto b)
    {
       if(a == b) return true;
            return relative_diff(a,b) < epsilon;
    };

    bool forward_and_back_correct = true;
    for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    {
       for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
       {
               for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
               {
                      const auto original_val = model.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      const auto res_val      = model.vertex_buffer[HEAT_FORWARD_AND_BACK][IDX(x,y,z)];
                      if(!in_eps_threshold(original_val,res_val))
                      {
                              if(forward_and_back_correct) fprintf(stderr,"forward and back wrong at (%zu,%zu,%zu) %.14e vs. %.14e\n",x,y,z,original_val,res_val);
			      forward_and_back_correct = false;
                      }
       	}
       }
    }

    bool poisson_correct = true;
    for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    {
       for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
       {
               for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
               {
                      const auto analytical_val = model.vertex_buffer[HEAT_INIT][IDX(x,y,z)]/(-3.0);
                      const auto res_val  = model.vertex_buffer[HEAT_SOLUTION][IDX(x,y,z)];
                      if(!in_eps_threshold(analytical_val,res_val))
                      {
                              if(poisson_correct) fprintf(stderr,"Poisson wrong at %.14e vs. %.14e\n",analytical_val,res_val);
                              poisson_correct = false;
                      }
                      const auto planar_res_val  = model.vertex_buffer[HEAT_PLANAR_SOLUTION][IDX(x,y,z)];
                      if(!in_eps_threshold(analytical_val,planar_res_val))
                      {
                              if(poisson_correct) fprintf(stderr,"Planar solution wrong at %.14e vs. %.14e diff: %.14e\n",analytical_val,planar_res_val,relative_diff(analytical_val,planar_res_val));
                              poisson_correct = false;
                      }
       	}
       }
    }
    bool xy_correct = true;
    /**
    {
    	acHostMeshRandomize(&model);
    	acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	        for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
    	        {
			model.vertex_buffer[HEAT_INIT][IDX(x,y,z)] = sin(k.x*spatial_pos.x)*sin(k.y*spatial_pos.y)*sin(k.z*spatial_pos.z)
		}
	   }
	}
        acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
	acGridSynchronizeStream(STREAM_ALL);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
    	           {
                      const auto src_val = model.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      const auto res_val = candidate.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      if(!in_eps_threshold(src_val,res_val))
		      {
                        if(xy_correct) fprintf(stderr,"XY back and forth not correct at %.14e vs. %.14e\n",src_val,res_val);
			xy_correct = false;
		      }
		   }
	   }
	}
    }
    **/
    {
    	acHostMeshRandomize(&model);
    	acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
        acDeviceFFTR2Planar(acGridGetDevice(),HEAT_INIT,HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG);
        acDeviceFFTBackwardTransformPlanar2R(acGridGetDevice(),HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG,HEAT_INIT);
	acGridSynchronizeStream(STREAM_ALL);
        acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
	acGridSynchronizeStream(STREAM_ALL);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
    	           {
                      const auto src_val = model.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      const auto res_val = candidate.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      if(!in_eps_threshold(src_val,res_val))
		      {
                        if(xy_correct) fprintf(stderr,"Planar back and forth not correct at (%zu,%zu,%zu) %.14e vs. %.14e, relative diff: %.14e\n",x,y,z,src_val,res_val,relative_diff(src_val,res_val));
			xy_correct = false;
		      }
		   }
	   }
	}
    }

    fprintf(stderr,"3d sin(kx)+sin(ky)+sin(kz)\n");
    bool threed_correct = true;
    {
    	acHostMeshRandomize(&model);
    	acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
        acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(three_dimensional_test_setup),1);
	acGridSynchronizeStream(STREAM_ALL);
        acDeviceFFTR2Planar(acGridGetDevice(),HEAT_INIT,HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG);
        acDeviceFFTR2PlanarBatched(acGridGetDevice(),HEAT_INIT_SINGLE,HEAT_PLANAR_REAL_SINGLE,HEAT_PLANAR_IMAG_SINGLE,1);
        acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(copy_single_to_output),1);
        acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
	acGridSynchronizeStream(STREAM_ALL);
	int number_of_nonzero_components = 0;
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
		   {
        		const AcReal real = candidate.vertex_buffer[HEAT_PLANAR_REAL][IDX(x,y,z)];
			const AcReal imag = candidate.vertex_buffer[HEAT_PLANAR_IMAG][IDX(x,y,z)];
        		const AcReal single_real = candidate.vertex_buffer[HEAT_PLANAR_REAL_SINGLE_OUTPUT][IDX(x,y,z)];
			const AcReal single_imag = candidate.vertex_buffer[HEAT_PLANAR_IMAG_SINGLE_OUTPUT][IDX(x,y,z)];
			if(fabs(real) > epsilon || fabs(imag) > epsilon)
			{
				++number_of_nonzero_components;
				if(number_of_nonzero_components < 100)
				{
					fprintf(stderr,"3d Fourier coeffs at (%zu,%zu,%zu): %.14e,%.14e\n",x,y,z,real,imag);
					fprintf(stderr,"3d Amplitude at (%zu,%zu,%zu): %.14e\n",x,y,z,sqrt(real*real + imag*imag));

					fprintf(stderr,"Single precision 3d Fourier coeffs at (%zu,%zu,%zu): %.14e,%.14e\n",x,y,z,single_real,single_imag);
					fprintf(stderr,"Single precision 3d Amplitude at (%zu,%zu,%zu): %.14e\n",x,y,z,sqrt(single_real*single_real + single_imag*single_imag));
				}
			}
		   }
	   }
	}
    	threed_correct &= (number_of_nonzero_components == 6);
	if(number_of_nonzero_components != 6)
	{
		fprintf(stderr,"Expected six non-zero components but got %d\n",number_of_nonzero_components);
	}
    }

    bool twod_correct = true;
    fprintf(stderr,"2d sin(kx)+sin(ky)\n");
    {
    	acHostMeshRandomize(&model);
    	acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
	const int3 pos = (int3){8,8,8};
	const AcReal3 k = get_wavevector(pos,model.info);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
    	           {
		      const AcReal3 spatial_pos = (AcReal3){(x-NGHOST)*info[AC_ds].x,(y-NGHOST)*info[AC_ds].y,(z-NGHOST)*info[AC_ds].z};
		      model.vertex_buffer[HEAT_INIT][IDX(x,y,z)] = sin(k.x*spatial_pos.x)+sin(k.y*spatial_pos.y);
		      model.vertex_buffer[HEAT_PLANAR_REAL][IDX(x,y,z)] = 0.0;
		      model.vertex_buffer[HEAT_PLANAR_IMAG][IDX(x,y,z)] = 0.0;
		   }
	   }
	}
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
	const int z_layer = 15;
        acDeviceFFTR2PlanarXY(acGridGetDevice(),HEAT_INIT,HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG,z_layer);
	acDeviceFFTBackwardTransformPlanar2RXY(acGridGetDevice(),HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG,HEAT_INIT,z_layer);
        acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
	acGridSynchronizeStream(STREAM_ALL);
	int number_of_nonzero_components = 0;
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
		   {
        		const AcReal real = candidate.vertex_buffer[HEAT_PLANAR_REAL][IDX(x,y,z)];
			const AcReal imag = candidate.vertex_buffer[HEAT_PLANAR_IMAG][IDX(x,y,z)];
			if(fabs(real) > epsilon || fabs(imag) > epsilon)
			{
				++number_of_nonzero_components;
				if(number_of_nonzero_components < 100)
				{
					fprintf(stderr,"2d Fourier coeffs at (%zu,%zu,%zu): %.14e,%.14e\n",x,y,z,real,imag);
					fprintf(stderr,"2d Amplitude at (%zu,%zu,%zu): %.14e\n",x,y,z,sqrt(real*real + imag*imag));
				}
    				twod_correct &= (z == z_layer);
			}
		   }
	   }
	}
    	twod_correct &= (number_of_nonzero_components == 4);
	if(number_of_nonzero_components != 4)
	{
		fprintf(stderr,"Expected four non-zero components but got %d\n",number_of_nonzero_components);
	}

    	acHostMeshRandomize(&model);
    	acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
        acDeviceFFTR2PlanarXY(acGridGetDevice(),HEAT_INIT,HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG,z_layer);
	acDeviceFFTBackwardTransformPlanar2RXY(acGridGetDevice(),HEAT_PLANAR_REAL,HEAT_PLANAR_IMAG,HEAT_INIT,z_layer);
        acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
	acGridSynchronizeStream(STREAM_ALL);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
		   {
        		const AcReal orig =           model.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
			const AcReal back_and_forth = candidate.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
			if(!in_eps_threshold(orig,back_and_forth))
			{
                              	if(forward_and_back_correct) fprintf(stderr,"2D back and forth wrong %.14e vs. %.14e, relative diff: %.14e\n",orig,back_and_forth,relative_diff(orig,back_and_forth));
    				forward_and_back_correct = false;
			}
		   }
	   }
	}
    }

    /**
    if(test_2d)
    {
    	acHostMeshRandomize(&model);
    	acHostMeshRandomize(&candidate);
        acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT, model);
	acGridSynchronizeStream(STREAM_ALL);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {

		//const int z = 8;
		//acDeviceFFTR2PlanarXY(acGridGetDevice(), HEAT_INIT, HEAT_PLANAR_REAL, HEAT_PLANAR_IMAG, z);
		//acDeviceFFTBackwardTransformPlanar2RXY(acGridGetDevice(),  HEAT_PLANAR_REAL, HEAT_PLANAR_IMAG, HEAT_INIT, z);
	   }
	}
        acDeviceStoreMesh(acGridGetDevice(), STREAM_DEFAULT, &candidate);
	acGridSynchronizeStream(STREAM_ALL);
    	for(auto x = comp_dims.n0.x; x < comp_dims.n1.x; ++x)
    	{
    	   for(auto y = comp_dims.n0.y; y < comp_dims.n1.y; ++y)
    	   {
    	           for(auto z = comp_dims.n0.z; z < comp_dims.n1.z; ++z)
    	           {
                      const auto src_val = model.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      const auto res_val = candidate.vertex_buffer[HEAT_INIT][IDX(x,y,z)];
                      if(!in_eps_threshold(src_val,res_val))
		      {
                        if(xy_correct) fprintf(stderr,"XY back and forth not correct at %.14e vs. %.14e\n",src_val,res_val);
			xy_correct = false;
		      }
		   }
	   }
	}
    }
    **/
 
    fprintf(stderr,"FORWARD AND BACK ... %s\n", forward_and_back_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"3D CORRECT       ... %s\n", threed_correct ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    fprintf(stderr,"2D CORRECT       ... %s\n", twod_correct   ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    bool correct = forward_and_back_correct && poisson_correct && xy_correct;
    int retval = correct ? AC_SUCCESS : AC_FAILURE;

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "FFT_TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return retval == AC_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
