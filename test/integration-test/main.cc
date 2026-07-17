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
#include "user_constants.h"
#include "errchk.h"
#include <gsl/gsl_integration.h>

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define NUM_INTEGRATION_STEPS (2)

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

void
read_data_to_arr(AcMeshInfo& info, const AcRealArrayParam arr, const char* filename, const AcIntParam N_param)
{
    std::vector<AcReal> data{};
    FILE *fp = fopen(filename, "r");
    if (fp != NULL) {
      AcReal entry;
      while (fscanf(fp, "%lf,", &entry) == 1)
      {
	  data.push_back(entry);
      }
      acPushToConfig(info,N_param,(int)data.size());
      AcReal* res = (AcReal*)malloc(sizeof(AcReal)*info[N_param]);
      for(int i = 0; i < info[N_param]; ++i) res[i] = data[i];
      info[arr] = res;
      fclose(fp);
    }
}

AcReal
linear_interpol(const AcReal& x0, const AcReal& x1, const AcReal& y0, const AcReal& y1, const AcReal& x) 
{
	return (y0*(x1-x) + y1*(x-x0))/(x1-x0);
}


int
main(int argc, char* argv[])
{


    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);


    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("integration.conf", &info);

    if(argc > 1) 
    {
	    info[AC_integration_points_x] = atoi(argv[1]);
    }
    if(argc > 2) 
    {
	    info[AC_integration_points_y] = atoi(argv[2]);
    }
    if(argc > 3) 
    {
	    info[AC_integration_points_z] = atoi(argv[3]);
    }
    if(argc > 4) 
    {
	    info[AC_integration_points_w] = atoi(argv[4]);
    }
    if(argc > 5) 
    {
	    info[AC_k] = atof(argv[5]);
    }

    if(info[AC_logspace_x])
    {
	    if(info[AC_integration_points_x] > 1)
	    {
	      info[AC_integration_start_x] = log(info[AC_integration_start_x]);
	      info[AC_integration_end_x] = log(info[AC_integration_end_x]);
	    }
    }
    if(info[AC_logspace_y])
    {
	    if(info[AC_integration_points_y] > 1)
	    {
	      info[AC_integration_start_y] = log(info[AC_integration_start_y]);
	      info[AC_integration_end_y] = log(info[AC_integration_end_y]);
	    }
    }
    if(info[AC_logspace_z])
    {
	    if(info[AC_integration_points_z] > 1)
	    {
	      info[AC_integration_start_z] = log(info[AC_integration_start_z]);
	      info[AC_integration_end_z] = log(info[AC_integration_end_z]);
	    }
    }
    if(info[AC_logspace_w])
    {
	    if(info[AC_integration_points_w] > 1)
	    {
	      info[AC_integration_start_w] = log(info[AC_integration_start_w]);
	      info[AC_integration_end_w] = log(info[AC_integration_end_w]);
	    }
    }

    acPushToConfig(info,AC_ngrid,(int3){info[AC_integration_points_x],info[AC_integration_points_y],info[AC_integration_points_z]});
    acPushToConfig(info,AC_nlocal_w,info[AC_integration_points_w]);
    
    acPushToConfig(info,AC_first_gridpoint,(AcReal3){info[AC_integration_start_x],info[AC_integration_start_y],info[AC_integration_start_z]});
    acPushToConfig(info,AC_first_gridpoint_w,info[AC_integration_start_w]);

    acPushToConfig(info,AC_len,(AcReal3){info[AC_integration_end_x]-info[AC_integration_start_x],info[AC_integration_end_y]-info[AC_integration_start_y],info[AC_integration_end_z]-info[AC_integration_start_z]});
    acPushToConfig(info,AC_len_w,info[AC_integration_end_w]-info[AC_integration_start_w]);
    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_LINEAR);
    int3 decomp = {1,1,1};
    if(info[AC_integration_points_x] > 1)
    {
	decomp.x = nprocs;
    }
    else if(info[AC_integration_points_y] > 1)
    {
	decomp.y = nprocs;
    }
    else if(info[AC_integration_points_z] > 1)
    {
	decomp.z = nprocs;
    }
    else
    {
	    fprintf(stderr,"Only points in w is not allowed!\n");
	    exit(EXIT_FAILURE);
    }
    acPushToConfig(info,AC_domain_decomposition,decomp);
    const int3 pid3d = acGetPid3D(pid,decomp,info);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_EXTERNAL);
    acPushToConfig(info,AC_domain_coordinates,pid3d);
    acHostUpdateParams(&info); 

    info.comm->handle = MPI_COMM_WORLD;

    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    /**
    const size_t size = (size_t)info[AC_nlocal].x*info[AC_nlocal].y*info[AC_nlocal].z*info[AC_nlocal_w];
    AcReal* data_arr = (AcReal*)malloc(sizeof(AcReal)*size);
    memset(data_arr,0.0,size*sizeof(AcReal));
    if(data_arr == NULL)
    {
	    fprintf(stderr,"Was not able top allocate!\n");
	    fflush(stderr);
	    exit(EXIT_FAILURE);
    }
    const int nx = info[AC_nlocal].x;
    const int ny = info[AC_nlocal].y;
    const int nz = info[AC_nlocal].z;
    const AcReal k = 1.0;
    const AcReal v = 1.0/3.0;
    for(int x = 0; x < info[AC_nlocal].x; ++x)
    {
      AcReal t1 = info[AC_ds].x*x + info[AC_first_gridpoint].x;
      for(int y = 0; y < info[AC_nlocal].y; ++y)
      {
        AcReal t2 = info[AC_ds].y*y+ info[AC_first_gridpoint].y;
        for(int z = 0; z < info[AC_nlocal].z; ++z)
        {
          for(int w = 0; w < info[AC_nlocal_w]; ++w)
          {
	    const AcReal val = exp(-0.5*k*k*(t1-t2)*(t1-t2)*v*v)*cos(k*(t1-t2))/(t1*t2);
	    const size_t idx = x + nx*(y + ny*(z + nz*(w)));
	    data_arr[idx] = val;
          }
        }
      }
    }

    info[DATA] = data_arr;
    **/
    // GPU alloc & compute
    const auto update_arr = [&](const int offset, const int Ngrid, const int N, const auto& arr, const auto& weights, const AcReal& start, const AcReal& len)
    {
      gsl_integration_glfixed_table *table = gsl_integration_glfixed_table_alloc(Ngrid);
      AcReal* x   = (AcReal*)malloc(sizeof(AcReal)*N);
      AcReal* xw  = (AcReal*)malloc(sizeof(AcReal)*N);
      for(int i = 0; i  < N; ++i)
      {
	if(i+offset >= Ngrid)
	{
		fprintf(stderr,"Something went wrong %d %d %d %d!\n",i,N,Ngrid,offset);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
        gsl_integration_glfixed_point(start, start+len, i+offset, &x[i], &xw[i], table);
	if(std::isnan(x[i]) || std::isnan(xw[i]))
	{
		fprintf(stderr,"Got nan in generation Gauss-Legendre points for %s!\n",get_name(arr));
		fprintf(stderr,"Start is: %.14e\n",start);
		fprintf(stderr,"End is: %.14e\n",start+len);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
      }
      info[arr] = x;
      info[weights] = xw;
      gsl_integration_glfixed_table_free(table);
    };

    read_data_to_arr(info, E_P_TABULATED, "e.dat", AC_N_tabulated);
    read_data_to_arr(info, P_TABULATED, "p.dat", AC_N_tabulated);
    if(info[E_P_TABULATED] != NULL && info[P_TABULATED] != NULL)
    {
            FILE* fp_p = fopen("p_res.dat","w");
            FILE* fp_e = fopen("e_res.dat","w");
	    AcReal* res = (AcReal*)malloc(sizeof(AcReal)*info[AC_integration_points_x]);
	    {
	      int i_right = 1;
	      AcReal val_left  = info[E_P_TABULATED][i_right-1];
	      AcReal val_right = info[E_P_TABULATED][i_right];
	      AcReal pos_left  = info[P_TABULATED][i_right-1];
	      AcReal pos_right = info[P_TABULATED][i_right];
	      for(int p = 0; p < info[AC_integration_points_x]; ++p)
	      {
	          AcReal p_pos = info[AC_integration_start_x] + info[AC_ds].x*p;
	          if(info[AC_logspace_x]) p_pos = exp(p_pos);
	          while(p_pos > pos_right)
	          {
	          	pos_left = pos_right;
	          	val_left = val_right;
	          	i_right++;
	          	pos_right = info[P_TABULATED][i_right];
	          	val_right = info[E_P_TABULATED][i_right];
	          }
	          res[p] = linear_interpol(pos_left,pos_right,val_left,val_right,p_pos);
		  fprintf(fp_p,"%.14e,",p_pos);
		  fprintf(fp_e,"%.14e,",res[p]);
	      }
	    }
	    fclose(fp_p);
	    fclose(fp_e);

	    info[E_P] = res;

	    const int n_x = info[AC_integration_points_x];
	    const int n_y = info[AC_integration_points_y];

	    res = (AcReal*)malloc(sizeof(AcReal)*n_x*n_y);
	    for(int p = 0; p < n_x; ++p)
	    {
	      for(int z = 0; z < n_y; ++z)
	      {

	        int i_right = info[AC_N_tabulated]-1;
	        AcReal val_left  = info[E_P_TABULATED][i_right-1];
	        AcReal val_right = info[E_P_TABULATED][i_right];
	        AcReal pos_left  = info[P_TABULATED][i_right-1];
	        AcReal pos_right = info[P_TABULATED][i_right];

	        AcReal p_pos = info[AC_integration_start_x] + info[AC_ds].x*p;
		if(info[AC_logspace_x]) p_pos = exp(p_pos);

	        AcReal z_pos = info[AC_integration_start_y] + info[AC_ds].y*z;
		if(info[AC_logspace_y]) z_pos = exp(z_pos);

		AcReal ptilde = sqrt(p*p + info[AC_k]*info[AC_k] - 2*p*info[AC_k]*z_pos);
		while(ptilde < pos_left)
		{
			pos_right = pos_left;
			val_right = val_left;
			i_right--;
			pos_left = info[P_TABULATED][i_right-1];
			val_left = info[E_P_TABULATED][i_right-1];
		}
		res[p + n_x*z] = linear_interpol(pos_left,pos_right,val_left,val_right,ptilde);
	      }
	    }
	    info[E_PTILDE] = res;
    }

    update_arr(info[AC_multigpu_offset].x,info[AC_ngrid].x,info[AC_nlocal].x,X,X_W,info[AC_first_gridpoint].x,info[AC_len].x);
    update_arr(info[AC_multigpu_offset].y,info[AC_ngrid].y,info[AC_nlocal].y,Y,Y_W,info[AC_first_gridpoint].y,info[AC_len].y);
    update_arr(info[AC_multigpu_offset].z,info[AC_ngrid].z,info[AC_nlocal].z,Z,Z_W,info[AC_first_gridpoint].z,info[AC_len].z);
    update_arr(0,info[AC_nlocal_w],info[AC_nlocal_w],W,W_W,info[AC_first_gridpoint_w],info[AC_len_w]);
    acGridInit(info);

    const auto integrate = [&](bool test_convergence)
    {

      AcReal res = 0.0;
      if(info[AC_trapezoidal])
      {
        acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(calc_integral_trap),1);
        res = acDeviceGetOutput(acGridGetDevice(),AC_integral_res);
        if(test_convergence)
        {
          fprintf(stderr,"Trapezoidal integral is: %.14e\n",res);
          FILE* fp = fopen("trapz.dat","a");
          fprintf(fp,"%.14e,",res);
          fclose(fp);
	}
      }

      if(info[AC_gauss_legendre])
      {
        const auto graph = acGetOptimizedDSLTaskGraph(calc_integral_gauss);
        const auto start = MPI_Wtime();
        acGridExecuteTaskGraph(graph,1);
        const auto end = MPI_Wtime();
        res = acDeviceGetOutput(acGridGetDevice(),AC_gauss_legendre_res);
        if(test_convergence)
        {
          fprintf(stderr,"Integral took: %.14e\n",end-start);
          fprintf(stderr,"Gauss Legendre integral is: %.14e\n",res);

          FILE* fp = fopen("gauss.dat","a");
          fprintf(fp,"%.14e,",res);
          fclose(fp);

          fp = fopen("N.dat","a");
          fprintf(fp,"%d,",info[AC_nlocal].x);
          fclose(fp);
        }
      }
      
      return res;
    };


    FILE *fp = fopen("k.dat", "r");
    bool first = true;
    if (fp != NULL) {
      FILE* fp_res = fopen("res.dat","w");
      AcReal k;
      while (fscanf(fp, "%lf,", &k) == 1)
      {
          acDeviceLoadScalarUniform(acGridGetDevice(),STREAM_DEFAULT,AC_k,k);
	  const auto res = integrate(false);
	  if(!first)
	  {
	    fprintf(fp_res,",");
	  }
	  first = false;
	  fprintf(fp_res,"%.14e",res);
      }
      fclose(fp);
      fclose(fp_res);
    }
    else
    {
	    integrate(true);
    }

    //free(data_arr);
    const int retval = AC_SUCCESS;
    acGridQuit();
    MPI_Finalize();
    finalized = true;
    if (pid == 0)
        fprintf(stderr, "Integration test complete: %s\n",
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
