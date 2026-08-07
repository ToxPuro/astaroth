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
static int nprocs, pid;

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

#include <iostream>

bool
mpi_initialized()
{
	int initialized;
	MPI_Initialized(&initialized);
	return initialized;
}
void
interpolate(AcMeshInfo info)
{
    if(info[E_P_TABULATED] != NULL && info[P_TABULATED] != NULL)
    {
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
			if(i_right == info[AC_N_tabulated])
			{
	    			if(pid == 0) 
				{
					fprintf(stderr,"Interpolating E_P went over the array on the right!\nMake sure your tabulated data covers the integration range\n");
					fprintf(stderr,"P: %.14e\n",p_pos);
					fprintf(stderr,"Maximum tabulated p: %.14e\n",info[P_TABULATED][AC_N_tabulated-1]);
					fprintf(stderr,"Tabulated array had %d points!\n",info[AC_N_tabulated]);
				}
				fflush(stderr);
	    			exit(EXIT_FAILURE);
			}
	          	pos_right = info[P_TABULATED][i_right];
	          	val_right = info[E_P_TABULATED][i_right];
	          }
	          info[E_P][p] = linear_interpol(pos_left,pos_right,val_left,val_right,p_pos);
		  //fprintf(fp_p,"%.14e,",p_pos);
		  //fprintf(fp_e,"%.14e,",res[p]);
	      }
	    }
	    //fclose(fp_p);
	    //fclose(fp_e);

	    if(info[AC_integrate_using_z])
            {
	      const int n_x = info[AC_integration_points_x];
	      const int n_y = info[AC_integration_points_y];
              const AcReal k2 = info[AC_k]*info[AC_k];
	      for(int z = 0; z < n_y; ++z)
	      {
	        int old_starting_index = 1;
	        AcReal z_pos = info[AC_integration_start_y] + info[AC_ds].y*z;
	        if(info[AC_logspace_y]) z_pos = exp(z_pos);
		bool first = true;
	        for(int p = 0; p < n_x; ++p)
	        {
	          int i_right = old_starting_index;
	          AcReal val_left  = info[E_P_TABULATED][i_right-1];
	          AcReal val_right = info[E_P_TABULATED][i_right];
	          AcReal pos_left  = info[P_TABULATED][i_right-1];
	          AcReal pos_right = info[P_TABULATED][i_right];

	          AcReal p_pos = info[AC_integration_start_x] + info[AC_ds].x*p;
	          if(info[AC_logspace_x]) p_pos = exp(p_pos);

	          AcReal ptilde = sqrt(p_pos*p_pos + k2 - 2*p_pos*info[AC_k]*z_pos);
	          while(ptilde > pos_right)
	          {
	          	pos_left = pos_right;
	          	val_left = val_right;
	          	i_right++;
	          	if(i_right == info[AC_N_tabulated])
	          	{
	      			if(pid == 0) 
	          		{
	          			fprintf(stderr,"Interpolating E_PTILDE went over the array on the right!\nMake sure your tabulated data covers the integration range\n");
	          			fprintf(stderr,"Ptilde: %.14e\n",ptilde);
	          			fprintf(stderr,"Maximum tabulated p: %.14e\n",info[P_TABULATED][info[AC_N_tabulated]-1]);
	          			fprintf(stderr,"Tabulated array had %d points!\n",info[AC_N_tabulated]);
	          		}
	          		fflush(stderr);
	      			exit(EXIT_FAILURE);
	          	}
	          	pos_right = info[P_TABULATED][i_right];
	          	val_right = info[E_P_TABULATED][i_right];
			if(first) old_starting_index = i_right;
	          }
	          info[E_PTILDE][p + n_x*z] = linear_interpol(pos_left,pos_right,val_left,val_right,ptilde);
		  first = false;
	        }
	      }
	    }
	    else
	    {
	      const int n_x = info[AC_integration_points_x];
	      const int n_y = info[AC_integration_points_y];

	      for(int pi = 0; pi < n_x; ++pi)
	      {
		char p_tilde_res_filename[10000];
		char e_tilde_res_filename[10000];
		sprintf(p_tilde_res_filename,"p_tilde_%d_res.dat",pi);
		sprintf(e_tilde_res_filename,"e_tilde_%d_res.dat",pi);

                //FILE* fp_p = fopen(p_tilde_res_filename,"w");
                //FILE* fp_e = fopen(e_tilde_res_filename,"w");

	        AcReal p= info[AC_integration_start_x] + info[AC_ds].x*pi;
	        if(info[AC_logspace_x]) p = exp(p);
		const AcReal l = std::abs(info[AC_k]-p);
		const AcReal u = info[AC_k]+p;
	      	int i_right = 1;
	      	AcReal val_left  = info[E_P_TABULATED][i_right-1];
	      	AcReal val_right = info[E_P_TABULATED][i_right];
	      	AcReal pos_left  = info[P_TABULATED][i_right-1];
	      	AcReal pos_right = info[P_TABULATED][i_right];
	        for(int yi = 0; yi < n_y; ++yi)
	        {
		  //if(pi == 0) fprintf(stderr,"P: %.14e\n",p);
	          AcReal y= info[AC_integration_start_y] + info[AC_ds].y*yi;
		  AcReal ptilde = 0.5*(u-l)*y+0.5*(u+l);
		  //if(pi == 1 || pi == 2)
		  //{
		  //        fprintf(stderr,"Ptilde: %.14e %d\n",ptilde,pi);
		  //}
	          while(ptilde > pos_right)
	          {
	          	pos_left = pos_right;
	          	val_left = val_right;
	          	i_right++;
			if(i_right == info[AC_N_tabulated])
			{
	    			if(pid == 0) 
				{
					fprintf(stderr,"Interpolating E_PTILDE went over the array on the right!\nMake sure your tabulated data covers the integration range\n");
					fprintf(stderr,"Ptilde: %.14e\n",ptilde);
					fprintf(stderr,"Maximum tabulated p: %.14e\n",info[P_TABULATED][AC_N_tabulated-1]);
					fprintf(stderr,"Tabulated array had %d points!\n",info[AC_N_tabulated]);
				}
				fflush(stderr);
	    			exit(EXIT_FAILURE);
			}
	          	pos_right = info[P_TABULATED][i_right];
	          	val_right = info[E_P_TABULATED][i_right];
	          }
	          info[E_PTILDE][pi+n_x*yi] = linear_interpol(pos_left,pos_right,val_left,val_right,ptilde);
		  //fprintf(fp_p,"%.14e,",ptilde);
		  //fprintf(fp_e,"%.14e,",res[pi+n_x*yi]);
	        }
		//fclose(fp_p);
		//fclose(fp_e);
	      }
	    }
    }
    acDeviceLoadMeshInfo(acGridGetDevice(),info);
}

int
integrate_main(AcMeshInfo info, AcReal* dst)
{

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
    
    if(!info[AC_integrate_using_z])
    {
	acPushToConfig(info,AC_integration_start_y,-1.0);
	acPushToConfig(info,AC_integration_end_y,1.0);
	//TP: I assume this is wanted
	acPushToConfig(info,AC_logspace_y,false);
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


    update_arr(info[AC_multigpu_offset].x,info[AC_ngrid].x,info[AC_nlocal].x,X,X_W,info[AC_first_gridpoint].x,info[AC_len].x);
    update_arr(info[AC_multigpu_offset].y,info[AC_ngrid].y,info[AC_nlocal].y,Y,Y_W,info[AC_first_gridpoint].y,info[AC_len].y);
    update_arr(info[AC_multigpu_offset].z,info[AC_ngrid].z,info[AC_nlocal].z,Z,Z_W,info[AC_first_gridpoint].z,info[AC_len].z);
    update_arr(0,info[AC_nlocal_w],info[AC_nlocal_w],W,W_W,info[AC_first_gridpoint_w],info[AC_len_w]);
    info[E_P] = (AcReal*)malloc(sizeof(AcReal)*info[AC_integration_points_x]);
    info[E_PTILDE] = (AcReal*)malloc(sizeof(AcReal)*info[AC_integration_points_x]*info[AC_integration_points_y]);
    //Precompute variables
    const AcReal inv_alp = 1.0/info[AC_alp];
    const AcReal alp2 = info[AC_alp]*(info[AC_a]+info[AC_b]);
    acPushToConfig(info,AC_inv_alp,inv_alp);
    acPushToConfig(info,AC_alp2,alp2);
    acPushToConfig(info,AA,std::pow(info[AC_a]+info[AC_b],inv_alp));
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


    if(info[AC_N_k] > 0)
    {
      for(int ki = 0; ki < info[AC_N_k]; ++ki)
      {
        const AcReal k = info[AC_K][ki];
	if(info[AC_model] == 0)
	{
	  interpolate(info);
	}
        acPushToConfig(info,AC_k,k);
        acDeviceLoadScalarUniform(acGridGetDevice(),STREAM_DEFAULT,AC_k,k);
	dst[ki] = integrate(false);
      }
    }
    else
    {
	    integrate(true);
    }
    acGridQuit();
    return 0;
}
#ifdef PYTHON_BINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

enum class ParamType { Real, Int, Bool };

struct ParamInfo {
    ParamType type;
    int index;
};

static const std::unordered_map<std::string, ParamInfo> param_map = [] {
    std::unordered_map<std::string, ParamInfo> m;

    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        m.emplace(realparam_names[i], ParamInfo{ParamType::Real, i});

    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        m.emplace(intparam_names[i], ParamInfo{ParamType::Int, i});

    for (int i = 0; i < NUM_BOOL_PARAMS; ++i)
        m.emplace(boolparam_names[i], ParamInfo{ParamType::Bool, i});

    return m;
}();

void populate_config(py::dict config,AcMeshInfo& info)
{
	for (auto item : config)
	{
	    std::string key = py::cast<std::string>(item.first);
	
	    auto it = param_map.find(key);
	    if (it == param_map.end())
	        continue;
	
	    switch (it->second.type)
	    {
	    case ParamType::Real:
	        acPushToConfig(info, AcRealParam(it->second.index),
	                       py::cast<AcReal>(item.second));
	        break;
	
	    case ParamType::Int:
	        acPushToConfig(info, AcIntParam(it->second.index),
	                       py::cast<int>(item.second));
	        break;
	
	    case ParamType::Bool:
	        acPushToConfig(info, AcBoolParam(it->second.index),
	                       py::cast<bool>(item.second));
	        break;
	    }
	}
}

void integrate(
		    py::dict config,
    		    py::array_t<AcReal, py::array::c_style | py::array::forcecast> p_tabulated,
    		    py::array_t<AcReal, py::array::c_style | py::array::forcecast> e_tabulated,
    		    py::array_t<AcReal, py::array::c_style | py::array::forcecast> k,
    		    py::array_t<AcReal, py::array::c_style | py::array::forcecast> res)
{
    if(!mpi_initialized())
    {
      MPI_Init(NULL,NULL);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);
    AcMeshInfo info = acInitInfo();
    populate_config(config, info);

    acPushToConfig(info,AC_periodic_grid,(AcBool3){false,false,false});
    acPushToConfig(info,AC_set_tpb,(int3){16,8,1});
    acPushToConfig(info,AC_thread_block_loop_factors,(int3){1,64,64});
    acPushToConfig(info,AC_no_logs,true);
    acPushToConfig(info,AC_gauss_legendre,true);

    acPushToConfig(info,AC_N_k,k.shape(0));
    acPushToConfig(info,AC_N_tabulated,p_tabulated.shape(0));
    acPushToConfig(info,AC_N_k,k.shape(0));
    info[E_P_TABULATED] = (AcReal*)e_tabulated.data();
    info[P_TABULATED] = (AcReal*)p_tabulated.data();
    info[AC_K] = (AcReal*)k.data();
    acHostUpdateParams(&info);
    integrate_main(info,(AcReal*)res.data());
}

PYBIND11_MODULE(ac_integrator, m)
{
    m.def("integrate", &integrate);
}
#endif


int
main(int argc, char* argv[])
{

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);


    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("integration.conf", &info);
    if(info[AC_integrate_using_z])
    {
       if(pid == 0) fprintf(stderr,"Using z as integration variable\n");
    }
    else
    {
	      if(pid == 0) fprintf(stderr,"Using ptilde as integration variable\n");
    }

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
    read_data_to_arr(info, E_P_TABULATED, "e.dat", AC_N_tabulated);
    read_data_to_arr(info, P_TABULATED, "p.dat", AC_N_tabulated);
    read_data_to_arr(info, AC_K, "k.dat", AC_N_k);
    AcReal* res = (AcReal*)malloc(sizeof(AcReal)*info[AC_N_k]);
    integrate_main(info,res);
    bool first = true;
    FILE* fp_res = fopen("res.dat","w");
    for(int ki = 0; ki < info[AC_N_k]; ++ki)
    {
      if(!first)
      {
        fprintf(fp_res,",");
      }
      first = false;
      fprintf(fp_res,"%.14e",res[ki]);

    }
    fclose(fp_res);
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
