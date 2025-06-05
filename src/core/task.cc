/*
    Copyright (C) 2020, Oskar Lappi

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
#if AC_MPI_ENABLED
/**
 * Quick overview of tasks
 *
 * Each halo segment is assigned a HaloExchangeTask.
 * A HaloExchangeTask sends local data as a halo to a neighbor
 * and receives halo data from a (possibly different) neighbor.
 *
 * Each shell segment is assigned a ComputeTask.
 * ComputeTasks integrate their segment of the domain.
 *
 * After a task has been completed, its dependent tasks can be started with notifyDependents()
 * E.g. ComputeTasks may depend on HaloExchangeTasks because they're waiting to receive data.
 * Vv.  HaloExchangeTasks may depend on ComputeTasks because they're waiting for data to send.
 *
 * This all happens in grid.cc:GridIntegrate
 */



#include "task.h"
#include "astaroth.h"

AcKernel acGetOptimizedKernel(const AcKernel, const VertexBufferArray vba);

#include "internal_device_funcs.h"
#include "util_funcs.h"
#include "astaroth_utils.h"
#include "grid_detail.h"
#include "analysis_grid_helpers.h"

#include <cassert>
#include <memory>
#include <mpi.h>
#include <stdlib.h>
#include <vector>
#include <cstring>

#include "decomposition.h" //getPid and friends
#include "errchk.h"
#include "kernels/kernels.h" //AcRealPacked, ComputeKernel

int
static ac_pid()
{
    int pid;
    MPI_Comm_rank(acGridMPIComm(), &pid);
    return pid;
}

AcMeshInfo
ac_get_info()
{
        return acDeviceGetLocalConfig(acGridGetDevice());
}

#define fatal(MESSAGE, ...) \
        { \
	acLogFromRootProc(ac_pid(),MESSAGE,__VA_ARGS__); \
	ERRCHK_ALWAYS(false); \
	} 

#define HALO_TAG_OFFSET (100) //"Namespacing" the MPI tag space to avoid collisions
#define MAX_HALO_TAG (10000) //"Namespacing" the MPI tag space to avoid collisions in case of multiple messages

#if AC_USE_HIP
template <typename T, typename... Args>
std::unique_ptr<T>
make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#else
using std::make_unique;
#endif


template <typename T>
T*
ptr_copy(const T* src, const int n)
{
	T* res = (T*)malloc(sizeof(T)*n);
	memcpy(res,src,sizeof(T)*n);
	return res;
}

template <typename T>
T*
merge_ptrs(const T* a, const T* b, const size_t a_n, const size_t b_n)
{
	T* res = (T*)malloc(sizeof(T)*(a_n+b_n));
	memcpy(res,a,sizeof(T)*a_n);
	memcpy(res+a_n,b,sizeof(T)*b_n);
	return res;
}

Volume
get_max_halo_size(const Field fields[], const size_t num_fields)
{
    Volume max_halo_size = (Volume){0,0,0};
    if(num_fields == 0) return max_halo_size;
    const auto info = ac_get_info();
    for(size_t field = 0; field < num_fields; ++field)
    {
	max_halo_size.x = max(max_halo_size.x,as_size_t(info[vtxbuf_halos[fields[field]]].x));
	max_halo_size.y = max(max_halo_size.y,as_size_t(info[vtxbuf_halos[fields[field]]].y));
	max_halo_size.z = max(max_halo_size.z,as_size_t(info[vtxbuf_halos[fields[field]]].z));
    }
    if(max_halo_size.x == 0 || max_halo_size.y == 0 || max_halo_size.z == 0)
    {
	fprintf(stderr,"In fields: ");
    	for(size_t field = 0; field < num_fields; ++field) fprintf(stderr,"%s,",field_names[fields[field]]);
	fprintf(stderr,"Halo size: %ld,%ld,%ld\n",max_halo_size.x,max_halo_size.y,max_halo_size.z);
    	ERRCHK_ALWAYS(max_halo_size.x > 0);
    	ERRCHK_ALWAYS(max_halo_size.y > 0);
    	ERRCHK_ALWAYS(max_halo_size.z > 0);
    }
    return max_halo_size;
}

AcTaskDefinition
acRayUpdate(const AcKernel kernel, const AcBoundary boundary, const int3 ray_direction,
		Field fields_in[], const size_t num_fields_in, Field fields_out[], const size_t num_fields_out,
  		KernelParamsLoader load_func
		)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_RAY_UPDATE;
    task_def.kernel_enum         = kernel;
    task_def.halo_sizes = (Volume){1,1,1};
    task_def.fields_in      = ptr_copy(fields_in,num_fields_in);
    task_def.num_fields_in  = num_fields_in;
    task_def.fields_out     = ptr_copy(fields_out,num_fields_out);
    task_def.num_fields_out = num_fields_out;
    task_def.ray_direction = ray_direction;
    task_def.boundary      = boundary;
    task_def.load_kernel_params_func = new LoadKernelParamsFunc({load_func});
    return task_def;
}

AcTaskDefinition
acCompute(const AcKernel kernel, 
		Field fields_in[], const size_t num_fields_in, Field fields_out[], const size_t num_fields_out,
		Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out
		)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_COMPUTE;
    task_def.kernel_enum         = kernel;
    const auto max_halo_in   = get_max_halo_size(fields_in,num_fields_in);
    const auto max_halo_out  = get_max_halo_size(fields_out,num_fields_out);
    const Volume max_halo = (Volume){
		    			max(max_halo_in.x,max_halo_out.x),
		    			max(max_halo_in.y,max_halo_out.y),
		    			max(max_halo_in.z,max_halo_out.z)
	    			    };
    task_def.halo_sizes = max_halo;
    task_def.fields_in      = ptr_copy(fields_in,num_fields_in);
    task_def.num_fields_in  = num_fields_in;
    task_def.fields_out     = ptr_copy(fields_out,num_fields_out);
    task_def.num_fields_out = num_fields_out;

    task_def.profiles_in      = ptr_copy(profiles_in,num_profiles_in);
    task_def.num_profiles_in  = num_profiles_in;
    task_def.profiles_reduce_out     = ptr_copy(profiles_out,num_profiles_out);
    task_def.num_profiles_reduce_out = num_profiles_out;

    task_def.load_kernel_params_func = new LoadKernelParamsFunc{[](ParamLoadingInfo){;}};
    return task_def;
}

AcTaskDefinition
acComputeWithParams(const AcKernel kernel, Field fields_in[], const size_t num_fields_in, Field fields_out[], const size_t num_fields_out, 
		    Profile profiles_in[], const size_t num_profiles_in, Profile profiles_reduce_out[], const size_t num_profiles_reduce_out,
		    Profile profiles_write_out[], const size_t num_profiles_write_out,
                    KernelReduceOutput outputs_in[], const size_t num_outputs_in, KernelReduceOutput outputs_out[], const size_t num_outputs_out, 
		    const Volume start, const Volume end, const int onion_level,
	            KernelParamsLoader load_func)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_COMPUTE;
    task_def.kernel_enum         = kernel;
    const auto max_halo_in   = get_max_halo_size(fields_in,num_fields_in);
    const auto max_halo_out  = get_max_halo_size(fields_out,num_fields_out);
    const Volume max_halo = (Volume){
		    			max(max_halo_in.x,max_halo_out.x),
		    			max(max_halo_in.y,max_halo_out.y),
		    			max(max_halo_in.z,max_halo_out.z)
	    			    };
    task_def.halo_sizes = max_halo;

    task_def.halo_sizes.x *= onion_level;
    task_def.halo_sizes.y *= onion_level;
    task_def.halo_sizes.z *= onion_level;

    task_def.start = start;
    task_def.end   = end;
    task_def.given_launch_bounds = (end.x-start.x > 0) && (end.y - start.y > 0)  && (end.z - start.z > 0);
	    			


    task_def.fields_in      = ptr_copy(fields_in,num_fields_in);
    task_def.num_fields_in  = num_fields_in;
    task_def.fields_out     = ptr_copy(fields_out,num_fields_out);
    task_def.num_fields_out = num_fields_out;

    task_def.profiles_in      = ptr_copy(profiles_in,num_profiles_in);
    task_def.num_profiles_in  = num_profiles_in;
    task_def.profiles_reduce_out     = ptr_copy(profiles_reduce_out,num_profiles_reduce_out);
    task_def.num_profiles_reduce_out = num_profiles_reduce_out;

    task_def.profiles_write_out     = ptr_copy(profiles_write_out,num_profiles_write_out);
    task_def.num_profiles_write_out = num_profiles_write_out;

    task_def.outputs_in      = ptr_copy(outputs_in,num_outputs_in);
    task_def.num_outputs_in  = num_outputs_in;
    task_def.outputs_out     = ptr_copy(outputs_out,num_outputs_out);
    task_def.num_outputs_out = num_outputs_out;

    task_def.load_kernel_params_func = new LoadKernelParamsFunc({load_func});
    return task_def;
}


AcTaskDefinition
acReduceInRayDirection(Field fields[], const size_t num_fields, const int3 ray_direction)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_RAY_REDUCE;
    task_def.fields_in      = ptr_copy(fields,num_fields);
    task_def.num_fields_in  = num_fields;
    task_def.fields_out = ptr_copy(fields,num_fields);
    task_def.num_fields_out = num_fields;
    task_def.halo_sizes = get_max_halo_size(fields,num_fields);
    task_def.ray_direction = ray_direction;
    task_def.sending   = true;
    task_def.receiving = true;
    return task_def;
}


AcTaskDefinition
acHaloExchange(Field fields[], const size_t num_fields)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_HALOEXCHANGE;
    task_def.fields_in      = ptr_copy(fields,num_fields);
    task_def.num_fields_in  = num_fields;
    task_def.fields_out = ptr_copy(fields,num_fields);
    task_def.num_fields_out = num_fields;
    task_def.halo_sizes = get_max_halo_size(fields,num_fields);
    task_def.ray_direction = (int3){0,0,0};
    task_def.sending   = true;
    task_def.receiving = true;
    task_def.boundary      = BOUNDARY_XYZ;
    task_def.include_boundaries = false;
    int* halo_types = (int*)malloc(sizeof(int)*num_fields);
    for(size_t i = 0; i < num_fields; ++i) halo_types[i] = 2;
    task_def.halo_types = halo_types;
    return task_def;
}

AcTaskDefinition
acHaloExchangeWithBounds(Field fields[], const size_t num_fields, const Volume start, const Volume end, const int3 ray_direction, const bool sending, const bool receiving, const AcBoundary boundary, const bool include_boundaries, const int halo_types[])
{
    AcTaskDefinition task_def = acHaloExchange(fields,num_fields);
    task_def.halo_types = ptr_copy(halo_types,num_fields);
    task_def.start = start;
    task_def.end   = end;
    task_def.ray_direction = ray_direction;
    task_def.sending       = sending;
    task_def.receiving     = receiving;
    task_def.boundary      = boundary;
    task_def.include_boundaries = include_boundaries;
    task_def.given_launch_bounds = (end.x-start.x > 0) && (end.y - start.y > 0)  && (end.z - start.z > 0);

    return task_def;
}


#include "kernel_input_param_str.h"

AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, const AcKernel kernel, const Field fields_in[], const size_t num_fields_in, const Field fields_out[], const size_t num_fields_out, const std::function<void(ParamLoadingInfo)> load_func)
{
    if((boundary & BOUNDARY_Z) && TWO_D)
    {
	    fatal("%s","Can't have Z boundary conditions in 2d simulation\n");
    }
    AcTaskDefinition task_def{};
    task_def.task_type              = TASKTYPE_BOUNDCOND;
    task_def.boundary               = boundary;
    task_def.kernel_enum            = kernel;
    //TP: done in case we have communicating bcs
    task_def.sending   = true;
    task_def.receiving = true;

    task_def.fields_in      = ptr_copy(fields_in,num_fields_in);
    task_def.num_fields_in  = num_fields_in;
    task_def.fields_out     = ptr_copy(fields_out,num_fields_out);
    task_def.num_fields_out = num_fields_out;
    task_def.fieldwise = strstr(kernel_input_param_strs[kernel],"Field");
    const auto max_halo_in   = get_max_halo_size(fields_in,num_fields_in);
    const auto max_halo_out  = get_max_halo_size(fields_out,num_fields_out);
    const Volume max_halo = (Volume){
		    			max(max_halo_in.x,max_halo_out.x),
		    			max(max_halo_in.y,max_halo_out.y),
		    			max(max_halo_in.z,max_halo_out.z)
	    			    };
    task_def.halo_sizes = max_halo;

    int* halo_types = (int*)malloc(sizeof(int)*num_fields_out);
    for(size_t i = 0; i < num_fields_out; ++i) halo_types[i] = 2;
    task_def.halo_types = halo_types;
    if (!strcmp(kernel_input_param_strs[kernel],"Field"))
    {
    	auto default_loader = [](ParamLoadingInfo p)
    	{
    	        acLoadKernelParams(*p.params,p.kernel,p.vtxbuf);
    	};
	task_def.load_kernel_params_func = new LoadKernelParamsFunc({default_loader});
    }
    else
    	task_def.load_kernel_params_func = new LoadKernelParamsFunc({load_func});
    if(!acDeviceGetLocalConfig(acGridGetDevice())[AC_allow_non_periodic_bcs_with_periodic_grid])
    {
    	if(kernel == BOUNDCOND_PERIODIC && !acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].x && boundary & BOUNDARY_X)
    	{
    	        fatal("%s","Periodic boundary condition in X even though AC_periodic_grid.x is false!!\n");
    	}
    	if(kernel != BOUNDCOND_PERIODIC && acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].x && boundary & BOUNDARY_X)
    	{
    	        fatal("%s","Non-periodic boundary condition in X even though AC_periodic_grid.x is true!!\n"
			    "To skip this check set AC_allow_non_periodic_bcs_with_periodic_grid to true!\n"
				);
    	}
    	if(kernel == BOUNDCOND_PERIODIC && !acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].y && boundary & BOUNDARY_Y)
    	{
    	        fatal("%s","Periodic boundary condition in Y even though AC_periodic_grid.y is false!!\n");
    	}
    	if(kernel != BOUNDCOND_PERIODIC && acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].y && boundary & BOUNDARY_Y)
    	{
    	        fatal("%s","Non-periodic boundary condition in Y even though AC_periodic_grid.y is true!!\n"
			    "To skip this check set AC_allow_non_periodic_bcs_with_periodic_grid to true!\n"
				);
    	}
    	if(kernel == BOUNDCOND_PERIODIC && !acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].z && boundary & BOUNDARY_Z)
    	{
    	        fatal("%s","Periodic boundary condition in Z even though AC_periodic_grid.z is false!!\n");
    	}
    	if(kernel != BOUNDCOND_PERIODIC && acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].z && boundary & BOUNDARY_Z)
    	{
    	        fatal("%s","Non-periodic boundary condition in Z even though AC_periodic_grid.z is true!!\n"
			    "To skip this check set AC_allow_non_periodic_bcs_with_periodic_grid to true!\n"
				);
    	}
    }
    return task_def;
}

AcTaskDefinition
acBoundaryConditionWithBounds(const AcBoundary boundary, const AcKernel kernel, const Field fields_in[], const size_t num_fields_in, const Field fields_out[], const size_t num_fields_out, const Volume start, const Volume end, const int halo_types[], const std::function<void(ParamLoadingInfo)> load_func)
{
	AcTaskDefinition task_def = acBoundaryCondition(boundary,kernel,fields_in,num_fields_in,fields_out,num_fields_out,load_func);
        task_def.start = start;
        task_def.end   = end;
	task_def.halo_types = ptr_copy(halo_types,num_fields_out);
        task_def.given_launch_bounds = (end.x-start.x > 0) && (end.y - start.y > 0)  && (end.z - start.z > 0);
	return task_def;
}
size_t
get_compute_output_position(const int id, const size_t start, const size_t ghost, const size_t nn, const bool boundary_included)
{
	int res = id == -1  ? (boundary_included ? start-ghost : start) : 
		id == 1  ? (boundary_included ? nn+start: nn+start-ghost) : 
		(boundary_included ? start: start+ghost);
	return as_size_t(res);
}

size_t
get_compute_output_dim(const int id, const size_t ghost, const size_t nn, const bool boundary_included)
{
	int res = id == 0 ? 
		 (boundary_included ? nn : nn - ghost*2) : 
		 ghost;
	return as_size_t(res);
}
size_t
get_exchange_output_pos(const int id, const size_t start, const size_t ghost, const size_t nn, const bool bottom_included)
{
	    if(bottom_included && id != 0) fatal("Bottom included but id was: %d\n",id);
	    if(bottom_included) return 0;
      	    return  id == -1  ? as_size_t((int)start-(int)ghost) : id == 1 ? start+nn : start;
}
size_t
get_exchange_input_pos(const int id, const size_t start, const size_t ghost, const size_t nn, const bool bottom_included)
{
	const auto output_pos = get_exchange_output_pos(id,start,ghost,nn,bottom_included);
	return id == -1 ? output_pos + ghost : id == 1 ? output_pos - ghost : output_pos;
}
size_t
get_exchange_output_dim(const int id, const size_t ghost, const size_t nn, const bool bottom_included, const bool top_included)
{
	if(bottom_included && id != 0) fatal("Bottom included but id was: %d\n",id);
	if(top_included && id != 0) fatal("Top included but id was: %d\n",id);
	size_t res = id == 0 ? nn : ghost;
	if(bottom_included) res += ghost;
	if(top_included)    res += ghost;
	return res;
}

Volume
get_compute_output_position(int3 id, Volume start, Volume ghosts, Volume nn, AcBoundary computes_on_boundary, const int max_facet_class)
{
      Volume res = (Volume){
	    get_compute_output_position(id.x,start.x,ghosts.x,nn.x,computes_on_boundary & BOUNDARY_X),
	    get_compute_output_position(id.y,start.y,ghosts.y,nn.y,computes_on_boundary & BOUNDARY_Y),
	    get_compute_output_position(id.z,start.z,ghosts.z,nn.z,computes_on_boundary & BOUNDARY_Z)
      };
      const int facet_class = std::abs(id.x) + std::abs(id.y) + std::abs(id.z);
      if(max_facet_class == 2 && facet_class == 2 && id.x == 0)
      {
	      res.x -= ghosts.x;
      }
      else if(max_facet_class == 1 && facet_class == 1 && id.y != 0)
      {
	      res.x -= ghosts.x;
	      res.z -= ghosts.z;
      }
      else if(max_facet_class == 1 && facet_class == 1 && id.z != 0)
      {
	      res.x -= ghosts.x;
      }
      else if(max_facet_class == 0 && facet_class == 0)
      {
	      res.x -= ghosts.x;
	      res.y -= ghosts.y;
	      res.z -= ghosts.z;
      }
      return res;
}
Volume
get_compute_output_dim(int3 id, Volume ghosts, Volume nn, AcBoundary computes_on_boundary, const int max_facet_class)
{
      Volume res = (Volume)
      {
	      get_compute_output_dim(id.x,ghosts.x,nn.x,computes_on_boundary & BOUNDARY_X),
	      get_compute_output_dim(id.y,ghosts.y,nn.y,computes_on_boundary & BOUNDARY_Y),
	      get_compute_output_dim(id.z,ghosts.z,nn.z,computes_on_boundary & BOUNDARY_Z)
      };
      const int facet_class = std::abs(id.x) + std::abs(id.y) + std::abs(id.z);
      if(max_facet_class == 2 && facet_class == 2 && id.x == 0)
      {
	      res.x += 2*ghosts.x;
      }
      else if(max_facet_class == 1 && facet_class == 1 && id.y != 0)
      {
	      res.x += 2*ghosts.x;
	      res.z += 2*ghosts.z;
      }
      else if(max_facet_class == 1 && facet_class == 1 && id.z != 0)
      {
	      res.x += 2*ghosts.x;
      }
      else if(max_facet_class == 0 && facet_class == 0)
      {
	      res.x += 2*ghosts.x;
	      res.y += 2*ghosts.y;
	      res.z += 2*ghosts.z;
      }
      if((computes_on_boundary & BOUNDARY_X) == 0)
      {
	      ERRCHK_ALWAYS(res.x <= nn.x);
      }
      if((computes_on_boundary & BOUNDARY_Y) == 0)
      {
	      ERRCHK_ALWAYS(res.y <= nn.y);
      }
      if((computes_on_boundary & BOUNDARY_Z) == 0)
      {
	      ERRCHK_ALWAYS(res.z <= nn.z);
      }
      return res;
}

Volume
get_compute_input_position(const int3 id, const Volume start, const Volume ghost, const Volume nn, const AcBoundary boundary_included, const AcBoundary depends_on_boundary, const int max_facet_class)
{
	//TP: the capping by zero is if one wants to compute a pointwise kernel on the halo based on the normal dependency rules the
	//    the input region is still one NGHOST radius surrounding the computation radius even if no stencils are used (this is needed to make multikernel launches safe
	//    i.e. kernel B accessess stencil neighbours so kernel C can not update its own point since its out can be the in of kernel B):
	//    The capping is unsafe if the user would try to use e.g. derx while including the halos but there is a separate safety check for that later
	//
	const auto output_position = get_compute_output_position(id,start,ghost,nn,boundary_included,max_facet_class);
	Volume res = output_position;
	if(depends_on_boundary & BOUNDARY_X) 
	{
		res.x = as_size_t(max((int)res.x - (int)ghost.x,0));
	}
	if(depends_on_boundary & BOUNDARY_Y) 
	{
		res.y = as_size_t(max((int)res.y - (int)ghost.y,0));
	}
	if(depends_on_boundary & BOUNDARY_Z) 
	{
		res.z = as_size_t(max((int)res.z - (int)ghost.z,0));
	}
	return res;
}

Volume
get_compute_input_dim(const int3 id, const Volume ghost, const Volume nn, const AcBoundary boundary_included, const AcBoundary depends_on_boundary, const int max_facet_class)
{
	auto res = get_compute_output_dim(id,ghost,nn,boundary_included,max_facet_class);
	if(depends_on_boundary & BOUNDARY_X) 
	{
		res.x += 2*ghost.x;
	}
	if(depends_on_boundary & BOUNDARY_Y) 
	{
		res.y += 2*ghost.y;
	}
	if(depends_on_boundary & BOUNDARY_Z) 
	{
		res.z += 2*ghost.z;
	}
	return res;
}

Region::Region(RegionFamily family_, int tag_, const AcBoundary depends_on_boundary, const AcBoundary boundary_included, Volume start, Volume nn, const Volume ghosts, const RegionMemoryInputParams mem_, const int max_comp_facet_class)
    : family(family_), tag(tag_) 
{
    halo = ghosts;
    comp_dims = nn;
    memory.profiles = {mem_.profiles, mem_.profiles + mem_.num_profiles};
    memory.reduce_outputs  = {mem_.reduce_outputs , mem_.reduce_outputs  + mem_.num_reduce_outputs};
    memory.fields = {};
    switch (family) {
    	case RegionFamily::Exchange_output: //Fallthrough
    	case RegionFamily::Exchange_input : {
	    for(auto& field : mem_.fields)
	    	if(vtxbuf_is_communicated[field]) memory.fields.push_back(field);
	    break;
	}
	default:
	    for(auto& field : mem_.fields)
		memory.fields.push_back(field);
	break;
      }
      id = tag_to_id(tag);
      // facet class 0 = inner core
      // facet class 1 = face
      // facet class 2 = edge
      // facet class 3 = corner
      facet_class = tag_to_facet_class(tag);
      ERRCHK_ALWAYS(facet_class <= 3);
      
      switch (family) {
      case RegionFamily::Compute_output: {
      const AcBoundary computes_on_boundary = boundary_included;
      // clang-format off
      position = get_compute_output_position(id,start,ghosts,nn,computes_on_boundary,max_comp_facet_class);
      // clang-format on
      dims = get_compute_output_dim(id,ghosts,nn,computes_on_boundary,max_comp_facet_class);
      if(dims.x == 0 || dims.y == 0 || dims.z == 0)
      {
	      fprintf(stderr,"Incorrect region dims: %zu,%zu,%zu\n",dims.x,dims.y,dims.z);
	      ERRCHK_ALWAYS(dims.x != 0 && dims.y != 0 && dims.z != 0);
      }
      break;
      }
      case RegionFamily::Compute_input: {
      const AcBoundary computes_on_boundary = boundary_included;
      // clang-format off
      position = get_compute_input_position(id,start,ghosts,nn,computes_on_boundary,depends_on_boundary,max_comp_facet_class);
      // clang-format on
      dims = get_compute_input_dim(id,ghosts,nn,computes_on_boundary,depends_on_boundary,max_comp_facet_class);

      if(dims.x == 0 || dims.y == 0 || dims.z == 0)
      {
	      fprintf(stderr,"Incorrect region dims: %zu,%zu,%zu\n",dims.x,dims.y,dims.z);
	      fprintf(stderr,"Id: %d,%d,%d\n",id.x,id.y,id.z);
	      fprintf(stderr,"Ghosts: %zu,%zu,%zu\n",ghosts.x,ghosts.y,ghosts.z);
	      ERRCHK_ALWAYS(dims.x != 0 && dims.y != 0 && dims.z != 0);
      }
      break;
      }
      case RegionFamily::Exchange_output: {
      // clang-format off
      position = {
	    get_exchange_output_pos(id.x,start.x,ghosts.x,nn.x,boundary_included & BOUNDARY_X_BOT),
	    get_exchange_output_pos(id.y,start.y,ghosts.y,nn.y,boundary_included & BOUNDARY_Y_BOT),
	    get_exchange_output_pos(id.z,start.z,ghosts.z,nn.z,boundary_included & BOUNDARY_Z_BOT),
      };
      // clang-format on
      dims = {
	      get_exchange_output_dim(id.x,ghosts.x,nn.x, boundary_included & BOUNDARY_X_BOT,boundary_included & BOUNDARY_X_TOP),
	      get_exchange_output_dim(id.y,ghosts.y,nn.y, boundary_included & BOUNDARY_Y_BOT,boundary_included & BOUNDARY_Y_TOP),
	      get_exchange_output_dim(id.z,ghosts.z,nn.z, boundary_included & BOUNDARY_Z_BOT,boundary_included & BOUNDARY_Z_TOP),
      	    };
      break;
      }
      case RegionFamily::Exchange_input: {
      position = {
	    get_exchange_input_pos(id.x,start.x,ghosts.x,nn.x,boundary_included & BOUNDARY_X_BOT),
	    get_exchange_input_pos(id.y,start.y,ghosts.y,nn.y,boundary_included & BOUNDARY_Y_BOT),
	    get_exchange_input_pos(id.z,start.z,ghosts.z,nn.z,boundary_included & BOUNDARY_Z_BOT),
      };
      //TP: input and output have same dims
      dims = {
	      get_exchange_output_dim(id.x,ghosts.x,nn.x, boundary_included & BOUNDARY_X_BOT,boundary_included & BOUNDARY_X_TOP),
	      get_exchange_output_dim(id.y,ghosts.y,nn.y, boundary_included & BOUNDARY_Y_BOT,boundary_included & BOUNDARY_Y_TOP),
	      get_exchange_output_dim(id.z,ghosts.z,nn.z, boundary_included & BOUNDARY_Z_BOT,boundary_included & BOUNDARY_Z_TOP),
      	    };
      break;
      }
      default: {
      ERROR("Unknown region family.");
      }
      }
      volume = dims.x * dims.y * dims.z;
      }
      
      Region::Region(RegionFamily family_, int3 id_, Volume position_, Volume nn, Volume halos, const RegionMemoryInputParams mem_)
      : Region{family_, id_to_tag(id_), BOUNDARY_XYZ, BOUNDARY_NONE, position_, nn, halos,mem_,3}
      {
      ERRCHK_ALWAYS(id_.x == id.x && id_.y == id.y && id_.z == id.z);
      }
      
      Region::Region(Volume position_, Volume dims_, Volume comp_dims_, Volume halos_, int tag_, const RegionMemory mem_, RegionFamily family_)
      : position(position_), dims(dims_), comp_dims(comp_dims_), halo(halos_), family(family_), tag(tag_)
      {
      std::vector<Field> fields{};
      switch (family) {
      case RegionFamily::Exchange_output: {} //Fallthrough
      case RegionFamily::Exchange_input : {
      	for(auto& field : mem_.fields)
      	    	if(vtxbuf_is_communicated[field]) fields.push_back(field);
      	break;
      }
      default:
      	for(auto& field : mem_.fields)
      		fields.push_back(field);
      	break;
      
      }
      id          = tag_to_id(tag);
      facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
      volume = dims.x*dims.y*dims.z;
      memory.fields = fields;
      memory.profiles = mem_.profiles;
      memory.reduce_outputs  = mem_.reduce_outputs;
      }



Region
Region::translate(int3 translation)
{
return Region(to_volume(this->position + translation), this->dims, this->comp_dims, this->halo, this->tag, this->memory,this->family);
}


bool
Region::overlaps(const Region* other) const
{

	const AcBool3 gem_overlaps = this->geometry_overlaps(other);
	const bool vtxbuffers_overlap = (gem_overlaps.x && gem_overlaps.y && gem_overlaps.z) && this->fields_overlap(other);


	bool reduce_outputs_overlap = false;
	for(auto output_1: this->memory.reduce_outputs)
		for(auto output_2: other->memory.reduce_outputs)
			reduce_outputs_overlap |= output_1.variable == output_2.variable;
	return vtxbuffers_overlap || reduce_outputs_overlap;
}
AcBool3
Region::geometry_overlaps(const Region* other) const
{
    //TP: We are conservative in cases where the computational dimensions differ since then normal geometry overlap rule do not really make sense.
    //    So if the dims differ we say the geometry always overlaps
    if(this->comp_dims != other->comp_dims) return (AcBool3){true,true,true};
    return 
    (AcBool3){
	   (this->position.x < other->position.x + other->dims.x) &&
           (other->position.x < this->position.x + this->dims.x),

           (this->position.y < other->position.y + other->dims.y) &&
           (other->position.y < this->position.y + this->dims.y),

           (this->position.z < other->position.z + other->dims.z) &&
           (other->position.z < this->position.z + this->dims.z)
    };
}
bool
Task::swaps_overlap(const Task* other)
{
	bool overlap = false;
	for(bool swap_1 : this->swap_offset)
		for(bool swap_2 : other->swap_offset)
			overlap |= (swap_1 == swap_2);
	return overlap;
}

bool
Region::fields_overlap(const Region* other) const
{
    for(auto field_1 : this->memory.fields)
	    for(auto field_2 : other->memory.fields)
		    if(field_1 == field_2) return true;
    return false;
}

AcBoundary
Region::boundary(uint3_64 decomp, int pid, AcProcMappingStrategy proc_mapping_strategy)
{
    int3 pid3d = getPid3D(pid, decomp, proc_mapping_strategy);
    return boundary(decomp, pid3d, id);
}

bool
Region::is_on_boundary(uint3_64 decomp, int pid, AcBoundary boundary, AcProcMappingStrategy proc_mapping_strategy)
{
    int3 pid3d = getPid3D(pid, decomp, proc_mapping_strategy);
    return is_on_boundary(decomp, pid3d, id, boundary);
}
// Static functions
int
Region::id_to_tag(int3 id)
{
    return ((3 + id.x) % 3) * 9 + ((3 + id.y) % 3) * 3 + (3 + id.z) % 3; }

int3
Region::tag_to_id(int _tag)
{
    int3 _id = (int3){(_tag) / 9, ((_tag) % 9) / 3, (_tag) % 3};
    _id.x    = _id.x == 2 ? -1 : _id.x;
    _id.y    = _id.y == 2 ? -1 : _id.y;
    _id.z    = _id.z == 2 ? -1 : _id.z;
    ERRCHK_ALWAYS(id_to_tag(_id) == _tag);
    return _id;
}

int
Region::tag_to_facet_class(int _tag)
{
	const int3 id = Region::tag_to_id(_tag);
	const int res = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
	ERRCHK_ALWAYS(res <= 3);
	return res;
}

AcBoundary
Region::boundary(uint3_64 decomp, int pid, int tag, AcProcMappingStrategy proc_mapping_strategy)
{
    int3 pid3d = getPid3D(pid, decomp, proc_mapping_strategy);
    int3 id    = tag_to_id(tag);
    return boundary(decomp, pid3d, id);
}

AcBoundary
Region::boundary(uint3_64 decomp, int3 pid3d, int3 id)
{
    int3 neighbor = pid3d + id;
    return (AcBoundary)((neighbor.x == -1 ? BOUNDARY_X_BOT : 0) |
                        (neighbor.x == (int)decomp.x ? BOUNDARY_X_TOP : 0) |
                        (neighbor.y == -1 ? BOUNDARY_Y_BOT : 0) |
                        (neighbor.y == (int)decomp.y ? BOUNDARY_Y_TOP : 0) |
                        (neighbor.z == -1 ? BOUNDARY_Z_BOT : 0) |
                        (neighbor.z == (int)decomp.z ? BOUNDARY_Z_TOP : 0));
}

bool
Region::is_on_boundary(uint3_64 decomp, int pid, int tag, AcBoundary boundary, AcProcMappingStrategy proc_mapping_strategy)
{
    int3 pid3d     = getPid3D(pid, decomp, proc_mapping_strategy);
    int3 region_id = tag_to_id(tag);
    return is_on_boundary(decomp, pid3d, region_id, boundary);
}

bool
Region::is_on_boundary(uint3_64 decomp, int3 pid3d, int3 id, AcBoundary boundary)
{
    AcBoundary b = Region::boundary(decomp, pid3d, id);
    return b & boundary ? true : false;
}

/* Task interface */
Task::Task(int order_, std::vector<Region> input_regions_, Region output_region_, AcTaskDefinition op,
           Device device_, std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : device(device_), vba(acDeviceGetVBA(device)), swap_offset(swap_offset_), state(wait_state), dep_cntr(), loop_cntr(),
      order(order_), active(true), boundary(BOUNDARY_NONE), input_regions(input_regions_),
      output_region(output_region_),
      input_parameters(op.parameters, op.parameters + op.num_parameters)
{
    MPI_Comm_rank(acGridMPIComm(), &rank);
}

void
Task::registerDependent(std::shared_ptr<Task> t, size_t offset)
{
    dependents.emplace_back(t, offset);
    t->registerPrerequisite(offset);
}

void
Task::registerPrerequisite(size_t offset)
{
    // Ensure targets exist
    if (offset >= dep_cntr.targets.size()) {
        size_t initial_val = dep_cntr.targets.empty() ? 0 : dep_cntr.targets.back();
        dep_cntr.targets.resize(offset + 1, initial_val);
    }
    for (; offset < dep_cntr.targets.size(); offset++) {
        dep_cntr.targets[offset]++;
    }
}

bool
Task::isPrerequisiteTo(std::shared_ptr<Task> other)
{
    for (auto dep : dependents) {
        if (dep.first.lock() == other) {
            return true;
        }
    }
    return false;
}

bool
Task::isComputeTask()      { return false;}

bool
Task::isHaloExchangeTask() { return false;}


void
Task::setIterationParams(size_t begin, size_t end)
{
    loop_cntr.i   = begin;
    loop_cntr.end = end;

    // Reset dependency counter, and ensure it has enough space
    dep_cntr.counts.resize(0);
    dep_cntr.counts.resize(end, 0);
}

bool
Task::isFinished()
{
    return loop_cntr.i == loop_cntr.end;
}

void
Task::update(std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> vtxbuf_swaps, const TraceFile* trace_file)
{
    if (isFinished())
        return;

    bool ready;
    if (state == wait_state) {
        // dep_cntr.targets contains a rising series of targets e.g. {5,10}. The reason that earlier
        // iterations of a task might have fewer prerequisites in the task graph because the
        // prerequisites would have been satisfied by work that was performed before the beginning
        // of the task graph execution.
        //
        // Therefore, in the example, dep_cntr.targets = {5,10}:
        // if the loop counter is 0 or 1, we choose targets[0] (5) and targets[1] (10) respecively
        // if the loop counter is greater than that (e.g. 3) we select the final target count (10).

        if (dep_cntr.targets.size() == 0) {
            ready = true;
        }
        else if (loop_cntr.i >= dep_cntr.targets.size()) {
            ready = (dep_cntr.counts[loop_cntr.i] == dep_cntr.targets.back());
        }
        else {
            ready = (dep_cntr.counts[loop_cntr.i] == dep_cntr.targets[loop_cntr.i]);
        }
    }
    else {
        ready = test();
    }

    if (ready) {
        advance(trace_file);
        if (state == wait_state) {
            swapVBA(vtxbuf_swaps);
            notifyDependents();
            loop_cntr.i++;
        }
    }
}

void
Task::notifyDependents()
{
    for (auto& dep : dependents) {
        std::shared_ptr<Task> dependent = dep.first.lock();
        dependent->satisfyDependency(loop_cntr.i + dep.second);
    }
}

void
Task::satisfyDependency(size_t iteration)
{
    if (iteration < loop_cntr.end) {
        dep_cntr.counts[iteration]++;
    }
}
void
set_device(const Device device)
{
    cudaSetDevice(acDeviceGetId(device));
}

void
Task::syncVBA()
{
    set_device(device);
    const auto device_vba = acDeviceGetVBA(device);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if (swap_offset[i]) {
            vba.on_device.in[i]  = device_vba.on_device.out[i];
            vba.on_device.out[i] = device_vba.on_device.in[i];
        }
        else {
            vba.on_device.in[i]  = device_vba.on_device.in[i];
            vba.on_device.out[i] = device_vba.on_device.out[i];
        }
    }
    for (int i = 0; i < NUM_PROFILES; ++i) {
        if (swap_offset[i+NUM_VTXBUF_HANDLES]) {
            vba.on_device.profiles.in[i]  = device_vba.on_device.profiles.out[i];
            vba.on_device.profiles.out[i] = device_vba.on_device.profiles.in[i];
        }
        else {
            vba.on_device.profiles.in[i]  = device_vba.on_device.profiles.in[i];
            vba.on_device.profiles.out[i] = device_vba.on_device.profiles.out[i];
        }
    }
    for(int i=0;i<NUM_WORK_BUFFERS; ++i)
        vba.on_device.w[i] = device_vba.on_device.w[i];
}

void
Task::swapVBA(std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> device_swaps)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        if (device_swaps[i]) {
            AcReal* tmp = vba.on_device.in[i];
            vba.on_device.in[i]   = vba.on_device.out[i];
            vba.on_device.out[i]  = tmp;
        }
    }
    for(int i = 0; i < NUM_PROFILES; ++i)
    {
	    if(device_swaps[i+NUM_VTXBUF_HANDLES])
	    {
            	AcReal* tmp = vba.on_device.profiles.in[i];
            	vba.on_device.profiles.in[i]   = vba.on_device.profiles.out[i];
            	vba.on_device.profiles.out[i]  = tmp;
	    }
    }
}

bool
Task::poll_stream()
{
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess) {
        return true;
    }
    if (err == cudaErrorNotReady) {
        return false;
    }
    fprintf(stderr,
            "CUDA error in task %s while polling CUDA stream"
            " (probably occured in the CUDA kernel):\n\t%s\n",
            name.c_str(), cudaGetErrorString(err));
    fflush(stderr);
    exit(EXIT_FAILURE);
    return false;
}

Volume
get_min_nn()
{
	return acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
}

Volume
get_local_nn()
{
	return acGetLocalNN(acDeviceGetLocalConfig(acGridGetDevice()));
}


/* Computation */
ComputeTask::ComputeTask(AcTaskDefinition op, int order_, int region_tag, Volume start, Volume dims, Device device_,
                         std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_,
			 const std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries, const int max_facet_class
			 )
    : Task(order_,
           {Region(RegionFamily::Compute_input, region_tag,  get_kernel_depends_on_boundaries(op.kernel_enum,fields_already_depend_on_boundaries), op.computes_on_halos, start, dims, op.halo_sizes, {std::vector<Field>(op.fields_in, op.fields_in + op.num_fields_in)  ,op.profiles_in, op.num_profiles_in,op.outputs_in,   op.num_outputs_in},max_facet_class)},
           Region(RegionFamily::Compute_output, region_tag, get_kernel_depends_on_boundaries(op.kernel_enum,fields_already_depend_on_boundaries), op.computes_on_halos, start,dims,  op.halo_sizes, {std::vector<Field>(op.fields_out, op.fields_out + op.num_fields_out),
		   merge_ptrs(op.profiles_reduce_out,op.profiles_write_out,op.num_profiles_reduce_out,op.num_profiles_write_out),
		   op.num_profiles_reduce_out + op.num_profiles_write_out,
		   op.outputs_out, op.num_outputs_out},max_facet_class),
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    {
	set_device(device);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    auto& input_region = input_regions[0];
    if(kernel_only_writes_profile(op.kernel_enum,PROFILE_X))
    {
	output_region.dims.y = 1;	
	output_region.dims.z = 1;	
    }

    else if(kernel_only_writes_profile(op.kernel_enum,PROFILE_Y))
    {
	output_region.dims.x = 1;	
	output_region.dims.z = 1;	
    }

    else if(kernel_only_writes_profile(op.kernel_enum,PROFILE_Z))
    {
	output_region.dims.x = 1;	
	output_region.dims.y = 1;	
    }
    else if(max_facet_class == 3)
    {
	const AcBoundary bc_dependencies = get_kernel_depends_on_boundaries(op.kernel_enum,fields_already_depend_on_boundaries);
	if(!(bc_dependencies & BOUNDARY_X) && !(op.computes_on_halos & BOUNDARY_X))
	{
            output_region.dims.x += 2*NGHOST;
            input_region.dims.x  += 2*NGHOST;

            output_region.position.x -= NGHOST;
            input_region.position.x  -= NGHOST;
	}
	if(!(bc_dependencies & BOUNDARY_Y) && !(op.computes_on_halos & BOUNDARY_Y))
	{
            output_region.dims.y += 2*NGHOST;
            input_region.dims.y  += 2*NGHOST;

            output_region.position.y -= NGHOST;
            input_region.position.y  -= NGHOST;
	}

	if(!(bc_dependencies & BOUNDARY_Z) && !(op.computes_on_halos & BOUNDARY_Z))
	{
            output_region.dims.z += 2*NGHOST;
            input_region.dims.z  += 2*NGHOST;

            output_region.position.z -= NGHOST;
            input_region.position.z  -= NGHOST;
	}
    }

    const auto [left_radius,right_radius] = get_kernel_radius(op.kernel_enum);
    bool in_bounds    =    (int)output_region.position.x-(int)left_radius.x >= 0
    			|| (int)output_region.position.y-(int)left_radius.y >= 0
    			|| (int)output_region.position.z-(int)left_radius.z >= 0

    			|| output_region.position.x+output_region.dims.x + right_radius.x <= input_region.position.x + input_region.dims.x
    			|| output_region.position.y+output_region.dims.y + right_radius.y <= input_region.position.y + input_region.dims.y
    			|| output_region.position.z+output_region.dims.z + right_radius.z <= input_region.position.z + input_region.dims.z
    			;
    if(!in_bounds)
    {
	    fprintf(stderr,"Out of bounds ComputeTask for %s!\n",kernel_names[op.kernel_enum]);
	    ERRCHK_ALWAYS(in_bounds);
    }
    syncVBA();

    // compute_func = compute_func_;

    params = KernelParameters{op.kernel_enum, stream, 0, output_region.position,
                              output_region.position + output_region.dims, op.load_kernel_params_func};
    name   = "Compute " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    task_type = TASKTYPE_COMPUTE;
}

ComputeTask::ComputeTask(AcTaskDefinition op, int order_, std::vector<Region> input_regions_, Region output_region_, Device device_,std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_,
	        std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries
		)
    : Task(order_,
	   input_regions_,
           output_region_,
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    (void)fields_already_depend_on_boundaries;
    {
	set_device(device);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }

    syncVBA();

    // compute_func = compute_func_;

    params = KernelParameters{op.kernel_enum, stream, 0, output_region.position,
                              output_region.position + output_region.dims,  op.load_kernel_params_func};
    name   = "Compute " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    task_type = TASKTYPE_COMPUTE;
}

ComputeTask::ComputeTask(AcTaskDefinition op, int order_, Region input_region_, Region output_region_, Device device_,std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_,
	        std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries
		)
    : Task(order_,
	   {input_region_},
           output_region_,
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    (void)fields_already_depend_on_boundaries;
    {
	set_device(device);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }

    syncVBA();

    // compute_func = compute_func_;

    params = KernelParameters{op.kernel_enum, stream, 0, output_region.position,
                              output_region.position + output_region.dims,  op.load_kernel_params_func};
    name   = "Compute " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    task_type = TASKTYPE_COMPUTE;
}

std::shared_ptr<ComputeTask>
ComputeTask::RayUpdate(AcTaskDefinition op, int order_, const int3 boundary_id,const int3 ray_direction, Device device_,std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_,
	        std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries)
{

    const auto get_input_region = [&](const int3 id)
    {
	return Region(RegionFamily::Exchange_output, Region::id_to_tag(id), BOUNDARY_NONE, BOUNDARY_NONE, get_min_nn(), get_local_nn(), (Volume){1,1,1}, {std::vector<Field>(op.fields_out,op.fields_out + op.num_fields_out),op.profiles_reduce_out, op.num_profiles_reduce_out ,op.outputs_out, op.num_outputs_out},3);
    };

    const int ndir = abs(ray_direction.x) + abs(ray_direction.y) + abs(ray_direction.z);
    std::vector<Region> input_regions{};
    if(ndir == 1)
    {
	input_regions.push_back(get_input_region(-boundary_id));
    }
    if(ndir == 2)
    {
	const int3 incoming_boundary_id = boundary_id - ray_direction;	    
	input_regions.push_back(get_input_region(incoming_boundary_id));
	int3 corner_id = incoming_boundary_id;

	if(incoming_boundary_id.x == 0 && ray_direction.x != 0) corner_id.x -= ray_direction.x;
	if(incoming_boundary_id.y == 0 && ray_direction.y != 0) corner_id.y -= ray_direction.y;
	if(incoming_boundary_id.z == 0 && ray_direction.z != 0) corner_id.z -= ray_direction.z;

	input_regions.push_back(get_input_region(corner_id));

    }
    if(ndir == 3)
    {
	    const int3 r = (int3)
	    {
		    (boundary_id.x != 0) ? ray_direction.x : 0,
		    (boundary_id.y != 0) ? ray_direction.y : 0,
		    (boundary_id.z != 0) ? ray_direction.z : 0
	    };
	    const int3 incoming_boundary_id = boundary_id - ray_direction;	    
	    input_regions.push_back(get_input_region(incoming_boundary_id));
	    input_regions.push_back(get_input_region(incoming_boundary_id-r));
	    if(incoming_boundary_id.x != 0)
	    {
		    const int3 boundary = (int3){incoming_boundary_id.x,0,0};
	    	    input_regions.push_back(get_input_region(boundary));
	    	    input_regions.push_back(get_input_region(boundary-r));
	    }
	    if(incoming_boundary_id.y != 0)
	    {
		    const int3 boundary = (int3){0,incoming_boundary_id.y,0};
	    	    input_regions.push_back(get_input_region(boundary));
	    	    input_regions.push_back(get_input_region(boundary-r));
	    }
	    if(incoming_boundary_id.z != 0)
	    {
		    const int3 boundary = (int3){0,0,incoming_boundary_id.z};
	    	    input_regions.push_back(get_input_region(boundary));
	    	    input_regions.push_back(get_input_region(boundary-r));
	    }
    }

    auto output_region = Region(RegionFamily::Exchange_input, Region::id_to_tag(boundary_id), BOUNDARY_NONE, BOUNDARY_NONE, get_min_nn(), get_local_nn(), (Volume){1,1,1}, {std::vector<Field>(op.fields_out,op.fields_out+op.num_fields_out),op.profiles_reduce_out, op.num_profiles_reduce_out ,op.outputs_out, op.num_outputs_out},3);
    //TP: avoid computing on corner points twice
    if(boundary_id.x != 0)
    {
            if(ray_direction.y == 1)
            {
        	output_region.dims.y -= 1;
            }
            if(ray_direction.y == -1)
            {
        	output_region.dims.y -= 1;
        	output_region.position.y += 1;
            }
    
            if(ray_direction.z == 1)
            {
        	output_region.dims.z -= 1;
            }
            if(ray_direction.z == -1)
            {
        	output_region.dims.z -= 1;
        	output_region.position.z += 1;
            }
    }
    if(boundary_id.y != 0)
    {
            if(ray_direction.z == 1)
            {
        	output_region.dims.z -= 1;
            }
            if(ray_direction.z == -1)
            {
        	output_region.dims.z -= 1;
        	output_region.position.z += 1;
            }
    }
    return std::make_shared<ComputeTask>(op,order_,input_regions, output_region,device_,swap_offset_,fields_already_depend_on_boundaries);
}


bool
ComputeTask::isComputeTask() { return true; }


AcKernel
ComputeTask::getKernel()
{
	return params.kernel_enum;
}

void
ComputeTask::compute()
{
    params.load_func->loader({&vba.on_device.kernel_input_params, device, (int)loop_cntr.i, {}, {}, params.kernel_enum});
    acLaunchKernel(params.kernel_enum, params.stream, params.start, params.end, vba);
}

bool
ComputeTask::test()
{
    switch (static_cast<ComputeState>(state)) {
    case ComputeState::Running: {
        return poll_stream();
    }
    default: {
        ERROR("ComputeTask in an invalid state.");
        return false;
    }
    }
}

void
ComputeTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<ComputeState>(state)) {
    case ComputeState::Waiting: {
        trace_file->trace(this, "waiting", "running");
        compute();
        state = static_cast<int>(ComputeState::Running);
        break;
    }
    case ComputeState::Running: {
        trace_file->trace(this, "running", "waiting");
        state = static_cast<int>(ComputeState::Waiting);
        break;
    }
    default:
        ERROR("ComputeTask in an invalid state.");
    }
}

/*  Communication   */

// HaloMessage contains all information needed to send or receive a single message
HaloMessage::HaloMessage(Volume dims, size_t num_vars, const int tag0, const int tag_, const std::vector<int> counterpart_ranks_, const HaloMessageType type_)
{
    type         = type_;
    length       = dims.x * dims.y * dims.z * num_vars;
    counterpart_ranks = counterpart_ranks_;

    tag = tag0 + tag_;
    ERRCHK_ALWAYS(tag < MAX_HALO_TAG);
    non_namespaced_tag = tag_;

    bytes = length * sizeof(AcRealPacked);
    if(type == HaloMessageType::Receive) 
    {
	    bytes *= counterpart_ranks.size();
    }
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data, bytes));
    if(!ac_get_info()[AC_use_cuda_aware_mpi])
    {
    	ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data_pinned, bytes));
    }
    std::vector<MPI_Request>empty{};
    requests = empty;
    for(size_t i = 0; i < counterpart_ranks.size() ; ++i)
	requests.push_back(MPI_REQUEST_NULL);
}

HaloMessage::~HaloMessage()
{
    MPI_Waitall(requests.size(),requests.data(), MPI_STATUSES_IGNORE);
    length = -1;
    cudaFree(data);
    if(!ac_get_info()[AC_use_cuda_aware_mpi])
    {
    	cudaFree(data_pinned);
    }
    data = NULL;
}

void
HaloMessage::pin(const Device device, const cudaStream_t stream)
{
    set_device(device);
    pinned       = true;
    ERRCHK_CUDA(cudaMemcpyAsync(data_pinned, data, bytes, cudaMemcpyDefault, stream));
}

void
HaloMessage::unpin(const Device device, const cudaStream_t stream)
{
    if (!pinned)
        return;

    set_device(device);
    pinned       = false;
    ERRCHK_CUDA(cudaMemcpyAsync(data, data_pinned, bytes, cudaMemcpyDefault, stream));
}

// HaloMessageSwapChain
HaloMessageSwapChain::HaloMessageSwapChain() {}

HaloMessageSwapChain::HaloMessageSwapChain(Volume dims, size_t num_vars, const int tag0, const int tag, const std::vector<int> counterpart_ranks, const HaloMessageType type)
    : buf_idx(SWAP_CHAIN_LENGTH - 1)
{
    buffers.reserve(SWAP_CHAIN_LENGTH);
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        buffers.emplace_back(dims, num_vars,tag0, tag, counterpart_ranks,type);
    }
}

void
HaloMessageSwapChain::update_counterpart_ranks(const std::vector<int> counterpart_ranks)
{
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
	    buffers[i].counterpart_ranks = counterpart_ranks;
    }

}

HaloMessage*
HaloMessageSwapChain::get_current_buffer()
{
    return &buffers[buf_idx];
}

HaloMessage*
HaloMessageSwapChain::get_fresh_buffer()
{
    buf_idx         = (buf_idx + 1) % SWAP_CHAIN_LENGTH;
    MPI_Waitall(buffers[buf_idx].requests.size(), buffers[buf_idx].requests.data(), MPI_STATUSES_IGNORE);
    return &buffers[buf_idx];
}

AcReal
shear_periodic_displacement_in_grid_cells()
{
        const auto info = ac_get_info();
        return acDeviceGetInput(acGridGetDevice(),AC_shear_delta_y)/info[AC_ds].y;
}

int
shear_periodic_displacement_in_processes()
{
        const auto info = ac_get_info();
	return int(shear_periodic_displacement_in_grid_cells())/info[AC_nlocal].y;
}

AcReal
shear_periodic_leftover_fraction()
{
        return shear_periodic_displacement_in_grid_cells() - int(shear_periodic_displacement_in_grid_cells());
}

AcShearInterpolationCoeffs
shear_periodic_interpolation_coeffs()
{
        const AcReal frac = shear_periodic_leftover_fraction();
        return
        {
                -          (frac+1.)*frac*(frac-1.)*(frac-2.)*(frac-3.)/120.,
                +(frac+2.)          *frac*(frac-1.)*(frac-2.)*(frac-3.)/24. ,
                -(frac+2.)*(frac+1.)     *(frac-1.)*(frac-2.)*(frac-3.)/12. ,
                +(frac+2.)*(frac+1.)*frac          *(frac-2.)*(frac-3.)/12. ,
                -(frac+2.)*(frac+1.)*frac*(frac-1.)          *(frac-3.)/24. ,
                +(frac+2.)*(frac+1.)*frac*(frac-1.)*(frac-2.)          /120.
        };
}


std::vector<int>
shear_periodic_get_rhs_recv_offsets()
{
	    const int y = shear_periodic_displacement_in_processes();
	    return {y-1,y,y+1,y+2};
}

std::vector<int>
shear_periodic_get_lhs_recv_offsets()
{
	    const int y = shear_periodic_displacement_in_processes();
	    return {-y-2,-y-1,-y,-y+1};
}

std::vector<int>
get_recv_counterpart_ranks(const Device device, const int rank, const int3 output_region_id, const bool shear_periodic)
{
    const auto proc_strategy = acDeviceGetLocalConfig(device)[AC_proc_mapping_strategy];
    const auto decomp = uint3_64(acDeviceGetLocalConfig(device)[AC_domain_decomposition]);
    const auto my_pid = getPid3D(rank, decomp, proc_strategy);
    const auto target_pid = my_pid + output_region_id;
    const auto get_pid = [&](const int3 pid3d)
    {
	    return getPid(pid3d,decomp,proc_strategy);
    };
    if(shear_periodic)
    {
	    const std::vector<int> offsets = 
	        		output_region_id.x == -1
	        		? shear_periodic_get_lhs_recv_offsets()
	        		: shear_periodic_get_rhs_recv_offsets();
	    std::vector<int> res{};
	    for(const int offset : offsets)
	    {
	            res.push_back(get_pid({target_pid.x, target_pid.y + offset, target_pid.z}));
	    }
	    return res;

    }
    return {get_pid(target_pid)};
}

std::vector<int>
get_send_counterpart_ranks(const Device device, const int rank, const int3 output_region_id, const bool shear_periodic)
{
    const auto proc_strategy = acDeviceGetLocalConfig(device)[AC_proc_mapping_strategy];
    const auto decomp = uint3_64(acDeviceGetLocalConfig(device)[AC_domain_decomposition]);
    const auto my_pid = getPid3D(rank, decomp, proc_strategy);
    const auto target_pid = my_pid + output_region_id;
    const auto get_pid = [&](const int3 pid3d)
    {
	    return getPid(pid3d,decomp,proc_strategy);
    };
    if(shear_periodic)
    {

	//We do the inverse of the shift the receiver does on the other boundary
	const std::vector<int> offsets = 
	    		output_region_id.x == -1
	    		? shear_periodic_get_rhs_recv_offsets()
	    		: shear_periodic_get_lhs_recv_offsets();
	std::vector<int> res{};
	for(const int offset : offsets)
	{
	        res.push_back(get_pid({target_pid.x, target_pid.y - offset, target_pid.z}));
	}
	return res;

    }
    return {get_pid(target_pid)};
}

bool
get_sending(const int3 direction, const int3 id)
{
	return 
		   ((direction.x != 0) && (direction.x == id.x)) ||
		   ((direction.y != 0) && (direction.y == id.y)) ||
		   ((direction.z != 0) && (direction.z == id.z)) ||
		   (direction == (int3){0,0,0});
}

bool
get_receiving(const int3 direction, const int3 id)
{
	return 
		   ((direction.x != 0) && (direction.x == -id.x)) ||
		   ((direction.y != 0) && (direction.y == -id.y)) ||
		   ((direction.z != 0) && (direction.z == -id.z)) ||
		   (direction == (int3){0,0,0});
}
std::vector<Field>
get_communicated_subset(const std::vector<Field> fields, const int halo_types[], const int facet_class)
{
	ERRCHK_ALWAYS(halo_types != NULL);
	std::vector<Field> res{};

	const bool include_corners = acDeviceGetLocalConfig(acGridGetDevice())[AC_include_3d_halo_corners];
	for(size_t i = 0; i < fields.size(); ++i)
	{
		if(facet_class <= halo_types[i] || include_corners) res.push_back(fields[i]);
	}
	return res;
}


// HaloExchangeTask
HaloExchangeTask::HaloExchangeTask(AcTaskDefinition op, int order_, const Volume start, const Volume dims, int tag_0, int halo_region_tag,
                                   AcGridInfo grid_info, Device device_,
                                   std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_, const bool shear_periodic_)
    : Task(order_,
	   {Region(RegionFamily::Exchange_input, halo_region_tag,  BOUNDARY_NONE, BOUNDARY_NONE, start, dims, op.halo_sizes, {std::vector<Field>(op.fields_in,  op.fields_in + op.num_fields_in) ,op.profiles_in, op.num_profiles_in ,op.outputs_in, op.num_outputs_in},3)},
           Region(RegionFamily::Exchange_output, halo_region_tag, BOUNDARY_NONE, shear_periodic_ ? BOUNDARY_Y: BOUNDARY_NONE, start, dims, op.halo_sizes, {std::vector<Field>(op.fields_out,op.fields_out+op.num_fields_out),op.profiles_reduce_out, op.num_profiles_reduce_out ,op.outputs_out, op.num_outputs_out},3),
           op, device_, swap_offset_),
      sending(op.sending ? get_sending(op.ray_direction  ,input_regions[0].id)     : false),
      receiving(op.receiving ? get_receiving(op.ray_direction,input_regions[0].id) : false),
      shear_periodic(shear_periodic_),

      // MPI tags are namespaced to avoid collisions with other MPI tasks
      //TP: in recv_buffers the dims of input is used instead of output since normally they are the same
      //    and in case of shear periodic then output_dims is larger than the message dims
      recv_buffers(input_regions[0].dims,  receiving ? input_regions[0].memory.fields.size() : 0,   tag_0, input_regions[0].tag, get_recv_counterpart_ranks(device_,rank,output_region.id,shear_periodic), HaloMessageType::Receive),
      send_buffers(input_regions[0].dims,  sending   ? output_region.memory.fields.size() : 0,   tag_0, Region::id_to_tag(-output_region.id),get_send_counterpart_ranks(device_,rank,output_region.id,shear_periodic), HaloMessageType::Send)
{
    auto& input_region = input_regions[0];
    // Create stream for packing/unpacking
    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: creating CUDA stream\n");
    {
	set_device(device);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: done creating CUDA stream\n");

    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: syncing VBA\n");
    syncVBA();
    acVerboseLogFromRootProc(rank, "Halo exchange task ctor: done syncing VBA\n");

    if(op.ray_direction != (int3){0,0,0})
    {
    	active = (sending || receiving);
    }
    else
    {
	const size_t active_fields = get_communicated_subset(output_region.memory.fields, op.halo_types, output_region.facet_class).size();
	active = active_fields > 0;
    	//active = ((acDeviceGetLocalConfig(device)[AC_include_3d_halo_corners]) || (int)output_region.facet_class <= 2) ? true : false;
    }
    name = "Halo exchange " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) +
           "," + std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) +
           ")";
    task_type = TASKTYPE_HALOEXCHANGE;

    //TP: HaloExchangeTasks usually have input and output regions on the same side of the boundary
    //Thus if you directly moving data through kernels you have to remap the output position to the other side of the boundary
    if(sendingToItself())
    {
	    const auto ghosts = op.halo_sizes;
	    const Volume mm = {grid_info.nn.x + ghosts.x, grid_info.nn.y + ghosts.y, grid_info.nn.z + ghosts.z};
	    output_region.position -= (int3){input_region.id.x*(int)mm.x, input_region.id.y*(int)mm.y, input_region.id.z*(int)mm.z};
	    output_region.id = -input_region.id;
    }

}
bool
HaloExchangeTask::isHaloExchangeTask(){ return true; }


HaloExchangeTask::~HaloExchangeTask()
{
    // Cancel last eager request
    // TP: this should not be needed anymore (no ongoing eager receive outside halo exchange) but also does not harm
    auto msg = recv_buffers.get_current_buffer();
    for(auto& request : msg->requests)
    {
    	if (request != MPI_REQUEST_NULL) {
    	    MPI_Cancel(&request);
    	}
    }

    set_device(device);
    // dependents.clear();
    cudaStreamDestroy(stream);
}

void
HaloExchangeTask::pack()
{
    ERRCHK(sending);
    auto msg = send_buffers.get_fresh_buffer();
    acKernelPackData(stream, vba, input_regions[0].position, input_regions[0].dims,
                             msg->data, input_regions[0].memory.fields.data(),
                             input_regions[0].memory.fields.size());
}

void
HaloExchangeTask::move()
{
        ERRCHK(sending && receiving);
	acKernelMoveData(stream, input_regions[0].position, output_region.position, input_regions[0].dims, output_region.dims, vba, input_regions[0].memory.fields.data(), input_regions[0].memory.fields.size());
}

void
HaloExchangeTask::unpack()
{

    ERRCHK(receiving);
    auto msg = recv_buffers.get_current_buffer();
    if(!ac_get_info()[AC_use_cuda_aware_mpi])
    {
    	msg->unpin(device, stream);
    }
    if(shear_periodic)
    {

	    const int local_displacement = int(shear_periodic_displacement_in_grid_cells()) % ac_get_info()[AC_nlocal].y;
	    
	    const int ny = ac_get_info()[AC_nlocal].y;
	    const int base_offset  = output_region.id.x == -1
		    			? ny*2
					: ny*1
					;
	    const int offset = output_region.id.x == -1
	    				? base_offset - local_displacement
					: base_offset + local_displacement
					;
	    //We shift by NGHOST since we update the whole side from 0 to AC_mlocal.y
	    const int final_offset = offset-NGHOST;
	    acKernelShearUnpackData(stream, msg->data, output_region.position, output_region.dims,
	                               vba, output_region.memory.fields.data(),
			    		output_region.memory.fields.size(),
					shear_periodic_interpolation_coeffs(),
					final_offset
					);
    }
    else
    {
	    acKernelUnpackData(stream, msg->data, output_region.position, output_region.dims,
	                               vba, output_region.memory.fields.data(),
			    		output_region.memory.fields.size());
    }
}

void
HaloExchangeTask::sync()
{
    cudaStreamSynchronize(stream);
}

bool
HaloExchangeTask::sendingToItself()
{
	int n_procs;
	MPI_Comm_size(acGridMPIComm(), &n_procs);
	//For now enable optim only if there is only a single proc
	//Because reasoning about kernel moves is too difficult for the async tasks in TaskGraph
	return !shear_periodic && rank == send_buffers.get_current_buffer()->counterpart_ranks[0] && n_procs == 1 && !acDeviceGetLocalConfig(device)[AC_skip_single_gpu_optim];
}

void
HaloExchangeTask::wait_recv()
{
    auto msg = recv_buffers.get_current_buffer();
    MPI_Waitall(msg->requests.size(), msg->requests.data(), MPI_STATUSES_IGNORE);
}

void
HaloExchangeTask::wait_send()
{
    auto msg = send_buffers.get_current_buffer();
    MPI_Waitall(msg->requests.size(), msg->requests.data(), MPI_STATUSES_IGNORE);
}

void
HaloExchangeTask::receiveDevice()
{
    // TODO: change these to debug log statements at high verbosity (there will be very many of
    // these outputs)
    if (rank == 0) {
        // fprintf(stderr, "receiveDevice, getting buffer\n");
    }
    auto msg = recv_buffers.get_fresh_buffer();
    if (rank == 0) {
        // fprintf(stderr, "calling MPI_Irecv\n");
    }

    for(size_t i = 0; i < msg->counterpart_ranks.size(); ++i)
    {
    	ERRCHK_ALWAYS(MPI_Irecv(msg->data + i*msg->length, msg->length, AC_REAL_MPI_TYPE, msg->counterpart_ranks[i],
    	          msg->tag + HALO_TAG_OFFSET + i*MAX_HALO_TAG, acGridMPIComm(), &msg->requests[i]) == MPI_SUCCESS);
    }
    if (rank == 0) {
        // fprintf(stderr, "Returned from MPI_Irecv\n");
    }
}

void
HaloExchangeTask::sendDevice()
{
    auto msg = send_buffers.get_current_buffer();
    sync();

    for(size_t i = 0; i < msg->counterpart_ranks.size(); ++i)
    {
    	ERRCHK_ALWAYS(MPI_Isend(msg->data, msg->length, AC_REAL_MPI_TYPE, msg->counterpart_ranks[i],
              msg->tag + HALO_TAG_OFFSET + i*MAX_HALO_TAG, acGridMPIComm(), &msg->requests[i]) == MPI_SUCCESS);
    }
}

void
HaloExchangeTask::exchangeDevice()
{
    // set_device(device);
    receiveDevice();
    sendDevice();
}

void
HaloExchangeTask::receiveHost()
{
    // TODO: change these to debug log statements at high verbosity (there will be very many of
    // these outputs)
    if (rank == 0) {
        // fprintf("receiveHost, getting buffer\n");
    }
    auto msg = recv_buffers.get_fresh_buffer();
    for(size_t i = 0; i < msg->counterpart_ranks.size(); ++i)
    {
        ERRCHK_ALWAYS(MPI_Irecv(msg->data_pinned + i*msg->length, msg->length, AC_REAL_MPI_TYPE, msg->counterpart_ranks[i],
                  msg->tag + HALO_TAG_OFFSET + i*MAX_HALO_TAG, acGridMPIComm(), &msg->requests[i]) == MPI_SUCCESS);
    }
    msg->pinned = true;
}

void
HaloExchangeTask::sendHost()
{
    auto msg = send_buffers.get_current_buffer();
    msg->pin(device, stream);
    sync();

    for(size_t i = 0; i < msg->counterpart_ranks.size(); ++i)
    {
        ERRCHK_ALWAYS(MPI_Isend(msg->data_pinned, msg->length, AC_REAL_MPI_TYPE, msg->counterpart_ranks[i],
              msg->tag + HALO_TAG_OFFSET + i*MAX_HALO_TAG, acGridMPIComm(), &msg->requests[i]) == MPI_SUCCESS);
    }
}

void
HaloExchangeTask::exchangeHost()
{
    // set_device(device);
    receiveHost();
    sendHost();
}

void
HaloExchangeTask::receive()
{
    // TODO: change these fprintfs to debug log statements at high verbosity (there will be very
    // many of these outputs)
    if(ac_get_info()[AC_use_cuda_aware_mpi])
    {
      if (rank == 0) {
          // fprintf(stderr, "receiveDevice()\n");
      }
      receiveDevice();
      if (rank == 0) {
          // fprintf(stderr, "returned from receiveDevice()\n");
      }
    }
    else
    {
      if (rank == 0) {
          // fprintf(stderr, "receiveHost()\n");
      }
      receiveHost();
      if (rank == 0) {
          // fprintf(stderr, "returned from receiveHost()\n");
      }
    }
}

void
HaloExchangeTask::send()
{
    if(ac_get_info()[AC_use_cuda_aware_mpi])
    {
    	sendDevice();
    }
    else
    {
    	sendHost();
    }
}

void
HaloExchangeTask::exchange()
{
    ERRCHK(sending && receiving);
    if(ac_get_info()[AC_use_cuda_aware_mpi])
    {
    	exchangeDevice();
    }
    else
    {
    	exchangeHost();
    }
}

bool
HaloExchangeTask::test()
{
    switch (static_cast<HaloExchangeState>(state)) {
    case HaloExchangeState::Moving: {
        return poll_stream();
    }
    case HaloExchangeState::Packing: {
        return poll_stream();
    }
    case HaloExchangeState::Unpacking: {
        return poll_stream();
    }
    case HaloExchangeState::Exchanging: {
        auto msg = recv_buffers.get_current_buffer();
        int request_complete;
        ERRCHK_ALWAYS(MPI_Testall(msg->requests.size(), msg->requests.data(), &request_complete, MPI_STATUS_IGNORE) == MPI_SUCCESS);
        return request_complete ? true : false;
    }
    default: {
        ERROR("HaloExchangeTask in an invalid state.");
        return false;
    }
    }
}

void
HaloExchangeTask::advance(const TraceFile* trace_file)
{
    if (shear_periodic) 
    {
	    recv_buffers.update_counterpart_ranks(get_recv_counterpart_ranks(device,ac_pid(), output_region.id, shear_periodic));
	    send_buffers.update_counterpart_ranks(get_send_counterpart_ranks(device,ac_pid(), output_region.id, shear_periodic));
    }
    //move directly inside cuda kernels
    if (sendingToItself())
    {
    	switch (static_cast<HaloExchangeState>(state)) {

    	case HaloExchangeState::Waiting:
    	    trace_file->trace(this, "waiting", "moving");
    	    move();
    	    state = static_cast<int>(HaloExchangeState::Moving);
    	    break;
    	case HaloExchangeState::Moving:
    	    trace_file->trace(this, "moving", "waiting");
	    sync();
    	    state = static_cast<int>(HaloExchangeState::Waiting);
    	    break;
    	default: /* Fallthrough */
            	ERROR("HaloExchangeTask in an invalid state.");
        }
        return;
    }

    switch (static_cast<HaloExchangeState>(state)) {
    case HaloExchangeState::Waiting:
        trace_file->trace(this, "waiting", "packing");
    	// Post receive before send, this avoids unexpected messages
	//
	//4-apr-2025: TP: moved eager receives here from initialization for two reasons:
	//	1. If using multiple taskgraphs this way we have less active MPI communications which seem to cause stability issues
	//	2. The src processes can change dynamically and not be known at init (shearing periodic bcs)
	//    If the important point is that no message is not expected than this should handle that easily
	//    And also now we do not have to cleanup the leftover receive
	if(receiving)
	{
        	receive();
	}
	if(sending)
	{
        	pack();
        	state = static_cast<int>(HaloExchangeState::Packing);
	}
	else
	{
        	state = static_cast<int>(HaloExchangeState::Exchanging);
	}
        break;
    case HaloExchangeState::Packing:
        trace_file->trace(this, "packing", "receiving");
        sync();
        send();
	if(receiving)
	{
        	state = static_cast<int>(HaloExchangeState::Exchanging);
	}
	else
	{
        	state = static_cast<int>(HaloExchangeState::Waiting);
	}
        break;
    case HaloExchangeState::Exchanging:
        trace_file->trace(this, "receiving", "unpacking");
        sync();
        unpack();
        state = static_cast<int>(HaloExchangeState::Unpacking);
        break;
    case HaloExchangeState::Unpacking:
        trace_file->trace(this, "unpacking", "waiting");
        sync();
        state = static_cast<int>(HaloExchangeState::Waiting);
        break;
    default:
        ERROR("HaloExchangeTask in an invalid state.");
    }
}

// HaloExchangeTask
MPIReduceTask::MPIReduceTask(AcTaskDefinition op, int order_, const Volume start, const Volume dims, int tag_0, int3 halo_region_id,
                                   AcGridInfo grid_info, Device device_,
                                   std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
	   {Region(RegionFamily::Exchange_input,  Region::id_to_tag(halo_region_id),  BOUNDARY_NONE, BOUNDARY_NONE, start, dims, op.halo_sizes, {std::vector<Field>(op.fields_in,  op.fields_in+op.num_fields_in) ,op.profiles_in, op.num_profiles_in ,op.outputs_in, op.num_outputs_in},3)},
           Region(RegionFamily::Exchange_output, Region::id_to_tag(-halo_region_id), BOUNDARY_NONE,  BOUNDARY_NONE, start, dims, op.halo_sizes, {std::vector<Field>(op.fields_out,op.fields_out + op.num_fields_out),op.profiles_reduce_out, op.num_profiles_reduce_out ,op.outputs_out, op.num_outputs_out},3),
           op, device_, swap_offset_),
      reduce_buffers(input_regions[0].dims,  input_regions[0].memory.fields.size(),  tag_0, input_regions[0].tag, get_recv_counterpart_ranks(device_,rank,output_region.id,false), HaloMessageType::Receive)
{
    // Create stream for packing/unpacking
    {
	(void)grid_info;
	set_device(device);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
}

MPIReduceTask::~MPIReduceTask()
{
    auto msg = reduce_buffers.get_current_buffer();
    for(auto& request : msg->requests)
    {
    	if (request != MPI_REQUEST_NULL) {
    	    MPI_Cancel(&request);
    	}
    }

    set_device(device);
    // dependents.clear();
    cudaStreamDestroy(stream);
}

bool
MPIReduceTask::test()
{
    switch (static_cast<MPIReduceTaskState>(state)) {
    case MPIReduceTaskState::Packing: {
        return poll_stream();
    }
    case MPIReduceTaskState::Unpacking: {
        return poll_stream();
    }
    case MPIReduceTaskState::Communicating: {
        auto msg = reduce_buffers.get_current_buffer();
        int request_complete;
        ERRCHK_ALWAYS(MPI_Testall(msg->requests.size(), msg->requests.data(), &request_complete, MPI_STATUS_IGNORE) == MPI_SUCCESS);
        return request_complete ? true : false;
    }
    default: {
        ERROR("MPIReduceTask in an invalid state.");
        return false;
    }
    }
}


void
MPIReduceTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<MPIReduceTaskState>(state)) {
    case MPIReduceTaskState::Waiting:
        trace_file->trace(this, "waiting", "packing");
        pack();
        state = static_cast<int>(MPIReduceTaskState::Packing);
        break;
    case MPIReduceTaskState::Packing:
    {
        trace_file->trace(this, "packing", "communicating");
        state = static_cast<int>(MPIReduceTaskState::Communicating);
	communicate();
        break;
    }
    case MPIReduceTaskState::Communicating:
    {
        trace_file->trace(this, "communicating", "unpacking");
        state = static_cast<int>(MPIReduceTaskState::Unpacking);
	communicate();
        break;
    }
    case MPIReduceTaskState::Unpacking:
    {
        trace_file->trace(this, "unpacking", "waiting");
        state = static_cast<int>(MPIReduceTaskState::Waiting);
	unpack();
        break;
    }

    default:
        ERROR("MPIReduceTask in an invalid state.");
    }
}

void
MPIReduceTask::pack()
{
    auto msg = reduce_buffers.get_fresh_buffer();
    acKernelPackData(stream, vba, input_regions[0].position, input_regions[0].dims,
                             msg->data, input_regions[0].memory.fields.data(),
                             input_regions[0].memory.fields.size());
}

void
MPIReduceTask::unpack()
{

    auto msg = reduce_buffers.get_current_buffer();
    acKernelUnpackData(stream, msg->data, output_region.position, output_region.dims,
                               vba, output_region.memory.fields.data(),
        	    		output_region.memory.fields.size());
}

void
MPIReduceTask::communicate()
{
   auto msg = reduce_buffers.get_current_buffer();
   const auto sub_comms = acGridMPISubComms();
   ERRCHK_ALWAYS(MPI_Iallreduce(MPI_IN_PLACE,&msg->data[0],msg->length,AC_REAL_MPI_TYPE,MPI_SUM,sub_comms.x, &msg->requests[0]) == MPI_SUCCESS);
}

ReduceTask::ReduceTask(AcTaskDefinition op, int order_, int region_tag, const Volume start, const Volume nn, Device device_,
                         std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           {Region(RegionFamily::Compute_input, region_tag,  BOUNDARY_NONE, op.computes_on_halos, start, nn, op.halo_sizes, {std::vector<Field>(op.fields_in, op.fields_in + op.num_fields_in),op.profiles_in, op.num_profiles_in ,op.outputs_in,  op.num_outputs_in},3)},
           Region(RegionFamily::Compute_output, region_tag, BOUNDARY_NONE, op.computes_on_halos, start, nn, op.halo_sizes, {std::vector<Field>(op.fields_out, op.fields_out + op.num_fields_out),op.profiles_reduce_out,op.num_profiles_reduce_out,op.outputs_out, op.num_outputs_out},3),
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    {
        set_device(device);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    ERRCHK_ALWAYS(!(op.num_profiles_in  == 0 && op.num_outputs_in  == 0));
    ERRCHK_ALWAYS(!(op.num_profiles_reduce_out == 0 && op.num_outputs_out == 0));
    ERRCHK_ALWAYS(op.num_profiles_reduce_out == op.num_profiles_in);
    ERRCHK_ALWAYS(op.num_outputs_out  == op.num_outputs_in);

    const Volume ghosts = op.halo_sizes;

    auto& input_region = input_regions[0];
    input_region.position = {start.x-ghosts.x,start.y-ghosts.y,start.z-ghosts.z};
    input_region.dims     = {nn.x+2*ghosts.x,nn.y+2*ghosts.y,nn.z+2*ghosts.z};

    output_region.position = {start.x-ghosts.x,start.y-ghosts.y,start.z-ghosts.z};
    output_region.dims     = {nn.x+2*ghosts.x,nn.y+2*ghosts.y,nn.z+2*ghosts.z};

    if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_X))
	    reduces_only_prof = PROFILE_X;
    else if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_Y))
	    reduces_only_prof = PROFILE_Y;
    else if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_Z))
	    reduces_only_prof = PROFILE_Z;

    syncVBA();
    nothing_to_communicate = input_region.memory.profiles.size() == 0;
    const auto& reduce_outputs = input_region.memory.reduce_outputs;
    for(size_t i = 0; i < reduce_outputs.size(); ++i)
    {
	    if(reduce_outputs[i].type == AC_REAL_TYPE)   nothing_to_communicate &= !real_output_is_global[reduce_outputs[i].variable];
	    if(reduce_outputs[i].type == AC_INT_TYPE)    nothing_to_communicate &= !int_output_is_global[reduce_outputs[i].variable];
#if AC_DOUBLE_PRECISION
	    if(reduce_outputs[i].type == AC_FLOAT_TYPE)  nothing_to_communicate &= !float_output_is_global[reduce_outputs[i].variable];
#endif
    }

    name   = "Reduce " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    for(int i = 0; i < NUM_OUTPUTS+NUM_PROFILES; ++i)
	    requests[i] = MPI_REQUEST_NULL;
    task_type = TASKTYPE_REDUCE;
}

bool
ReduceTask::test()
{
    switch (static_cast<ReduceState>(state)) {
    case ReduceState::Reducing: {
        return poll_stream();
    }
    case ReduceState::Loading: {
        return poll_stream();
    }
    case ReduceState::Communicating: {
        int requests_completed;
	ERRCHK_ALWAYS(MPI_Testall(NUM_OUTPUTS+NUM_PROFILES,requests,&requests_completed, MPI_STATUS_IGNORE) == MPI_SUCCESS);
        return requests_completed ? true : false;
    }
    default: {
        ERROR("ReduceTask in an invalid state.");
        return false;
    }
    }
}

void
ReduceTask::reduce()
{
	const auto& reduce_outputs = input_regions[0].memory.reduce_outputs;
     	const auto nn = output_region.comp_dims;

	if constexpr (NUM_PROFILES != 0)
	{
		for(const auto& prof : input_regions[0].memory.profiles)
		{
		    auto reduce_buf = acDeviceGetProfileReduceBuffer(device,prof);
		    auto dst        = acDeviceGetProfileBuffer(device,prof);
		    if(prof_types[prof] == PROFILE_X && reduces_only_prof == PROFILE_X && output_region.id.x == -1)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst,
					   stream,
					   (Volume){0,0,0},
					   (Volume){
						NGHOST,
					   	reduce_buf.src.shape.y,
					   	reduce_buf.src.shape.z,
						},
					   (Volume){0,0,0},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_X && reduces_only_prof == PROFILE_X && output_region.id.x == 0)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst+NGHOST,
					   stream,
					   (Volume){NGHOST,0,0},
					   (Volume){
						nn.x+NGHOST,
					   	reduce_buf.src.shape.y,
					   	reduce_buf.src.shape.z
						},
					   (Volume){0,0,NGHOST},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						nn.x+NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_X && reduces_only_prof == PROFILE_X && output_region.id.x == 1)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst+NGHOST+nn.x,
					   stream,
					   (Volume){NGHOST+nn.x,0,0},
					   (Volume){
						nn.x+2*NGHOST,
					   	reduce_buf.src.shape.y,
					   	reduce_buf.src.shape.z
						},
					   (Volume){0,0,NGHOST+nn.x},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						nn.x+2*NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_Y && reduces_only_prof == PROFILE_Y && output_region.id.y == -1)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst,
					   stream,
					   (Volume){0,0,0},
					   (Volume){
					   	reduce_buf.src.shape.x,
						NGHOST,
					   	reduce_buf.src.shape.z,
						},
					   (Volume){0,0,0},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_Y && reduces_only_prof == PROFILE_Y && output_region.id.y == 0)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst+NGHOST,
					   stream,
					   (Volume){0,NGHOST,0},
					   (Volume){
					   	reduce_buf.src.shape.x,
						nn.y+NGHOST,
					   	reduce_buf.src.shape.z
						},
					   (Volume){0,0,NGHOST},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						nn.y+NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_Y && reduces_only_prof == PROFILE_Y && output_region.id.y == 1)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst+NGHOST+nn.y,
					   stream,
					   (Volume){0,NGHOST+nn.y,0},
					   (Volume){
					   	reduce_buf.src.shape.x,
						nn.y+2*NGHOST,
					   	reduce_buf.src.shape.z
						},
					   (Volume){0,0,NGHOST+nn.y},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						nn.y+2*NGHOST
					   }
				    );
		    }


		    else if(prof_types[prof] == PROFILE_Z && reduces_only_prof == PROFILE_Z && output_region.id.z == -1)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst,
					   stream,
					   (Volume){0,0,0},
					   (Volume){
					   	reduce_buf.src.shape.x,
					   	reduce_buf.src.shape.y,
						NGHOST
						},
					   (Volume){0,0,0},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_Z && reduces_only_prof == PROFILE_Z && output_region.id.z == 0)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst+3,
					   stream,
					   (Volume){0,0,NGHOST},
					   (Volume){
					   	reduce_buf.src.shape.x,
					   	reduce_buf.src.shape.y,
						nn.z+NGHOST
						},
					   (Volume){0,0,NGHOST},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						nn.z+NGHOST
					   }
				    );
		    }
		    else if(prof_types[prof] == PROFILE_Z && reduces_only_prof == PROFILE_Z && output_region.id.z == 1)
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst+NGHOST+nn.z,
					   stream,
					   (Volume){0,0,NGHOST+nn.z},
					   (Volume){
					   	reduce_buf.src.shape.x,
					   	reduce_buf.src.shape.y,
						nn.z+2*NGHOST
						},
					   (Volume){0,0,NGHOST+nn.z},
					   (Volume)
					   {
					   	reduce_buf.transposed.shape.x,
					   	reduce_buf.transposed.shape.y,
						nn.z+2*NGHOST
					   }
				    );
				
		    }
		    else
		    {
		    		acReduceProfileWithBounds(prof,
					   reduce_buf,
					   dst,
					   stream,
					   (Volume){0,0,0},
					   get_volume_from_shape(reduce_buf.src.shape),
					   (Volume){0,0,0},
					   get_volume_from_shape(reduce_buf.transposed.shape)
				    );
		    }
		}
	}

    	for(size_t i = 0; i < reduce_outputs.size(); ++i)
    	{
	    const auto var    = reduce_outputs[i].variable;
	    const auto op     = reduce_outputs[i].op;
	    const auto kernel = reduce_outputs[i].kernel;
	    if(reduce_outputs[i].type == AC_REAL_TYPE)
	    	acDeviceFinishReduceRealStream(device,stream,&local_res_real[i],kernel,op,(AcRealOutputParam)var);
	    else if(reduce_outputs[i].type == AC_INT_TYPE)
	    	acDeviceFinishReduceIntStream(device,stream,&local_res_int[i],kernel,op,(AcIntOutputParam)var);
#if AC_DOUBLE_PRECISION
	    else if(reduce_outputs[i].type == AC_FLOAT_TYPE)
	    	acDeviceFinishReduceFloatStream(device,stream,&local_res_float[i],kernel,op,(AcFloatOutputParam)var);
#endif
	    else if(reduce_outputs[i].type == AC_PROF_TYPE) {}
	    else
	    {
		    fprintf(stderr,"Unknown variable type: %d\n",reduce_outputs[i].type);
		    exit(EXIT_FAILURE);
	    }
    	}
}

static MPI_Op
to_mpi_op(const AcReduceOp op)
{
	switch(op)
	{
		case(REDUCE_SUM):
			return MPI_SUM;
		case(REDUCE_MIN):
			return MPI_MIN;
		case(REDUCE_MAX):
			return MPI_MAX;
		case(NO_REDUCE):
			fatal("%s","Should not call to_mpi_op for NO_REDUCE\n");
	}
	fatal("%s","No mapping from AcReduceOp to MPI_Op\n");
}

void
ReduceTask::communicate()
{
   const auto nn = acGetLocalNN(acDeviceGetLocalConfig(device));
   const auto sub_comms = acGridMPISubComms();
   const auto grid_comm = acGridMPIComm();
   if constexpr(NUM_PROFILES != 0)
   {
   	for(const auto& prof: input_regions[0].memory.profiles)
   	{
   	        const MPI_Comm comm =
   	     	   	prof_types[prof] == PROFILE_X ? sub_comms.yz :
   	     	   	prof_types[prof] == PROFILE_Y ? sub_comms.xz :
   	     	   	prof_types[prof] == PROFILE_Z ? sub_comms.xy :

   	     	   	(prof_types[prof] == PROFILE_XY || prof_types[prof] == PROFILE_YX) ? sub_comms.z :
   	     	   	(prof_types[prof] == PROFILE_XZ || prof_types[prof] == PROFILE_ZX) ? sub_comms.y :
   	     	   	(prof_types[prof] == PROFILE_YZ || prof_types[prof] == PROFILE_ZY) ? sub_comms.x :
   	     		MPI_COMM_NULL;

   	        if(reduces_only_prof != PROFILE_NONE)
   	        {
   	     	   const auto n_size = 
   	     		   reduces_only_prof == PROFILE_X ? nn.x :
   	     		   reduces_only_prof == PROFILE_Y ? nn.y :
   	     		   reduces_only_prof == PROFILE_Z ? nn.z :
   	     		   0;

   	     	   const auto id = 
   	     		   reduces_only_prof == PROFILE_X ? output_region.id.x :
   	     		   reduces_only_prof == PROFILE_Y ? output_region.id.y :
   	     		   reduces_only_prof == PROFILE_Z ? output_region.id.z :
   	     		   0;

   	     	   if(id == -1)
   	     	   {
   	        		MPI_Iallreduce(MPI_IN_PLACE,
   	     		   acDeviceGetProfileBuffer(device,prof),
   	     		   NGHOST,
   	     		   AC_REAL_MPI_TYPE,
   	     		   MPI_SUM,
   	     		   comm,
   	     		   &requests[NUM_OUTPUTS + prof]);
   	     	   }
   	     	   else if(id == 0)
   	     	   {
   	        		MPI_Iallreduce(MPI_IN_PLACE,
   	     		   acDeviceGetProfileBuffer(device,prof)+NGHOST,
   	     		   n_size,
   	     		   AC_REAL_MPI_TYPE,
   	     		   MPI_SUM,
   	     		   comm,
   	     		   &requests[NUM_OUTPUTS + prof]);
   	     	   }
   	     	   else if(id == 1)
   	     	   {
   	        		MPI_Iallreduce(MPI_IN_PLACE,
   	     		   acDeviceGetProfileBuffer(device,prof)+NGHOST+n_size,
   	     		   NGHOST,
   	     		   AC_REAL_MPI_TYPE,
   	     		   MPI_SUM,
   	     		   comm,
   	     		   &requests[NUM_OUTPUTS + prof]);

   	     	   }
   	        }
   	        else
   	        {
   	        	MPI_Iallreduce(MPI_IN_PLACE,
   	     		   acDeviceGetProfileBuffer(device,prof),
   	                        prof_size(prof,as_size_t(acDeviceGetLocalConfig(acGridGetDevice())[AC_mlocal])),
   	     		   AC_REAL_MPI_TYPE,
   	     		   MPI_SUM,
   	     		   comm,
   	     		   &requests[NUM_OUTPUTS + prof]);
   	        }
   	}
   }
  const auto& reduce_outputs = input_regions[0].memory.reduce_outputs;
  for(size_t i = 0; i < reduce_outputs.size(); ++i)
  {
	  const int var = reduce_outputs[i].variable;
	  if(real_output_is_global[var]  && reduce_outputs[i].type == AC_REAL_TYPE)   MPI_Allreduce(MPI_IN_PLACE, &local_res_real[i],1,AC_REAL_MPI_TYPE,to_mpi_op(reduce_outputs[i].op),grid_comm);
	  if(int_output_is_global[var]   && reduce_outputs[i].type == AC_INT_TYPE)    MPI_Iallreduce(MPI_IN_PLACE, &local_res_int[i],1,MPI_INT,to_mpi_op(reduce_outputs[i].op),grid_comm, &requests[i]);
#if AC_DOUBLE_PRECISION
	  if(float_output_is_global[var] && reduce_outputs[i].type == AC_FLOAT_TYPE)  MPI_Iallreduce(MPI_IN_PLACE, &local_res_float[i],1,MPI_FLOAT,to_mpi_op(reduce_outputs[i].op),grid_comm, &requests[i]);
#endif
    }
}
void
ReduceTask::load_outputs()
{
	const auto& reduce_outputs = input_regions[0].memory.reduce_outputs;
    	for(size_t i = 0; i < reduce_outputs.size(); ++i)
    	{
	    const int var = reduce_outputs[i].variable;
	    //TP: local reduce variables have been already loaded to the device after the local reduction
	    if(reduce_outputs[i].type == AC_REAL_TYPE)
	    {
	    	acDeviceSetOutput(device,(AcRealOutputParam)var,local_res_real[i]);
	    	if(real_output_is_global[var]) 
		{
			acLoadRealReduceRes(stream,(AcRealOutputParam)var,&local_res_real[i]);
		}
	    }
	    else if(reduce_outputs[i].type == AC_INT_TYPE)
	    {
	    	acDeviceSetOutput(device,(AcIntOutputParam)var,local_res_int[i]);
	    	if(int_output_is_global[var]) acLoadIntReduceRes(stream,(AcIntOutputParam)var,&local_res_int[i]);
	    }
#if AC_DOUBLE_PRECISION
	    else if(reduce_outputs[i].type == AC_FLOAT_TYPE)
	    {
	    	acDeviceSetOutput(device,(AcFloatOutputParam)var,local_res_float[i]);
	    	if(float_output_is_global[var]) acLoadFloatReduceRes(stream,(AcFloatOutputParam)var,&local_res_float[i]);
	    }
#endif
	    else if(reduce_outputs[i].type == AC_PROF_TYPE)
	    	;
	    else
	    {
	    	fprintf(stderr,"Unknown reduce output type: %ld,%d\n",i,reduce_outputs[i].type);
		exit(EXIT_FAILURE);
	    }
    	}
}

void
ReduceTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<ReduceState>(state)) {
    case ReduceState::Waiting:
        trace_file->trace(this, "waiting", "reducing");
        reduce();
        state = static_cast<int>(ReduceState::Reducing);
        break;
    case ReduceState::Reducing:
    {
	if(nothing_to_communicate)
	{
        	trace_file->trace(this, "reducing", "waiting");
        	state = static_cast<int>(ReduceState::Waiting);
		load_outputs();
        	break;
	}
	else
	{
        	trace_file->trace(this, "reducing", "communicating");
        	state = static_cast<int>(ReduceState::Communicating);
		communicate();
        	break;
	}
    }
    case ReduceState::Communicating:
    {

        trace_file->trace(this, "communicating", "loading");
        state = static_cast<int>(ReduceState::Loading);
	load_outputs();
        break;
    }

    case ReduceState::Loading:
    {
        trace_file->trace(this, "loading", "waiting");
        state = static_cast<int>(ReduceState::Waiting);
        break;
    }

    default:
        ERROR("ReduceTask in an invalid state.");
    }
}



BoundaryConditionTask::BoundaryConditionTask(
    AcTaskDefinition op, int3 boundary_normal_, int order_, int region_tag, const Volume start, const Volume nn, Device device_,
    std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           {Region(RegionFamily::Exchange_input, region_tag,  BOUNDARY_NONE, BOUNDARY_NONE, start, nn,op.halo_sizes, {std::vector<Field>(op.fields_in, op.fields_in + op.num_fields_in)  ,op.profiles_in , op.num_profiles_in , op.outputs_in,  op.num_outputs_in},3)},
           Region(RegionFamily::Exchange_output, region_tag, BOUNDARY_NONE, BOUNDARY_NONE, start, nn,op.halo_sizes, {std::vector<Field>(op.fields_out, op.fields_out + op.num_fields_out),op.profiles_reduce_out, op.num_profiles_reduce_out, op.outputs_out, op.num_outputs_out},3),
           op, device_, swap_offset_),
       boundary_normal(boundary_normal_),
       fieldwise(op.fieldwise)
{
    // TODO: the input regions for some of these will be weird, because they will depend on the
    // ghost zone of other fields
    //  This is not currently reflected

    // Create stream for boundary condition task
    {
        set_device(device);
        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }
    syncVBA();


    const auto local_config = acDeviceGetLocalConfig(device);
    acAnalysisBCInfo x_info =
	    		  boundary_normal.x == -1 ? acAnalysisGetBCInfo(local_config,op.kernel_enum,BOUNDARY_X_BOT) :
	    		  boundary_normal.x == 1  ? acAnalysisGetBCInfo(local_config,op.kernel_enum,BOUNDARY_X_TOP) :
	    		  (acAnalysisBCInfo){false,false};
    acAnalysisBCInfo y_info =
	    		  boundary_normal.y == -1 ? acAnalysisGetBCInfo(local_config,op.kernel_enum,BOUNDARY_Y_BOT) :
	    		  boundary_normal.y == 1  ? acAnalysisGetBCInfo(local_config,op.kernel_enum,BOUNDARY_Y_TOP) :
	    		  (acAnalysisBCInfo){false,false};
    acAnalysisBCInfo z_info =
	    		  boundary_normal.z == -1 ? acAnalysisGetBCInfo(local_config,op.kernel_enum,BOUNDARY_Z_BOT) :
	    		  boundary_normal.z == 1  ? acAnalysisGetBCInfo(local_config,op.kernel_enum,BOUNDARY_Z_TOP) :
	    		  (acAnalysisBCInfo){false,false};



    int3 translation = int3{(int)(output_region.dims.x + 1) * (-boundary_normal.x),
                            (int)(output_region.dims.y + 1) * (-boundary_normal.y),
                            (int)(output_region.dims.z + 1) * (-boundary_normal.z)};

    boundary_dims = {
        boundary_normal.x == 0 ? output_region.dims.x : 1,
        boundary_normal.y == 0 ? output_region.dims.y : 1,
        boundary_normal.z == 0 ? output_region.dims.z : 1,
    };
    ERRCHK_ALWAYS(output_region.halo.x > 0);
    ERRCHK_ALWAYS(output_region.halo.y > 0);
    ERRCHK_ALWAYS(output_region.halo.z > 0);

    ERRCHK_ALWAYS(boundary_dims.x > 0);
    ERRCHK_ALWAYS(boundary_dims.y > 0);
    ERRCHK_ALWAYS(boundary_dims.z > 0);

    auto& input_region = input_regions[0];

    // TODO: input_region is now set twice, overwritten here
    auto input_fields = input_region.memory.fields;
    input_region = Region(output_region.translate(translation));
    input_region.memory.fields = input_fields;
    const auto ghosts = op.halo_sizes;

    if(boundary_normal.x == -1 && x_info.larger_input)
    	input_region.position.x -= ghosts.x;
    if(boundary_normal.x == 1 && x_info.larger_input)
    	input_region.dims.x += ghosts.x;

    if(boundary_normal.y == -1 && y_info.larger_input)
    	output_region.position.y -= ghosts.y;
    if(boundary_normal.y == 1  && y_info.larger_input)
    	input_region.dims.y += ghosts.y;

    if(boundary_normal.z == -1 && z_info.larger_input)
    	input_region.position.z -= ghosts.z;
    if(boundary_normal.z == 1  && z_info.larger_input)
    	input_region.dims.z += ghosts.z;

    if(boundary_normal.x == -1 && x_info.larger_output)
    	output_region.dims.x += 1;
    if(boundary_normal.x == 1  && x_info.larger_output)
    	output_region.position.x -= 1;

    if(boundary_normal.y == -1 && y_info.larger_output)
    	output_region.dims.y += 1;
    if(boundary_normal.x == 1  && y_info.larger_output)
    	output_region.position.y -= 1;

    if(boundary_normal.z == -1 && z_info.larger_output)
    	output_region.dims.z += 1;
    if(boundary_normal.z == 1  && z_info.larger_output)
    	output_region.position.z -= 1;

    ERRCHK_ALWAYS(output_region.dims.x > 0);
    ERRCHK_ALWAYS(output_region.dims.y > 0);
    ERRCHK_ALWAYS(output_region.dims.z > 0);


    std::string kernel_name = std::string(kernel_names[op.kernel_enum]);
    name = "Boundary condition " + kernel_name + " " + std::to_string(order_) + ".(" +
           std::to_string(output_region.id.x) + "," + std::to_string(output_region.id.y) + "," +
           std::to_string(output_region.id.z) + ")" + ".(" + std::to_string(boundary_normal.x) +
           "," + std::to_string(boundary_normal.y) + "," + std::to_string(boundary_normal.z) + ")";
    task_type = TASKTYPE_BOUNDCOND;
    params = KernelParameters{op.kernel_enum, stream, 0, output_region.position,
                              output_region.position + output_region.dims, op.load_kernel_params_func};
}


void
BoundaryConditionTask::populate_boundary_region()
{
     const auto nmin = to_volume(ac_get_info()[AC_nmin]);
     const auto ghost = output_region.halo;
     const auto nn = output_region.comp_dims;

     if(fieldwise)
     {
     	for (auto variable : output_region.memory.fields) {
     		params.load_func->loader({&vba.on_device.kernel_input_params, device, (int)loop_cntr.i, boundary_normal, variable, params.kernel_enum});
     		const int3 region_id = output_region.id;
     		const Volume start = {(region_id.x == 1 ? nmin.x+ nn.x
     		                                           : region_id.x == -1 ? nmin.x-ghost.x : nmin.x),
     		                         (region_id.y == 1 ? nmin.y+ nn.y
     		                                           : region_id.y == -1 ? nmin.y-ghost.y : nmin.y),
     		                         (region_id.z == 1 ? nmin.z + nn.z
     		                                           : region_id.z == -1 ? nmin.z-ghost.z : nmin.z)};
     		const Volume end = start + boundary_dims;
     		acLaunchKernel(acGetOptimizedKernel(params.kernel_enum,vba), params.stream, start, end, vba);
     	}
     }
     else
     {
     		const int3 region_id = output_region.id;
     		const Volume start = {(region_id.x == 1 ? nmin.x + nn.x
     		                                           : region_id.x == -1 ? nmin.x-ghost.x : nmin.x),
     		                         (region_id.y == 1 ? nmin.y + nn.y
     		                                           : region_id.y == -1 ? nmin.y-ghost.y : nmin.y),
     		                         (region_id.z == 1 ? nmin.z + nn.z
     		                                           : region_id.z == -1 ? nmin.z-ghost.z : nmin.z)};
     		const Volume end = start + boundary_dims;
     		acLaunchKernel(acGetOptimizedKernel(params.kernel_enum,vba), params.stream, start, end, vba);
     }
}


bool
BoundaryConditionTask::test()
{
    switch (static_cast<BoundaryConditionState>(state)) {
    case BoundaryConditionState::Running: {
        return poll_stream();
    }
    default: {
        ERROR("BoundaryConditionTask in an invalid state.");
        return false;
    }
    }
}

void
BoundaryConditionTask::advance(const TraceFile* trace_file)
{
    switch (static_cast<BoundaryConditionState>(state)) {
    case BoundaryConditionState::Waiting:
        trace_file->trace(this, "waiting", "running");
        populate_boundary_region();
        state = static_cast<int>(BoundaryConditionState::Running);
        break;
    case BoundaryConditionState::Running:
        trace_file->trace(this, "running", "waiting");
        state = static_cast<int>(BoundaryConditionState::Waiting);
        break;
    default:
        ERROR("BoundaryConditionTask in an invalid state.");
    }
}


AcBoundary
boundary_from_normal(int3 normal)
{
    return (
        AcBoundary)((normal.x == -1 ? BOUNDARY_X_BOT : 0) | (normal.x == 1 ? BOUNDARY_X_TOP : 0) |
                    (normal.y == -1 ? BOUNDARY_Y_BOT : 0) | (normal.y == 1 ? BOUNDARY_Y_TOP : 0) |
                    (normal.z == -1 ? BOUNDARY_Z_BOT : 0) | (normal.z == 1 ? BOUNDARY_Z_TOP : 0));
}

int3
normal_from_boundary(AcBoundary boundary)
{
    return int3{((BOUNDARY_X_TOP & boundary) != 0) - ((BOUNDARY_X_BOT & boundary) != 0),
                ((BOUNDARY_Y_TOP & boundary) != 0) - ((BOUNDARY_Y_BOT & boundary) != 0),
                ((BOUNDARY_Z_TOP & boundary) != 0) - ((BOUNDARY_Z_BOT & boundary) != 0)};
}

#endif // AC_MPI_ENABLED
