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

#ifndef AC_INSIDE_AC_LIBRARY 
#define AC_INSIDE_AC_LIBRARY 
#endif


#include "task.h"
#include "astaroth.h"
#include "user_builtin_non_scalar_constants.h"

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

#define fatal(MESSAGE, ...) \
        { \
	acLogFromRootProc(ac_pid(),MESSAGE,__VA_ARGS__); \
	ERRCHK_ALWAYS(false); \
	} 

#define HALO_TAG_OFFSET (100) //"Namespacing" the MPI tag space to avoid collisions

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

AcTaskDefinition
acCompute(const AcKernel kernel, 
		Field fields_in[], const size_t num_fields_in, Field fields_out[], const size_t num_fields_out,
		Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out
		)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_COMPUTE;
    task_def.kernel_enum         = kernel;

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
	            std::function<void(ParamLoadingInfo)> load_func)
{
    AcTaskDefinition task_def{};
    task_def.task_type      = TASKTYPE_COMPUTE;
    task_def.kernel_enum         = kernel;

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
acSync()
{
    AcTaskDefinition task_def{};
    task_def.task_type = TASKTYPE_SYNC;
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

    task_def.fields_in      = ptr_copy(fields_in,num_fields_in);
    task_def.num_fields_in  = num_fields_in;
    task_def.fields_out     = ptr_copy(fields_out,num_fields_out);
    task_def.num_fields_out = num_fields_out;
    task_def.fieldwise = strstr(kernel_input_param_strs[kernel],"Field");
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
    if(kernel == BOUNDCOND_PERIODIC && !acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].x && boundary & BOUNDARY_X)
    {
	    fatal("%s","Periodic boundary condition in X even though AC_periodic_grid.x is false!!\n");
    }
    if(kernel != BOUNDCOND_PERIODIC && acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].x && boundary & BOUNDARY_X)
    {
	    fatal("%s","Non-periodic boundary condition in X even though AC_periodic_grid.x is true!!\n");
    }
    if(kernel == BOUNDCOND_PERIODIC && !acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].y && boundary & BOUNDARY_Y)
    {
	    fatal("%s","Periodic boundary condition in Y even though AC_periodic_grid.y is false!!\n");
    }
    if(kernel != BOUNDCOND_PERIODIC && acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].y && boundary & BOUNDARY_Y)
    {
	    fatal("%s","Non-periodic boundary condition in Y even though AC_periodic_grid.y is true!!\n");
    }
    if(kernel == BOUNDCOND_PERIODIC && !acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].z && boundary & BOUNDARY_Z)
    {
	    fatal("%s","Periodic boundary condition in Z even though AC_periodic_grid.z is false!!\n");
    }
    if(kernel != BOUNDCOND_PERIODIC && acDeviceGetLocalConfig(acGridGetDevice())[AC_periodic_grid].z && boundary & BOUNDARY_Z)
    {
	    fatal("%s","Non-periodic boundary condition in Z even though AC_periodic_grid.z is true!!\n");
    }
    return task_def;
}
size_t
get_compute_output_position(const int id, const size_t ghost, const size_t nn, const bool boundary_included)
{
	return id == -1  ? (boundary_included ? 0 : ghost) : 
		id == 1  ? (boundary_included ? nn+ghost : nn) : 
		(boundary_included ? ghost : ghost*2);
}

size_t
get_compute_output_dim(const int id, const size_t ghost, const size_t nn, const bool boundary_included)
{

	return id == 0 ? 
		 (boundary_included ? nn : nn - ghost*2) : 
		 ghost;
}
size_t
get_compute_input_position(const int id, const size_t ghost, const size_t nn, const bool boundary_included, const bool depends_on_boundary)
{
	const auto output_position = get_compute_output_position(id,ghost,nn,boundary_included);
	if(depends_on_boundary) return output_position - ghost;
	else return output_position;
}
size_t
get_compute_input_dim(const int id, const size_t ghost, const size_t nn, const bool boundary_included, const bool depends_on_boundary)
{
	const auto output_dims = get_compute_output_dim(id,ghost,nn,boundary_included);
	if(depends_on_boundary) return output_dims + 2*ghost;
	else return output_dims;
}


Region::Region(RegionFamily family_, int tag_, const AcBoundary depends_on_boundary, const AcBoundary computes_on_boundary, Volume nn, const RegionMemoryInputParams mem_)
    : family(family_), tag(tag_) 
{
    const Volume ghosts = to_volume(acDeviceGetLocalConfig(acGridGetDevice())[AC_nmin]);
    memory.profiles = {mem_.profiles, mem_.profiles + mem_.num_profiles};
    memory.reduce_outputs  = {mem_.reduce_outputs , mem_.reduce_outputs  + mem_.num_reduce_outputs};
    memory.fields = {};
    switch (family) {
    	case RegionFamily::Exchange_output: //Fallthrough
    	case RegionFamily::Exchange_input : {
    	    for(size_t i = 0; i < mem_.num_fields; ++i)
	    	if(vtxbuf_is_communicated[mem_.fields[i]]) memory.fields.push_back(mem_.fields[i]);
	    break;
	}
	default:
	    for(size_t i = 0; i < mem_.num_fields; ++i)
		memory.fields.push_back(mem_.fields[i]);
	break;
      }
      id = tag_to_id(tag);
      // facet class 0 = inner core
      // facet class 1 = face
      // facet class 2 = edge
      // facet class 3 = corner
      facet_class = (id.x == 0 ? 0 : 1) + (id.y == 0 ? 0 : 1) + (id.z == 0 ? 0 : 1);
      ERRCHK_ALWAYS(facet_class <= 3);
      
      switch (family) {
      case RegionFamily::Compute_output: {
      // clang-format off
      position = {
	    get_compute_output_position(id.x,ghosts.x,nn.x,computes_on_boundary & BOUNDARY_X),
	    get_compute_output_position(id.y,ghosts.y,nn.y,computes_on_boundary & BOUNDARY_Y),
	    get_compute_output_position(id.z,ghosts.z,nn.z,computes_on_boundary & BOUNDARY_Z)
      };
      // clang-format on
      dims = {
	      get_compute_output_dim(id.x,ghosts.x,nn.x,computes_on_boundary & BOUNDARY_X),
	      get_compute_output_dim(id.y,ghosts.y,nn.y,computes_on_boundary & BOUNDARY_Y),
	      get_compute_output_dim(id.z,ghosts.z,nn.z,computes_on_boundary & BOUNDARY_Z),
      };
      if(dims.x == 0 || dims.y == 0 || dims.z == 0)
      {
	      fprintf(stderr,"Incorrect region dims: %zu,%zu,%zu\n",dims.x,dims.y,dims.z);
	      ERRCHK_ALWAYS(dims.x != 0 && dims.y != 0 && dims.z != 0);
      }
      break;
      }
      case RegionFamily::Compute_input: {
      // clang-format off
      position = {
	    get_compute_input_position(id.x,ghosts.x,nn.x,computes_on_boundary & BOUNDARY_X,depends_on_boundary & BOUNDARY_X),
	    get_compute_input_position(id.y,ghosts.y,nn.y,computes_on_boundary & BOUNDARY_Y,depends_on_boundary & BOUNDARY_Y),
	    get_compute_input_position(id.z,ghosts.z,nn.z,computes_on_boundary & BOUNDARY_Z,depends_on_boundary & BOUNDARY_Z)
      };
      // clang-format on
      dims = {
	    get_compute_input_dim(id.x,ghosts.x,nn.x,computes_on_boundary & BOUNDARY_X,depends_on_boundary & BOUNDARY_X),
	    get_compute_input_dim(id.y,ghosts.y,nn.y,computes_on_boundary & BOUNDARY_Y,depends_on_boundary & BOUNDARY_Y),
	    get_compute_input_dim(id.z,ghosts.z,nn.z,computes_on_boundary & BOUNDARY_Z,depends_on_boundary & BOUNDARY_Z)
      };

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
      	    id.x == -1  ? 0 : id.x == 1 ? ghosts.x+ nn.x :  ghosts.x,
      	    id.y == -1  ? 0 : id.y == 1 ? ghosts.y + nn.y : ghosts.y,
      	    id.z == -1  ? 0 : id.z == 1 ? ghosts.z + nn.z : ghosts.z};
      // clang-format on
      dims = {id.x == 0 ? nn.x : ghosts.x , id.y == 0 ? nn.y : ghosts.y,
      	      id.z == 0 ? nn.z : ghosts.z};
      break;
      }
      case RegionFamily::Exchange_input: {
      position = {id.x == 1 ? nn.x : ghosts.x, id.y == 1 ? nn.y : ghosts.y,
      		  id.z == 1 ? nn.z : ghosts.z};
      dims = {id.x == 0 ? nn.x : ghosts.x, id.y == 0 ? nn.y : ghosts.y,
      	      id.z == 0 ? nn.z : ghosts.z};
      break;
      }
      default: {
      ERROR("Unknown region family.");
      }
      }
      volume = dims.x * dims.y * dims.z;
      }
      
      Region::Region(RegionFamily family_, int3 id_, Volume nn, const RegionMemoryInputParams mem_)
      : Region{family_, id_to_tag(id_), BOUNDARY_XYZ, BOUNDARY_NONE, nn, mem_}
      {
      ERRCHK_ALWAYS(id_.x == id.x && id_.y == id.y && id_.z == id.z);
      }
      
      Region::Region(Volume position_, Volume dims_, int tag_, const RegionMemory mem_, RegionFamily family_)
      : position(position_), dims(dims_), family(family_), tag(tag_)
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

Region::Region(Volume position_, Volume dims_, int tag_, const RegionMemory mem_)
: Region{position_, dims_, tag_, mem_, RegionFamily::None}{}



Region
Region::translate(int3 translation)
{
return Region(to_volume(this->position + translation), this->dims, this->tag, this->memory);
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
    int3 pid3d = getPid3D(pid, decomp, (int)proc_mapping_strategy);
    return boundary(decomp, pid3d, id);
}

bool
Region::is_on_boundary(uint3_64 decomp, int pid, AcBoundary boundary, AcProcMappingStrategy proc_mapping_strategy)
{
    int3 pid3d = getPid3D(pid, decomp, (int)proc_mapping_strategy);
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

AcBoundary
Region::boundary(uint3_64 decomp, int pid, int tag, AcProcMappingStrategy proc_mapping_strategy)
{
    int3 pid3d = getPid3D(pid, decomp, (int)proc_mapping_strategy);
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
    int3 pid3d     = getPid3D(pid, decomp, (int)proc_mapping_strategy);
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
Task::Task(int order_, Region input_region_, Region output_region_, AcTaskDefinition op,
           Device device_, std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : device(device_), vba(acDeviceGetVBA(device)), swap_offset(swap_offset_), state(wait_state), dep_cntr(), loop_cntr(),
      order(order_), active(true), boundary(BOUNDARY_NONE), input_region(input_region_),
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

/* Computation */
ComputeTask::ComputeTask(AcTaskDefinition op, int order_, int region_tag, Volume nn, Device device_,
                         std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Compute_input, region_tag,  get_kernel_depends_on_boundaries(op.kernel_enum), op.computes_on_halos, nn, {op.fields_in, op.num_fields_in  ,op.profiles_in, op.num_profiles_in,op.outputs_in,   op.num_outputs_in}),
           Region(RegionFamily::Compute_output, region_tag, get_kernel_depends_on_boundaries(op.kernel_enum), op.computes_on_halos, nn,{op.fields_out, op.num_fields_out,
		   merge_ptrs(op.profiles_reduce_out,op.profiles_write_out,op.num_profiles_reduce_out,op.num_profiles_write_out),
		   op.num_profiles_reduce_out + op.num_profiles_write_out,
		   op.outputs_out, op.num_outputs_out}),
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
    {
	set_device(device);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }

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
    else
    {
	if(!(get_kernel_depends_on_boundaries(op.kernel_enum) & BOUNDARY_X) && !(op.computes_on_halos & BOUNDARY_X))
	{
            output_region.dims.x += 2*NGHOST;
            input_region.dims.x  += 2*NGHOST;

            output_region.position.x -= NGHOST;
            input_region.position.x  -= NGHOST;
	}
	if(!(get_kernel_depends_on_boundaries(op.kernel_enum) & BOUNDARY_Y) && !(op.computes_on_halos & BOUNDARY_Y))
	{
            output_region.dims.y += 2*NGHOST;
            input_region.dims.y  += 2*NGHOST;

            output_region.position.y -= NGHOST;
            input_region.position.y  -= NGHOST;
	}

	if(!(get_kernel_depends_on_boundaries(op.kernel_enum) & BOUNDARY_Z) && !(op.computes_on_halos & BOUNDARY_Z))
	{
            output_region.dims.z += 2*NGHOST;
            input_region.dims.z  += 2*NGHOST;

            output_region.position.z -= NGHOST;
            input_region.position.z  -= NGHOST;
	}
    }


    syncVBA();

    // compute_func = compute_func_;

    params = KernelParameters{op.kernel_enum, stream, 0, output_region.position,
                              output_region.position + output_region.dims, op.load_kernel_params_func};
    name   = "Compute " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) + "," +
           std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) + ")";
    task_type = TASKTYPE_COMPUTE;
}

ComputeTask::ComputeTask(AcTaskDefinition op, int order_, Region input_region_, Region output_region_, Device device_,std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           input_region_,
           output_region_,
           op, device_, swap_offset_)
{
    // stream = device->streams[STREAM_DEFAULT + region_tag];
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
HaloMessage::HaloMessage(Volume dims, size_t num_vars)
{
    length       = dims.x * dims.y * dims.z * num_vars;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data, bytes));
#if !(USE_CUDA_AWARE_MPI)
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data_pinned, bytes));
#endif
    request = MPI_REQUEST_NULL;
}

HaloMessage::~HaloMessage()
{
    if (request != MPI_REQUEST_NULL) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    length = -1;
    cudaFree(data);
#if !(USE_CUDA_AWARE_MPI)
    cudaFree(data_pinned);
#endif
    data = NULL;
}

#if !(USE_CUDA_AWARE_MPI)
void
HaloMessage::pin(const Device device, const cudaStream_t stream)
{
    set_device(device);
    pinned       = true;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA(cudaMemcpyAsync(data_pinned, data, bytes, cudaMemcpyDefault, stream));
}

void
HaloMessage::unpin(const Device device, const cudaStream_t stream)
{
    if (!pinned)
        return;

    set_device(device);
    pinned       = false;
    size_t bytes = length * sizeof(AcRealPacked);
    ERRCHK_CUDA(cudaMemcpyAsync(data, data_pinned, bytes, cudaMemcpyDefault, stream));
}
#endif

// HaloMessageSwapChain
HaloMessageSwapChain::HaloMessageSwapChain() {}

HaloMessageSwapChain::HaloMessageSwapChain(Volume dims, size_t num_vars)
    : buf_idx(SWAP_CHAIN_LENGTH - 1)
{
    buffers.reserve(SWAP_CHAIN_LENGTH);
    for (int i = 0; i < SWAP_CHAIN_LENGTH; i++) {
        buffers.emplace_back(dims, num_vars);
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
    MPI_Request req = buffers[buf_idx].request;
    if (req != MPI_REQUEST_NULL) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    return &buffers[buf_idx];
}



// HaloExchangeTask
HaloExchangeTask::HaloExchangeTask(AcTaskDefinition op, int order_, int tag_0, int halo_region_tag,
                                   AcGridInfo grid_info, uint3_64 decomp, Device device_,
                                   std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Exchange_input, halo_region_tag,  BOUNDARY_NONE, BOUNDARY_NONE, grid_info.nn,  {op.fields_in,  op.num_fields_in ,op.profiles_in, op.num_profiles_in ,op.outputs_in, op.num_outputs_in}),
           Region(RegionFamily::Exchange_output, halo_region_tag, BOUNDARY_NONE, BOUNDARY_NONE, grid_info.nn, {op.fields_out,op.num_fields_out,op.profiles_reduce_out, op.num_profiles_reduce_out ,op.outputs_out, op.num_outputs_out}),
           op, device_, swap_offset_),
      recv_buffers(output_region.dims, op.num_fields_in),
      send_buffers(input_region.dims, op.num_fields_out)
{
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

    const auto proc_strategy = acDeviceGetLocalConfig(device)[AC_proc_mapping_strategy];
    counterpart_rank = getPid(getPid3D(rank, decomp, proc_strategy) + output_region.id, decomp, proc_strategy);
    // MPI tags are namespaced to avoid collisions with other MPI tasks
    send_tag = tag_0 + input_region.tag;
    recv_tag = tag_0 + Region::id_to_tag(-output_region.id);

    // Post receive immediately, this avoids unexpected messages
    active = ((acDeviceGetLocalConfig(device)[AC_include_3d_halo_corners]) || output_region.facet_class != 3) ? true : false;
    if (active) {
        acVerboseLogFromRootProc(rank, "Halo exchange task ctor: posting early receive\n");
        receive();
        acVerboseLogFromRootProc(rank, "Halo exchange task ctor: done posting early receive\n");
    }
    name = "Halo exchange " + std::to_string(order_) + ".(" + std::to_string(output_region.id.x) +
           "," + std::to_string(output_region.id.y) + "," + std::to_string(output_region.id.z) +
           ")";
    task_type = TASKTYPE_HALOEXCHANGE;

    //TP: HaloExchangeTasks usually have input and output regions on the same side of the boundary
    //Thus if you directly moving data through kernels you have to remap the output position to the other side of the boundary
    if(sendingToItself())
    {
	    const auto ghosts = acDeviceGetLocalConfig(device)[AC_nmin];
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
    auto msg = recv_buffers.get_current_buffer();
    if (msg->request != MPI_REQUEST_NULL) {
        MPI_Cancel(&msg->request);
    }

    set_device(device);
    // dependents.clear();
    cudaStreamDestroy(stream);
}

void
HaloExchangeTask::pack()
{
    auto msg = send_buffers.get_fresh_buffer();
    acKernelPackData(stream, vba, input_region.position, input_region.dims,
                             msg->data, input_region.memory.fields.data(),
                             input_region.memory.fields.size());
}

void
HaloExchangeTask::move()
{
	acKernelMoveData(stream, input_region.position, output_region.position, input_region.dims, output_region.dims, vba, input_region.memory.fields.data(), input_region.memory.fields.size());
}

void
HaloExchangeTask::unpack()
{

    auto msg = recv_buffers.get_current_buffer();
#if !(USE_CUDA_AWARE_MPI)
    msg->unpin(device, stream);
#endif
    acKernelUnpackData(stream, msg->data, output_region.position, output_region.dims,
                               vba, output_region.memory.fields.data(),
		    		output_region.memory.fields.size());
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
	return rank == counterpart_rank && n_procs == 1 && !acDeviceGetLocalConfig(device)[AC_skip_single_gpu_optim];
}

void
HaloExchangeTask::wait_recv()
{
    auto msg = recv_buffers.get_current_buffer();
    MPI_Wait(&msg->request, MPI_STATUS_IGNORE);
}

void
HaloExchangeTask::wait_send()
{
    auto msg = send_buffers.get_current_buffer();
    MPI_Wait(&msg->request, MPI_STATUS_IGNORE);
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

    ERRCHK_ALWAYS(MPI_Irecv(msg->data, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              recv_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request) == MPI_SUCCESS);
    if (rank == 0) {
        // fprintf(stderr, "Returned from MPI_Irecv\n");
    }
}

void
HaloExchangeTask::sendDevice()
{
    auto msg = send_buffers.get_current_buffer();
    sync();
    ERRCHK_ALWAYS(MPI_Isend(msg->data, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              send_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request) == MPI_SUCCESS);
}

void
HaloExchangeTask::exchangeDevice()
{
    // set_device(device);
    receiveDevice();
    sendDevice();
}

#if !(USE_CUDA_AWARE_MPI)
void
HaloExchangeTask::receiveHost()
{
    // TODO: change these to debug log statements at high verbosity (there will be very many of
    // these outputs)
    if (rank == 0) {
        // fprintf("receiveHost, getting buffer\n");
    }
    auto msg = recv_buffers.get_fresh_buffer();
    if (rank == 0) {
        // fprintf("Called MPI_Irecv\n");
    }
    MPI_Irecv(msg->data_pinned, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              recv_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request);
    if (rank == 0) {
        // fprintf("Returned from MPI_Irecv\n");
    }
    msg->pinned = true;
}

void
HaloExchangeTask::sendHost()
{
    auto msg = send_buffers.get_current_buffer();
    msg->pin(device, stream);
    sync();
    MPI_Isend(msg->data_pinned, msg->length, AC_REAL_MPI_TYPE, counterpart_rank,
              send_tag + HALO_TAG_OFFSET, acGridMPIComm(), &msg->request);
}
void
HaloExchangeTask::exchangeHost()
{
    // set_device(device);
    receiveHost();
    sendHost();
}
#endif

void
HaloExchangeTask::receive()
{
    // TODO: change these fprintfs to debug log statements at high verbosity (there will be very
    // many of these outputs)
#if USE_CUDA_AWARE_MPI
    if (rank == 0) {
        // fprintf(stderr, "receiveDevice()\n");
    }
    receiveDevice();
    if (rank == 0) {
        // fprintf(stderr, "returned from receiveDevice()\n");
    }
#else
    if (rank == 0) {
        // fprintf(stderr, "receiveHost()\n");
    }
    receiveHost();
    if (rank == 0) {
        // fprintf(stderr, "returned from receiveHost()\n");
    }
#endif
}

void
HaloExchangeTask::send()
{
#if USE_CUDA_AWARE_MPI
    sendDevice();
#else
    sendHost();
#endif
}

void
HaloExchangeTask::exchange()
{
#if USE_CUDA_AWARE_MPI
    exchangeDevice();
#else
    exchangeHost();
#endif
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
        ERRCHK_ALWAYS(MPI_Test(&msg->request, &request_complete, MPI_STATUS_IGNORE) == MPI_SUCCESS);
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
        pack();
        state = static_cast<int>(HaloExchangeState::Packing);
        break;
    case HaloExchangeState::Packing:
        trace_file->trace(this, "packing", "receiving");
        sync();
        send();
        state = static_cast<int>(HaloExchangeState::Exchanging);
        break;
    case HaloExchangeState::Exchanging:
        trace_file->trace(this, "receiving", "unpacking");
        sync();
        unpack();
        state = static_cast<int>(HaloExchangeState::Unpacking);
        break;
    case HaloExchangeState::Unpacking:
        trace_file->trace(this, "unpacking", "waiting");
        receive();
        sync();
        state = static_cast<int>(HaloExchangeState::Waiting);
        break;
    default:
        ERROR("HaloExchangeTask in an invalid state.");
    }
}




SyncTask::SyncTask(AcTaskDefinition op, int order_, Volume nn, Device device_,
                   std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
	//TP: this will make tasks with larger than necessary regions in case of two dimensional setup
	//but that does not really harm since the whole point of this task is to be a dummy that synchronizes
    : Task(order_,
           Region({0,0,0},(Volume){(size_t)2*NGHOST+nn.x,(size_t)2*NGHOST+nn.y,(size_t)2*NGHOST+nn.z}, 0, {}),
           Region({0,0,0},(Volume){(size_t)2*NGHOST+nn.x,(size_t)2*NGHOST+nn.y,(size_t)2*NGHOST+nn.z}, 0, {}),
           op, device_, swap_offset_)
{

    // Synctask is on default stream
    {
        stream = 0;
    }
    syncVBA();

    name      = "SyncTask" + std::to_string(order_);
    task_type = TASKTYPE_SYNC;
}

bool
SyncTask::test()
{
    // always ready
    return true;
}

void
SyncTask::advance(const TraceFile* trace_file)
{
    //no tracing for now simply silence unused warning
    (void)trace_file;
    //Synchronize everything
    acGridSynchronizeStream(STREAM_ALL);
}

ReduceTask::ReduceTask(AcTaskDefinition op, int order_, int region_tag, Volume nn, Device device_,
                         std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Compute_input, region_tag,  BOUNDARY_NONE, op.computes_on_halos, nn, {op.fields_in, op.num_fields_in  ,op.profiles_in, op.num_profiles_in ,op.outputs_in,  op.num_outputs_in}),
           Region(RegionFamily::Compute_output, region_tag, BOUNDARY_NONE, op.computes_on_halos, nn,{op.fields_out, op.num_fields_out,op.profiles_reduce_out,op.num_profiles_reduce_out,op.outputs_out, op.num_outputs_out}),
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

    const Volume ghosts = to_volume(acDeviceGetLocalConfig(acGridGetDevice())[AC_nmin]);

    input_region.position = {0,0,0};
    input_region.dims     = {nn.x+2*ghosts.x,nn.y+2*ghosts.y,nn.z+2*ghosts.z};

    output_region.position = {0,0,0};
    output_region.dims     = {nn.x+2*ghosts.x,nn.y+2*ghosts.y,nn.z+2*ghosts.z};

    if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_X))
	    reduces_only_prof = PROFILE_X;
    else if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_Y))
	    reduces_only_prof = PROFILE_Y;
    else if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_Z))
	    reduces_only_prof = PROFILE_Z;

    syncVBA();

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
	const auto& reduce_outputs = input_region.memory.reduce_outputs;
     	const auto nn = acGetLocalNN(acDeviceGetLocalConfig(device));

	if constexpr (NUM_PROFILES != 0)
	{
		for(const auto& prof : input_region.memory.profiles)
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
void
ReduceTask::communicate()
{
   const auto nn = acGetLocalNN(acDeviceGetLocalConfig(device));
   if constexpr(NUM_PROFILES != 0)
   {
   	for(const auto& prof: input_region.memory.profiles)
   	{
   	        const auto sub_comms = acGridMPISubComms();
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
}
void
ReduceTask::load_outputs()
{
	const auto& reduce_outputs = input_region.memory.reduce_outputs;
    	for(size_t i = 0; i < reduce_outputs.size(); ++i)
    	{
	    if(reduce_outputs[i].type == AC_REAL_TYPE)
	    	acDeviceSetOutput(device,(AcRealOutputParam)reduce_outputs[i].variable,local_res_real[i]);
	    else if(reduce_outputs[i].type == AC_INT_TYPE)
	    	acDeviceSetOutput(device,(AcIntOutputParam)reduce_outputs[i].variable,local_res_int[i]);
#if AC_DOUBLE_PRECISION
	    else if(reduce_outputs[i].type == AC_FLOAT_TYPE)
	    	acDeviceSetOutput(device,(AcFloatOutputParam)reduce_outputs[i].variable,local_res_float[i]);
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
	if(input_region.memory.profiles.size() == 0)
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

        trace_file->trace(this, "communicating", "waiting");
        state = static_cast<int>(ReduceState::Waiting);
	load_outputs();
        break;
    }
    default:
        ERROR("ReduceTask in an invalid state.");
    }
}



BoundaryConditionTask::BoundaryConditionTask(
    AcTaskDefinition op, int3 boundary_normal_, int order_, int region_tag, Volume nn, Device device_,
    std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset_)
    : Task(order_,
           Region(RegionFamily::Exchange_input, region_tag,  BOUNDARY_NONE, BOUNDARY_NONE, nn,  {op.fields_in, op.num_fields_in  ,op.profiles_in , op.num_profiles_in , op.outputs_in,  op.num_outputs_in}),
           Region(RegionFamily::Exchange_output, region_tag, BOUNDARY_NONE, BOUNDARY_NONE, nn, {op.fields_out, op.num_fields_out,op.profiles_reduce_out, op.num_profiles_reduce_out, op.outputs_out, op.num_outputs_out}),
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

    // TODO: input_region is now set twice, overwritten here
    auto input_fields = input_region.memory.fields;
    input_region = Region(output_region.translate(translation));
    input_region.memory.fields = input_fields;
    const auto ghosts = local_config[AC_nmin];

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
     const auto nn = acGetLocalNN(acDeviceGetLocalConfig(device));
     const auto ghosts = acDeviceGetLocalConfig(device)[AC_nmin];
     if(fieldwise)
     {
     	for (auto variable : output_region.memory.fields) {
     		params.load_func->loader({&vba.on_device.kernel_input_params, device, (int)loop_cntr.i, boundary_normal, variable, params.kernel_enum});
     		const int3 region_id = output_region.id;
     		const Volume start = {(region_id.x == 1 ? ghosts.x+ nn.x
     		                                           : region_id.x == -1 ? 0 : ghosts.x),
     		                         (region_id.y == 1 ? ghosts.y+ nn.y
     		                                           : region_id.y == -1 ? 0 : ghosts.y),
     		                         (region_id.z == 1 ? ghosts.z + nn.z
     		                                           : region_id.z == -1 ? 0 : ghosts.z)};
     		const Volume end = start + boundary_dims;
     		acLaunchKernel(acGetOptimizedKernel(params.kernel_enum,vba), params.stream, start, end, vba);
     	}
     }
     else
     {
     		const int3 region_id = output_region.id;
     		const Volume start = {(region_id.x == 1 ? ghosts.x + nn.x
     		                                           : region_id.x == -1 ? 0 : ghosts.x),
     		                         (region_id.y == 1 ? ghosts.y + nn.y
     		                                           : region_id.y == -1 ? 0 : ghosts.y),
     		                         (region_id.z == 1 ? ghosts.z + nn.z
     		                                           : region_id.z == -1 ? 0 : ghosts.z)};
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
