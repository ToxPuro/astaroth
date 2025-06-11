#if AC_MPI_ENABLED 

#include "task.h"

#include <algorithm>
#include <cstring> //memcpy
#include <mpi.h>
#include <queue>
#include <vector>
#include <stack>


#include "decomposition.h" //getPid3D, morton3D
#include "errchk.h"
#include "math_utils.h"
#include "timer_hires.h"
AcMeshInfo
get_info()
{
	return acDeviceGetLocalConfig(acGridGetDevice());
}



#include "analysis_grid_helpers.h"
#include "taskgraph_kernels.h"
#include "taskgraph_bc_handles.h"
#include "taskgraph_kernel_bcs.h"

static int ac_pid()
{
	int res{};
	MPI_Comm_rank(acGridMPIComm(),&res);
	return res;
}

#define fatal(MESSAGE, ...) \
        { \
	acLogFromRootProc(ac_pid(),MESSAGE,__VA_ARGS__); \
	exit(EXIT_FAILURE); \
	} 

bool
is_bc_taskgraph(const AcDSLTaskGraph graph)
{
	for(const auto& bc : DSLTaskGraphKernelBoundaries[graph])
		if (bc != BOUNDARY_NONE) return true;
	return false;
}

typedef struct
{
	AcKernel kernel;
	AcBoundary boundary;
	std::vector<Field> in;
	std::vector<Field> out;
	acAnalysisBCInfo info;
	size_t topological_order;
	int3 id;
} BoundCond;

typedef std::array<std::array<BoundCond,27>,NUM_VTXBUF_HANDLES> FieldBCs;

typedef struct
{
	std::vector<Field> in;
	std::vector<Field> out;
} KernelFields;

KernelFields
get_kernel_fields(const AcKernel kernel)
{
		const auto info = get_kernel_analysis_info();
		KernelFields res{};
		for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
		{
			bool in = info[kernel].read_fields[i]             || info[kernel].field_has_stencil_op[i] || kernel == BOUNDCOND_PERIODIC;
			for(int ray = 0; ray < NUM_RAYS; ++ray)
				in |= info[kernel].ray_accessed[i][ray];
			if(in)    res.in.push_back((Field)i);
			if(info[kernel].written_fields[i]          || kernel == BOUNDCOND_PERIODIC)    res.out.push_back((Field)i);
		}
		return res;
}
typedef struct
{
	std::vector<Profile> in;
	std::vector<Profile> reduce_out;
	std::vector<Profile> write_out;
} KernelProfiles;

typedef struct
{
	std::vector<KernelReduceOutput> in;
	std::vector<KernelReduceOutput> out;
}
KernelReduceOutputs;

typedef struct
{
	KernelFields   fields;
	KernelProfiles profiles;
	KernelReduceOutputs reduce_outputs;
} KernelOutputs;

KernelProfiles
get_kernel_profiles(const AcKernel kernel)
{
		const auto info = get_kernel_analysis_info();
		KernelProfiles res{};
		for(int i = 0; i < NUM_PROFILES; ++i)
		{
			if(info[kernel].read_profiles[i] || info[kernel].profile_has_stencil_op[i])       res.in.push_back((Profile)i);
			if(info[kernel].reduced_profiles[i])    res.reduce_out.push_back((Profile)i);
			if(info[kernel].written_profiles[i])    res.write_out.push_back((Profile)i);
		}
		return res;
}
KernelReduceOutputs
get_kernel_reduce_outputs(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	KernelReduceOutputs res{};
	for(size_t i = 0; i < info[kernel].n_reduce_outputs; ++i)
	{
		res.out.push_back(info[kernel].reduce_outputs[i]);
	}
	for(size_t i = 0; i < info[kernel].n_reduce_inputs; ++i)
	{
		res.in.push_back(info[kernel].reduce_inputs[i]);
	}
	return res;
}
std::vector<KernelFields>
get_kernel_fields(const std::vector<AcKernel> kernels)
{
		std::vector<KernelFields> res{};
		for(const auto& kernel : kernels) res.push_back(get_kernel_fields(kernel));
		return res;
}
KernelOutputs
get_kernel_outputs(const AcKernel kernel)
{
	return 
	{
		get_kernel_fields(kernel),
		get_kernel_profiles(kernel),
		get_kernel_reduce_outputs(kernel)
	};
}


template <typename T>
bool
overlaps(const std::vector<T>& a,const std::vector<T>& b)
{
	for(auto& elem : a)
		if(std::find(b.begin(), b.end(), elem) != b.end()) return true;
	return false;
}
std::vector<size_t>
generate_topological_order(const std::vector<BoundCond>& bcs, const char* bc_name)
{
	std::vector<std::vector<bool>> dependency_matrix(bcs.size());
	std::vector<bool> empty_row(bcs.size());
	for(size_t i = 0; i < bcs.size(); ++i) dependency_matrix[i] = empty_row;

	//TP: at the moment this is somewhat conservative since if j only writes to the boundary and i does not read from boundary there is no dependency
	//TP: relax when cannot determine a topological order anymore
	for(size_t i = 0; i < bcs.size(); ++i)
		for(size_t j = 0; j < bcs.size(); ++j)
			dependency_matrix[i][j] = i != j && overlaps(bcs[i].in,bcs[j].out) && (bcs[i].boundary & bcs[j].boundary);
	//Kahn's algorithm
	std::queue<size_t> vertices_under_work;

	auto no_incoming_edges = [&](const size_t current_vertex)
	{
		bool res = true;
		for(size_t j = 0; j < bcs.size(); ++j)
			res &= !dependency_matrix[current_vertex][j];
		return res;
	};

	auto get_dependent_vertices = [&](const size_t current_vertex)
	{
		std::vector<size_t> res{};
		for(size_t i = 0; i < bcs.size(); ++i)
			if(dependency_matrix[i][current_vertex]) res.push_back(i);
		return res;
	};

	for(size_t i = 0; i < bcs.size(); ++i)
	{
		if(no_incoming_edges(i)) vertices_under_work.push(i);
	}
	if(vertices_under_work.size() == 0)
		fatal("Cannot continue: BC dependencies of %s do not form a DAG\n",bc_name);

	std::vector<size_t> res{};
	while(vertices_under_work.size() != 0)
	{
		auto current_vertex = vertices_under_work.front();
		vertices_under_work.pop();
		res.push_back(current_vertex);
		for(const auto& vertex : get_dependent_vertices(current_vertex))
		{
			dependency_matrix[vertex][current_vertex] = false;
			if (no_incoming_edges(vertex)) vertices_under_work.push(vertex);
		}
	}
	for(size_t i = 0; i < bcs.size(); ++i)
		for(size_t j = 0; j < bcs.size(); ++j)
			if (dependency_matrix[i][j])
				fatal("Cannot continue: BC dependencies of %s do not form a DAG\n",bc_name);
	return res;
}

std::vector<AcBoundary>
get_boundaries()
{
		const auto info = get_info();
		std::vector<AcBoundary> boundaries{};
		if(!info[AC_dimension_inactive].x)
		{
			boundaries.push_back(BOUNDARY_X_TOP);
			boundaries.push_back(BOUNDARY_X_BOT);
		}
		if(!info[AC_dimension_inactive].y)
		{
			boundaries.push_back(BOUNDARY_Y_TOP);
			boundaries.push_back(BOUNDARY_Y_BOT);
		}
		if(!info[AC_dimension_inactive].z)
		{
			boundaries.push_back(BOUNDARY_Z_TOP);
			boundaries.push_back(BOUNDARY_Z_BOT);
		}
		return boundaries;
}
bool
bc_output_fields_overlap(const std::vector<BoundCond>& bcs)
{
	std::array<std::vector<Field>,6> fields_written{};
	std::array<std::vector<const char*>,6> bc_names{};
	auto add_to_written_fields = [&](const int i, const Field field, const BoundCond bc)
	{
		const auto it = std::find(fields_written[i].begin(), fields_written[i].end(),field);
		const bool already_written =  it != fields_written[i].end();
		if(already_written)
		{
			const ptrdiff_t index = it - fields_written[i].begin();			
			fatal("Cannot continue: Field %s written both from %s and %s\n",field_names[field],bc_names[i][index],kernel_names[bc.kernel]);
		}
		fields_written[i].push_back(field);
		bc_names[i].push_back(kernel_names[bc.kernel]);
		return already_written;
	};
	for(auto& bc : bcs)
	{
		auto fields = get_kernel_fields(bc.kernel);
		for(auto& field: fields.out)
		{
			const auto boundaries = get_boundaries();
			for(int i = 0; i < 6; ++i)
			{
				if(bc.boundary & boundaries[i])
				{
					if (add_to_written_fields(i,field,bc)) return true;
				}
			}
		}
	}
	return false;
}
KernelParamsLoader
get_loader(const int graph, const int call_index)
{
	#include "user_loaders.h"
	return  DSLTaskGraphKernelLoaders[graph][call_index];
}


std::vector<AcKernel>
get_optimized_kernels(const AcDSLTaskGraph graph, const bool filter_unnecessary_ones)
{
	auto kernel_calls = DSLTaskGraphKernels[graph];
	std::vector<AcKernel> res{};
	for(size_t call_index = 0; call_index < kernel_calls.size(); ++call_index)
	{
		VertexBufferArray vba{};
		auto loader = get_loader(graph,call_index);
    		loader({&vba.on_device.kernel_input_params, acGridGetDevice(), {}, {}, {}, kernel_calls[call_index]});
		const AcKernel optimized_kernel = acGetOptimizedKernel(kernel_calls[call_index],vba);
		if(filter_unnecessary_ones)
		{
			auto outputs = get_kernel_outputs(optimized_kernel);
			if(outputs.fields.out.size() == 0 && outputs.profiles.write_out.size() == 0 && outputs.profiles.reduce_out.size() == 0 && outputs.reduce_outputs.out.size() == 0) 
			{
				res.push_back(AC_NULL_KERNEL);
				continue;
			}
		}
		res.push_back(optimized_kernel);
	}
	return res;
}


std::vector<BoundCond>
get_boundconds(const AcDSLTaskGraph bc_graph, const bool optimized)
{

	std::vector<BoundCond> res{};
	const std::vector<AcKernel>   kernels    = optimized ?
							get_optimized_kernels(bc_graph,false) :
							DSLTaskGraphKernels[bc_graph];
	const std::vector<AcBoundary> boundaries = DSLTaskGraphKernelBoundaries[bc_graph];
	for(size_t i = 0; i < kernels.size(); ++i)
	std::vector<acAnalysisBCInfo> bc_infos{};
	for(size_t i = 0; i < kernels.size(); ++i)
	{
		auto bc_info = acAnalysisGetBCInfo(get_info(),kernels[i],boundaries[i]);
		auto fields = get_kernel_fields(kernels[i]);
		res.push_back((BoundCond){kernels[i],boundaries[i],fields.in,fields.out,bc_info,0,(int3){0,0,0}});
	}
	const bool bcs_overlap = bc_output_fields_overlap(res);
	if(bcs_overlap)
		fatal("Cannot continue: overlap in output fields in BCs of %s\n",taskgraph_names[bc_graph]);
	auto topological_order = generate_topological_order(res,taskgraph_names[bc_graph]);
	for(size_t i = 0; i < kernels.size(); ++i)
	{
		for(size_t j = 0; j < kernels.size(); ++j)
		{
			if (topological_order[j] == i)
				res[i].topological_order = j;
		}
	}
	return res;
}

std::tuple<Volume,Volume>
get_launch_bounds_from_fields(const std::vector<Field> in_fields,  const std::vector<Field> out_fields)
{
	Volume start = (Volume){0,0,0};
	Volume end   = (Volume){0,0,0};
	const auto get_start_end = [&start,&end](const std::vector<Field> fields)
	{
		if(fields.size() > 0)
		{
			const int3 dims = get_info()[vtxbuf_dims[fields[0]]];
			bool all_same = true;
			for(size_t i = 1; i < fields.size(); ++i)
			{
				all_same &= get_info()[vtxbuf_dims[fields[i]]] == dims;
			}
			if(all_same && dims != get_info()[AC_mlocal])
			{
				start = to_volume(get_info()[AC_nmin]);
				end   = to_volume(dims) - to_volume(get_info()[AC_nmin]);
			}
		}
	};
	if(out_fields.size() > 0) get_start_end(out_fields);
	else if(in_fields.size() > 0) get_start_end(in_fields);
	return {start,end};
}

void
log_launch_bounds(FILE* stream, std::vector<Field> in_fields, std::vector<Field> out_fields)
{
	const auto [start,end] = get_launch_bounds_from_fields(in_fields,out_fields);
	if(end.x > 0)
	{
		if(!ac_pid())
		{
			fprintf(stream,",{%ld,%ld,%ld}",start.x,start.y,start.z);
			fprintf(stream,",{%ld,%ld,%ld}",end.x,end.y,end.z);
		}
	}
}
void
log_boundcond(FILE* stream, const BoundCond bc)
{
	auto log_fields = [&](const auto& fields)
	{
		if(!ac_pid()) fprintf(stream,"{");
		for(const auto& field : fields)
			if(!ac_pid()) fprintf(stream, "%s,",field_names[field]);
		if(!ac_pid()) fprintf(stream,"}");
	};
	if(!ac_pid()) fprintf(stream,"BoundCond(%s,%s,",ac_boundary_to_str(bc.boundary), kernel_names[bc.kernel]);
	log_fields(bc.in);
	if(!ac_pid()) fprintf(stream,",");
	log_fields(bc.out);
	log_launch_bounds(stream,bc.in,bc.out);
	if(!ac_pid()) fprintf(stream,"{%d,%d,%d}",bc.id.x,bc.id.y,bc.id.z);
	if(!ac_pid()) fprintf(stream,")\n");
}

AcTaskDefinition
gen_bc(FILE* stream, const BoundCond bc, const std::vector<facet_class_range> halo_types)
{
	log_boundcond(stream,bc);
	const auto [bc_start,bc_end] = get_launch_bounds_from_fields(bc.in,bc.out);
	return acBoundaryCondition(bc.boundary,bc.kernel,bc.in,bc.out,(Volume)bc_start,(Volume)bc_end,halo_types,bc.id);
}

std::vector<AcTaskDefinition>
acGetDSLBCTaskGraphOps(const AcDSLTaskGraph bc_graph, const bool optimized)
{
	FILE* stream = !ac_pid() ? fopen("taskgraph_log.txt","a") : NULL;
	if (!ac_pid()) fprintf(stream,"%s Ops:\n",taskgraph_names[bc_graph]);
	std::vector<AcTaskDefinition> res{};
	auto bcs = get_boundconds(bc_graph, optimized);
	//TP: insert boundconds in topological order
	for(size_t i = 0; i < bcs.size(); ++i)
	{
		for(auto& bc : bcs)
			if(bc.topological_order == i)
			{
				std::vector<facet_class_range> halo_types{};
				for(size_t j = 0; j < bc.out.size(); ++j) halo_types.push_back((facet_class_range){1,3});
				res.push_back(gen_bc(stream,bc,halo_types));
			}
	}

	if(!ac_pid()) fclose(stream);
	return res;
}
bool
id_and_bc_overlap(const int x, const int y, const int z, const AcBoundary boundary)
{
	bool overlaps = false;
	overlaps |= ((boundary & BOUNDARY_X_BOT) != 0 && x == -1);
	overlaps |= ((boundary & BOUNDARY_X_TOP) != 0 && x == +1);
	overlaps |= ((boundary & BOUNDARY_Y_BOT) != 0 && y == -1);
	overlaps |= ((boundary & BOUNDARY_Y_TOP) != 0 && y == +1);
	overlaps |= ((boundary & BOUNDARY_Z_BOT) != 0 && z == -1);
	overlaps |= ((boundary & BOUNDARY_Z_TOP) != 0 && z == +1);
	return overlaps;
}


FieldBCs
get_field_boundconds(const AcDSLTaskGraph bc_graph, const bool optimized)
{
	const auto bcs = get_boundconds(bc_graph,optimized);
	const std::vector<AcBoundary> boundaries = {BOUNDARY_X_TOP, BOUNDARY_X_BOT, BOUNDARY_Y_TOP, BOUNDARY_Y_BOT, BOUNDARY_Z_TOP, BOUNDARY_Z_BOT};

	FieldBCs res;
	BoundCond empty_bc = {AC_NULL_KERNEL,BOUNDARY_NONE,{}, {}, {},0,(int3){0,0,0}};
	for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
		for(int j = 0; j < 27; ++j)
			res[i][j] = empty_bc;

	for(auto& bc : bcs)
	{
		for(int x = -1; x <= 1; ++x)
			for(int y = -1; y <= 1; ++y)
				for(int z = -1; z <= 1; ++z)
				{
					const int index = (x+1) + 3*((y+1)+3*(z+1));
					if(x == 0 && y == 0 && z == 0) continue;
					if(!id_and_bc_overlap(x,y,z,bc.boundary)) continue;
					for(auto& field : bc.out) 
					{
						res[(int)field][index] = bc;
						res[(int)field][index].id = (int3){x,y,z};
					}
				}

	}
	return res;
}



typedef struct
{
	AcKernel kernel;
	std::function<void(ParamLoadingInfo step_info)> loader;
} KernelCall;

static AcTaskDefinition
gen_taskgraph_kernel_entry(const KernelCall call, int onion_level, FILE* stream);

void
check_field_boundconds(const FieldBCs field_boundconds, const std::vector<Field> fields)
{
	for(const auto& field : fields)
	{
		if(!vtxbuf_is_communicated[field]) continue;
		for(int x = -1; x <= 1; ++x)
			for(int y = -1; y <= 1; ++y)
				for(int z = -1; z <= 1; ++z)
				{
					const int index = (x+1) + 3*((y+1)+3*(z+1));
					if(x == 0 && y == 0 && z == 0) continue;
					if(field_boundconds[field][index].kernel  == AC_NULL_KERNEL)
						fatal("FATAL AC ERROR: Missing boundcond for field %s at(%d,%d,%d)\n",field_names[field],x,y,z)
				}
	}
}
void
log_halo_types(
		const std::vector<facet_class_range> halo_types,
		FILE* stream
	      )
{
		if(!ac_pid()) fprintf(stream,",{");
		for(const auto& halo_type: halo_types)
			if(!ac_pid()) fprintf(stream, "(%d,%d),",halo_type.min,halo_type.max);
		if(!ac_pid()) fprintf(stream,"}");
}
std::vector<AcTaskDefinition>
gen_halo_exchange(
		const std::vector<Field>& output_fields,
		const FieldBCs field_boundconds,
		const int3 direction,
		const AcBoundary boundary,
		const std::vector<facet_class_range> halo_types,
		const bool before_kernel_call,
		FILE* stream
		)
{
		(void)field_boundconds;
		std::vector<AcTaskDefinition> res{};
		if(direction == (int3){0,0,0} && !before_kernel_call) return res;
		if(output_fields.size() == 0) return res;
		auto log_fields = [&](const auto& input_fields)
		{
			if(!ac_pid()) fprintf(stream,"{");
			for(const auto& field : input_fields)
				if(!ac_pid()) fprintf(stream, "%s,",field_names[field]);
			if(!ac_pid()) fprintf(stream,"}");
		};

		if(!ac_pid()) fprintf(stream, "Halo(");
		log_fields(output_fields);

		log_launch_bounds(stream,output_fields,output_fields);
		if(direction != (int3){0,0,0})
		{
			if(!ac_pid()) fprintf(stream,",Ray_direction(%d,%d,%d)",direction.x,direction.y,direction.z);
		}
		const std::tuple<Volume,Volume> bounds = get_launch_bounds_from_fields(output_fields,output_fields);
		const Volume start = std::get<0>(bounds);
		const Volume end = std::get<0>(bounds);
		const bool sending   = (direction == (int3){0,0,0} || !before_kernel_call); 
		const bool receiving = (direction == (int3){0,0,0} || before_kernel_call); 
		if(!ac_pid()) fprintf(stream,",sending = %d,receiving = %d",sending,receiving);
		if(!ac_pid()) fprintf(stream, ",%s",ac_boundary_to_str(boundary));
		log_halo_types(halo_types,stream);
		if(!ac_pid()) fprintf(stream, ")\n");
		res.push_back(acHaloExchange(output_fields,start,end,direction,sending,receiving,boundary,halo_types));
		return res;
}
std::vector<AcTaskDefinition>
gen_periodic_bcs(
		const std::vector<Field>& output_fields,
		const FieldBCs field_boundconds,
		const int3 direction,
		const AcBoundary boundary,
		const std::vector<facet_class_range> halo_types,
		const bool before_kernel_call,
		FILE* stream
		)
{
		const auto info = get_info();
		std::vector<AcTaskDefinition> res{};
		if(direction == (int3){0,0,0} && !before_kernel_call) return res;
		if(output_fields.size() == 0) return res;
		auto log_fields = [&](const auto& input_fields)
		{
			if(!ac_pid()) fprintf(stream,"{");
			for(const auto& field : input_fields)
				if(!ac_pid()) fprintf(stream, "%s,",field_names[field]);
			if(!ac_pid()) fprintf(stream,"}");
		};

		const std::tuple<Volume,Volume> bounds = get_launch_bounds_from_fields(output_fields,output_fields);
		const Volume start = std::get<0>(bounds);
		const Volume end = std::get<0>(bounds);
		const Field one_communicated_field = output_fields[0];
		const auto x_boundcond = !info[AC_dimension_inactive].x ? field_boundconds[one_communicated_field][0] : (BoundCond){};
		const auto y_boundcond = !info[AC_dimension_inactive].y ? field_boundconds[one_communicated_field][2] : (BoundCond){};
		const auto z_boundcond = !info[AC_dimension_inactive].z ? field_boundconds[one_communicated_field][4] : (BoundCond){};
		
		const bool x_periodic = ((boundary & BOUNDARY_X) != 0) && !info[AC_dimension_inactive].x && x_boundcond.kernel == BOUNDCOND_PERIODIC;
		const bool y_periodic = ((boundary & BOUNDARY_Y) != 0) && !info[AC_dimension_inactive].y && y_boundcond.kernel == BOUNDCOND_PERIODIC;
		const bool z_periodic = ((boundary & BOUNDARY_Z) != 0) && !info[AC_dimension_inactive].z && z_boundcond.kernel == BOUNDCOND_PERIODIC;

		//TP: for some reason specifying periodic bc as a single task gives better perf as of 8.1.2025
		const bool all_periodic = x_periodic && y_periodic && z_periodic;
		std::vector<facet_class_range> oned_halo_types{};
		for(size_t i = 0; i < halo_types.size(); ++i) oned_halo_types.push_back((facet_class_range){1,1});
		std::vector<facet_class_range> twod_halo_types{};
		for(size_t i = 0; i < halo_types.size(); ++i) twod_halo_types.push_back((facet_class_range){1,min(2,halo_types[i].max)});

		if(all_periodic)
		{
				res.push_back(acBoundaryCondition(BOUNDARY_XYZ,BOUNDCOND_PERIODIC,output_fields,start,end,halo_types));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_XYZ,");
				log_fields(output_fields);
				log_launch_bounds(stream,output_fields,output_fields);
				log_halo_types(halo_types,stream);
				if(!ac_pid()) fprintf(stream,")\n");
		}
		else if(x_periodic && y_periodic)
		{
				res.push_back(acBoundaryCondition(BOUNDARY_XY,BOUNDCOND_PERIODIC,output_fields,start,end,twod_halo_types));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_XY,");
				log_fields(output_fields);
				log_launch_bounds(stream,output_fields,output_fields);
				log_halo_types(twod_halo_types,stream);
				if(!ac_pid()) fprintf(stream,")\n");
		}
		else if(x_periodic && z_periodic)
		{
				res.push_back(acBoundaryCondition(BOUNDARY_XZ,BOUNDCOND_PERIODIC,output_fields,start,end,twod_halo_types));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_XZ,");
				log_fields(output_fields);
				log_launch_bounds(stream,output_fields,output_fields);
				log_halo_types(twod_halo_types,stream);
				if(!ac_pid()) fprintf(stream,")\n");
		}
		else if(y_periodic && z_periodic)
		{
				res.push_back(acBoundaryCondition(BOUNDARY_YZ,BOUNDCOND_PERIODIC,output_fields,start,end,twod_halo_types));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_YZ,");
				log_fields(output_fields);
				log_launch_bounds(stream,output_fields,output_fields);
				log_halo_types(twod_halo_types,stream);
				if(!ac_pid()) fprintf(stream,")\n");
		}
		else if(x_periodic)
		{
			res.push_back(acBoundaryCondition(BOUNDARY_X,BOUNDCOND_PERIODIC,output_fields,start,end,oned_halo_types));
			if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_X,");
			log_fields(output_fields);
			log_launch_bounds(stream,output_fields,output_fields);
			log_halo_types(oned_halo_types,stream);
			if(!ac_pid()) fprintf(stream,")\n");
		}
		else if(y_periodic)
		{
			res.push_back(acBoundaryCondition(BOUNDARY_Y,BOUNDCOND_PERIODIC,output_fields,start,end,oned_halo_types));
			if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_Y,");
			log_fields(output_fields);
			log_launch_bounds(stream,output_fields,output_fields);
			log_halo_types(oned_halo_types,stream);
			if(!ac_pid()) fprintf(stream,")\n");
		}
		else if(z_periodic)
		{
			res.push_back(acBoundaryCondition(BOUNDARY_Z,BOUNDCOND_PERIODIC,output_fields,start,end,oned_halo_types));
			if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_Z,");
			log_fields(output_fields);
			log_launch_bounds(stream,output_fields,output_fields);
			log_halo_types(oned_halo_types,stream);
			if(!ac_pid()) fprintf(stream,")\n");
		}
		return res;
}

int
id_to_arr_index(const int x,const int y,const int z) {return (x+1)+3*((y+1)+3*(z+1));};
std::vector<AcTaskDefinition>
gen_halo_exchange_and_boundconds(
		const std::vector<Field>& fields,
		const std::vector<Field>& communicated_fields,
		const std::array<std::array<bool,NUM_FIELDS>,27>& communicated_regions,
		const std::array<facet_class_range,NUM_FIELDS>& halo_types,
		const FieldBCs field_boundconds,
		const std::array<std::vector<int3>,NUM_FIELDS> field_ray_directions,
		const bool before_kernel_call,
		const bool no_communication,
		FILE* stream
		)
{
		//TP: this is because bcs of higher facet class can depend on the bcs of the lower facet classes
		//Take for example symmetric bc. At (-1,1,0) with boundary_normal (1,0,0) the bc would depend on (0,1,0)
		//If that bc was generated first there would be no dependency between the bcs and cause a race condition
		std::array<int,27> correct_region_order{
			id_to_arr_index(0,0,0),

			id_to_arr_index(+1,0,0),
			id_to_arr_index(0,+1,0),
			id_to_arr_index(0,0,+1),
			id_to_arr_index(-1,0,0),
			id_to_arr_index(0,-1,0),
			id_to_arr_index(0,0,-1),

			id_to_arr_index(+1,+1,0),
			id_to_arr_index(+1,-1,0),
			id_to_arr_index(-1,+1,0),
			id_to_arr_index(-1,-1,0),
			id_to_arr_index(+1,0,+1),
			id_to_arr_index(+1,0,-1),
			id_to_arr_index(-1,0,+1),
			id_to_arr_index(-1,0,-1),
			id_to_arr_index(0,+1,+1),
			id_to_arr_index(0,+1,-1),
			id_to_arr_index(0,-1,+1),
			id_to_arr_index(0,-1,-1),

			id_to_arr_index(+1,+1,+1),
			id_to_arr_index(-1,+1,+1),
			id_to_arr_index(+1,-1,+1),
			id_to_arr_index(-1,-1,+1),
			id_to_arr_index(+1,+1,-1),
			id_to_arr_index(-1,+1,-1),
			id_to_arr_index(+1,-1,-1),
			id_to_arr_index(-1,-1,-1),
		};
 
		std::array<int,NUM_FIELDS> communicated_boundaries{};
		for(size_t field = 0; field < NUM_FIELDS; ++field)
		{
			for(int x = -1; x <= 1; ++x)
				for(int y = -1; y <= 1; ++y)
					for(int z = -1; z <= 1; ++z)
					{
						const int index = id_to_arr_index(x,y,z);
						if(!communicated_regions[index][field]) continue;
						if(x == -1) communicated_boundaries[field] |= BOUNDARY_X_BOT;
						if(x == +1) communicated_boundaries[field] |= BOUNDARY_X_TOP;
						if(y == -1) communicated_boundaries[field] |= BOUNDARY_Y_BOT;
						if(y == +1) communicated_boundaries[field] |= BOUNDARY_Y_TOP;
						if(z == -1) communicated_boundaries[field] |= BOUNDARY_Z_BOT;
						if(z == +1) communicated_boundaries[field] |= BOUNDARY_Z_TOP;
					}
		}
		for(const auto& field : fields)
		{
			if(!vtxbuf_is_communicated[field]) fatal("%s","Internal AC bug: gen_halo_exchange_and_boundconds takes in only communicated fields!\n");
		}
		check_field_boundconds(field_boundconds,communicated_fields);
		std::vector<AcTaskDefinition> empty{};
		const std::vector<AcBoundary> boundaries = get_boundaries();
	        const auto info = get_info();
		std::array<std::array<bool,27>,NUM_FIELDS>  field_boundconds_processed{};
		std::array<std::array<bool,27>,NUM_FIELDS>  field_boundconds_dependencies_included{};

		std::vector<Field> output_fields{};
		for(auto& field : fields)
		{
			bool communicated = std::find(communicated_fields.begin(), communicated_fields.end(), field) != communicated_fields.end();
			if(communicated) output_fields.push_back(field);
		}
		if(output_fields.size() == 0 || no_communication) return empty;

		auto pop_same = [&](auto& src, const auto& info_arr, auto& dst)
		{
			dst.push_back(src.front());
			const auto info_var = info_arr[dst[0]];
			src.pop_front();
			auto it = src.begin();
			while (it != src.end()) {
			    if (info_arr[*it] == info_var) {
			        dst.push_back(std::move(*it));
				//Erase returns the next iterator
			        it = src.erase(it);
			    } else {
			        ++it;
			    }
			}
			return dst;
		};
		auto gen_comm_task = [&](const auto& func,auto& res) 
		{
			std::deque<Field> out_fields{};
			std::array<int3,NUM_FIELDS> fields_dims{};
			for(int field = 0; field < NUM_FIELDS; ++field) fields_dims[field] = info[vtxbuf_dims[field]];
			for(auto& field : output_fields) out_fields.push_back(field);
			while(!out_fields.empty())
			{
				std::vector<Field> same_dims_fields{};
				pop_same(out_fields,fields_dims,same_dims_fields);
				for(int x_dir = -1; x_dir <= 1; ++x_dir)
				{
					for(int y_dir = -1; y_dir <= 1; ++y_dir)
					{
						for(int z_dir = -1; z_dir <= 1; ++z_dir)
						{
							const int3 dir = (int3){x_dir,y_dir,z_dir};
							std::deque<Field> same_direction_fields{};
							for(auto& field : same_dims_fields)
							{
								if(std::find(field_ray_directions[field].begin(), field_ray_directions[field].end(),dir) != field_ray_directions[field].end()) 
									same_direction_fields.push_back(field);
							}
							while(!same_direction_fields.empty())
							{
								std::vector<Field> same_boundary_communicated_fields{}; 
								pop_same(same_direction_fields,communicated_boundaries,same_boundary_communicated_fields);
								std::vector<facet_class_range> out_halo_types{};
								for(auto& field: same_boundary_communicated_fields) out_halo_types.push_back(halo_types[field]);
								const std::vector<AcTaskDefinition> tasks = func(same_boundary_communicated_fields,field_boundconds,dir,AcBoundary(communicated_boundaries[same_boundary_communicated_fields[0]]),out_halo_types,before_kernel_call,stream);
								for(auto& elem: tasks) res.push_back(elem);
							}
						}
					}
				}
			}
		};
		std::vector<AcTaskDefinition> halo_exchanges{};
		std::vector<AcTaskDefinition> bc_tasks{};
		gen_comm_task(gen_halo_exchange,halo_exchanges);
		gen_comm_task(gen_periodic_bcs,bc_tasks);
		if(before_kernel_call)
		{
			for(size_t region = 0; region < 27; ++region)
                	        for(const auto& field : fields)
				{
					bool communicated = std::find(communicated_fields.begin(), communicated_fields.end(), field) != communicated_fields.end() && communicated_regions[region][field];
                	                field_boundconds_processed[field][region]  =  !communicated  || field_boundconds[field][region].kernel == BOUNDCOND_PERIODIC;
                	                field_boundconds_dependencies_included[field][region]  =  !communicated  || field_boundconds[field][region].kernel == BOUNDCOND_PERIODIC;
				}


			//TP: it can be that for the next compute steps only A needs to be updated on the boundaries but to update A on the boundaries
			//One has to first update B on the boundary even though B is not actually used in the domain
			std::array<std::vector<Field>,27> field_bcs_called{};

			for(size_t i = 0; i < 27; ++i)
				field_bcs_called[i] = communicated_fields;
			{
                		bool all_are_processed = false;
				bool made_progress = true;
				while(!all_are_processed)
				{
					if(!made_progress) fatal("%s","STUCK IN A INFINITE LOOP!\n");
					made_progress = false;
					for(int j = 1; j < 27; ++j)
					{
						const int index = correct_region_order[j];
						BoundCond processed_boundcond{AC_NULL_KERNEL,BOUNDARY_NONE,{}, {}, {},0,(int3){0,0,0}};
                				for(const auto& field : fields)
                				        processed_boundcond = !field_boundconds_dependencies_included[field][index] ?  field_boundconds[field][index] : processed_boundcond;
                				if(processed_boundcond.kernel == AC_NULL_KERNEL) continue;

						for(const auto& field : processed_boundcond.in)
						{
							if(!processed_boundcond.info.larger_input) continue;
							if(std::find(field_bcs_called[index].begin(), field_bcs_called[index].end(), field) != field_bcs_called[index].end()) continue;
							field_bcs_called[index].push_back(field);
							field_boundconds_dependencies_included[field][index] = false;
							field_boundconds_processed[field][index] = false;
							made_progress = true;
						}
						for(const auto& field : processed_boundcond.out)
						{
							field_boundconds_dependencies_included[field][index] = true;
							made_progress = true;
						}
					}
                		        all_are_processed = true;
                                        for(const auto& field : fields)
					{
						for(int j = 1; j < 27; ++j)
						{
							const int index = correct_region_order[j];
                		       			all_are_processed &= field_boundconds_dependencies_included[field][index];
						}
					}
				}
			}

			{
                		bool all_are_processed = false;
				bool made_progress = true;
                		while(!all_are_processed)
                		{
					if(!made_progress) fatal("%s","STUCK IN A INFINITE LOOP!\n");
					made_progress = false;
					for(int j = 1; j < 27; ++j)
					{
						const int index = correct_region_order[j];
						BoundCond processed_boundcond{AC_NULL_KERNEL,BOUNDARY_NONE,{}, {}, {},0,(int3){0,0,0}};
                		        	for(const auto& field : fields)
                		        	{
							if(!vtxbuf_is_communicated[field]) continue;
							//TP: always insert boundary conditions with the smaller topological order first
							if(processed_boundcond.kernel != AC_NULL_KERNEL && processed_boundcond.topological_order < field_boundconds[field][index].topological_order) continue;
                		        	        processed_boundcond = !field_boundconds_processed[field][index] ?  field_boundconds[field][index] : processed_boundcond;
                		        	}
                		        	if(processed_boundcond.kernel == AC_NULL_KERNEL) continue;

						int new_boundary = 0;
						for(auto field : processed_boundcond.out)
						{
							new_boundary |= (processed_boundcond.boundary & communicated_boundaries[field]);
						}
						processed_boundcond.boundary = AcBoundary(new_boundary);
						std::vector<facet_class_range> facet_classes{};
						for(auto& field: processed_boundcond.out) facet_classes.push_back(halo_types[field]);
						bc_tasks.push_back(gen_bc(stream,processed_boundcond,facet_classes));
						made_progress = true;
						for(const auto& field : processed_boundcond.out)
						{
							field_boundconds_processed[field][index] = true;
						}
					}
                		        all_are_processed = true;
                                        for(const auto& field : fields)
					{
						for(int j = 1; j < 27; ++j)
						{
						  const int index = correct_region_order[j];
                		        	  all_are_processed &= field_boundconds_processed[field][index];
						}
					}
                		}
			}
		}
		std::vector<AcTaskDefinition> res{};
		for(auto task: halo_exchanges) res.push_back(task);
		for(auto task: bc_tasks) res.push_back(task);
		return res;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
void
compute_next_level_set(T1& dst, const T2& kernel_calls, T3& field_written_to,const T4& call_level_set, const T5& info)
{
	std::array<bool,NUM_FIELDS> field_consumed{};
	std::fill(field_consumed.begin(), field_consumed.end(),false);
	std::fill(field_written_to.begin(), field_written_to.end(),false);
	//TP: padded since cray compiler does not like zero sized arrays when debug flags are on
	std::array<bool,NUM_PROFILES+1> profile_consumed{};
	std::fill(profile_consumed.begin(), profile_consumed.end(),false);
	for(size_t i = 0; i < kernel_calls.size(); ++i)
	{
		if(call_level_set[i] == -1)
		{
		  const int kernel_index = (int)kernel_calls[i];
		  bool can_compute = true;
		  for(size_t j = 0; j < NUM_FIELDS; ++j)
		  {
			bool field_accessed = info[kernel_index].read_fields[j] || info[kernel_index].field_has_stencil_op[j];
			for(int ray = 0; ray < NUM_RAYS; ++ray)
				field_accessed |= info[kernel_index].ray_accessed[j][ray];
		  	can_compute &= !(field_consumed[j] && field_accessed);
		  	field_consumed[j] |= info[kernel_index].written_fields[j];
		  }
		  for(int j = 0; j < NUM_PROFILES; ++j)
		  {
			const bool profile_accessed = info[kernel_index].read_profiles[j] || info[kernel_index].profile_has_stencil_op[j];
			can_compute &= !(profile_consumed[j] && profile_accessed);
			profile_consumed[j] |= (info[kernel_index].reduced_profiles[j] || info[kernel_index].written_profiles[j]);
		  }
		  for(size_t j = 0; j < NUM_FIELDS; ++j)
		  	field_written_to[j] |= (can_compute && info[kernel_index].written_fields[j]);
		  if(can_compute) dst[i] = true;
		}
	}
};

typedef struct
{
	std::vector<KernelCall> calls;
	std::vector<Field> fields_communicated_before;
	std::array<std::array<bool,NUM_FIELDS>,27> communicated_regions;
	std::array<facet_class_range,NUM_FIELDS> halo_types;
} level_set;


// Combine hash helper function
template <typename T>
void hash_combine(std::size_t &seed, const T &value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash for a single vector
struct VectorHash {
    template <typename T>
    std::size_t operator()(const std::vector<T> &vec) const {
        std::size_t seed = 0;
        for (const auto &elem : vec) {
            hash_combine(seed, elem);
        }
        return seed;
    }
};
using KeyType = std::tuple<std::vector<AcKernel>,std::vector<AcKernel>,Volume,Volume>;
struct KeyHash {
    std::size_t operator()(const KeyType &key) const {
        const auto &[vec1, vec2,start,end] = key;
        std::size_t seed = 0;
        hash_combine(seed, VectorHash{}(vec1));
        hash_combine(seed, VectorHash{}(vec2));
        hash_combine(seed, start.x);
        hash_combine(seed, start.y);
        hash_combine(seed, start.z);
        hash_combine(seed, end.x);
        hash_combine(seed, end.y);
        hash_combine(seed, end.z);
        return seed;
    }
};

// Equality comparator for the tuple of vectors
struct KeyEqual {
    bool operator()(const KeyType &lhs,
                    const KeyType &rhs) const {
        return lhs == rhs; 
    }
};


std::vector<level_set>
gen_level_sets(const AcDSLTaskGraph graph, const bool optimized)
{
	auto kernel_calls = optimized ?
				get_optimized_kernels(graph,true) :
				DSLTaskGraphKernels[graph];
	auto kernel_call_computes_profile_across_halos = compute_kernel_call_computes_profile_across_halos(kernel_calls);
	const auto info = get_kernel_analysis_info();
	constexpr int MAX_TASKS = 100;
	int n_level_sets = 0;
	std::array<int,MAX_TASKS> call_level_set{};
	std::array<std::array<std::array<bool, NUM_VTXBUF_HANDLES>,27>,MAX_TASKS> field_needs_to_be_communicated_before_level_set{};
	std::array<std::array<int, MAX_TASKS>,NUM_VTXBUF_HANDLES> halo_types_level_set{};

	std::fill(call_level_set.begin(),call_level_set.end(),-1);
	
	bool all_processed = false;


	std::array<bool, NUM_FIELDS> field_out_from_last_level_set{};
	std::array<bool, NUM_FIELDS> field_out_from_level_set{};
	std::array<bool, NUM_KERNELS> next_level_set{};

	std::array<std::array<bool, NUM_FIELDS>,27> field_need_halo_to_be_in_sync{};
	std::array<std::array<bool, NUM_FIELDS>,27> field_halo_in_sync{};
	std::array<std::array<bool, NUM_VTXBUF_HANDLES>,27> field_need_to_communicate{};

	while(!all_processed)
	{
		for(size_t i = 0; i < 27; ++i)
		{
			std::fill(field_need_halo_to_be_in_sync[i].begin(),field_need_halo_to_be_in_sync[i].end(), false);
			std::fill(field_need_to_communicate[i].begin(),field_need_to_communicate[i].end(), false);
		}
		std::fill(next_level_set.begin(),next_level_set.end(), false);
		compute_next_level_set(next_level_set, kernel_calls,field_out_from_level_set,call_level_set,info);
		bool none_advanced = true;
		for(size_t i = 0; i  < next_level_set.size(); ++i)
			none_advanced &= !next_level_set[i];
		if(none_advanced)
			fatal("Bug in acGetDSLTaskGraphOps. Aborting: None advanced in level set %d\n",n_level_sets)
		for(size_t i = 0; i  < kernel_calls.size(); ++i)
		{
			if(next_level_set[i])
			{
				bool computes_across_halos = false;
				for(int j = 0; j< NUM_PROFILES; ++j)
					computes_across_halos  |= (kernel_call_computes_profile_across_halos[i][j] != BOUNDARY_NONE);

				call_level_set[i] = n_level_sets;
				const int k = (int)kernel_calls[i];
				for(size_t j = 0; j < NUM_FIELDS ; ++j)
				{
					for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
					{
						if(info[k].stencils_accessed[j][stencil])
						{
							for(int x = -1; x <= 1; ++x)
								for(int y = -1; y <= 1; ++y)
									for(int z = -1; z <= 1; ++z)
									{
										const int index = id_to_arr_index(x,y,z);
										const AcBoundary boundary = get_stencil_boundaries(Stencil(stencil));
										if(x == 0 && y == 0 && z == 0) continue;
										if(x == -1 && (boundary & BOUNDARY_X_BOT) == 0) continue;
										if(x == +1 && (boundary & BOUNDARY_X_TOP) == 0) continue;
										if(y == -1 && (boundary & BOUNDARY_Y_BOT) == 0) continue;
										if(y == +1 && (boundary & BOUNDARY_Y_TOP) == 0) continue;
										if(z == -1 && (boundary & BOUNDARY_Z_BOT) == 0) continue;
										if(z == +1 && (boundary & BOUNDARY_Z_TOP) == 0) continue;
										field_need_halo_to_be_in_sync[index][j] |= true;
									}
							halo_types_level_set[j][n_level_sets] = max(halo_types_level_set[j][n_level_sets],get_stencil_halo_type(Stencil(stencil)));
						}
					}
					if(info[k].read_fields[j] && computes_across_halos)
					{
						for(int x = -1; x <= 1; ++x)
							for(int y = -1; y <= 1; ++y)
								for(int z = -1; z <= 1; ++z)
									{
										const int index = id_to_arr_index(x,y,z);
										if(x == 0 && y ==  0 && z == 0) continue;
										field_need_halo_to_be_in_sync[index][j] |= true;
									}
					}
					
					for(int ray = 0; ray < NUM_RAYS; ++ray)
					{
						if(info[k].ray_accessed[j][ray])
						{
							for(int x = -1; x <= 1; ++x)
								for(int y = -1; y <= 1; ++y)
									for(int z = -1; z <= 1; ++z)
									{
										if(x == 0 && y == 0 && z == 0) continue;
										const int index = id_to_arr_index(x,y,z);
										const AcBoundary boundary = get_ray_boundaries(AcRay(ray));
										if(x == -1 && (boundary & BOUNDARY_X_BOT) == 0) continue;
										if(x == +1 && (boundary & BOUNDARY_X_TOP) == 0) continue;
										if(y == -1 && (boundary & BOUNDARY_Y_BOT) == 0) continue;
										if(y == +1 && (boundary & BOUNDARY_Y_TOP) == 0) continue;
										if(z == -1 && (boundary & BOUNDARY_Z_BOT) == 0) continue;
										if(z == +1 && (boundary & BOUNDARY_Z_TOP) == 0) continue;
										field_need_halo_to_be_in_sync[index][j] |= true;
									}
						}
					}
				}
			}
		}
		for(size_t j = 0; j < NUM_FIELDS; ++j)
		{
		    if(field_out_from_last_level_set[j])
		    {
			    for(size_t region = 0; region < 27; ++region)
			    	field_halo_in_sync[region][j] = 0;
		    }
		}
		for(size_t j = 0; j < NUM_FIELDS; ++j)
		{
		    for(size_t region = 0; region < 27; ++region)
		    {
		    	field_need_to_communicate[region][j] |= (!field_halo_in_sync[region][j] && field_need_halo_to_be_in_sync[region][j]);
		    }
		}
		for(size_t j = 0; j < NUM_FIELDS; ++j)
		{
		    for(size_t region = 0; region < 27; ++region)
		    {
			field_halo_in_sync[region][j] |= field_need_to_communicate[region][j];
			field_needs_to_be_communicated_before_level_set[n_level_sets] = field_need_to_communicate;
		    }

		}
		auto swap_tmp = field_out_from_level_set;
		field_out_from_level_set = field_out_from_last_level_set;
		field_out_from_last_level_set = swap_tmp;
		++n_level_sets;
		all_processed = true;
		for(size_t k = 0; k < kernel_calls.size(); ++k)
			all_processed &= (call_level_set[k] != -1);
		if((size_t) n_level_sets > kernel_calls.size())
			fatal("Bug in acGetDSLTaskGraphOps. Aborting: %d\n",n_level_sets)
	}
	
	std::vector<level_set> level_sets{};
	for(int level_set_index = 0; level_set_index < n_level_sets; ++level_set_index)
	{
		std::vector<KernelCall> level_set_calls{};
		for(size_t call = 0; call < kernel_calls.size(); ++call) 
		{
			if(call_level_set[call] == level_set_index)
			{
				level_set_calls.push_back((KernelCall){kernel_calls[call], get_loader(graph,call)});
			}

		}

		std::vector<Field> tmp{};
		std::array<std::array<bool,NUM_FIELDS>,27> regions{};
		std::array<facet_class_range,NUM_FIELDS> halo_types{};
		for(int x = -1; x <= 1; ++x)
		{
			for(int y = -1; y <= 1; ++y)
				for(int z = -1; z <= 1; ++z)
				{
					for(size_t i = 0; i < NUM_FIELDS; ++i)
					{
						Field field = static_cast<Field>(i);
						const int index = id_to_arr_index(x,y,z);
						regions[index][i] = field_needs_to_be_communicated_before_level_set[level_set_index][index][i];
						if(regions[index][i] && std::find(tmp.begin(), tmp.end(),field) == tmp.end())
						{
							tmp.push_back(field);
						}
					}
				}
		}

		for(size_t i = 0; i < NUM_FIELDS; ++i) halo_types[i] = (facet_class_range){1,halo_types_level_set[i][level_set_index]};

		level_sets.push_back((level_set){level_set_calls,tmp,regions,halo_types});
	}
	return level_sets;
}
bool
level_set_has_overlap_in_input_and_output(const level_set& set)
{
	std::vector<Field>   in_fields{};
	std::vector<Field>   out_fields{};
	std::vector<Profile> in_profiles{};
	std::vector<Profile> out_profiles{};
	for(const auto& call : set.calls)
	{
		auto fields   = get_kernel_fields(call.kernel);
		auto profiles = get_kernel_profiles(call.kernel);

		for(const auto& field : fields.in)  
		{
			if(std::find(out_fields.begin(), out_fields.end(),field) != out_fields.end())
				return true;
		}

		for(const auto& profile: profiles.in)  
		{
			if(std::find(out_profiles.begin(), out_profiles.end(),profile) != out_profiles.end())
				return true;
		}

		for(const auto& field : fields.in)     in_fields.push_back(field);
		for(const auto& field : fields.out)    out_fields.push_back(field);
		for(const auto& profile: profiles.in)  in_profiles.push_back(profile);
		for(const auto& profile: profiles.reduce_out) out_profiles.push_back(profile);
		for(const auto& profile: profiles.write_out)  out_profiles.push_back(profile);
	}
	return false;
}
int
get_fused_kernel(const AcKernel a, const AcKernel b)
{
	const std::string a_name {kernel_names[a]};
	const std::string b_name {kernel_names[b]};
	const std::string fused_name = a_name + "_FUSED_" + b_name;
	for(int kernel = 0; kernel < NUM_KERNELS; ++kernel)
	{
		const std::string tmp_name{kernel_names[kernel]};
		if(tmp_name == fused_name) return kernel;
	
	}
	return -1;
}

std::vector<level_set>
fuse_calls_in_level_sets(const std::vector<level_set>& base_level_sets)
{
	std::vector<level_set> res{};
	for(const auto& set : base_level_sets)
	{
		level_set tmp = set;
		bool fused = true;
		while(fused)
		{
			fused = false;
			for(size_t i = 0; i < tmp.calls.size(); ++i)
				for(size_t j = 0; j < tmp.calls.size(); ++j)
				{
					if(fused) continue;
					const int fused_kernel = get_fused_kernel(tmp.calls[i].kernel,tmp.calls[j].kernel);
					if(fused_kernel != -1)
					{
						tmp.calls.erase(tmp.calls.begin() + max(i,j));
						tmp.calls.erase(tmp.calls.begin() + min(i,j));

						const std::function<void(ParamLoadingInfo step_info)> empty_loader = [](const ParamLoadingInfo){};
						tmp.calls.push_back((KernelCall){(AcKernel)fused_kernel,empty_loader});
					}
				}
		}
		res.push_back(tmp);
	}
	return res;
}
bool
fuse_calls_between_level_sets(std::vector<level_set>& level_sets)
{
	for(size_t i = 0; i < level_sets.size(); ++i)
		for(size_t j = i+1; j < level_sets.size(); ++j)
		{
			auto& a_level = level_sets[i];
			auto& b_level = level_sets[j];
			//TP: for now take the conservative approximation that it is safe to move calls to previous level set
			//if in the current level set nothing is also an input and output
			//TODO: one could be more precise that a kernel call is safe to move between level sets if no outputs of it are inputs to other calls in the same level set
			if(level_set_has_overlap_in_input_and_output(b_level)) continue;
			for(size_t a_index = 0; a_index < a_level.calls.size(); ++a_index)
				for(size_t b_index = 0; b_index < b_level.calls.size(); ++b_index)
				{
					const auto a_kernel = a_level.calls[a_index].kernel;
					const auto b_kernel = b_level.calls[b_index].kernel;
					const int fused_kernel = get_fused_kernel(a_kernel,b_kernel);
					if(fused_kernel != -1)
					{
						b_level.calls.erase(b_level.calls.begin() + b_index);	
						const std::function<void(ParamLoadingInfo step_info)> empty_loader = [](const ParamLoadingInfo){};
						a_level.calls[a_index] = ((KernelCall){(AcKernel)fused_kernel,empty_loader});
						return true;
					}
				}
		}
	return false;
}

std::vector<level_set>
get_level_sets(const AcDSLTaskGraph graph, const bool optimized)
{
	auto level_sets = gen_level_sets(graph,optimized);
	bool fused_call_between_level_sets = true;
	while(fused_call_between_level_sets)
	{
		fused_call_between_level_sets = false;
		level_sets = fuse_calls_in_level_sets(level_sets);
		fused_call_between_level_sets = fuse_calls_between_level_sets(level_sets);
	}
	return level_sets;
}
std::array<std::vector<int3>,NUM_FIELDS>
get_field_ray_directions(const std::vector<AcKernel> kernels)
{

	std::array<std::vector<int3>,NUM_FIELDS> field_ray_directions{};
	const auto info = get_kernel_analysis_info();
	for(auto& kernel : kernels)
	{
		for(int field = 0; field < NUM_FIELDS; ++field)
		{
			for(int ray = 0; ray < NUM_RAYS; ++ray)
			{
				if(info[kernel].ray_accessed[field][ray])
				{
					field_ray_directions[field].push_back(ray_directions[ray]);
				}
			}
		}
	}
	for(int field = 0; field < NUM_FIELDS; ++field)
	{
		if(field_ray_directions[field].size() == 0) field_ray_directions[field].push_back((int3){0,0,0});
	}

	return field_ray_directions;
}


std::vector<AcTaskDefinition>
acGetDSLTaskGraphOps(const AcDSLTaskGraph graph, const bool optimized, const bool no_communication)
{
	if(is_bc_taskgraph(graph))
		return acGetDSLBCTaskGraphOps(graph,optimized);
	const auto info = get_kernel_analysis_info();
	const FieldBCs  field_boundconds = get_field_boundconds(DSLTaskGraphBCs[graph],optimized);
	std::vector<AcTaskDefinition> res{};
	auto level_sets = get_level_sets(graph,optimized);	

	FILE* stream = !ac_pid() ? fopen("taskgraph_log.txt","a") : NULL;
	if (!ac_pid()) fprintf(stream,"%s Ops:\n",taskgraph_names[graph]);
	std::array<bool,NUM_FIELDS> field_written_out_before{};
	auto kernel_calls = optimized ?
				get_optimized_kernels(graph,true) :
				DSLTaskGraphKernels[graph];
	const auto field_ray_directions = get_field_ray_directions(kernel_calls);
	for(size_t current_level_set_index = 0; current_level_set_index < level_sets.size(); ++current_level_set_index)
	{
		const auto& current_level_set = level_sets[current_level_set_index];
		std::vector<Field> communicated_fields_not_written_to{};
		std::vector<Field> communicated_fields_written_to{};
		for(size_t i = 0; i < NUM_FIELDS; ++i)
		{
			if(!vtxbuf_is_communicated[i]) continue;
			Field field = static_cast<Field>(i);
			if(field_written_out_before[i])
				communicated_fields_written_to.push_back(field);
			else
				communicated_fields_not_written_to.push_back(field);
		}
		auto ops = gen_halo_exchange_and_boundconds(
		  communicated_fields_written_to,
		  current_level_set.fields_communicated_before,
		  current_level_set.communicated_regions,
		  current_level_set.halo_types,
		  field_boundconds,
		  field_ray_directions,
		  true,
		  no_communication,
		  stream
		);
		for(const auto& op : ops) res.push_back(op);
		ops = gen_halo_exchange_and_boundconds(
		  communicated_fields_not_written_to,
		  current_level_set.fields_communicated_before,
		  current_level_set.communicated_regions,
		  current_level_set.halo_types,
		  field_boundconds,
		  field_ray_directions,
		  true,
		  no_communication,
		  stream
		);
		for(const auto& op : ops) res.push_back(op);

		bool level_set_has_fixed_boundary = true;
		for(auto& call : current_level_set.calls)
			level_set_has_fixed_boundary &= kernel_has_fixed_boundary[call.kernel];
		if(level_set_has_fixed_boundary)
		{
			for(auto& call : current_level_set.calls)
				if(!kernel_has_fixed_boundary[call.kernel]) fatal("%s\n", "TODO: kernels that have fixed boundaries should not be in the same level set as those that do not have\n");
		}

		std::vector<Field> input_fields_not_communicated{};
		for(auto& call : current_level_set.calls)
		{
			auto fields = get_kernel_fields(call.kernel);
			for(auto& field : fields.in)
			{
				if(std::find(current_level_set.fields_communicated_before.begin(), current_level_set.fields_communicated_before.end(), field) == current_level_set.fields_communicated_before.end())
					input_fields_not_communicated.push_back(field);
			}

		}
		if(!level_set_has_fixed_boundary)
		{
	                const std::vector<AcBoundary> boundaries = get_boundaries();
			for(auto& field : input_fields_not_communicated)
			{
                        	for(size_t boundcond = 0; boundcond < boundaries.size(); ++boundcond)
                        	{
					if(field_boundconds[field][boundcond].info.larger_output)
					{
						std::vector<facet_class_range> halo_types{};
						for(size_t j = 0; j < field_boundconds[field][boundcond].out.size(); ++j) halo_types.push_back((facet_class_range){1,1});
						res.push_back(gen_bc(stream,field_boundconds[field][boundcond],halo_types));
					}

                        	}
			}
		}
		for(auto& call : current_level_set.calls)
		{
			if(call.kernel == AC_NULL_KERNEL) continue;
			res.push_back(gen_taskgraph_kernel_entry(call,current_level_set_index+1,stream));
			for(size_t field = 0; field < NUM_FIELDS; ++field)
				field_written_out_before[field] |= info[call.kernel].written_fields[field];

		}

		ops = gen_halo_exchange_and_boundconds(
		  communicated_fields_written_to,
		  current_level_set.fields_communicated_before,
		  current_level_set.communicated_regions,
		  current_level_set.halo_types,
		  field_boundconds,
		  field_ray_directions,
		  false,
		  no_communication,
		  stream
		);
		for(const auto& op : ops) res.push_back(op);
		ops = gen_halo_exchange_and_boundconds(
		  communicated_fields_not_written_to,
		  current_level_set.fields_communicated_before,
		  current_level_set.communicated_regions,
		  current_level_set.halo_types,
		  field_boundconds,
		  field_ray_directions,
		  false,
		  no_communication,
		  stream
		);
		for(const auto& op : ops) res.push_back(op);
	}
	if (!ac_pid()) fprintf(stream,"\n");
	if (!ac_pid()) fclose(stream);
	return res;
}

static std::unordered_map<KeyType, AcTaskGraph*, KeyHash, KeyEqual> task_graphs{};

AcResult
acGridClearTaskGraphCache()
{
	const std::unordered_map<KeyType, AcTaskGraph*, KeyHash, KeyEqual> empty_graphs{};
	for(auto [key,graph] : task_graphs)
		acGridDestroyTaskGraph(graph);
	task_graphs.clear();
	return AC_SUCCESS;
}

AcTaskGraph*
acGetOptimizedDSLTaskGraphWithBounds(const AcDSLTaskGraph graph, const Volume start, const Volume end, const bool bcs_everywhere)
{
	auto optimized_kernels = get_optimized_kernels(graph,false);
	auto optimized_bcs      = get_optimized_kernels(DSLTaskGraphBCs[graph],false);
	KeyType key = std::make_tuple(optimized_kernels,optimized_bcs,start,end);
	if(task_graphs.find(key) != task_graphs.end())
		return task_graphs[key];

	auto ops = acGetDSLTaskGraphOps(graph,true,bcs_everywhere);
	auto res = acGridBuildTaskGraph(ops,start,end);
	task_graphs[key] = res;
	return res;
}

AcTaskGraph*
acGetOptimizedDSLTaskGraph(const AcDSLTaskGraph graph)
{
	return acGetOptimizedDSLTaskGraphWithBounds(graph,
			to_volume(get_info()[AC_nmin]),
			to_volume(get_info()[AC_nlocal_max]),
			false
			);
}

AcTaskGraph*
acGetDSLTaskGraphWithBounds(const AcDSLTaskGraph graph, const Volume start, const Volume end)
{
	return acGridBuildTaskGraph(acGetDSLTaskGraphOps(graph,false,false),start,end);
}

AcTaskGraph*
acGetDSLTaskGraph(const AcDSLTaskGraph graph)
{
	return acGetDSLTaskGraphWithBounds(graph,
			to_volume(get_info()[AC_nmin]),
			to_volume(get_info()[AC_nlocal_max]));
}
#include "user_constants.h"
static AcTaskDefinition
gen_taskgraph_kernel_entry(const KernelCall call, int onion_level, FILE* stream)
{
	constexpr int max_onion_level = 1;
	onion_level = min(onion_level,max_onion_level);
	auto log = [&](const auto& elems)
	{
		if(!ac_pid()) fprintf(stream,"{");
		for(const auto& elem: elems)
			if(!ac_pid()) fprintf(stream, "%s,",get_name(elem));
		if(!ac_pid()) fprintf(stream,"}");
	};
	auto[fields, profiles, reduce_outputs] = get_kernel_outputs(call.kernel);
	if(!ac_pid()) fprintf(stream,"%s(",kernel_names[call.kernel]);
	log(fields.in);
	if(!ac_pid()) fprintf(stream,",");
	log(fields.out);
	if(!ac_pid()) fprintf(stream,",");
	log(profiles.in);
	if(!ac_pid()) fprintf(stream,",");
	log(profiles.reduce_out);
	if(!ac_pid()) fprintf(stream,",");
	log(profiles.write_out);
	if(!ac_pid()) fprintf(stream,",");

	if(!ac_pid()) fprintf(stream,"{");
	for(auto& e : reduce_outputs.in)
	{
		if(e.type == AC_REAL_TYPE)
			if(!ac_pid()) fprintf(stream, "%s,",get_name((AcRealOutputParam)e.variable));	
		if(e.type == AC_INT_TYPE)
			if(!ac_pid()) fprintf(stream, "%s,",get_name((AcIntOutputParam)e.variable));	
#if AC_DOUBLE_PRECISION
		if(e.type == AC_FLOAT_TYPE)
			if(!ac_pid()) fprintf(stream, "%s,",get_name((AcFloatOutputParam)e.variable));	
#endif
	}
	if(!ac_pid()) fprintf(stream,"}");

	if(!ac_pid()) fprintf(stream,",");

	if(!ac_pid()) fprintf(stream,"{");
	for(auto& e : reduce_outputs.out)
	{
		if(e.type == AC_REAL_TYPE)
			if(!ac_pid()) fprintf(stream, "%s,",get_name((AcRealOutputParam)e.variable));	
		if(e.type == AC_INT_TYPE)
			if(!ac_pid()) fprintf(stream, "%s,",get_name((AcIntOutputParam)e.variable));	
#if AC_DOUBLE_PRECISION
		if(e.type == AC_FLOAT_TYPE)
			if(!ac_pid()) fprintf(stream, "%s,",get_name((AcFloatOutputParam)e.variable));	
#endif
	}
	if(!ac_pid()) fprintf(stream,"}");
	log_launch_bounds(stream,fields.in,fields.out);
	if(!ac_pid())  fprintf(stream,",%d",onion_level);
	if(!ac_pid()) fprintf(stream,")\n");
	const auto [start,end] = get_launch_bounds_from_fields(fields.in,fields.out);
	return acCompute(call.kernel,fields.in,fields.out,profiles.in,profiles.reduce_out,profiles.write_out,reduce_outputs.in,reduce_outputs.out,start,end,onion_level,call.loader);
}
#endif // AC_MPI_ENABLED
