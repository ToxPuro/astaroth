#if AC_MPI_ENABLED 

#include "astaroth.h"
#include "astaroth_utils.h"
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

KernelAnalysisInfo
get_kernel_analysis_info()
{
	KernelAnalysisInfo res;
	acAnalysisGetKernelInfo(get_info(),&res);
	return res;
}
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
} BoundCond;

typedef std::array<std::array<BoundCond,6>,NUM_VTXBUF_HANDLES> FieldBCs;

typedef struct
{
	std::vector<Field> in;
	std::vector<Field> out;
} KernelFields;

KernelFields
get_kernel_fields(const AcKernel kernel)
{
		const KernelAnalysisInfo info = get_kernel_analysis_info();
		KernelFields res{};
		for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
		{
			if(info.read_fields[kernel][i]    || kernel == BOUNDCOND_PERIODIC)    res.in.push_back((Field)i);
			if(info.written_fields[kernel][i] || kernel == BOUNDCOND_PERIODIC)    res.out.push_back((Field)i);
		}
		return res;
}
typedef struct
{
	std::vector<Profile> in;
	std::vector<Profile> out;
} KernelProfiles;

KernelProfiles
get_kernel_profiles(const AcKernel kernel)
{
		const KernelAnalysisInfo info = get_kernel_analysis_info();
		KernelProfiles res{};
		for(int i = 0; i < NUM_PROFILES; ++i)
		{
			if(info.read_profiles[kernel][i]   )    res.in.push_back((Profile)i);
			if(info.written_profiles[kernel][i] || info.reduced_profiles[kernel][i])    res.out.push_back((Profile)i);
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
const char*
boundary_str(const AcBoundary bc)
{
       return
               bc == BOUNDARY_X_BOT ? "BOUNDARY_X_BOT" :
               bc == BOUNDARY_Y_BOT ? "BOUNDARY_Y_BOT" :
               bc == BOUNDARY_X_TOP ? "BOUNDARY_X_TOP" :
               bc == BOUNDARY_Y_TOP ? "BOUNDARY_Y_TOP" :
               bc == BOUNDARY_Z_BOT ? "BOUNDARY_Z_BOT" :
               bc == BOUNDARY_Z_TOP ? "BOUNDARY_Z_TOP" :
               NULL;
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
bool
bc_output_fields_overlap(const std::vector<BoundCond>& bcs)
{
	std::array<std::vector<Field>,6> fields_written{};
	auto add_to_written_fields = [&](const int i, const Field field)
	{
		const bool already_written = std::find(fields_written[i].begin(), fields_written[i].end(),field) != fields_written[i].end();
		fields_written[i].push_back(field);
		return already_written;
	};
	for(auto& bc : bcs)
	{
		auto fields = get_kernel_fields(bc.kernel);
		for(auto& field: fields.out)
		{
			if(bc.boundary & BOUNDARY_X_TOP)
				if (add_to_written_fields(0,field)) return true;
			if(bc.boundary & BOUNDARY_X_BOT)
				if (add_to_written_fields(1,field)) return true;
			if(bc.boundary & BOUNDARY_Y_TOP)
				if (add_to_written_fields(2,field)) return true;
			if(bc.boundary & BOUNDARY_Y_BOT)
				if (add_to_written_fields(3,field)) return true;
			if(bc.boundary & BOUNDARY_Z_TOP)
				if (add_to_written_fields(4,field)) return true;
			if(bc.boundary & BOUNDARY_Z_BOT)
				if (add_to_written_fields(5,field)) return true;
		}
	}
	return false;
}

std::vector<BoundCond>
get_boundconds(const AcDSLTaskGraph bc_graph)
{

	std::vector<BoundCond> res{};
	const std::vector<AcKernel>   kernels    = DSLTaskGraphKernels[bc_graph];
	const std::vector<AcBoundary> boundaries = DSLTaskGraphKernelBoundaries[bc_graph];
	for(size_t i = 0; i < kernels.size(); ++i)
	std::vector<acAnalysisBCInfo> bc_infos{};
	for(size_t i = 0; i < kernels.size(); ++i)
	{
		auto bc_info = acAnalysisGetBCInfo(get_info(),kernels[i],boundaries[i]);
		auto fields = get_kernel_fields(kernels[i]);
		res.push_back((BoundCond){kernels[i],boundaries[i],fields.in,fields.out,bc_info,0});
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
void
log_boundcond(FILE* stream, const BoundCond bc, const AcBoundary boundary)
{
	auto log_fields = [&](const auto& fields)
	{
		if(!ac_pid()) fprintf(stream,"{");
		for(const auto& field : fields)
			if(!ac_pid()) fprintf(stream, "%s,",field_names[field]);
		if(!ac_pid()) fprintf(stream,"}");
	};
	if(!ac_pid()) fprintf(stream,"BoundCond(%s,%s,",boundary_str(boundary), kernel_names[bc.kernel]);
	log_fields(bc.in);
	if(!ac_pid()) fprintf(stream,",");
	log_fields(bc.out);
	if(!ac_pid()) fprintf(stream,")\n");
}
std::vector<AcTaskDefinition>
acGetDSLBCTaskGraphOps(const AcDSLTaskGraph bc_graph)
{
	FILE* stream = !ac_pid() ? fopen("taskgraph_log.txt","a") : NULL;
	if (!ac_pid()) fprintf(stream,"%s Ops:\n",taskgraph_names[bc_graph]);
	std::vector<AcTaskDefinition> res{};
	auto bcs = get_boundconds(bc_graph);
	//TP: insert boundconds in topological order
	for(size_t i = 0; i < bcs.size(); ++i)
		for(auto& bc : bcs)
			if(bc.topological_order == i)
			{
				log_boundcond(stream,bc,bc.boundary);
				res.push_back(acBoundaryCondition(bc.boundary,bc.kernel,bc.in,bc.out));
			}

	if(!ac_pid()) fclose(stream);
	return res;
}


FieldBCs
get_field_boundconds(const AcDSLTaskGraph bc_graph)
{
	const auto bcs = get_boundconds(bc_graph);
	const std::vector<AcBoundary> boundaries = {BOUNDARY_X_TOP, BOUNDARY_X_BOT, BOUNDARY_Y_TOP, BOUNDARY_Y_BOT, BOUNDARY_Z_TOP, BOUNDARY_Z_BOT};

	FieldBCs res{};
	BoundCond empty_bc = {NUM_KERNELS,BOUNDARY_NONE,{}, {}, {},0};
	for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
		for(int j = 0; j < 6; ++j)
			res[i][j] = empty_bc;

	for(auto& bc : bcs)
	{
		for(size_t i = 0; i < boundaries.size(); ++i) 
			if(bc.boundary & boundaries[i])
				for(auto& field : bc.out) res[(int)field][i] = bc;

	}
	return res;
}


void
check_field_boundconds(const FieldBCs field_boundconds)
{
	const std::vector<AcBoundary> boundaries_to_check  = 
	TWO_D ? (std::vector<AcBoundary>){BOUNDARY_X_TOP, BOUNDARY_X_BOT, BOUNDARY_Y_TOP, BOUNDARY_Y_BOT}
	      : (std::vector<AcBoundary>){BOUNDARY_X_TOP, BOUNDARY_X_BOT, BOUNDARY_Y_TOP, BOUNDARY_Y_BOT, BOUNDARY_Z_TOP, BOUNDARY_Z_BOT};
	for(size_t field = 0; field < NUM_VTXBUF_HANDLES; ++field)
	{
		if(!vtxbuf_is_communicated[field]) continue;
		for(size_t bc = 0; bc < boundaries_to_check.size(); ++bc)
			if(field_boundconds[field][bc].kernel  == NUM_KERNELS)
				fatal("FATAL AC ERROR: Missing boundcond for field %s at boundary %s\n",field_names[field], boundary_str(boundaries_to_check[bc]))
	}
}

typedef struct
{
	AcKernel kernel;
	std::function<void(ParamLoadingInfo step_info)> loader;
} KernelCall;

static AcTaskDefinition
gen_taskgraph_kernel_entry(const KernelCall call, FILE* stream);

std::vector<AcTaskDefinition>
gen_halo_exchange_and_boundconds(
		const std::vector<Field>& fields,
		const std::vector<Field>& communicated_fields,
		const FieldBCs field_boundconds,
		FILE* stream
		)
{
 
		auto log_fields = [&](const auto& input_fields)
		{
			if(!ac_pid()) fprintf(stream,"{");
			for(const auto& field : input_fields)
				if(!ac_pid()) fprintf(stream, "%s,",field_names[field]);
			if(!ac_pid()) fprintf(stream,"}");
		};
		std::vector<AcTaskDefinition> res{};
		constexpr int num_boundaries = TWO_D ? 4 : 6;
		std::vector<AcBoundary> boundaries = TWO_D ? (std::vector<AcBoundary>){BOUNDARY_X_TOP, BOUNDARY_X_BOT, BOUNDARY_Y_TOP, BOUNDARY_Y_BOT} : (std::vector<AcBoundary>){BOUNDARY_X_TOP, BOUNDARY_X_BOT, BOUNDARY_Y_TOP, BOUNDARY_Y_BOT, BOUNDARY_Z_TOP, BOUNDARY_Z_BOT};
		std::array<std::array<bool,num_boundaries>,NUM_ALL_FIELDS>  field_boundconds_processed{};

		std::vector<Field> output_fields{};
		for(auto& field : fields)
		{
			bool communicated = std::find(communicated_fields.begin(), communicated_fields.end(), field) != communicated_fields.end();
			if(communicated) output_fields.push_back(field);
		}
		if(output_fields.size() > 0)
		{

			if(!ac_pid()) fprintf(stream, "Halo(");
			log_fields(output_fields);
			if(!ac_pid()) fprintf(stream, ")\n");
			res.push_back(acHaloExchange(output_fields));
			const Field one_communicated_field = output_fields[0];
			const auto x_boundcond = field_boundconds[one_communicated_field][0];
			const auto y_boundcond = field_boundconds[one_communicated_field][1];
			const auto z_boundcond = num_boundaries > 4 ? field_boundconds[one_communicated_field][4] : (BoundCond){};
			
			const bool x_periodic = x_boundcond.kernel == BOUNDCOND_PERIODIC;
			const bool y_periodic = y_boundcond.kernel == BOUNDCOND_PERIODIC;
			const bool z_periodic = num_boundaries > 4 && z_boundcond.kernel == BOUNDCOND_PERIODIC;

			if(x_periodic)
			{
				res.push_back(acBoundaryCondition(BOUNDARY_X,BOUNDCOND_PERIODIC,output_fields));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_X,");
				log_fields(output_fields);
				if(!ac_pid()) fprintf(stream,")\n");
			}
			if(y_periodic)
			{
				res.push_back(acBoundaryCondition(BOUNDARY_Y,BOUNDCOND_PERIODIC,output_fields));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_Y,");
				log_fields(output_fields);
				if(!ac_pid()) fprintf(stream,")\n");
			}
			if(z_periodic)
			{
				res.push_back(acBoundaryCondition(BOUNDARY_Z,BOUNDCOND_PERIODIC,output_fields));
				if(!ac_pid()) fprintf(stream,"Periodic(BOUNDARY_Z,");
				log_fields(output_fields);
				if(!ac_pid()) fprintf(stream,")\n");
			}

			for(size_t boundcond = 0; boundcond < boundaries.size(); ++boundcond)
                                for(const auto& field : fields)
				{
					bool communicated = std::find(communicated_fields.begin(), communicated_fields.end(), field) != communicated_fields.end();
                                        field_boundconds_processed[field][boundcond]  =  !communicated  || field_boundconds[field][boundcond].kernel == BOUNDCOND_PERIODIC;
				}
                        bool all_are_processed = false;
                        while(!all_are_processed)
                        {
                                for(size_t boundcond = 0; boundcond < boundaries.size(); ++boundcond)
                                {
					BoundCond processed_boundcond{NUM_KERNELS,BOUNDARY_NONE,{}, {}, {},0};
                                	for(const auto& field : fields)
                                        {
						if(!vtxbuf_is_communicated[field]) continue;
						//TP: always insert boundary conditions with the smaller topological order first
						if(processed_boundcond.kernel != NUM_KERNELS && processed_boundcond.topological_order < field_boundconds[field][boundcond].topological_order) continue;
                                                processed_boundcond = !field_boundconds_processed[field][boundcond] ?  field_boundconds[field][boundcond] : processed_boundcond;
                                        }
                                        if(processed_boundcond.kernel == NUM_KERNELS) continue;

					log_boundcond(stream,processed_boundcond,boundaries[boundcond]);
					res.push_back(acBoundaryCondition(boundaries[boundcond],processed_boundcond.kernel,processed_boundcond.in,processed_boundcond.out));
					for(const auto& field : processed_boundcond.out)
						field_boundconds_processed[field][boundcond] = true;
					bool has_dependency = false;
					const char* dependent_bc = NULL;
					const char* preq_field   = NULL;
					for(const auto& field : processed_boundcond.in)
					{
						if(std::find(communicated_fields.begin(), communicated_fields.end(), field) != communicated_fields.end()) continue;
						const bool depend = processed_boundcond.info.larger_input;
						if(depend)
						{
							has_dependency = true;
							dependent_bc = kernel_names[processed_boundcond.kernel];
							preq_field   = field_names[field];
						}
					}

					if(has_dependency)
						fatal("BC of %s needs to be called since %s depends on it\nTODO implement this --- now easier with an actual example use case ---",preq_field,dependent_bc)

                                }
                                all_are_processed = true;
                                for(size_t boundcond = 0; boundcond < boundaries.size(); ++boundcond)
                                	for(const auto& field : fields)
                                                all_are_processed &= field_boundconds_processed[field][boundcond];
                        }
		}
		return res;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
void
compute_next_level_set(T1& dst, const T2& kernel_calls, T3& field_written_to,const T4& call_level_set, const T5& info)
{
	std::array<bool,NUM_ALL_FIELDS> field_consumed{};
	std::fill(field_consumed.begin(), field_consumed.end(),false);
	std::fill(field_written_to.begin(), field_written_to.end(),false);
	std::array<bool,NUM_PROFILES> profile_consumed{};
	std::fill(profile_consumed.begin(), profile_consumed.end(),false);
	for(size_t i = 0; i < kernel_calls.size(); ++i)
	{
		if(call_level_set[i] == -1)
		{
		  const int kernel_index = (int)kernel_calls[i];
		  bool can_compute = true;
		  for(size_t j = 0; j < NUM_ALL_FIELDS; ++j)
		  {
			const bool field_accessed = info.read_fields[kernel_index][j] || info.field_has_stencil_op[kernel_index][j];
		  	can_compute &= !(field_consumed[j] && field_accessed);
		  	field_consumed[j] |= info.written_fields[kernel_index][j];
		  }
		  for(int j = 0; j < NUM_PROFILES; ++j)
		  {
			const bool profile_accessed = info.read_profiles[kernel_index][j] || info.reduced_profiles[kernel_index][j];
			can_compute &= !profile_consumed[j] || !profile_accessed;
			profile_consumed[j] |= info.reduced_profiles[kernel_index][j];
		  }
		  for(size_t j = 0; j < NUM_ALL_FIELDS; ++j)
		  	field_written_to[j] |= (can_compute && info.written_fields[kernel_index][j]);
		  if(can_compute) dst[i] = true;
		}
	}
};

typedef struct
{
	std::vector<KernelCall> calls;
	std::vector<Field> fields_communicated_before;
} level_set;

#include "user_loaders.h"
std::vector<level_set>
gen_level_sets(const AcDSLTaskGraph graph)
{
	auto kernel_calls = DSLTaskGraphKernels[graph];
	const KernelAnalysisInfo info = get_kernel_analysis_info();
	constexpr int MAX_TASKS = 100;
	int n_level_sets = 0;
	std::array<int,MAX_TASKS> call_level_set{};
	std::array<std::array<bool, MAX_TASKS>,NUM_VTXBUF_HANDLES> field_needs_to_be_communicated_before_level_set{};

	std::fill(call_level_set.begin(),call_level_set.end(),-1);
	
        std::array<bool, NUM_VTXBUF_HANDLES> field_need_to_communicate{};
	bool all_processed = false;

	std::array<bool, NUM_ALL_FIELDS> field_halo_in_sync{};
	std::array<bool, NUM_ALL_FIELDS> field_out_from_last_level_set{};
	std::array<bool, NUM_ALL_FIELDS> field_out_from_level_set{};
	std::array<bool, NUM_ALL_FIELDS> field_stencil_ops_at_next_level_set{};
	std::array<bool, NUM_KERNELS> next_level_set{};

	while(!all_processed)
	{
		std::fill(field_stencil_ops_at_next_level_set.begin(),field_stencil_ops_at_next_level_set.end(), false);
		std::fill(field_need_to_communicate.begin(),field_need_to_communicate.end(), false);
		std::fill(next_level_set.begin(),next_level_set.end(), 0);
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
				call_level_set[i] = n_level_sets;
				const int k = (int)kernel_calls[i];
				for(size_t j = 0; j < NUM_ALL_FIELDS ; ++j)
					field_stencil_ops_at_next_level_set[j] |= info.field_has_stencil_op[k][j];
			}
		}
		for(size_t j = 0; j < NUM_ALL_FIELDS; ++j)
		    field_halo_in_sync[j] &= !field_out_from_last_level_set[j];
		for(size_t j = 0; j < NUM_ALL_FIELDS; ++j)
		    field_need_to_communicate[j] |= (!field_halo_in_sync[j] && field_stencil_ops_at_next_level_set[j]);
		for(size_t j = 0; j < NUM_ALL_FIELDS; ++j)
		{
			field_halo_in_sync[j] |= field_need_to_communicate[j];
			field_needs_to_be_communicated_before_level_set[j][n_level_sets] = field_need_to_communicate[j];

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
				level_set_calls.push_back((KernelCall){kernel_calls[call], DSLTaskGraphKernelLoaders[graph][call]});
			}

		}

		std::vector<Field> tmp{};
		for(size_t i = 0; i < NUM_ALL_FIELDS; ++i)
		{
			Field field = static_cast<Field>(i);
			if(field_needs_to_be_communicated_before_level_set[i][level_set_index])
				tmp.push_back(field);
		}
		level_sets.push_back((level_set){level_set_calls,tmp});
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
		for(const auto& profile: profiles.out) out_profiles.push_back(profile);
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
get_level_sets(const AcDSLTaskGraph graph)
{
	auto level_sets = gen_level_sets(graph);
	bool fused_call_between_level_sets = true;
	while(fused_call_between_level_sets)
	{
		fused_call_between_level_sets = false;
		level_sets = fuse_calls_in_level_sets(level_sets);
		fused_call_between_level_sets = fuse_calls_between_level_sets(level_sets);
	}
	return level_sets;
}

std::vector<AcTaskDefinition>
acGetDSLTaskGraphOps(const AcDSLTaskGraph graph)
{
	if(is_bc_taskgraph(graph))
		return acGetDSLBCTaskGraphOps(graph);
	const KernelAnalysisInfo info = get_kernel_analysis_info();
	const FieldBCs  field_boundconds = get_field_boundconds(DSLTaskGraphBCs[graph]);
	check_field_boundconds(field_boundconds);
	std::vector<AcTaskDefinition> res{};
	auto kernel_calls = DSLTaskGraphKernels[graph];

	auto level_sets = get_level_sets(graph);	

	FILE* stream = !ac_pid() ? fopen("taskgraph_log.txt","a") : NULL;
	if (!ac_pid()) fprintf(stream,"%s Ops:\n",taskgraph_names[graph]);
	std::array<bool,NUM_ALL_FIELDS> field_written_out_before{};
	for(const auto& current_level_set : level_sets)
	{
		std::vector<Field> fields_not_written_to{};
		std::vector<Field> fields_written_to{};
		for(size_t i = 0; i < NUM_ALL_FIELDS; ++i)
		{
			Field field = static_cast<Field>(i);
			if(field_written_out_before[i])
				fields_written_to.push_back(field);
			else
				fields_not_written_to.push_back(field);
		}
		auto ops = gen_halo_exchange_and_boundconds(
		  fields_written_to,
		  current_level_set.fields_communicated_before,
		  field_boundconds,
		  stream
		);
		for(const auto& op : ops) res.push_back(op);
		ops = gen_halo_exchange_and_boundconds(
		  fields_not_written_to,
		  current_level_set.fields_communicated_before,
		  field_boundconds,
		  stream
		);
		for(const auto& op : ops) res.push_back(op);

		bool level_set_has_fixed_boundary = true;
		for(auto& call : current_level_set.calls)
			level_set_has_fixed_boundary &= kernel_has_fixed_boundary[call.kernel];

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
			for(auto& field : input_fields_not_communicated)
			{
				bool need_to_call_bc = false;
				need_to_call_bc |= field_boundconds[field][0].info.larger_output;
				need_to_call_bc |= field_boundconds[field][1].info.larger_output;
				need_to_call_bc |= field_boundconds[field][2].info.larger_output;
				need_to_call_bc |= field_boundconds[field][3].info.larger_output;
				need_to_call_bc |= field_boundconds[field][4].info.larger_output;
				need_to_call_bc |= field_boundconds[field][5].info.larger_output;
				if(need_to_call_bc)
					fatal("%s","BC that sets the actual boundary needs to be inserted even though the ghost zones are not needed\nTODO implement this --- now easier with an actual example use case ---")

			}
		}
		else
		{
			for(auto& call : current_level_set.calls)
				if(!kernel_has_fixed_boundary[call.kernel]) fatal("%s\n", "TODO: kernels that have fixed boundaries should not be in the same level set as those that do not have\n");
		}
		for(auto& call : current_level_set.calls)
		{
			res.push_back(gen_taskgraph_kernel_entry(call,stream));
			for(size_t field = 0; field < NUM_ALL_FIELDS; ++field)
				field_written_out_before[field] |= info.written_fields[call.kernel][field];

		}
	}
	if (!ac_pid()) fprintf(stream,"\n");
	if (!ac_pid()) fclose(stream);
	return res;
}

AcTaskGraph*
acGetDSLTaskGraph(const AcDSLTaskGraph graph)
{
	return acGridBuildTaskGraph(
		acGetDSLTaskGraphOps(graph)
			);
}

#include "user_constants.h"
static AcTaskDefinition
gen_taskgraph_kernel_entry(const KernelCall call, FILE* stream)
{

	auto log = [&](const auto& elems)
	{
		if(!ac_pid()) fprintf(stream,"{");
		for(const auto& elem: elems)
			if(!ac_pid()) fprintf(stream, "%s,",get_name(elem));
		if(!ac_pid()) fprintf(stream,"}");
	};
	auto fields = get_kernel_fields(call.kernel);
	auto profiles = get_kernel_profiles(call.kernel);
	if(!ac_pid()) fprintf(stream,"%s(",kernel_names[call.kernel]);
	log(fields.in);
	if(!ac_pid()) fprintf(stream,",");
	log(fields.out);
	if(!ac_pid()) fprintf(stream,",");
	log(profiles.in);
	if(!ac_pid()) fprintf(stream,",");
	log(profiles.out);
	if(!ac_pid()) fprintf(stream,")\n");
	return acCompute(call.kernel,fields.in,fields.out,profiles.in,profiles.out,call.loader);
}
#endif // AC_MPI_ENABLED
