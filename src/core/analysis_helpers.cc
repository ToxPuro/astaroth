#include <stddef.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include "acc_runtime.h"
#include "acreal.h"
#include "astaroth.h"
#include "astaroth_analysis.h"
#include "builtin_enums.h"
#include "errchk.h"
#include "host_datatypes.h"

// clang-format off
#include "user_defines.h"
// clang-format on

AcResult acAnalysisGetKernelInfo(const AcMeshInfo info, KernelAnalysisInfo* dst);
[[maybe_unused]] constexpr int AC_IN_BOUNDS_WRITE      = (1 << 0);
[[maybe_unused]] constexpr int AC_OUT_OF_BOUNDS_WRITE  = (1 << 1);
[[maybe_unused]] constexpr int AC_WRITE_TO_INPUT  = (1 << 2);

std::vector<KernelAnalysisInfo>
get_kernel_analysis_info(const AcMeshInfo info)
{
	std::vector<KernelAnalysisInfo> res(NUM_KERNELS,KernelAnalysisInfo{});
	acAnalysisGetKernelInfo(info,res.data());
	return res;
}

KernelAnalysisInfo
get_kernel_analysis_info(const AcMeshInfo info, const AcKernel kernel)
{
	return acAnalysisGetKernelInfoSingle(info,kernel);
}

KernelAnalysisInfo
get_kernel_analysis_info(const AcMeshInfo info, const AcKernel kernel,const acKernelInputParams input_params)
{
       return acAnalysisGetKernelInfoSingleWithInputParams(info,kernel,input_params);
}

bool
kernel_has_profile_stencil_ops(const KernelAnalysisInfo info)
{
	for(int i = 0; i  < NUM_PROFILES; ++i)
		if(info.profile_has_stencil_op[i]) return true;
	return false;
}

bool
kernel_has_stencil_ops(const KernelAnalysisInfo info)
{
	for(int i = 0; i  < NUM_FIELDS; ++i)
		if(info.field_has_stencil_op[i]) return true;
	for(int i = 0; i  < NUM_PROFILES; ++i)
		if(info.profile_has_stencil_op[i]) return true;
	return false;
}
bool
kernel_updates_vtxbuf(const KernelAnalysisInfo info)
{
	for(int i = 0; i < NUM_FIELDS; ++i)
		if(info.written_fields[i]) return true;
	return false;
}

bool
kernel_writes_to_input(const Field field, const KernelAnalysisInfo info)
{
	return (info.written_fields[field] & AC_WRITE_TO_INPUT) != 0;
}

bool
kernel_writes_to_output(const Field field,const KernelAnalysisInfo info)
{
	return info.written_fields[field] && ((info.written_fields[field] & AC_WRITE_TO_INPUT) == 0);
}

bool
is_raytracing_kernel(const KernelAnalysisInfo info)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(info.ray_accessed[field][ray]) return true;
	}
	return false;
}

AcBool3
raytracing_step_direction(const KernelAnalysisInfo info)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(info.ray_accessed[field][ray])
			{
				if(ray_directions[ray].z != 0) return (AcBool3){false,false,true};
				if(ray_directions[ray].y != 0) return (AcBool3){false,true,false};
				if(ray_directions[ray].x != 0) return (AcBool3){true,false,false};
			}
	}
	return (AcBool3){false,false,false};
}

AcBoundary
get_ray_boundaries(const AcRay ray)
{
	int res = 0;
	if(ray_directions[ray].x == +1) res |= BOUNDARY_X_BOT;
	if(ray_directions[ray].x == -1) res |= BOUNDARY_X_TOP;

	if(ray_directions[ray].y ==  +1) res |= BOUNDARY_Y_BOT;
	if(ray_directions[ray].y ==  -1) res |= BOUNDARY_Y_TOP;

	if(ray_directions[ray].z == +1) res |= BOUNDARY_Z_BOT;
	if(ray_directions[ray].z == -1) res |= BOUNDARY_Z_TOP;
	return AcBoundary(res);
}

AcBoundary
stencil_accesses_boundaries(const AcMeshInfo info, const Stencil stencil)
{
	[[maybe_unused]] auto DCONST = [&](const auto& param)
	{
		return info[param];
	};
        #include "coeffs.h"
	auto stencil_accesses_z_ghost_zone = [&]()
	{
	    bool res = false;
	    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
	      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
	        for (int width = 0; width < STENCIL_WIDTH; ++width) {
		  const bool dmid = (depth == (STENCIL_DEPTH-1)/2);
	          res |= !dmid && (stencils[stencil][depth][height][width] != (AcReal)0.0);
	        }
	      }
	    }
	    return res;
	};

	auto stencil_accesses_y_ghost_zone = [&]()
	{
	  // Check which stencils are invalid for profiles
	  // (computed in a new array to avoid side effects).
	    bool res = false;
	    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
	      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
	        for (int width = 0; width < STENCIL_WIDTH; ++width) {
		  const bool hmid = (height == (STENCIL_HEIGHT-1)/2);
	          res |= !hmid && (stencils[stencil][depth][height][width] != (AcReal)0.0);
	        }
	      }
	    }
	    return res;
	};

	auto stencil_accesses_x_ghost_zone = [&]()
	{
	  // Check which stencils are invalid for profiles
	  // (computed in a new array to avoid side effects).
	    bool res = false;
	    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
	      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
	        for (int width = 0; width < STENCIL_WIDTH; ++width) {
		  const bool wmid = (width == (STENCIL_WIDTH-1)/2);
	          res |= !wmid && (stencils[stencil][depth][height][width] != (AcReal)0.0);
	        }
	      }
	    }
	    return res;
	};

	int res = BOUNDARY_NONE;
	if(stencil_accesses_x_ghost_zone())
		res |= BOUNDARY_X;
	if(stencil_accesses_y_ghost_zone())
		res |= BOUNDARY_Y;
	if(stencil_accesses_z_ghost_zone())
		res |= BOUNDARY_Z;
	return (AcBoundary)res;
}

std::array<int,NUM_FIELDS>
get_fields_kernel_depends_on_boundaries(const AcMeshInfo config, const std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries, const KernelAnalysisInfo info)
{
	std::array<int,NUM_FIELDS> res{};
	for(int j = 0; j < NUM_FIELDS; ++j)
       		res[j] = fields_already_depend_on_boundaries[j];
	for(int j = 0; j < NUM_FIELDS; ++j)
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
			if(info.stencils_accessed[j][stencil])
				res[j] |= stencil_accesses_boundaries(config,Stencil(stencil)); 
	return res;
}

AcBoundary
get_kernel_depends_on_boundaries(const AcMeshInfo config, const std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries, const KernelAnalysisInfo info)
{
	//TP: this is because if kernel A uses stencils kernel B has to wait for A to finish on neighbours to avoid overwriting A's input
	//TP: this is somewhat conservative since if A does not use stencils B has more dependency then needed
	//TP: but I guess if A and B do not have stencils they are anyways so fast that it does not matter that much
	int res = 0;
	for(int j = 0; j < NUM_FIELDS; ++j)
	{
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
		{
			if(info.stencils_accessed[j][stencil])
			{
				res |= stencil_accesses_boundaries(config, Stencil(stencil));
			}
		}
		if(info.written_fields[j])
		{
			res |= fields_already_depend_on_boundaries[j];
		}
		for(int ray = 0; ray < NUM_RAYS; ++ray)
		{
			if(info.ray_accessed[j][ray])
			{
				res |= get_ray_boundaries(AcRay(ray));
			}

		}
	}
	for(int j = 0; j < NUM_PROFILES; ++j)
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
			if(info.stencils_accessed[j+NUM_ALL_FIELDS][stencil])
				res |= stencil_accesses_boundaries(config, Stencil(stencil));


	return AcBoundary(res);

}
std::vector<AcBoundary>
get_kernel_depends_on_boundaries(const AcMeshInfo config, const std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries, const KernelAnalysisInfo* info)
{
	std::vector<AcBoundary> res{};
	for(size_t i = 0; i < NUM_KERNELS; ++i)
		res.push_back(get_kernel_depends_on_boundaries(config, fields_already_depend_on_boundaries,info[i]));
	return res;
}

bool
kernel_reduces_scalar(const KernelAnalysisInfo info)
{
	for(size_t i = 0; i < info.n_reduce_outputs; ++i)
		if(info.reduce_outputs[i].type != AC_PROF_TYPE) return true;
	return false;
}

bool
kernel_reduces_something(const KernelAnalysisInfo info)
{
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info.reduced_profiles[i]) return true;
	if(kernel_reduces_scalar(info)) return true;
	return false;
}


bool
kernel_writes_profile(const AcProfileType prof_type, const KernelAnalysisInfo info)
{
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info.written_profiles[i] && prof_types[i] == prof_type) return true;
	return false;
}

bool
kernel_reduces_only_profiles(const AcProfileType prof_type, const KernelAnalysisInfo info)
{
	if(kernel_reduces_scalar(info)) return false;
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info.reduced_profiles[i] && prof_types[i] != prof_type) return false;
	return true;
}

bool
kernel_does_only_profile_reductions(const AcProfileType prof_type, const KernelAnalysisInfo info)
{
	if(kernel_updates_vtxbuf(info)) return false;
	if(kernel_has_stencil_ops(info)) return false;
	return kernel_reduces_only_profiles(prof_type,info);
}

bool
kernel_calls_ray(const AcRay ray, const KernelAnalysisInfo info)
{
	for(int field = 0; field < NUM_FIELDS; ++field)
	{
		if(info.ray_accessed[field][ray]) return true;
	}
	return false;
}

bool
kernel_uses_rays(const KernelAnalysisInfo info)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		if(kernel_calls_ray(AcRay(ray),info)) return true;
	}
	return false;
}

bool
kernel_only_writes_profile(const AcProfileType prof_type, const KernelAnalysisInfo info)
{
	if(kernel_reduces_something(info)) return false;
	if(kernel_updates_vtxbuf(info))    return false;
	if(kernel_updates_vtxbuf(info))    return false;
	if(kernel_uses_rays(info))    return false;
	const std::array<AcProfileType,9> profile_types =
	{
		PROFILE_X,
		PROFILE_Y,
		PROFILE_Z,

		PROFILE_XY,
		PROFILE_XZ,

		PROFILE_YX,
		PROFILE_YZ,

		PROFILE_ZX,
		PROFILE_ZY,
	};
	for(const auto& type : profile_types)
		if(prof_type != type && kernel_writes_profile(type,info)) return false;
	return true;
}

//TP: padded since cray compiler does not like zero sized arrays when debug flags are on
std::vector<std::array<AcBoundary,NUM_PROFILES+1>>
compute_kernel_call_computes_profile_across_halos_static(const AcMeshInfo config, const std::vector<AcKernel>& calls, const std::vector<KernelAnalysisInfo>& info)
{
	std::vector<std::array<AcBoundary,NUM_PROFILES+1>> res{};
	for(size_t i = 0; i  < calls.size(); ++i)
	{
		std::array<AcBoundary,NUM_PROFILES+1> computes_profile_across_halos{};
		for(int prof = 0; prof < NUM_PROFILES; ++prof) computes_profile_across_halos[prof] = BOUNDARY_NONE;
		res.push_back(computes_profile_across_halos);
	}

	for(size_t i = 0; i  < calls.size(); ++i)
	{
		const int k = calls[i];
		for(int prof = 0; prof < NUM_PROFILES; ++prof)
		{
			for(size_t stencil = 0; stencil < NUM_STENCILS; ++stencil)
			{
				if(info[k].stencils_accessed[NUM_ALL_FIELDS+prof][stencil])
				{
					int defining_call = -1;
					for(int j = i-1; j >= 0; --j)
					{
						if(info[calls[j]].written_profiles[prof] || info[calls[j]].reduced_profiles[prof])
						{
							defining_call = j;
							break;
						}
					}
					if(defining_call != -1)
						res[defining_call][prof] = (AcBoundary) ((int)res[defining_call][prof] | stencil_accesses_boundaries(config, (Stencil)stencil));
				}
			}
		}
	}
	return res;
}

//TP: padded since cray compiler does not like zero sized arrays when debug flags are on
std::vector<std::array<AcBoundary,NUM_PROFILES+1>>
compute_kernel_call_computes_profile_across_halos(const AcMeshInfo config, const std::vector<AcKernel>& calls, const std::vector<KernelAnalysisInfo>& info)
{
	std::vector<std::array<AcBoundary,NUM_PROFILES+1>> res{};
	for(size_t i = 0; i  < calls.size(); ++i)
	{
		std::array<AcBoundary,NUM_PROFILES+1> computes_profile_across_halos{};
		for(int prof = 0; prof < NUM_PROFILES; ++prof) computes_profile_across_halos[prof] = BOUNDARY_NONE;
		res.push_back(computes_profile_across_halos);
	}
	ERRCHK_ALWAYS(calls.size() == info.size());

	for(size_t i = 0; i  < calls.size(); ++i)
	{
		for(int prof = 0; prof < NUM_PROFILES; ++prof)
		{
			for(size_t stencil = 0; stencil < NUM_STENCILS; ++stencil)
			{
				if(info[i].stencils_accessed[NUM_ALL_FIELDS+prof][stencil])
				{
					int defining_call = -1;
					for(int j = i-1; j >= 0; --j)
					{
						if(info[j].written_profiles[prof] || info[j].reduced_profiles[prof])
						{
							defining_call = j;
							break;
						}
					}
					if(defining_call != -1)
						res[defining_call][prof] = (AcBoundary) ((int)res[defining_call][prof] | stencil_accesses_boundaries(config, (Stencil)stencil));
				}
			}
		}
	}
	return res;
}

std::tuple<int3,int3>
get_stencil_dims(const AcMeshInfo info, const Stencil stencil)
{
	[[maybe_unused]] auto DCONST = [&](const auto& param)
	{
		return info[param];
	};
	#include "coeffs.h"
	int3 min_radius = (int3){0,0,0};
	int3 max_radius = (int3){0,0,0};
	for(int x = -NGHOST; x <= NGHOST; ++x)
	{
		for(int y = -NGHOST; y <= NGHOST; ++y)
		{
			for(int z = -NGHOST; z <= NGHOST; ++z)
			{
				if(double(stencils[stencil][x+NGHOST][y+NGHOST][z+NGHOST]) != 0.0)
				{
					min_radius.x = std::min(min_radius.x,x);
					min_radius.y = std::min(min_radius.y,y);
					min_radius.z = std::min(min_radius.z,z);

					max_radius.x = std::max(max_radius.x,x);
					max_radius.y = std::max(max_radius.y,y);
					max_radius.z = std::max(max_radius.z,z);
				}
			}
		}
	}
	return std::tuple<int3,int3>{min_radius,max_radius};
}
int
get_stencil_halo_type(const AcMeshInfo info, const Stencil stencil)
{
	const auto [min,max] = get_stencil_dims(info, stencil);
	const int x = (min.x != 0 || max.x != 0) ? 1 : 0;
	const int y = (min.y != 0 || max.y != 0) ? 1 : 0;
	const int z = (min.z != 0 || max.z != 0) ? 1 : 0;
	return x+y+z;
}

bool
kernel_calls_stencil(const Stencil stencil, const KernelAnalysisInfo info)
{
	for(int field = 0; field < NUM_FIELDS; ++field)
	{
		if(info.stencils_accessed[field][stencil]) return true;
	}
	return false;
}
std::tuple<int3,int3>
get_kernel_radius(const AcMeshInfo config, const KernelAnalysisInfo info)
{
	int3 min_radius = (int3){0,0,0};
	int3 max_radius = (int3){0,0,0};
	for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
	{
		if(kernel_calls_stencil(Stencil(stencil),info))
		{
			const auto [stencil_min_radius,stencil_max_radius] = get_stencil_dims(config, Stencil(stencil));
			min_radius.x = std::min(stencil_min_radius.x,min_radius.x);
			min_radius.y = std::min(stencil_min_radius.y,min_radius.y);
			min_radius.z = std::min(stencil_min_radius.z,min_radius.z);

			max_radius.x = std::max(stencil_max_radius.x,max_radius.x);
			max_radius.y = std::max(stencil_max_radius.y,max_radius.y);
			max_radius.z = std::max(stencil_max_radius.z,max_radius.z);
		}
	}
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		if(kernel_calls_ray(AcRay(ray),info))
		{
			min_radius.x = std::min(-ray_directions[ray].x,min_radius.x);
			min_radius.y = std::min(-ray_directions[ray].y,min_radius.y);
			min_radius.z = std::min(-ray_directions[ray].z,min_radius.z);

			max_radius.x = std::max(ray_directions[ray].x,max_radius.x);
			max_radius.y = std::max(ray_directions[ray].y,max_radius.y);
			max_radius.z = std::max(ray_directions[ray].z,max_radius.z);
		}
	}
	return std::tuple<int3,int3>{min_radius,max_radius};
}
