static std::vector<KernelAnalysisInfo>
get_kernel_analysis_info()
{
	std::vector<KernelAnalysisInfo> res(NUM_KERNELS,KernelAnalysisInfo{});
	const auto& info = acDeviceGetLocalConfig(acGridGetDevice());
	acAnalysisGetKernelInfo(info,res.data());
	return res;
}
static UNUSED bool
kernel_has_profile_stencil_ops(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i  < NUM_PROFILES; ++i)
		if(info[kernel].profile_has_stencil_op[i]) return true;
	return false;
}

static bool
kernel_has_stencil_ops(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i  < NUM_FIELDS; ++i)
		if(info[kernel].field_has_stencil_op[i]) return true;
	for(int i = 0; i  < NUM_PROFILES; ++i)
		if(info[kernel].profile_has_stencil_op[i]) return true;
	return false;
}
static bool
kernel_updates_vtxbuf(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_FIELDS; ++i)
		if(info[kernel].written_fields[i]) return true;
	return false;
}

static bool
is_raytracing_kernel(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(info[kernel].ray_accessed[field][ray]) return true;
	}
	return false;
}

static UNUSED AcBool3
raytracing_step_direction(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(info[kernel].ray_accessed[field][ray])
			{
				if(ray_directions[ray].z != 0) return (AcBool3){false,false,true};
				if(ray_directions[ray].y != 0) return (AcBool3){false,true,false};
				if(ray_directions[ray].x != 0) return (AcBool3){true,false,false};
			}
	}
	return (AcBool3){false,false,false};
}


static UNUSED AcBoundary
get_kernel_depends_on_boundaries(const AcKernel kernel, std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries)
{
	//TP: this is because if kernel A uses stencils kernel B has to wait for A to finish on neighbours to avoid overwriting A's input
	//TP: this is somewhat conservative since if A does not use stencils B has more dependency then needed
	//TP: but I guess if A and B do not have stencils they are anyways so fast that it does not matter that much
	if(kernel_updates_vtxbuf(kernel)) return BOUNDARY_XYZ;
	if(is_raytracing_kernel(kernel))  return BOUNDARY_XYZ;
	const auto info = get_kernel_analysis_info();
	int res = 0;
	for(int j = 0; j < NUM_FIELDS; ++j)
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
			if(info[kernel].stencils_accessed[j][stencil])
			{
				res |= acDeviceStencilAccessesBoundaries(acGridGetDevice(), Stencil(stencil));
				fields_already_depend_on_boundaries[j] |= acDeviceStencilAccessesBoundaries(acGridGetDevice(), Stencil(stencil));
			}
	for(int j = 0; j < NUM_PROFILES; ++j)
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
			if(info[kernel].stencils_accessed[j+NUM_ALL_FIELDS][stencil])
				res |= acDeviceStencilAccessesBoundaries(acGridGetDevice(), Stencil(stencil));

	return AcBoundary(res);

}

static UNUSED std::vector<AcBoundary>
get_kernel_depends_on_boundaries(std::array<int,NUM_FIELDS>& fields_already_depend_on_boundaries)
{
	std::vector<AcBoundary> res{};
	for(size_t i = 0; i < NUM_KERNELS; ++i)
		res.push_back(get_kernel_depends_on_boundaries(AcKernel(i), fields_already_depend_on_boundaries));
	return res;
}
static bool
kernel_reduces_scalar(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(size_t i = 0; i < info[kernel].n_reduce_outputs; ++i)
		if(info[kernel].reduce_outputs[i].type != AC_PROF_TYPE) return true;
	return false;
}
static bool
kernel_reduces_something(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info[kernel].reduced_profiles[i]) return true;
	if(kernel_reduces_scalar(kernel)) return true;
	return false;
}


static bool
kernel_writes_profile(const AcKernel kernel, const AcProfileType prof_type)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info[kernel].written_profiles[i] && prof_types[i] == prof_type) return true;
	return false;
}

static bool
kernel_reduces_only_profiles(const AcKernel kernel, const AcProfileType prof_type)
{
	if(kernel_reduces_scalar(kernel)) return false;
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info[kernel].reduced_profiles[i] && prof_types[i] != prof_type) return false;
	return true;
}

static UNUSED bool
kernel_does_only_profile_reductions(const AcKernel kernel, const AcProfileType prof_type)
{
	if(kernel_updates_vtxbuf(kernel)) return false;
	if(kernel_has_stencil_ops(kernel)) return false;
	return kernel_reduces_only_profiles(kernel,prof_type);
}


static UNUSED bool
kernel_only_writes_profile(const AcKernel kernel, const AcProfileType prof_type)
{
	if(kernel_reduces_something(kernel)) return false;
	if(kernel_updates_vtxbuf(kernel))    return false;
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
		if(prof_type != type && kernel_writes_profile(kernel,type)) return false;
	return true;
}

//TP: padded since cray compiler does not like zero sized arrays when debug flags are on
static UNUSED std::vector<std::array<AcBoundary,NUM_PROFILES+1>>
compute_kernel_call_computes_profile_across_halos(const std::vector<AcKernel>& calls)
{
	std::vector<std::array<AcBoundary,NUM_PROFILES+1>> res{};
	for(size_t i = 0; i  < calls.size(); ++i)
	{
		std::array<AcBoundary,NUM_PROFILES+1> computes_profile_across_halos{};
		for(int prof = 0; prof < NUM_PROFILES; ++prof) computes_profile_across_halos[prof] = BOUNDARY_NONE;
		res.push_back(computes_profile_across_halos);
	}

	const auto info = get_kernel_analysis_info();
	for(size_t i = 0; i  < calls.size(); ++i)
	{
		const auto k = calls[i];
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
						res[defining_call][prof] = (AcBoundary) ((int)res[defining_call][prof] | acDeviceStencilAccessesBoundaries(acGridGetDevice(), (Stencil)stencil));
				}
			}
		}
	}
	return res;
}
static std::tuple<int3,int3>
get_stencil_dims(const Stencil stencil)
{
	const auto info = acDeviceGetLocalConfig(acGridGetDevice());
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
				if(stencils[stencil][x+NGHOST][y+NGHOST][z+NGHOST] != 0.0)
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

static bool
kernel_calls_ray(const AcKernel kernel, const AcRay ray)
{
	const auto info = get_kernel_analysis_info();
	for(int field = 0; field < NUM_FIELDS; ++field)
	{
		if(info[kernel].ray_accessed[field][ray]) return true;
	}
	return false;
}
static bool
kernel_calls_stencil(const AcKernel kernel, const Stencil stencil)
{
	const auto info = get_kernel_analysis_info();
	for(int field = 0; field < NUM_FIELDS; ++field)
	{
		if(info[kernel].stencils_accessed[field][stencil]) return true;
	}
	return false;
}
static UNUSED std::tuple<int3,int3>
get_kernel_radius(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	int3 min_radius = (int3){0,0,0};
	int3 max_radius = (int3){0,0,0};
	for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
	{
		if(kernel_calls_stencil(kernel,Stencil(stencil)))
		{
			const auto [stencil_min_radius,stencil_max_radius] = get_stencil_dims(Stencil(stencil));
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
		if(kernel_calls_ray(kernel,AcRay(ray)))
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
