static KernelAnalysisInfo
get_kernel_analysis_info()
{
	KernelAnalysisInfo res;
	const auto& info = acDeviceGetLocalConfig(acGridGetDevice());
	acAnalysisGetKernelInfo(info.params,&res);
	return res;
}

static bool
kernel_has_profile_stencil_ops(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i  < NUM_PROFILES; ++i)
		if(info.profile_has_stencil_op[kernel][i]) return true;
	return false;
}

static bool
kernel_has_stencil_ops(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i  < NUM_FIELDS; ++i)
		if(info.field_has_stencil_op[kernel][i]) return true;
	for(int i = 0; i  < NUM_PROFILES; ++i)
		if(info.profile_has_stencil_op[kernel][i]) return true;
	return false;
}
static bool
kernel_updates_vtxbuf(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_FIELDS; ++i)
		if(info.written_fields[kernel][i]) return true;
	return false;
}

static AcBoundary
get_kernel_depends_on_boundaries(const AcKernel kernel)
{
	//TP: this is because if kernel A uses stencils kernel B has to wait for A to finish on neighbours to avoid overwriting A's input
	//TP: this is somewhat conservative since if A does not use stencils B has more dependency then needed
	//TP: but I guess if A and B do not have stencils they are anyways so fast that it does not matter that much
	if(kernel_updates_vtxbuf(kernel)) return BOUNDARY_XYZ;
	const auto info = get_kernel_analysis_info();
	int res = 0;
	for(int j = 0; j < NUM_FIELDS; ++j)
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
			if(info.stencils_accessed[kernel][j][stencil])
				res |= acDeviceStencilAccessesBoundaries(acGridGetDevice(), Stencil(stencil));
	for(int j = 0; j < NUM_PROFILES; ++j)
		for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
			if(info.stencils_accessed[kernel][j+NUM_ALL_FIELDS][stencil])
				res |= acDeviceStencilAccessesBoundaries(acGridGetDevice(), Stencil(stencil));
	return AcBoundary(res);

}

static std::vector<AcBoundary>
get_kernel_depends_on_boundaries()
{
	std::vector<AcBoundary> res{};
	for(size_t i = 0; i < NUM_KERNELS; ++i)
		res.push_back(get_kernel_depends_on_boundaries(AcKernel(i)));
	return res;
}
static bool
kernel_reduces_scalar(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(size_t i = 0; i < info.n_reduce_outputs[kernel]; ++i)
		if(info.reduce_outputs[kernel][i].type != AC_PROF_TYPE) return true;
	return false;
}
static bool
kernel_reduces_something(const AcKernel kernel)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info.reduced_profiles[kernel][i]) return true;
	if(kernel_reduces_scalar(kernel)) return true;
	return false;
}


static bool
kernel_writes_profile(const AcKernel kernel, const AcProfileType prof_type)
{
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info.written_profiles[kernel][i] && prof_types[i] == prof_type) return true;
	return false;
}

static bool
kernel_reduces_only_profiles(const AcKernel kernel, const AcProfileType prof_type)
{
	if(kernel_reduces_scalar(kernel)) return false;
	const auto info = get_kernel_analysis_info();
	for(int i = 0; i < NUM_PROFILES; ++i)
		if(info.reduced_profiles[kernel][i] && prof_types[i] != prof_type) return false;
	return true;
}

static bool
kernel_does_only_profile_reductions(const AcKernel kernel, const AcProfileType prof_type)
{
	if(kernel_updates_vtxbuf(kernel)) return false;
	if(kernel_has_stencil_ops(kernel)) return false;
	return kernel_reduces_only_profiles(kernel,prof_type);
}


static bool
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

static std::vector<std::array<AcBoundary,NUM_PROFILES>>
compute_kernel_call_computes_profile_across_halos(const std::vector<AcKernel>& calls)
{
	std::vector<std::array<AcBoundary,NUM_PROFILES>> res{};
	for(size_t i = 0; i  < calls.size(); ++i)
	{
		std::array<AcBoundary,NUM_PROFILES> computes_profile_across_halos{};
		for(int prof = 0; prof < NUM_PROFILES; ++prof) computes_profile_across_halos[i] = BOUNDARY_NONE;
		res.push_back(computes_profile_across_halos);
	}

	const KernelAnalysisInfo info = get_kernel_analysis_info();
	for(size_t i = 0; i  < calls.size(); ++i)
	{
		const auto k = calls[i];
		for(size_t prof = 0; prof < NUM_PROFILES; ++prof)
		{
			for(size_t stencil = 0; stencil < NUM_STENCILS; ++stencil)
			{
				if(info.stencils_accessed[k][NUM_ALL_FIELDS+prof][stencil])
				{
					int defining_call = -1;
					for(int j = i-1; j >= 0; --j)
					{
						if(info.written_profiles[calls[j]][prof] || info.reduced_profiles[calls[j]][prof])
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
