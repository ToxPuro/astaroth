static KernelAnalysisInfo
get_kernel_analysis_info()
{
	KernelAnalysisInfo res;
	const auto& info = acDeviceGetLocalConfig(acGridGetDevice());
	acAnalysisGetKernelInfo(info.params,&res);
	return res;
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
