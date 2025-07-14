#include "acc_runtime.h"
#include "stencil_accesses.h"
bool
is_raytracing_kernel(const AcKernel kernel)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(incoming_ray_value_accessed[kernel][field][ray]) return true;
	}
	return false;
}

int
num_fields_ray_accessed_read_and_written(const AcKernel kernel)
{
	int res = 0;
	for(int field = 0; field < NUM_ALL_FIELDS; ++field)
	{
		if(write_called[kernel][field] || stencils_accessed[kernel][field][0])
		{
			res++;
			continue;
		}
		for(int ray = 0; ray < NUM_RAYS; ++ray)
		{
			if(
			    incoming_ray_value_accessed[kernel][field][ray]
			    || outgoing_ray_value_accessed[kernel][field][ray]
			    )
			{
				res++;
				continue;
			}
		}
	}
	return res;
}

AcBool3
raytracing_step_direction(const AcKernel kernel)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(incoming_ray_value_accessed[kernel][field][ray])
			{
				if(ray_directions[ray].z != 0) return (AcBool3){false,false,true};
				if(ray_directions[ray].y != 0) return (AcBool3){false,true,false};
				if(ray_directions[ray].x != 0) return (AcBool3){true,false,false};
			}
	}
	return (AcBool3){false,false,false};
}

AcBool3
raytracing_directions(const AcKernel kernel)
{
	AcBool3 res = (AcBool3){false,false,false};
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(incoming_ray_value_accessed[kernel][field][ray])
			{
				res.x |= ray_directions[ray].x != 0;
				res.y |= ray_directions[ray].y != 0;
				res.z |= ray_directions[ray].z != 0;
			}
	}
	return res;
}

int
raytracing_number_of_directions(const AcKernel kernel)
{
	const auto dirs = raytracing_directions(kernel);
	return dirs.x+dirs.y+dirs.z;
}

bool
is_coop_raytracing_kernel(const AcKernel kernel)
{
	return is_raytracing_kernel(kernel) && (raytracing_number_of_directions(kernel) > 1);
}
bool
profile_is_reduced(const Profile profile)
{
	for(size_t kernel = 0;  kernel < NUM_KERNELS; ++kernel)
	{
		if(reduced_profiles[kernel][profile]) return true;
	}
	return false;
}
