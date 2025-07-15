#ifdef __cplusplus
extern "C"
{
#endif
bool
is_raytracing_kernel(const AcKernel kernel);
int
num_fields_ray_accessed_read_and_written(const AcKernel kernel);
AcBool3
raytracing_step_direction(const AcKernel kernel);
AcBool3
raytracing_directions(const AcKernel kernel);
int
raytracing_number_of_directions(const AcKernel kernel);
bool
is_coop_raytracing_kernel(const AcKernel kernel);
bool
profile_is_reduced(const Profile profile);
#ifdef __cplusplus
}
#endif
