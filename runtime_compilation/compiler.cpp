#include "acc_runtime.h"
#include "user_array_dims.h"

template <const bool is_real, typename T, typename F>
void
load_arrays(const char* type, const bool* is_loaded, const T* const* vals, const int* num_dims, const int3* dims, const int n_elems, F names, FILE* fp, const T default_value)
{
	for(int i = 0; i < n_elems; ++i)
	{
		if(num_dims[i] == 1)
		{
			fprintf(fp,"const %s %s = {",type,names[i]);
			for(int j = 0; j < dims[i].x; ++j)
			{
				const T val = is_loaded[i] ? vals[i][j] : default_value;
				if constexpr (is_real)
					fprintf(fp,"%.14e",val);
				else
					fprintf(fp,"%d",val);
				if(j < dims[i].x-1) fprintf(fp,"%s",",");
			}
			fprintf(fp,"%s","}\n");
		}
		else if(num_dims[i] == 2)
		{
			fprintf(fp,"const %s %s = {", type, names[i]);
			for(int y = 0; y < dims[i].y; ++y)
			{
				fprintf(fp,"%s","{");
				for(int x = 0; x < dims[i].x; ++x)
				{
					const T val = is_loaded[i] ? vals[i][x + y*dims[i].x] : default_value;
					if constexpr (is_real)
						fprintf(fp,"%.14e",val);
					else
						fprintf(fp,"%d",val);
					if(x < dims[i].x-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"%s","}");
				if(y < dims[i].y-1) fprintf(fp,"%s",",");
			}
			fprintf(fp,"}\n");
		}
	}
}
template <const bool is_real, typename T, typename F>
void
load_scalars(const char* type, const bool* is_loaded, const T* vals, const int n_elems, F names, FILE* fp, const T default_value)
{
	for(int i = 0; i < n_elems; ++i)
	{
		const T val = (is_loaded[i]) ? vals[i] : default_value;
		if constexpr(is_real)
			fprintf(fp,"const %s %s = %.14e\n",type, names[i], val);
		else
			fprintf(fp,"const %s %s = %d\n",type, names[i], val);
	}
}
template <const bool is_real, typename T, typename F>
void
load_vecs(const char* type, const bool* is_loaded, const T* vals, const int n_elems, F names, FILE* fp, const T default_value)
{
	for(int i = 0; i < n_elems; ++i)
	{
		const T val = (is_loaded[i]) ? vals[i] : default_value;
		if constexpr(is_real)
			fprintf(fp,"const %s %s = real3(%.14e, %.14e, %.14e) \n",type, names[i], val.x, val.y, val.z);
		else
			fprintf(fp,"const %s %s = int3(%d, %d, %d) \n",type, names[i], val.x, val.y, val.z);
	}
}
void
acCompile(const char* compilation_string, const AcCompInfo info)
{
	FILE* fp = fopen(AC_OVERRIDES_PATH,"w");
	load_scalars<true>("real",info.is_loaded.real_params,info.config.real_params,NUM_REAL_COMP_PARAMS,real_comp_param_names,fp,(AcReal) NAN);
	load_scalars<false>("int",info.is_loaded.int_params,info.config.int_params,NUM_INT_COMP_PARAMS,int_comp_param_names,fp,0);
	load_scalars<false>("bool",info.is_loaded.bool_params,info.config.bool_params,NUM_BOOL_COMP_PARAMS,bool_comp_param_names,fp,false);

	load_vecs<true>("real3",info.is_loaded.real3_params,info.config.real3_params,NUM_REAL3_COMP_PARAMS,real3_comp_param_names,fp, (AcReal3){(AcReal) NAN,(AcReal) NAN,(AcReal) NAN});
	load_vecs<false>("int3",info.is_loaded.int3_params,info.config.int3_params,NUM_INT3_COMP_PARAMS,int3_comp_param_names,fp, (int3){0,0,0});

	load_arrays<true>("real", info.is_loaded.real_arrays,info.config.real_arrays,real_array_num_dims + NUM_REAL_ARRAYS, real_array_dims + NUM_REAL_ARRAYS, NUM_REAL_COMP_ARRAYS, real_array_comp_param_names,fp,(AcReal)NAN);
	load_arrays<false>("int", info.is_loaded.int_arrays,info.config.int_arrays,int_array_num_dims + NUM_INT_ARRAYS, int_array_dims + NUM_INT_ARRAYS, NUM_INT_COMP_ARRAYS, int_array_comp_param_names,fp,0);
	load_arrays<false>("bool", info.is_loaded.bool_arrays,info.config.bool_arrays,bool_array_num_dims + NUM_BOOL_ARRAYS, bool_array_dims + NUM_BOOL_ARRAYS, NUM_BOOL_COMP_ARRAYS, bool_array_comp_param_names,fp,false);
	fclose(fp);
	char cmd[10000];

	sprintf(cmd,"rm -rf %s",runtime_astaroth_build_path);
	int retval = system(cmd);
	if(retval)
	{
		fflush(stdout);
		fflush(stderr);
		fprintf(stderr,"Fatal error was not able to remove build directory: %s\n",runtime_astaroth_build_path);
		exit(EXIT_FAILURE);
	}

	sprintf(cmd,"mkdir %s",runtime_astaroth_build_path);
	retval = system(cmd);
	if(retval)
	{
		fflush(stdout);
		fflush(stderr);
		fprintf(stderr,"%s","Fatal error was not able to make build directory\n");
		exit(EXIT_FAILURE);
	}


	sprintf(cmd,"cd %s && cmake %s -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON %s && make -j",runtime_astaroth_build_path, AC_BASE_PATH, compilation_string);
	retval = system(cmd);
	if(retval)
	{
		fflush(stdout);
		fflush(stderr);
		fprintf(stderr,"%s","Fatal error was not able to go into build directory\n");
		exit(EXIT_FAILURE);
	}

}
