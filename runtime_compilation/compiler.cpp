#include "acc_runtime.h"
#include "user_array_dims.h"



template <typename P>
auto
get_default_value()
{
	//if constexpr(std::is_same<P,AcRealCompArrayParam>::value) return (AcReal)NAN;
	//if constexpr(std::is_same<P,AcIntCompArrayParam>::value)  return (int)0;
	//if constexpr(std::is_same<P,AcBoolCompArrayParam>::value) return (bool)false;

	//if constexpr(std::is_same<P,AcRealCompParam>::value)  return (AcReal)NAN;
	//if constexpr(std::is_same<P,AcIntCompParam>::value)   return (int)0;
	//if constexpr(std::is_same<P,AcBoolCompParam>::value)  return (bool)false;
	//if constexpr(std::is_same<P,AcReal3CompParam>::value) return (AcReal3){(AcReal) NAN,(AcReal) NAN,(AcReal) NAN};
	//if constexpr(std::is_same<P,AcInt3CompParam>::value)  return (int3){0,0,0};
#include"get_default_value.h"
}

template <typename V>
const char*
get_datatype(){}

template <>
const char*
get_datatype<int>()     {return "int";};

template <>
const char*
get_datatype<bool>()    {return "bool";};


template <>
const char*
get_datatype<AcReal>()  {return "real";};

char*
to_str(const int value)
{
	char* res = (char*)malloc(sizeof(char)*4098);
	sprintf(res,"%d\n",value);
	return res;
}

char*
to_str(const AcReal value)
{
	char* res = (char*)malloc(sizeof(char)*4098);
	sprintf(res,"%.14e\n",value);
	return res;
}

char*
to_str(const bool value)
{
	char* res = (char*)malloc(sizeof(char)*4098);
	sprintf(res,"%d\n",value);
	return res;
}



template <typename V>
char*
to_str(const V value, const char* name)
{
	char* res = (char*)malloc(sizeof(char)*4098);
	char* val_str = to_str(value);
	sprintf(res,"%s %s = %s\n",get_datatype<V>(), name, val_str);
	free(val_str);
	return res;
}

#include "to_str_funcs.h"

template <typename V>
const char*
get_value_type(V value)
{
	(void)value;
	return get_datatype<V>();
}


template <typename P>
struct load_arrays
{
	void operator()(const AcCompInfo info, FILE* fp)
	{
		auto default_value = get_default_value<P>();
		const char* type = get_value_type(default_value);
		for(P array : get_params<P>())
		{
			const int n_dims = get_array_n_dims(array);
			const char* name = get_array_name(array);
			const bool is_loaded = get_is_loaded(array,info.is_loaded);
			auto* loaded_val = get_loaded_val(array,info.config);
			if(n_dims == 1)
			{
				fprintf(fp,"const %s %s = [",type,name);
				const int3 dims = get_array_dims(array);
				for(int j = 0; j < dims.x; ++j)
				{
					auto val = is_loaded ? loaded_val[j] : default_value;
					char* val_string = to_str(val);
					fprintf(fp,"%s",val_string);
					free(val_string);
					if(j < dims.x-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"%s","]\n");
			}
			else if(n_dims == 2)
			{
				fprintf(fp,"const %s %s = [", type, name);
				const int3 dims = get_array_dims(array);
				for(int y = 0; y < dims.y; ++y)
				{
					fprintf(fp,"%s","[");
					for(int x = 0; x < dims.x; ++x)
					{
						auto val = is_loaded ? loaded_val[x + y*dims.x] : default_value;
						char* val_string = to_str(val);
						fprintf(fp,"%s",val_string);
						free(val_string);
						if(x < dims.x-1) fprintf(fp,"%s",",");
					}
					fprintf(fp,"%s","]");
					if(y < dims.y-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"]\n");
			}
		}
	}
};
template <typename P>
struct load_scalars
{
	void operator()(const AcCompInfo info, FILE* fp)
	{
		for(P var : get_params<P>())
		{
			auto val =  get_is_loaded(var,info.is_loaded) ? get_loaded_val(var,info.config) : get_default_value<P>();
			char* res = to_str(val,get_param_name(var));
			fprintf(fp,"%s",res);
			free(res);
		}
	}
};
void
acCompile(const char* compilation_string, const AcCompInfo info)
{
	FILE* fp = fopen(AC_OVERRIDES_PATH,"w");

	AcScalarCompTypes::run<load_scalars>(info, fp);
	AcArrayCompTypes::run<load_arrays>(info, fp);

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
