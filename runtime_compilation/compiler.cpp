#include "acc_runtime.h"
#include "astaroth_runtime_compilation.h"
#include <string>



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
std::string
get_datatype(){}

template <>
std::string
get_datatype<int>()     {return "int";};

template <>
std::string
get_datatype<bool>()    {return "bool";};


template <>
std::string
get_datatype<AcReal>()  {return "real";};

std::string
to_str(const int value)
{
	return std::to_string(value);
}

std::string
to_str(const AcReal value)
{
	char* tmp;
	asprintf(&tmp,"%.14e\n",value);
	std::string res = tmp;
	free(tmp);
	return res;
}

std::string
to_str(const bool value)
{
	char* tmp;
	asprintf(&tmp,"%s\n",(value) ? "true" : "false");
	std::string res = tmp;
	free(tmp);
	return res;
}



template <typename V>
std::string
to_str(const V value, const char* name)
{
	std::string val_str = to_str(value);
	std::string name_str = name;
	return "const " + get_datatype<V>() + " " + name + " = " + val_str + ";\n";
}

#include "to_str_funcs.h"

template <typename V>
std::string
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
		std::string type = get_value_type(default_value);
		for(P array : get_params<P>())
		{
			const int n_dims = get_array_n_dims(array);
			const char* name = get_array_name(array);
			const bool is_loaded = get_is_loaded(array,info.is_loaded);
			auto* loaded_val = get_loaded_val(array,info.config);
			if(n_dims == 1)
			{
				fprintf(fp,"const %s %s = [",type.c_str(),name);
				const AcArrayDims dims = get_array_dims(array);
				for(int j = 0; j < dims.len[0]; ++j)
				{
					auto val = is_loaded ? loaded_val[j] : default_value;
					std::string val_string = to_str(val);
					fprintf(fp,"%s",val_string.c_str());
					if(j < dims.len[0]-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"%s","]\n");
			}
			else if(n_dims == 2)
			{
				fprintf(fp,"const %s %s = [", type.c_str(), name);
				const AcArrayDims dims = get_array_dims(array);
				for(int y = 0; y < dims.len[1]; ++y)
				{
					fprintf(fp,"%s","[");
					for(int x = 0; x < dims.len[0]; ++x)
					{
						auto val = is_loaded ? loaded_val[x + y*dims.len[0]] : default_value;
						std::string val_string = to_str(val);
						fprintf(fp,"%s",val_string.c_str());
						if(x < dims.len[0]-1) fprintf(fp,"%s",",");
					}
					fprintf(fp,"%s","]");
					if(y < dims.len[1]-1) fprintf(fp,"%s",",");
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
			std::string res = to_str(val,get_param_name(var));
			fprintf(fp,"%s",res.c_str());
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


	sprintf(cmd,"cd %s && cmake %s -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DACC_COMPILER_PATH=%s %s  && make -j",
			runtime_astaroth_build_path, AC_BASE_PATH, acc_compiler_path, compilation_string);
	retval = system(cmd);
	if(retval)
	{
		fflush(stdout);
		fflush(stderr);
		fprintf(stderr,"%s","Fatal error was not able to go into build directory\n");
		exit(EXIT_FAILURE);
	}

}
