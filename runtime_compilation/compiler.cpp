#include "astaroth.h"
#include "astaroth_runtime_compilation.h"
#include <string>
#if AC_MPI_ENABLED
#include "../src/core/decomposition.h"
#endif
#include <sys/stat.h>



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
get_datatype(){return{};}

template <>
std::string
get_datatype<int>()     {return "int";};

template <>
std::string
get_datatype<bool>()    {return "bool";};


template <>
std::string
get_datatype<AcReal>()  {return "real";};

template <>
std::string
get_datatype<long>()  {return "long";};

template <>
std::string
get_datatype<long long>()  {return "long long";};


std::string
to_str(const int value)
{
	return std::to_string(value);
}

std::string
to_str(const long value)
{
	return std::to_string(value);
}

std::string
to_str(const long long value)
{
	return std::to_string(value);
}

std::string
to_str(const AcReal value)
{
	char* tmp;
	asprintf(&tmp,"%.17g\n",value);
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
	return "override const " + get_datatype<V>() + " " + name + " = " + val_str + ";\n";
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
			const bool is_loaded = info.is_loaded[array];
			auto* loaded_val = info.config[array];
			if(n_dims == 1)
			{
				fprintf(fp,"override const %s %s = [",type.c_str(),name);
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
				fprintf(fp,"override const %s %s = [", type.c_str(), name);
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
			auto val =  info.is_loaded[var] ? info.config[var] : get_default_value<P>();
			std::string res = to_str(val,get_param_name(var));
			fprintf(fp,"%s",res.c_str());
		}
	}
};
#if AC_MPI_ENABLED
static uint3_64
get_decomp(const MPI_Comm comm, const AcCompInfo info)
{

    const auto global_config = info.config;
    int nprocs;
    MPI_Comm_size(comm, &nprocs);
    switch((AcDecomposeStrategy)global_config.int_params[AC_decompose_strategy])
    {
	    case AcDecomposeStrategy::External:
		return static_cast<uint3_64>(global_config.int3_params[AC_domain_decomposition]);
	    default:
		return decompose(nprocs,(AcDecomposeStrategy)global_config.int_params[AC_decompose_strategy]);

    }
    return (uint3_64){0,0,0};
}

void
decompose_info(const MPI_Comm comm, AcCompInfo& info)
{
  auto decomp = get_decomp(comm,info);
  auto& config = info.config;
  ERRCHK_ALWAYS(config.int_params[AC_nx] % decomp.x == 0);
  ERRCHK_ALWAYS(config.int_params[AC_ny] % decomp.y == 0);
#if TWO_D == 0
  ERRCHK_ALWAYS(config.int_params[AC_nz] % decomp.z == 0);
#else
  ERRCHK_ALWAYS(config.int_params[AC_nz]  == 1);
#endif
  const int3 nn = (int3){config.int_params[AC_nx], config.int_params[AC_ny], config.int_params[AC_nz]};
  const int submesh_nx = nn.x / decomp.x;
  const int submesh_ny = nn.y / decomp.y;
  const int submesh_nz = nn.z / decomp.z;

  acLoadCompInfo(AC_nx,submesh_nx,&info);
  acLoadCompInfo(AC_ny,submesh_ny,&info);
#if TWO_D == 0
  acLoadCompInfo(AC_nz,submesh_nz,&info);
#endif
  acLoadCompInfo(AC_domain_decomposition,(int3){(int)decomp.x, (int)decomp.y, (int)decomp.z},&info);
  acHostUpdateBuiltinCompParams(&info);
}

#endif

void
check_that_built_ins_loaded(const AcCompInfo info)
{
	ERRCHK_ALWAYS(info.is_loaded[AC_nx] || info.is_loaded[AC_nxgrid]);
	ERRCHK_ALWAYS(info.is_loaded[AC_ny] || info.is_loaded[AC_nygrid]);
	ERRCHK_ALWAYS(info.is_loaded[AC_nz] || info.is_loaded[AC_nzgrid]);

#if AC_MPI_ENABLED
	ERRCHK_ALWAYS(info.is_loaded[AC_proc_mapping_strategy]);
	ERRCHK_ALWAYS(info.is_loaded[AC_decompose_strategy]);
	ERRCHK_ALWAYS(info.is_loaded[AC_MPI_comm_strategy]);
	if(info.config[AC_decompose_strategy]  ==  (int)AcDecomposeStrategy::External)
		ERRCHK_ALWAYS(info.is_loaded.int3_params[AC_domain_decomposition]);
#endif
}
void
acLoadRunConstsBase(const char* filename, AcMeshInfo info)
{
	FILE* fp = fopen(filename,"w");
	AcScalarCompTypes::run<load_scalars>(info.run_consts, fp);
	AcArrayCompTypes::run<load_arrays>(info.run_consts, fp);
	fclose(fp);
}

void
acLoadRunConsts(AcMeshInfo info)
{
	acLoadRunConstsBase(AC_OVERRIDES_PATH,info);
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}
void
acCompile(const char* compilation_string, AcMeshInfo mesh_info)
{
	AcCompInfo info = mesh_info.run_consts;
	check_that_built_ins_loaded(info);
	acHostUpdateBuiltinCompParams(&info);
#if AC_MPI_ENABLED
	ERRCHK_ALWAYS(mesh_info.comm != MPI_COMM_NULL);
	int pid;
	MPI_Comm_rank(mesh_info.comm,&pid);
	decompose_info(mesh_info.comm,info);
#else
	const int pid = 0;
#endif
	mesh_info.run_consts = info;
	if(pid == 0)
	{
		acLoadRunConstsBase("tmp_astaroth_run_consts.h",mesh_info);
		char cmd[10000];
		sprintf(cmd,"diff tmp_astaroth_run_consts.h %s",AC_OVERRIDES_PATH);
		const bool overrides_exists = file_exists(AC_OVERRIDES_PATH);
		const bool loaded_different = 
				        overrides_exists	
					? system(cmd) : true;
		acLoadRunConstsBase(AC_OVERRIDES_PATH,mesh_info);
		if(loaded_different)
		{
			if(loaded_different && overrides_exists)
				fprintf(stderr,"Loaded different run_const values; recompiling\n");
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
		}
		sprintf(cmd,"cd %s",
		       runtime_astaroth_build_path);
		int retval = system(cmd);
		if(retval)
		{
			fprintf(stderr,"%s","Fatal error was not able to go into build directory\n");
			fflush(stdout);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
		sprintf(cmd,"cmake --help > /dev/null");
		retval = system(cmd);
		if(retval)
		{
			fprintf(stderr,"%s","Did not find cmake\n");
			fflush(stdout);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
		sprintf(cmd,"cd %s && cmake -DREAD_OVERRIDES=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DACC_COMPILER_PATH=%s %s %s",
		       runtime_astaroth_build_path, acc_compiler_path, compilation_string,astaroth_base_path);
		retval = system(cmd);
		if(retval)
		{
			fprintf(stderr,"%s %d\n","Fatal was not able to run cmake:",retval);
			fprintf(stderr,"Cmake command: %s\n",cmd);
			fflush(stdout);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
		sprintf(cmd,"cd %s && make -j",runtime_astaroth_build_path);
		retval = system(cmd);
		if(retval)
		{
			fprintf(stderr,"%s","Fatal was not able to compile\n");
			fflush(stdout);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
	}
#if AC_MPI_ENABLED
	MPI_Barrier(mesh_info.comm);
#endif
}
