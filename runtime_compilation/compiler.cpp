#include "astaroth.h"
#include "astaroth_runtime_compilation.h"
#include <string>
#if AC_MPI_ENABLED
#include "../src/core/decomposition.h"
#endif
#include <sys/stat.h>
#include "../src/config_helpers.h"

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
  const auto decomp = get_decomp(comm,info);
  const int3 int3_decomp = (int3){(int)decomp.x,(int)decomp.y,(int)decomp.z};
  auto& config = info.config;
  ERRCHK_ALWAYS(config[AC_nlocal].x % decomp.x == 0);
  ERRCHK_ALWAYS(config[AC_nlocal].y % decomp.y == 0);
  ERRCHK_ALWAYS(config[AC_nlocal].z % decomp.z == 0);

  config[AC_nlocal].x = config[AC_nlocal].x/int3_decomp.x;
  config[AC_nlocal].y = config[AC_nlocal].y/int3_decomp.y;
  config[AC_nlocal].z = config[AC_nlocal].z/int3_decomp.z;

  acLoadCompInfo(AC_domain_decomposition,(int3){(int)decomp.x, (int)decomp.y, (int)decomp.z},&info);
  acHostUpdateBuiltinCompParams(&info);
}

#endif

void
check_that_built_ins_loaded(const AcCompInfo info)
{
	ERRCHK_ALWAYS(info.is_loaded[AC_nlocal] || info.is_loaded[AC_ngrid]);

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
	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp,"override const", true);
	AcArrayCompTypes::run<load_comp_arrays>(info.run_consts, fp,"override const", true);
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
