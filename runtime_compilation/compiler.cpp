#include <stdlib.h>

#include <string>

#include "astaroth.h"
#include "astaroth_runtime_compilation.h"
#if AC_MPI_ENABLED
#include "../src/core/decomposition/decomposition.h"
#endif
#include <sys/stat.h>
#include <unistd.h>

#include "src/config_helpers.h"

#if AC_MPI_ENABLED
static uint3_64
get_decomp(const MPI_Comm comm, const AcMeshInfo config)
{

    int nprocs;
    MPI_Comm_size(comm, &nprocs);
    switch(config[AC_decompose_strategy])
    {
	    case AC_DECOMPOSE_STRATEGY_EXTERNAL:
		return static_cast<uint3_64>(config[AC_domain_decomposition]);
	    default:
		return decompose(nprocs,(AcDecomposeStrategy)config[AC_decompose_strategy]);

    }
    return (uint3_64){0,0,0};
}

void
decompose_info(const MPI_Comm comm, AcMeshInfo& config)
{
  const auto decomp = get_decomp(comm,config);
  //TP: is not run_const anymore since for some reason gives bad performance
  //TODO: find out why!
  const int3 int3_decomp = (int3){(int)decomp.x,(int)decomp.y,(int)decomp.z};
  if(!config[AC_allow_non_divisible_grid])
  {
  	ERRCHK_ALWAYS(config[AC_ngrid].x % decomp.x == 0);
  	ERRCHK_ALWAYS(config[AC_ngrid].y % decomp.y == 0);
  	ERRCHK_ALWAYS(config[AC_ngrid].z % decomp.z == 0);
  }

  acPushToConfig(config,AC_nlocal,
		  	(int3)
			{
				config[AC_ngrid].x/int3_decomp.x,
				config[AC_ngrid].y/int3_decomp.y,
				config[AC_ngrid].z/int3_decomp.z
			}
		  );
  acLoadCompInfo(AC_domain_decomposition,(int3){(int)decomp.x, (int)decomp.y, (int)decomp.z},&config.run_consts);
}

#endif

void
check_that_built_ins_loaded(const AcCompInfo info)
{
  	//TP: are not run_const anymore since for some reason gives bad performance
  	//TODO: find out why!
	//ERRCHK_ALWAYS(info.is_loaded[AC_nlocal] || info.is_loaded[AC_ngrid]);

#if AC_MPI_ENABLED
	ERRCHK_ALWAYS(info.is_loaded[AC_proc_mapping_strategy]);
	ERRCHK_ALWAYS(info.is_loaded[AC_decompose_strategy]);
	ERRCHK_ALWAYS(info.is_loaded[AC_MPI_comm_strategy]);
	if(info.config[AC_decompose_strategy]  ==  AC_DECOMPOSE_STRATEGY_EXTERNAL)
		ERRCHK_ALWAYS(info.is_loaded.int3_params[AC_domain_decomposition]);
#endif
}
void
acLoadRunConstsBase(const char* filename, AcMeshInfo info)
{
	FILE* fp = fopen(filename,"w");
	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp,"override const", true);
	AcArrayCompTypes::run<load_comp_arrays>(info, fp,"override const", true);
	fclose(fp);
}
const char* dynamic_base_path   = astaroth_base_path;
const char* dynamic_binary_path = astaroth_binary_path;
const char* dynamic_acc_compiler_path = NULL;

static std::string
get_astaroth_base_path()
{
	return std::string(dynamic_base_path);
}

static std::string
get_astaroth_binary_path()
{
	return std::string(dynamic_binary_path);
}
static std::string
runtime_astaroth_build_path()
{
	return get_astaroth_binary_path() + std::string("/runtime_build");
}
static std::string
acc_compiler_path()
{
	if(dynamic_acc_compiler_path != NULL) return dynamic_acc_compiler_path;
	return get_astaroth_binary_path() + std::string("/acc-runtime/acc/acc");
}

static std::string
previous_cmake_options_path()
{
	return get_astaroth_binary_path() + std::string("/runtime_build/previous_cmake_options");
}

static std::string
ac_overrides_path()
{
	return runtime_astaroth_build_path() + std::string("/overrides.h");
}


void
acLoadRunConsts(AcMeshInfo info)
{
	acLoadRunConstsBase(ac_overrides_path().c_str(),info);
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}
void
check_for_cmake()
{
   char cmd[2*10000];
   sprintf(cmd,"cmake --help > /dev/null");
   const int retval = system(cmd);
   if(retval)
   {
   	fprintf(stderr,"%s","Did not find cmake\n");
   	fflush(stdout);
   	fflush(stderr);
   	exit(EXIT_FAILURE);
   }
}

const char*
get_cmake_options(const char* user_cmake_options)
{
    const char* command_line_options = RUNTIME_CMAKE_OPTIONS;
    static char* options             = NULL;
    asprintf(&options,
             "%s -DCMAKE_BUILD_TYPE=%s -DREAD_OVERRIDES=ON -DBUILD_SHARED_LIBS=ON "
             "-DOPENMP_ENABLED=OFF -DCPU_BUILD=%s -DDSL_MODULE_DIR=%s "
             "-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DACC_COMPILER_PATH=%s "
             "-DAC_STENCIL_ORDER=%d %s",
             command_line_options,
#ifdef NDEBUG
             "Release",
#else
             "Debug",
#endif
             AC_CPU_BUILD ? "ON" : "OFF", DSL_MODULE_DIR, acc_compiler_path().c_str(),
             AC_STENCIL_ORDER, user_cmake_options);
    return options;
}

AcResult
run_cmake(const char* user_cmake_options, const char* log_dst)
{
  
  char cmd[2*10000];
  const char* options = get_cmake_options(user_cmake_options);
  sprintf(cmd,"cd %s && cmake %s ",runtime_astaroth_build_path().c_str(),options);
#if AC_USE_HIP
#else
  //TP: needed to ensure nvcc does not write under /tmp which on compute nodes does not have enough memory
  setenv("TMPDIR", dynamic_binary_path, 1);
  strcat(cmd,"-DCMAKE_CUDA_FLAGS=-objtemp ");
#endif
  strcat(cmd,get_astaroth_base_path().c_str());
  if(log_dst)
  {
	  strcat(cmd," > ");
  	  strcat(cmd,log_dst);
	  strcat(cmd," 2>&1");
  }
  const int retval = system(cmd);
  if(retval)
  {
  	fprintf(stderr,"%s %d\n","Fatal was not able to run cmake:",retval);
  	fprintf(stderr,"Cmake command: %s\n",cmd);
  	fflush(stdout);
  	fflush(stderr);
	return AC_FAILURE;
  }
  sprintf(cmd,"echo %s > %s", get_cmake_options(user_cmake_options), previous_cmake_options_path().c_str());
  const int echo_retval = system(cmd);
  if(echo_retval)
  {
  	fprintf(stderr,"%s %d\n","Was not able to store cmake options:",retval);
  	fprintf(stderr,"Storing command: %s\n",cmd);
  	fflush(stdout);
  	fflush(stderr);
	return AC_FAILURE;
  }
  return AC_SUCCESS;
}

AcResult
acCompileFromRootProc(const char* user_cmake_options, const char* target, AcMeshInfo mesh_info)
{
	acLoadRunConstsBase("tmp_astaroth_run_consts.h",mesh_info);
	char cmd[2*10000];
	char cwd[5024];
	if (getcwd(cwd, sizeof(cwd)) == NULL) {
		fprintf(stderr,"Failed to get current working directory!\n");
		exit(EXIT_FAILURE);
	}
	char log_buffer[10024];

	if(mesh_info.runtime_compilation_log_dst == NULL)
		sprintf(log_buffer,"%s","/dev/stderr");
	else if (mesh_info.runtime_compilation_log_dst[0] == '/')
		sprintf(log_buffer,"%s",mesh_info.runtime_compilation_log_dst);
	else
		sprintf(log_buffer,"%s/%s",cwd,mesh_info.runtime_compilation_log_dst);
	const char* log_dst =  (mesh_info.runtime_compilation_log_dst == NULL) ? NULL : log_buffer;
	check_for_cmake();
	sprintf(cmd,"diff tmp_astaroth_run_consts.h %s",ac_overrides_path().c_str());
	const bool runtime_build_existed = file_exists(runtime_astaroth_build_path().c_str());
	const bool previous_build_exists = file_exists(ac_overrides_path().c_str()) && runtime_build_existed;
	const bool loaded_different = 
			       previous_build_exists 
				? system(cmd) : true;
	system("rm tmp_astaroth_run_consts.h");
	const bool stored_cmake = file_exists(previous_cmake_options_path().c_str());
	sprintf(cmd,"echo %s | diff - %s",get_cmake_options(user_cmake_options),previous_cmake_options_path().c_str());
	const bool different_cmake_string =  stored_cmake ? system(cmd) : true;
	const bool compile = !previous_build_exists || loaded_different || different_cmake_string;
	char logging_to_message[100000];
	if(mesh_info.runtime_compilation_skip_autotuning)
	{
		char autotune_csv_filename[10000];
		sprintf(autotune_csv_filename, "%s/acc-runtime/api/autotune.csv",runtime_astaroth_build_path().c_str());
		if(file_exists(autotune_csv_filename))
		{
			sprintf(cmd,"mv %s .",autotune_csv_filename);
			int retval = system(cmd);
			if(previous_build_exists && retval)
			{
				fflush(stdout);
				fflush(stderr);
				fprintf(stderr,"Was not able to copy previous autotuning results.\nWill autotune again.\n");
			}
		}
	}
	if(log_dst) sprintf(logging_to_message," logging to %s",log_dst);
	else        sprintf(logging_to_message,"%s","");
	if(!previous_build_exists)
	{
		fprintf(stderr,"acCompile: Compiling Astaroth;%s\n",logging_to_message);
	}
	if(compile)
	{
		if(loaded_different && previous_build_exists)  
		{
			fprintf(stderr,"acCompile: Loaded different run_const values; recompiling;%s\n",logging_to_message);
		}
		else if(previous_build_exists && different_cmake_string && stored_cmake) 
		{
			fprintf(stderr,"acCompile: Gave different cmake options; recompiling;%s\n",logging_to_message);
		}
		sprintf(cmd,"rm -rf %s",runtime_astaroth_build_path().c_str());
		int retval = system(cmd);
		if(retval)
		{
			fflush(stdout);
			fflush(stderr);
			fprintf(stderr,"Fatal error was not able to remove build directory: %s\n",runtime_astaroth_build_path().c_str());
			return AC_FAILURE;
		}

		sprintf(cmd,"mkdir -p %s",runtime_astaroth_build_path().c_str());
		retval = system(cmd);
		if(retval)
		{
			fflush(stdout);
			fflush(stderr);
			fprintf(stderr,"%s","Fatal error was not able to make build directory\n");
			return AC_FAILURE;
		}
		acLoadRunConstsBase(ac_overrides_path().c_str(),mesh_info);
		if(run_cmake(user_cmake_options,log_dst) != AC_SUCCESS) return AC_FAILURE;
	}
	else
	{
		if(mesh_info.runtime_compilation_skip_make_if_nothing_has_changed)
		{
		        fprintf(stderr,"acCompile: No changes in input; Skipping compilation;\n");
			return AC_SUCCESS;
		}
		fprintf(stderr,"acCompile: No changes in input; running make in case DSL code has changed;%s\n",logging_to_message);
	}
	sprintf(cmd,"cd %s",
	       runtime_astaroth_build_path().c_str());
	int retval = system(cmd);
	if(retval)
	{
		fprintf(stderr,"%s","Fatal error was not able to go into build directory\n");
		fflush(stdout);
		fflush(stderr);
		return AC_FAILURE;
	}
	if(log_dst)
	{
		sprintf(cmd,"cd %s && make %s -j >> %s 2>&1",runtime_astaroth_build_path().c_str(),target,log_dst);
	}
	else
	{
		sprintf(cmd,"cd %s && make %s -j",runtime_astaroth_build_path().c_str(),target);
	}
	retval = system(cmd);
	if(retval)
	{
		fprintf(stderr,"%s","Fatal was not able to compile\n");
		if(log_dst)
		{
			fprintf(stderr,"Check %s for the compilation log!\n",log_dst);
		}
		fflush(stdout);
		fflush(stderr);
		return AC_FAILURE;
	}
	if(mesh_info.runtime_compilation_skip_autotuning)
	{
		char autotune_csv_filename[100000];
		sprintf(autotune_csv_filename, "autotune.csv");
		if(file_exists(autotune_csv_filename))
		{
			sprintf(cmd,"mv autotune.csv %s/acc-runtime/api",runtime_astaroth_build_path().c_str());
			if(system(cmd) && previous_build_exists)
			{
				fflush(stdout);
				fflush(stderr);
				fprintf(stderr,"Was not able to copy stored autotuning results.\nWill autotune again.\n");
			}
		}
	}
	return AC_SUCCESS;
}

AcResult
acCompile(const char* user_cmake_options, const char* target, AcMeshInfo mesh_info)
{
        ac_unset_floating_point_exceptions();
	if(mesh_info.runtime_compilation_build_path) dynamic_binary_path = mesh_info.runtime_compilation_build_path;
	if(mesh_info.runtime_compilation_base_path)  dynamic_base_path  = mesh_info.runtime_compilation_base_path;
	if(mesh_info.acc_compiler_path)  dynamic_acc_compiler_path  = mesh_info.acc_compiler_path;
	check_that_built_ins_loaded(mesh_info.run_consts);
	acHostUpdateParams(&mesh_info);
#if AC_MPI_ENABLED
        int mpi_has_been_initialized{};
        MPI_Initialized(&mpi_has_been_initialized);
        if(!mpi_has_been_initialized)
        {
                int provided{};
                MPI_Init_thread(NULL,NULL,MPI_THREAD_MULTIPLE,&provided);
                ERRCHK_ALWAYS(provided >= MPI_THREAD_MULTIPLE);
        }
	ERRCHK_ALWAYS(mesh_info.comm != NULL && mesh_info.comm->handle != MPI_COMM_NULL);
	int pid;
	MPI_Comm_rank(mesh_info.comm->handle,&pid);
	if(!mpi_has_been_initialized && pid == 0)
	{
          fprintf(stderr, "acCompile: MPI was not initialized so called MPI_Init_thread with MPI_THREAD_MULTIPLE\n");
	}
	decompose_info(mesh_info.comm->handle,mesh_info);
#else
	const int pid = 0;
#endif
	acHostUpdateParams(&mesh_info);
	AcResult res = AC_SUCCESS;
	if(pid == 0)
	{
		res = acCompileFromRootProc(user_cmake_options,target,mesh_info);
	}
#if AC_MPI_ENABLED
	MPI_Barrier(mesh_info.comm->handle);
#endif
        ac_restore_floating_point_exceptions();
	return res;
}
