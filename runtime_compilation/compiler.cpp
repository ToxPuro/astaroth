#include <stdlib.h>

#include <string>

#include "ac_helpers.h"
#include "astaroth.h"
#include "astaroth_runtime_compilation.h"
#include "astaroth_helpers.h"
#if AC_MPI_ENABLED
#include "../src/core/decomposition/decomposition.h"
#endif
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif

const size_t max_string_size = 400000;


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

static void
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

static void
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
	acStoreRunConsts(info,ac_overrides_path().c_str());
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

static void
check_for_cmake()
{
   static char cmd[402000];
   snprintf(cmd,max_string_size,"cmake --help > /dev/null");
   const int retval = system(cmd);
   if(retval)
   {
   	fprintf(stderr,"%s","Did not find cmake\n");
   	fflush(stdout);
   	fflush(stderr);
   	exit(EXIT_FAILURE);
   }
}

static const char*
get_cmake_options(const char* user_cmake_options)
{
    const char* command_line_options = RUNTIME_CMAKE_OPTIONS;
    static char* options             = NULL;
    asprintf(&options,
             "%s -DCMAKE_BUILD_TYPE=%s -DREAD_OVERRIDES=ON -DBUILD_SHARED_LIBS=ON "
             "-DOPENMP_ENABLED=OFF -DCPU_BUILD=%s -DDSL_MODULE_DIR=%s "
             "-DACC_COMPILER_PATH=%s -DAC_STENCIL_ORDER=%d %s",
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

static AcResult
run_cmake(const char* user_cmake_options, const char* log_dst)
{
  
  static char cmd[402000];
  const char* options = get_cmake_options(user_cmake_options);
  snprintf(cmd,sizeof(cmd),"cd %s && cmake %s ",runtime_astaroth_build_path().c_str(),options);
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
  snprintf(cmd,max_string_size,"echo %s > %s", get_cmake_options(user_cmake_options), previous_cmake_options_path().c_str());
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

static AcResult
acCompileFromRootProc(const char* user_cmake_options, const char* target, AcMeshInfo mesh_info)
{
	acStoreRunConsts(mesh_info,"tmp_astaroth_run_consts.h");
	static char cmd[max_string_size];
	char cwd[5024];
	if (getcwd(cwd, sizeof(cwd)) == NULL) {
		fprintf(stderr,"Failed to get current working directory!\n");
		exit(EXIT_FAILURE);
	}
	static char log_buffer[402000];

	if(mesh_info.runtime_compilation_log_dst == NULL)
		snprintf(log_buffer,max_string_size,"%s","/dev/stderr");
	else if (mesh_info.runtime_compilation_log_dst[0] == '/')
		snprintf(log_buffer,max_string_size,"%s",mesh_info.runtime_compilation_log_dst);
	else
		snprintf(log_buffer,max_string_size,"%s/%s",cwd,mesh_info.runtime_compilation_log_dst);
	const char* log_dst =  (mesh_info.runtime_compilation_log_dst == NULL) ? NULL : log_buffer;
	check_for_cmake();
	snprintf(cmd,max_string_size,"diff tmp_astaroth_run_consts.h %s",ac_overrides_path().c_str());
	const bool runtime_build_existed = file_exists(runtime_astaroth_build_path().c_str());
	const bool previous_build_exists = file_exists(ac_overrides_path().c_str()) && runtime_build_existed;
	const bool loaded_different = 
			       previous_build_exists 
				? system(cmd) : true;
	system("rm tmp_astaroth_run_consts.h");
	const bool stored_cmake = file_exists(previous_cmake_options_path().c_str());
	snprintf(cmd,max_string_size,"echo %s | diff - %s",get_cmake_options(user_cmake_options),previous_cmake_options_path().c_str());
	const bool different_cmake_string =  stored_cmake ? system(cmd) : true;
	const bool compile = !previous_build_exists || loaded_different || different_cmake_string;
	static char logging_to_message[402000];
	if(mesh_info.runtime_compilation_skip_autotuning)
	{
		static char autotune_csv_filename[402000];
		snprintf(autotune_csv_filename,max_string_size, "%s/acc-runtime/api/autotune.csv",runtime_astaroth_build_path().c_str());
		if(file_exists(autotune_csv_filename))
		{
			snprintf(cmd,max_string_size+1000,"mv %s .",autotune_csv_filename);
			int retval = system(cmd);
			if(previous_build_exists && retval)
			{
				fflush(stdout);
				fflush(stderr);
				fprintf(stderr,"Was not able to copy previous autotuning results.\nWill autotune again.\n");
			}
		}
	}
	if(log_dst) snprintf(logging_to_message,max_string_size+1000," logging to %s",log_dst);
	else        snprintf(logging_to_message,max_string_size,"%s","");
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
		snprintf(cmd,max_string_size,"rm -rf %s",runtime_astaroth_build_path().c_str());
		int retval = system(cmd);
		if(retval)
		{
			fflush(stdout);
			fflush(stderr);
			fprintf(stderr,"Fatal error was not able to remove build directory: %s\n",runtime_astaroth_build_path().c_str());
			return AC_FAILURE;
		}

		snprintf(cmd,max_string_size,"mkdir -p %s",runtime_astaroth_build_path().c_str());
		retval = system(cmd);
		if(retval)
		{
			fflush(stdout);
			fflush(stderr);
			fprintf(stderr,"%s","Fatal error was not able to make build directory\n");
			return AC_FAILURE;
		}
		acStoreRunConsts(mesh_info,ac_overrides_path().c_str());
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
	snprintf(cmd,max_string_size,"cd %s",
	       runtime_astaroth_build_path().c_str());
	int retval = system(cmd);
	if(retval)
	{
		fprintf(stderr,"%s","Fatal error was not able to go into build directory\n");
		fflush(stdout);
		fflush(stderr);
		return AC_FAILURE;
	}

        // Chaining this many %s parameters is quite gnarly but this way we can avoid even more
        // conditionals or allocations on the heap.
        snprintf(cmd, max_string_size,
		"cd %s && cmake --build . -j%s%s%s%s%s",
		runtime_astaroth_build_path().c_str(),
		strcmp(target, "") != 0 ? " -t " : "",
		strcmp(target, "") != 0 ? target : "",
		log_dst ? " >>" : "",
		log_dst ? log_dst : "",
		log_dst ? " 2>&1" : "");

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
		static char autotune_csv_filename[402000];
		snprintf(autotune_csv_filename,max_string_size, "autotune.csv");
		if(file_exists(autotune_csv_filename))
		{
			snprintf(cmd,max_string_size,"mv autotune.csv %s/acc-runtime/api",runtime_astaroth_build_path().c_str());
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

enum AcAffinity
{
	ALL,
	ORIGINAL
};


void
set_affinity(const AcAffinity affinity)
{
#if defined(__linux__)
  static bool got_affinity = false;
  static cpu_set_t original_affinity{};
  if(affinity == ALL)
  {
   got_affinity = sched_getaffinity(0,sizeof(original_affinity),&original_affinity) == 0;
   if(got_affinity)
   {
   	cpu_set_t all_cpus;
   	CPU_ZERO(&all_cpus);
   	for(int i = 0; i < CPU_SETSIZE; ++i) CPU_SET(i,&all_cpus);
   	//TP: the kernel drops the CPUs not allowed by the cpuset of the job
   	sched_setaffinity(0,sizeof(all_cpus),&all_cpus);
   }
  }
  else if(affinity == ORIGINAL)
  {
    if(got_affinity) sched_setaffinity(0,sizeof(original_affinity),&original_affinity);
  }
#endif
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
		//TP: the compilation (cmake, make -j, acc and the compilers) is run by child processes of this thread
		//    which inherit its CPU affinity. If the caller has bound this thread to a single core,
		//    e.g. OpenMP with OMP_PROC_BIND set binds the initial thread to the first place,
		//    the whole parallel build would be run on that single core.
		//    Hence allow the compilation to use all CPUs available to the process and restore the affinity afterwards.
		set_affinity(ALL);
		res = acCompileFromRootProc(user_cmake_options,target,mesh_info);
		set_affinity(ORIGINAL);
	}
#if AC_MPI_ENABLED
	MPI_Barrier(mesh_info.comm->handle);
	MPI_Bcast(&res, 1, MPI_INT, 0, mesh_info.comm->handle);
#endif
        ac_restore_floating_point_exceptions();
	return res;
}

#include "astaroth_lib.h"

static std::vector<AcLibHandle> handles{};

AcLibHandle
acLibLoadLibrary(const AcMeshInfo* info, const char* lib_path, const char* lib_filename,
                 int* counter)
{
#ifdef __APPLE__
    const char* lib_extension = "dylib";
#else
    const char* lib_extension = "so";
#endif

    char* original_lib_path = NULL;
    if (asprintf(&original_lib_path, "%s/runtime_build/%s/%s.%s",
                 info->runtime_compilation_build_path ? info->runtime_compilation_build_path
                                                      : astaroth_binary_path,
                 lib_path, lib_filename, lib_extension) == -1) {
        fprintf(stderr, "Fatal error while preparing path to library %s\n", lib_filename);
        exit(EXIT_FAILURE);
    }

    const char* new_lib_path = acLibraryVersion(original_lib_path, *counter, info->comm);
    ++counter;

    void* lib_handle = dlopen(new_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib_handle) {
        fprintf(stderr, "Fatal error while loading library %s: %s\n", new_lib_path, dlerror());
        exit(EXIT_FAILURE);
    }
    handles.push_back(lib_handle);

    return lib_handle;
}

AcResult
acLoadLibrary(const AcMeshInfo info)
{
    static int counter = 0;
    AcLibHandle handle =  acLibLoadLibrary(&info, "src/core", "libastaroth_core", &counter);

    LOAD_DSYM(acDeviceGetVertexBufferPtrs);
    LOAD_DSYM(acDeviceGetLocalConfig);
    LOAD_DSYM(acDeviceFinishReduceInt);
    LOAD_DSYM(acDeviceMemGetInfo);
    LOAD_DSYM(acKernelFlushInt);
    LOAD_DSYM(acAnalysisGetKernelInfo);
    LOAD_DSYM(acAnalysisCheckForDSLErrors);
    LOAD_DSYM(acDeviceSwapAllProfileBuffers);
    LOAD_DSYM(acDeviceFFTR2C);
    LOAD_DSYM(acDeviceFFTC2R);
    LOAD_DSYM(acDeviceFFTR2Planar);
    LOAD_DSYM(acDeviceFFTR2PlanarBatched);
    LOAD_DSYM(acDeviceFFTR2HermitianPlanarBatched);
    LOAD_DSYM(acDeviceFFTBackwardTransformPlanar);
    LOAD_DSYM(acDeviceFFTBackwardTransformPlanar2R);
    LOAD_DSYM(acDeviceFFTR2PlanarXY);
    LOAD_DSYM(acDeviceFFTBackwardTransformPlanar2RXY);
#if AC_MPI_ENABLED
    *(void**)(&BASE_FUNC_NAME(acBoundaryCondition)) = dlsym(handle, "acBoundaryCondition");
    LOAD_DSYM(ac_MPI_Barrier);
    LOAD_DSYM(ac_MPI_Comm_rank);
    LOAD_DSYM(ac_MPI_Comm_size);
    LOAD_DSYM(ac_MPI_Finalize);
    LOAD_DSYM(ac_MPI_Init);
    LOAD_DSYM(ac_MPI_Init_thread);
    LOAD_DSYM(acGridInitialized);
    LOAD_DSYM(acGridMPIComm);
    LOAD_DSYM(acGridMPISubComms);
    LOAD_DSYM(acGridDecomposeMeshInfo);
    LOAD_DSYM(acGridGetLocalMeshInfo);
    LOAD_DSYM(acGridQuit);
    LOAD_DSYM(acGridGetDevice);
    LOAD_DSYM(acGridRandomize);
    LOAD_DSYM(acGridSynchronizeStream);
    LOAD_DSYM(acGridLoadScalarUniform);
    LOAD_DSYM(acGridLoadVectorUniform);
    LOAD_DSYM(acGridLoadIntUniform);
    LOAD_DSYM(acGridLoadInt3Uniform);
    LOAD_DSYM(acGridLoadMesh);
    LOAD_DSYM(acGridStoreMesh);
#ifdef AC_INTEGRATION_ENABLED
    LOAD_DSYM(acGridIntegrate);
#endif
    LOAD_DSYM(acGridSwapBuffers);
    LOAD_DSYM(acGridHaloExchange);
    LOAD_DSYM(acGridPeriodicBoundconds);
    LOAD_DSYM(acGridReduceScal);
    LOAD_DSYM(acGridReduceVec);
    LOAD_DSYM(acGridReduceVecScal);
    LOAD_DSYM(acGridReduceXY);
    LOAD_DSYM(acGridAccessMeshOnDiskSynchronous);
    LOAD_DSYM(acGridDiskAccessLaunch);
    LOAD_DSYM(acGridWriteSlicesToDiskLaunch);
    LOAD_DSYM(acGridWriteSlicesToDiskSynchronous);
    LOAD_DSYM(acGridWriteSlicesToDiskCollectiveSynchronous);
    LOAD_DSYM(acGridWriteMeshToDiskLaunch);
    LOAD_DSYM(acGridDiskAccessSync);
    LOAD_DSYM(acGridReadVarfileToMesh);
    LOAD_DSYM(acHaloExchangeBoundary);
    LOAD_DSYM(acPeriodicRay);
    *(void**)(&BASE_FUNC_NAME(acRayUpdate))          = dlsym(handle, "acRayUpdate");
    *(void**)(&BASE_FUNC_NAME(acComputeWithParams))  = dlsym(handle, "acComputeWithParams");
    *(void**)(&BASE_FUNC_NAME(acCompute))            = dlsym(handle, "acCompute");
    *(void**)(&BASE_FUNC_NAME(acHaloExchange))       = dlsym(handle, "acHaloExchange");
    *(void**)(&BASE_FUNC_NAME(acScan))               = dlsym(handle, "acScan");
    *(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraph)) = dlsym(handle, "acGridBuildTaskGraph");
    *(void**)(&BASE_FUNC_NAME(
        acGridBuildTaskGraphWithBounds)) = dlsym(handle, "acGridBuildTaskGraphWithBounds");
    LOAD_DSYM(acGridDestroyTaskGraph);
    LOAD_DSYM(acGridClearTaskGraphCache);
    LOAD_DSYM(acGetDSLTaskGraph);
    *(void**)(&BASE_FUNC_NAME(acGetOptimizedDSLTaskGraph)) = dlsym(handle,
                                                                   "acGetOptimizedDSLTaskGraph");
    LOAD_DSYM(acGetDSLTaskGraphWithBounds);
    LOAD_DSYM(acGetOptimizedDSLTaskGraphWithBounds);
    LOAD_DSYM(acGetComputeStepsBCs);
    LOAD_DSYM(acGridAccessMeshOnDiskSynchronousDistributed);
    LOAD_DSYM(acGridAccessMeshOnDiskSynchronousCollective);
    LOAD_DSYM(acGridGetDefaultTaskGraph);
    LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsX);
    LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsY);
    LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsZ);
    LOAD_DSYM(acGridTaskGraphIsEmpty);
    LOAD_DSYM(acGridExecuteTaskGraph);
    LOAD_DSYM(acGridExecuteTaskGraphBase);
    LOAD_DSYM(acGridFinalizeReduceLocal);
    LOAD_DSYM(acGridFinalizeReduce);
    LOAD_DSYM(acGridLaunchKernel);
    LOAD_DSYM(acGridLoadStencil);
    LOAD_DSYM(acGridStoreStencil);
    LOAD_DSYM(acGridLoadStencils);
    LOAD_DSYM(acGridStoreStencils);
    LOAD_DSYM(acGridInitBase);
#endif
    LOAD_DSYM(acGetLocalNN);
    LOAD_DSYM(acGetLocalMM);
    LOAD_DSYM(acGetGridNN);
    LOAD_DSYM(acGetGridMM);
    LOAD_DSYM(acGetMinNN);
    LOAD_DSYM(acGetMaxNN);
    LOAD_DSYM(acGetGridMaxNN);
    LOAD_DSYM(acGetLengths);
#include "device_load_uniform_loads.h"
#include "ac_push_to_config_loads.h"
    LOAD_DSYM(acGetKernelId);
    LOAD_DSYM(acGetKernelIdByName);
    LOAD_DSYM(acCheckDeviceAvailability);
    LOAD_DSYM(acGetNumDevicesPerNode);
    LOAD_DSYM(acGetNumFields);
    LOAD_DSYM(acGetFieldHandle);
    LOAD_DSYM(acGetFieldName);
    LOAD_DSYM(acFieldIsAuxiliary);
    LOAD_DSYM(acGetNode);
    LOAD_DSYM(acNodeCreate);
    LOAD_DSYM(acNodeDestroy);
    LOAD_DSYM(acNodePrintInfo);
    LOAD_DSYM(acNodeQueryDeviceConfiguration);
    LOAD_DSYM(acNodeAutoOptimize);
    LOAD_DSYM(acNodeSynchronizeStream);
    LOAD_DSYM(acNodeSynchronizeVertexBuffer);
    LOAD_DSYM(acNodeSynchronizeMesh);
    LOAD_DSYM(acNodeSwapBuffers);
    LOAD_DSYM(acNodeLoadConstant);
    LOAD_DSYM(acNodeLoadVertexBufferWithOffset);
    LOAD_DSYM(acNodeLoadMeshWithOffset);
    LOAD_DSYM(acNodeLoadVertexBuffer);
    LOAD_DSYM(acNodeLoadMesh);
    LOAD_DSYM(acNodeSetVertexBuffer);
    LOAD_DSYM(acNodeStoreVertexBufferWithOffset);
    LOAD_DSYM(acNodeStoreMeshWithOffset);
    LOAD_DSYM(acNodeStoreVertexBuffer);
    LOAD_DSYM(acNodeStoreMesh);
    LOAD_DSYM(acNodeIntegrateSubstep);
    LOAD_DSYM(acNodeIntegrate);
    LOAD_DSYM(acNodeIntegrateGBC);
    LOAD_DSYM(acNodePeriodicBoundcondStep);
    LOAD_DSYM(acNodePeriodicBoundconds);
    LOAD_DSYM(acNodeGeneralBoundcondStep);
    LOAD_DSYM(acNodeGeneralBoundconds);
    LOAD_DSYM(acNodeReduceScal);
    LOAD_DSYM(acNodeReduceVec);
    LOAD_DSYM(acNodeReduceVecScal);
    LOAD_DSYM(acDeviceCreate);
    LOAD_DSYM(acDeviceDestroy);
    LOAD_DSYM(acDeviceResetMesh);
    LOAD_DSYM(acDevicePrintInfo);
    LOAD_DSYM(acDeviceSynchronizeStream);
    LOAD_DSYM(acDeviceSwapBuffer);
    LOAD_DSYM(acDeviceSwapBuffers);
    LOAD_DSYM(acDeviceLoadScalarUniform);
    LOAD_DSYM(acDeviceLoadVectorUniform);
    LOAD_DSYM(acDeviceLoadIntUniform);
    LOAD_DSYM(acDeviceLoadBoolUniform);
    LOAD_DSYM(acDeviceLoadInt3Uniform);
    LOAD_DSYM(acDeviceStoreScalarUniform);
    LOAD_DSYM(acDeviceStoreVectorUniform);
    LOAD_DSYM(acDeviceStoreIntUniform);
    LOAD_DSYM(acDeviceStoreBoolUniform);
    LOAD_DSYM(acDeviceStoreInt3Uniform);
    LOAD_DSYM(acDeviceLoadMeshInfo);
    LOAD_DSYM(acDeviceLoadVertexBufferWithOffset);
    LOAD_DSYM(acDeviceLoadMeshWithOffset);
    LOAD_DSYM(acDeviceLoadVertexBuffer);
    LOAD_DSYM(acDeviceLoadMesh);
    LOAD_DSYM(acDeviceSetVertexBuffer);
    LOAD_DSYM(acDeviceFlushOutputBuffers);
    LOAD_DSYM(acDeviceStoreVertexBufferWithOffset);
    LOAD_DSYM(acDeviceGetConfig);
    LOAD_DSYM(acDeviceGetKernelInputParamsObject);
    LOAD_DSYM(acDeviceStoreMeshWithOffset);
    LOAD_DSYM(acDeviceStoreVertexBuffer);
    LOAD_DSYM(acDeviceStoreMesh);
    LOAD_DSYM(acDeviceTransferVertexBufferWithOffset);
    LOAD_DSYM(acDeviceTransferMeshWithOffset);
    LOAD_DSYM(acDeviceTransferVertexBuffer);
    LOAD_DSYM(acDeviceTransferMesh);
    LOAD_DSYM(acDeviceIntegrateSubstep);
    LOAD_DSYM(acDevicePeriodicBoundcondStep);
    LOAD_DSYM(acDevicePeriodicBoundconds);
    LOAD_DSYM(acDeviceGeneralBoundcondStep);
    LOAD_DSYM(acDeviceGeneralBoundconds);
    LOAD_DSYM(acDeviceReduceScalNoPostProcessing);
    LOAD_DSYM(acDeviceReduceScal);
    LOAD_DSYM(acDeviceReduceVecNoPostProcessing);
    LOAD_DSYM(acDeviceReduceVec);
    LOAD_DSYM(acDeviceReduceVecScalNoPostProcessing);
    LOAD_DSYM(acDeviceReduceVecScal);
    LOAD_DSYM(acDeviceUpdate);
    LOAD_DSYM(acDeviceGetKernelOutput);
    LOAD_DSYM(acDeviceLaunchKernel);
    LOAD_DSYM(acDeviceBenchmarkKernel);
    LOAD_DSYM(acDeviceLoadStencil);
    LOAD_DSYM(acDeviceLoadStencils);
    LOAD_DSYM(acDeviceLoadStencilsFromConfig);
    LOAD_DSYM(acDeviceStoreStencil);
    LOAD_DSYM(acDeviceVolumeCopy);
#include "device_get_input_loads.h"
#include "device_get_output_loads.h"
#include "device_set_input_loads.h"
#include "get_vtxbufs_loads.h"

    *(void**)(&acDeviceGetIntOutput)  = dlsym(handle, "acDeviceGetIntOutput");
    *(void**)(&acDeviceGetRealInput)  = dlsym(handle, "acDeviceGetRealInput");
    *(void**)(&acDeviceGetIntInput)   = dlsym(handle, "acDeviceGetIntInput");
    *(void**)(&acDeviceGetRealOutput) = dlsym(handle, "acDeviceGetRealOutput");
    LOAD_DSYM(acHostMeshRandomize);
    LOAD_DSYM(acHostGridMeshRandomize);
    LOAD_DSYM(acHostMeshDestroy);

    LOAD_DSYM(acVerifyCompatibility);
    LOAD_DSYM(acDeviceLoadRealReduceRes);
    // Runtime functions
    LOAD_DSYM(acKernelFlush);
    LOAD_DSYM(acVBAReset);
    LOAD_DSYM(acVBACreate);
    LOAD_DSYM(acAllocateArrays);
    LOAD_DSYM(acUpdateArrays);
    LOAD_DSYM(acVBADestroy);
    LOAD_DSYM(acRandInitAlt);
    LOAD_DSYM(acRandQuit);
    LOAD_DSYM(acLaunchKernel);
    LOAD_DSYM(acBenchmarkKernel);
    LOAD_DSYM(acLoadStencil);
    LOAD_DSYM(acStoreStencil);
    LOAD_DSYM(acLoadRealUniform);
    LOAD_DSYM(acLoadRealArrayUniform);
    LOAD_DSYM(acLoadReal3Uniform);
    LOAD_DSYM(acLoadIntUniform);
    LOAD_DSYM(acLoadIntUniform);
    LOAD_DSYM(acLoadIntArrayUniform);
    LOAD_DSYM(acLoadBoolUniform);
    LOAD_DSYM(acLoadIntArrayUniform);
    LOAD_DSYM(acLoadInt3Uniform);
    LOAD_DSYM(acStoreRealUniform);
    LOAD_DSYM(acStoreReal3Uniform);
    LOAD_DSYM(acStoreIntUniform);
    LOAD_DSYM(acStoreBoolUniform);
    LOAD_DSYM(acStoreInt3Uniform);
    LOAD_DSYM(acKernelLaunchGetLastTPB);
    LOAD_DSYM(acGetOptimizedKernel);
    LOAD_DSYM(acGetKernelReduceScratchPadSize);
    LOAD_DSYM(acGetKernelReduceScratchPadMinSize);
    LOAD_DSYM(acGetKernels);
    LOAD_DSYM(acGetOptimTPB);
    LOAD_DSYM(acRuntimeQuit);
    LOAD_DSYM(acGetRealScratchpadSize);

    const AcResult is_compatible = acVerifyCompatibility(sizeof(AcMesh), sizeof(AcMeshInfo),
                                                         sizeof(AcCompInfo), NUM_REAL_PARAMS,
                                                         NUM_INT_PARAMS, NUM_BOOL_PARAMS,
                                                         NUM_REAL_ARRAYS, NUM_INT_ARRAYS,
                                                         NUM_BOOL_ARRAYS);
    if (is_compatible == AC_FAILURE) {
        fprintf(stderr, "Library is not compatible\n");
        exit(EXIT_FAILURE);
    }
    return AC_SUCCESS;
}

AcResult
acCloseLibrary()
{
    for(size_t i = 0; i < handles.size(); ++i)
    {
      const int success = (handles[i] != NULL) ? dlclose(handles[i]) : 0;
      if(success) handles[i] = NULL;
      else return AC_FAILURE;
    }
    handles = std::vector<AcLibHandle>{};
    return AC_SUCCESS;
}
