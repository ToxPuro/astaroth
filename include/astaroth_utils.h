/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Functions for loading and updating AcMeshInfo.
 *
 */
#pragma once
#include "astaroth.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

typedef struct {
    AcReal model;
    AcReal candidate;
    long double abs_error;
    long double ulp_error;
    long double rel_error;
    AcReal maximum_magnitude;
    AcReal minimum_magnitude;
    int x;
    int y;
    int z;
} Error;

#if AC_RUNTIME_COMPILATION

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) static return_type (*func_name) __VA_ARGS__
#endif

#else

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) return_type func_name __VA_ARGS__
#endif
#endif

/** Loads data from the config file */
FUNC_DEFINE(AcResult, acLoadConfig, (const char* config_path, AcMeshInfo* config));

/** */
FUNC_DEFINE(AcResult, acHostVertexBufferSet,(const VertexBufferHandle handle, const AcReal value, AcMesh* mesh));

/** */
FUNC_DEFINE(AcResult, acHostMeshSet,(const AcReal value, AcMesh* mesh));

/** */
FUNC_DEFINE(AcResult, acHostMeshApplyPeriodicBounds,(AcMesh* mesh));

/** */
FUNC_DEFINE(AcResult, acHostMeshApplyConstantBounds,(const AcReal value, AcMesh* mesh));

/** */
FUNC_DEFINE(AcResult, acHostMeshClear,(AcMesh* mesh));

/** Applies a full integration step on host mesh using the compact 2N RK3 scheme. The boundaries are
 * not updated after the final substep. A call to acHostMeshApplyPeriodicBounds is required if this
 * is not desired. NOTE: applies boundary conditions on the mesh before the initial substep. */
FUNC_DEFINE(AcResult, acHostIntegrateStep,(AcMesh mesh, const AcReal dt));

/** */
FUNC_DEFINE(AcReal, acHostReduceScal,(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a));

/** */
FUNC_DEFINE(AcReal, acHostReduceVec,(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b, const VertexBufferHandle c));
/** */
FUNC_DEFINE(AcReal, acHostReduceVecScal,(const AcMesh mesh, const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b, const VertexBufferHandle c, const VertexBufferHandle d));

FUNC_DEFINE(bool, acEvalError,(const char* label, const Error error));

FUNC_DEFINE(AcResult, acVerifyMesh,(const char* label, const AcMesh model, const AcMesh candidate));
//AcResult (*acVerifyMesh)(const char*, const AcMesh, const AcMesh);

FUNC_DEFINE(AcResult, acMeshDiffWriteSliceZ,(const char* path, const AcMesh model, const AcMesh candidate, const size_t z));

FUNC_DEFINE(AcResult, acMeshDiffWrite,(const char* path, const AcMesh model, const AcMesh candidate));

FUNC_DEFINE(AcResult, acHostMeshWriteToFile,(const AcMesh mesh, const size_t id));

FUNC_DEFINE(AcResult, acHostMeshReadFromFile,(const size_t id, AcMesh* mesh));

FUNC_DEFINE(Error, acGetError,(const AcReal model, const AcReal candidate));

#if AC_RUNTIME_COMPILATION
#include "astaroth_lib.h"
static AcLibHandle __attribute__((unused)) acLoadUtils()
{
 	void* handle = dlopen(runtime_astaroth_utils_path,RTLD_NOW);
	if(!handle)
	{
    		fprintf(stderr,"%s","Fatal error was not able to load Astaroth utils\n"); 
		exit(EXIT_FAILURE);
	}
	*(void**)(&acLoadConfig) = dlsym(handle,"acLoadConfig");
	if(!acLoadConfig) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLoadConfig");
	*(void**)(&acHostVertexBufferSet) = dlsym(handle,"acHostVertexBufferSet");
	if(!acHostVertexBufferSet) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostVertexBufferSet");
	*(void**)(&acHostMeshSet) = dlsym(handle,"acHostMeshSet");
	if(!acHostMeshSet) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshSet");
	*(void**)(&acHostMeshApplyPeriodicBounds) = dlsym(handle,"acHostMeshApplyPeriodicBounds");
	if(!acHostMeshApplyPeriodicBounds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshApplyPeriodicBounds");
	*(void**)(&acHostMeshApplyConstantBounds) = dlsym(handle,"acHostMeshApplyConstantBounds");
	if(!acHostMeshApplyConstantBounds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshApplyConstantBounds");
	*(void**)(&acHostMeshClear) = dlsym(handle,"acHostMeshClear");
	if(!acHostMeshClear) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshClear");
	*(void**)(&acHostReduceScal) = dlsym(handle,"acHostReduceScal");
	if(!acHostReduceScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostReduceScal");
	*(void**)(&acHostReduceVec) = dlsym(handle,"acHostReduceVec");
	if(!acHostReduceVec) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostReduceVec");
	*(void**)(&acHostReduceVecScal) = dlsym(handle,"acHostReduceVecScal");
	if(!acHostReduceVecScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostReduceVecScal");
	*(void**)(&acEvalError) = dlsym(handle,"acEvalError");
	if(!acEvalError) fprintf(stderr,"Astaroth error: was not able to load %s\n","acEvalError");
	*(void**)(&acVerifyMesh) = dlsym(handle,"acVerifyMesh");
	if(!acVerifyMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVerifyMesh");
	*(void**)(&acMeshDiffWriteSliceZ) = dlsym(handle,"acMeshDiffWriteSliceZ");
	if(!acMeshDiffWriteSliceZ) fprintf(stderr,"Astaroth error: was not able to load %s\n","acMeshDiffWriteSliceZ");
	*(void**)(&acMeshDiffWrite) = dlsym(handle,"acMeshDiffWrite");
	if(!acMeshDiffWrite) fprintf(stderr,"Astaroth error: was not able to load %s\n","acMeshDiffWrite");
	*(void**)(&acHostMeshWriteToFile) = dlsym(handle,"acHostMeshWriteToFile");
	if(!acHostMeshWriteToFile) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshWriteToFile");
	*(void**)(&acHostMeshReadFromFile) = dlsym(handle,"acHostMeshReadFromFile");
	if(!acHostMeshReadFromFile) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshReadFromFile");
	*(void**)(&acGetError) = dlsym(handle,"acGetError");
	if(!acGetError) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetError");

	*(void**)(&acHostIntegrateStep) = dlsym(handle,"acHostIntegrateStep");
	if(!acHostIntegrateStep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostIntegrateStep");

//#ifdef __cplusplus
//	return AcLibHandle(handle);
//#else
//	return handle;
//#endif
	return handle;
}
#endif
#ifdef __cplusplus
} // extern "C"
#endif

#define AC_RED   "\x1B[31m"
#define AC_GRN   "\x1B[32m"
#define AC_YEL   "\x1B[33m"
#define AC_BLU   "\x1B[34m"
#define AC_MAG   "\x1B[35m"
#define AC_CYN   "\x1B[36m"
#define AC_WHT   "\x1B[37m"
#define AC_COL_RESET "\x1B[0m"
