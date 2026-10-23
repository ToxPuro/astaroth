/*
    Copyright (C) 2014-2026, Johannes Pekkila, Touko Puro, Miikka Vaisala.

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
#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ac_mpi.h"
#include "acc_runtime.h"
#include "acreal.h"
#include "astaroth_analysis.h"
#include "astaroth_base.h"
#include "astaroth_device.h"
// The device headers are not needed for interfacing with Astaroth
#include "astaroth_device_headers.h"
#include "astaroth_grid.h"
#include "astaroth_helpers.h"
#include "astaroth_legacy.h"
#include "astaroth_logging.h"
#include "astaroth_node.h"
#include "astaroth_runtime_compilation.h"
#include "builtin_enums.h"
#include "errchk.h"
#include "func_define.h"
#include "host_datatypes.h"

// clang-format off
#include "user_defines.h"
// clang-format on


AC_BEGIN_C_DECLARATIONS

int3
acDecompose(const uint64_t target, const AcMeshInfo info);
int3
acGetPid3D(const uint64_t pid, const int3 decomp, const AcMeshInfo info);
int
acGetPid(const int3 pid, const int3 decomp, const AcMeshInfo info);

#include "get_vtxbufs_declares.h"

AcResult
acLoadLibrary(const AcMeshInfo info);
AcResult
acCloseLibrary();

static UNUSED AcCompInfo
acInitCompInfo()
{
    AcCompInfo res;
    // TP: initially nothing is loaded and if they are not loaded their values
    // might as well be zero since then a default value is used for them
    memset(&res, 0, sizeof(res));
    return res;
}

static UNUSED AcMeshInfo
acInitInfo()
{
    AcMeshInfo res;
    // TP: this is useful for the following reasons:
    // All enums are initialized by default to the first enum value
    // All array ptrs are initialized to nulls
    // All booleans are initialized to false
    // All booleans about whether values are loaded are false
    memset(&res, 0, sizeof(res));
    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    // TP: for backwards compatibility set original datatypes to all ones as before
    memset(&res.int_params, (uint8_t)0xFF, sizeof(res.int_params));
    memset(&res.real_params, (uint8_t)0xFF, sizeof(res.real_params));
    memset(&res.int3_params, (uint8_t)0xFF, sizeof(res.int3_params));
    memset(&res.real3_params, (uint8_t)0xFF, sizeof(res.real3_params));
    memset(&res.complex_params, (uint8_t)0xFF, sizeof(res.complex_params));

#if AC_MPI_ENABLED
    res.comm         = (AcCommunicator*)malloc(sizeof(AcCommunicator));
    res.comm->handle = MPI_COMM_NULL;
#endif
    res.run_consts = acInitCompInfo();
    return res;
}


#include "ac_push_to_config_decl.h"


static UNUSED AcMesh
acInitMesh()
{
    AcMesh res;
    for (size_t j = 0; j < NUM_VTXBUF_HANDLES; ++j) {
        res.vertex_buffer[j] = NULL;
    }
    res.info = acInitInfo();
    return res;
}

static const char*
acGetDefaultConfig()
{
    // if (filepath == NULL) return NULL;

    return AC_DEFAULT_CONFIG;
    // return AC_SUCCESS;
}

AC_END_C_DECLARATIONS

#ifdef __cplusplus

template <typename P, typename V>
void
acPushToConfig(AcMeshInfo& config, P param, V val)
{
	(void)param;
	(void)val;
        if constexpr(IsCompParam<P>())
        {
        	  config.run_consts.config[param] = val;
        	  config.run_consts.is_loaded[param] = true;
        }
        else if constexpr(IsParam<P>())
        {
      	  config[param] = val;
      	  config.is_loaded[param] = true;
        }
}

static AcResult
acUpdateDecompositionParams(AcMeshInfo* dst)
{
#if AC_MPI_ENABLED
	int nprocs{};
	int rank{};
	ERRCHK_ALWAYS(dst->comm != NULL && dst->comm->handle != MPI_COMM_NULL);
	MPI_Comm_size(dst->comm->handle,&nprocs);
	MPI_Comm_rank(dst->comm->handle,&rank);
	const int3 decomp = acDecompose(nprocs,*dst);
	const int3 pid3d = acGetPid3D(rank,decomp,*dst);
	acPushToConfig((*dst),AC_domain_coordinates,pid3d);
	acPushToConfig((*dst),AC_domain_decomposition,decomp);
	return AC_SUCCESS;
#else
	return AC_FAILURE;
#endif
}

#endif
