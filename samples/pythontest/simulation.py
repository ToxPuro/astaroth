

'''
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

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
'''

# Example draft of who simulation mode would run in Python
# Based on astaroth/samples/standalone_mpi/main.cc 

from mpi4py import MPI # For booting up MPI in Python.
import astaroth        # Astaroth API invocation as a C extension.
import actools         # Python library for helpful host-level tools.

import numpy           # Other useful Python libraries. 

pid = MPI.COMM_WORLD.Get_rank()

if pid == 0:
    astaroth.acMeshCreate(mesh.info, mesh.mesh)
    actools.acmesh_init_to(mesh)

# STREAM_DEFAULT, RTYPE_MIN, RTYPE_MAX, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY,
# VTXBUF_UUZ, not written in pythonic way yet 
astaroth.acGridInit(mesh.info)
astaroth.acGridLoadMesh(STREAM_DEFAULT, mesh.mesh)

for t_step in range(0, 100): 
    dt = actools.calc_timestep(mesh)
    astaroth.cGridIntegrate(STREAM_DEFAULT, dt)

    astaroth.acGridReduceVec(STREAM_DEFAULT, RTYPE_MIN, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, uumin)
    astaroth.acGridReduceVec(STREAM_DEFAULT, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, uumax)
    astaroth.acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, uurms)

    print("Step %d, dt: %g\n" % (t_step, dt))
    print("%*s: %.3e, %.3e, %.3e\n UU" % (uumin, uumax, uurms))

    for vtxbuf in mesh.VTXBUF_HANDLES: #mesh.VTXBUF_HANDLES would be a list of handles
        astaroth.acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, vtxbuf, scalmin)
        astaroth.acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, vtxbuf, scalmax)
        astaroth.acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, vtxbuf, scalrms)

        print("%*s: %.3e, %.3e, %.3e\n" % (mesh.vtxbuf_names(vtxbuf), scalmin, scalmax, scalrms))

if (pid == 0)
    astaroth.acMeshDestroy(mesh.mesh)

astaroth.acGridQuit()

