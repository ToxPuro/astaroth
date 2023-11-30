'''
    Copyright (C) 2014-2023, Johannes Pekkila, Miikka Vaisala.

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

import astar.data as ad
import astar.visual as vis
import numpy as np 
import pylab as plt 
import pandas as pd

# ---------------------------------------------------------------------------
# This Python script does a number of check on the simulation data. 
# 
# NOTE: Right now it focused on testing special reduction but can be added to
# feature other checks when we need to. 
# ---------------------------------------------------------------------------

#
# This is a test which tests windowed and other special reduction by comparing
# them to equivalent Python based reductions
#

meshdir = 'output-snapshots/'

mesh_file_numbers, xsplit, ysplit, zsplit = ad.read.parse_directory(meshdir)
print(mesh_file_numbers)
print(xsplit, ysplit, zsplit)
maxfiles = np.amax(mesh_file_numbers)
for i in mesh_file_numbers[:2]:
    mesh = ad.read.Mesh(i, fdir=meshdir, xsplit=xsplit, ysplit=ysplit, zsplit=zsplit)
    mesh.Bfield(trim=True)

    max_lnrho = np.amax(mesh.lnrho)
    max_uu_x  = np.amax(mesh.uu[0])
    max_uu_y  = np.amax(mesh.uu[1])
    max_uu_z  = np.amax(mesh.uu[2])
    max_aa_x  = np.amax(mesh.aa[0])
    max_aa_y  = np.amax(mesh.aa[1])
    max_aa_z  = np.amax(mesh.aa[2])
    max_bb_x  = np.amax(mesh.bb[0])
    max_bb_y  = np.amax(mesh.bb[1])
    max_bb_z  = np.amax(mesh.bb[2])

    min_lnrho = np.amin(mesh.lnrho)
    min_uu_x  = np.amin(mesh.uu[0])
    min_uu_y  = np.amin(mesh.uu[1])
    min_uu_z  = np.amin(mesh.uu[2])
    min_aa_x  = np.amin(mesh.aa[0])
    min_aa_y  = np.amin(mesh.aa[1])
    min_aa_z  = np.amin(mesh.aa[2])
    min_bb_x  = np.amin(mesh.bb[0])
    min_bb_y  = np.amin(mesh.bb[1])
    min_bb_z  = np.amin(mesh.bb[2])

    rms_lnrho = np.sqrt(np.mean(mesh.lnrho**2.0))
    rms_uu_x  = np.sqrt(np.mean(mesh.uu[0]**2.0))
    rms_uu_y  = np.sqrt(np.mean(mesh.uu[1]**2.0))
    rms_uu_z  = np.sqrt(np.mean(mesh.uu[2]**2.0))
    rms_aa_x  = np.sqrt(np.mean(mesh.aa[0]**2.0))
    rms_aa_y  = np.sqrt(np.mean(mesh.aa[1]**2.0))
    rms_aa_z  = np.sqrt(np.mean(mesh.aa[2]**2.0))
    rms_bb_x  = np.sqrt(np.mean(mesh.bb[0]**2.0))
    rms_bb_y  = np.sqrt(np.mean(mesh.bb[1]**2.0))
    rms_bb_z  = np.sqrt(np.mean(mesh.bb[2]**2.0))

    print("max_lnrho = %e" % max_lnrho)
    print("min_lnrho = %e" % min_lnrho)
    print("rms_lnrho = %e" % rms_lnrho)

    print("max_uu_x = %e, max_uu_y = %e, max_uu_z = %e" % (max_uu_x, max_uu_y, max_uu_z))
    print("min_uu_x = %e, min_uu_y = %e, min_uu_z = %e" % (min_uu_x, min_uu_y, min_uu_z))
    print("rms_uu_x = %e, rms_uu_y = %e, rms_uu_z = %e" % (rms_uu_x, rms_uu_y, rms_uu_z))

    print("max_aa_x = %e, max_aa_y = %e, max_aa_z = %e" % (max_aa_x, max_aa_y, max_aa_z))
    print("min_aa_x = %e, min_aa_y = %e, min_aa_z = %e" % (min_aa_x, min_aa_y, min_aa_z))
    print("rms_aa_x = %e, rms_aa_y = %e, rms_aa_z = %e" % (rms_aa_x, rms_aa_y, rms_aa_z))

    print("max_bb_x = %e, max_bb_y = %e, max_bb_z = %e" % (max_bb_x, max_bb_y, max_bb_z))
    print("min_bb_x = %e, min_bb_y = %e, min_bb_z = %e" % (min_bb_x, min_bb_y, min_bb_z))
    print("rms_bb_x = %e, rms_bb_y = %e, rms_bb_z = %e" % (rms_bb_x, rms_bb_y, rms_bb_z))

SimulationDiagnostics =  ad.read.TimeSeries(pandas=True)

print(SimulationDiagnostics.ts_dataframe)
