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

df_ts_snapshots = pd.DataFrame()

mesh_file_numbers, xsplit, ysplit, zsplit = ad.read.parse_directory(meshdir)
print(mesh_file_numbers)
print(xsplit, ysplit, zsplit)
maxfiles = np.amax(mesh_file_numbers)
#for i in mesh_file_numbers[:5]:

get_window = True
for i in mesh_file_numbers:
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

    if get_window:
        #Base on the test, this is correct!
        xx = mesh.xx - mesh.minfo.contents['AC_center_x'] - 3.0*mesh.minfo.contents['AC_dsx'] 
        yy = mesh.yy - mesh.minfo.contents['AC_center_y'] - 3.0*mesh.minfo.contents['AC_dsy'] 
        zz = mesh.zz - mesh.minfo.contents['AC_center_z'] - 3.0*mesh.minfo.contents['AC_dsz']
        xx = xx[3:-3] 
        yy = yy[3:-3] 
        zz = zz[3:-3] 

        xx_alt = (np.arange(mesh.minfo.contents['AC_mx']) - 3)*mesh.minfo.contents['AC_dsx'] 
        yy_alt = (np.arange(mesh.minfo.contents['AC_my']) - 3)*mesh.minfo.contents['AC_dsy'] 
        zz_alt = (np.arange(mesh.minfo.contents['AC_mz']) - 3)*mesh.minfo.contents['AC_dsz']

        xx_alt = xx_alt - mesh.minfo.contents['AC_center_x']
        yy_alt = yy_alt - mesh.minfo.contents['AC_center_y']
        zz_alt = zz_alt - mesh.minfo.contents['AC_center_z']

        xcomp = mesh.xx - mesh.minfo.contents['AC_center_x'] - 3.0*mesh.minfo.contents['AC_dsx']

        print("xx_alt") 
        print(xx_alt) 
        print("xx = xx[3:-3]")
        print(xx)
        print("mesh.xx - 3.0*mesh.minfo.contents['AC_dsx']")
        print(xcomp)
        #break

        xx_grid, yy_grid, zz_grid = np.meshgrid(xx, yy, zz)

        xx_grid = xx_grid.astype(np.longdouble) 
        yy_grid = yy_grid.astype(np.longdouble) 
        zz_grid = zz_grid.astype(np.longdouble) 

        rr_grid = np.sqrt(xx_grid**2.0 + yy_grid**2.0 + zz_grid**2.0)


        rscale = mesh.minfo.contents['AC_window_radius']
       
        window_radial = np.zeros_like(rr_grid)
        window_radial[np.where(rr_grid <= rscale)] = 1.0

        window_gaussian = np.exp(-(rr_grid/rscale)**2.0)

        get_window = False

    max_uu_x_wg = np.amax(mesh.uu[0].astype(np.longdouble)*window_gaussian) 
    min_uu_x_wg = np.amin(mesh.uu[0].astype(np.longdouble)*window_gaussian) 
    sum_uu_x_wg =  np.sum(mesh.uu[0].astype(np.longdouble)*window_gaussian) 
    max_uu_x_wl = np.amax(mesh.uu[0].astype(np.longdouble)*window_radial) 
    min_uu_x_wl = np.amin(mesh.uu[0].astype(np.longdouble)*window_radial) 
    sum_uu_x_wl =  np.sum(mesh.uu[0].astype(np.longdouble)*window_radial) 

    max_uu_y_wg = np.amax(mesh.uu[1].astype(np.longdouble)*window_gaussian) 
    min_uu_y_wg = np.amin(mesh.uu[1].astype(np.longdouble)*window_gaussian) 
    sum_uu_y_wg =  np.sum(mesh.uu[1].astype(np.longdouble)*window_gaussian) 
    max_uu_y_wl = np.amax(mesh.uu[1].astype(np.longdouble)*window_radial) 
    min_uu_y_wl = np.amin(mesh.uu[1].astype(np.longdouble)*window_radial) 
    sum_uu_y_wl =  np.sum(mesh.uu[1].astype(np.longdouble)*window_radial) 

    max_uu_z_wg = np.amax(mesh.uu[2].astype(np.longdouble)*window_gaussian) 
    min_uu_z_wg = np.amin(mesh.uu[2].astype(np.longdouble)*window_gaussian) 
    sum_uu_z_wg =  np.sum(mesh.uu[2].astype(np.longdouble)*window_gaussian) 
    max_uu_z_wl = np.amax(mesh.uu[2].astype(np.longdouble)*window_radial) 
    min_uu_z_wl = np.amin(mesh.uu[2].astype(np.longdouble)*window_radial) 
    sum_uu_z_wl =  np.sum(mesh.uu[2].astype(np.longdouble)*window_radial) 

    column_names = ["time",
                    "max_lnrho", "min_lnrho", "rms_lnrho", 
                    "max_uu_x",  "max_uu_y",  "max_uu_z", 
                    "min_uu_x",  "min_uu_y",  "min_uu_z", 
                    "rms_uu_x",  "rms_uu_y",  "rms_uu_z", 
                    "max_aa_x",  "max_aa_y",  "max_aa_z", 
                    "min_aa_x",  "min_aa_y",  "min_aa_z", 
                    "rms_aa_x",  "rms_aa_y",  "rms_aa_z", 
                    "max_bb_x",  "max_bb_y",  "max_bb_z", 
                    "min_bb_x",  "min_bb_y",  "min_bb_z", 
                    "rms_bb_x",  "rms_bb_y",  "rms_bb_z",
                    "max_uu_x_wg", "max_uu_x_wl", 
                    "sum_uu_x_wg", "sum_uu_x_wl", 
                    "min_uu_x_wg", "min_uu_x_wl", 
                    "max_uu_y_wg", "max_uu_y_wl", 
                    "sum_uu_y_wg", "sum_uu_y_wl", 
                    "min_uu_y_wg", "min_uu_y_wl", 
                    "max_uu_z_wg", "max_uu_z_wl", 
                    "sum_uu_z_wg", "sum_uu_z_wl", 
                    "min_uu_z_wg", "min_uu_z_wl"] 

    values_line = [[mesh.timestamp,
                    max_lnrho, min_lnrho, rms_lnrho,
                    max_uu_x , max_uu_y,  max_uu_z, 
                    min_uu_x,  min_uu_y,  min_uu_z, 
                    rms_uu_x,  rms_uu_y,  rms_uu_z, 
                    max_aa_x,  max_aa_y,  max_aa_z, 
                    min_aa_x,  min_aa_y,  min_aa_z, 
                    rms_aa_x,  rms_aa_y,  rms_aa_z, 
                    max_bb_x,  max_bb_y,  max_bb_z, 
                    min_bb_x,  min_bb_y,  min_bb_z, 
                    rms_bb_x,  rms_bb_y,  rms_bb_z,
                    max_uu_x_wg, max_uu_x_wl,  
                    sum_uu_x_wg, sum_uu_x_wl,  
                    min_uu_x_wg, min_uu_x_wl,  
                    max_uu_y_wg, max_uu_y_wl,  
                    sum_uu_y_wg, sum_uu_y_wl,  
                    min_uu_y_wg, min_uu_y_wl,  
                    max_uu_z_wg, max_uu_z_wl,  
                    sum_uu_z_wg, sum_uu_z_wl,  
                    min_uu_z_wg, min_uu_z_wl]] 

    df_line = pd.DataFrame(values_line,  
                           columns=column_names)

    print(df_line)
    df_ts_snapshots = pd.concat([df_ts_snapshots, df_line])


SimulationDiagnostics =  ad.read.TimeSeries(pandas=True)

print(SimulationDiagnostics.ts_dataframe)

print(df_ts_snapshots)

plt.figure()
plt.plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_max'], '-')
plt.plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_x'], 'o')

plt.figure()
plt.plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_min'], '-')  
plt.plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_x'], 'o')

plt.figure()
plt.plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_rms'], '-')  
plt.plot(df_ts_snapshots['time'], df_ts_snapshots['rms_uu_x'], 'o')

plt.show()

fig, axs = plt.subplots(3,3, figsize=(16.0, 9.0))

axs[0,0].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_max_wg'], '-', label='VTXBUF_UUX_max_wg')
axs[0,0].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_max_wl'], '--', label='VTXBUF_UUX_max_wl')
axs[0,0].plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_x_wg'], 'o', label='max_uu_x_wg')
axs[0,0].plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_x_wl'], 'x', label='max_uu_x_wl')
axs[0,0].legend()

axs[0,1].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUY_max_wg'], '-', label='VTXBUF_UUY_max_wg')
axs[0,1].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUY_max_wl'], '--', label='VTXBUF_UUY_max_wl')
axs[0,1].plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_y_wg'], 'o', label='max_uu_y_wg')
axs[0,1].plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_y_wl'], 'x', label='max_uu_y_wl')
axs[0,1].legend()

axs[0,2].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUZ_max_wg'], '-', label='VTXBUF_UUZ_max_wg')
axs[0,2].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUZ_max_wl'], '--', label='VTXBUF_UUZ_max_wl')
axs[0,2].plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_z_wg'], 'o', label='max_uu_z_wg')
axs[0,2].plot(df_ts_snapshots['time'], df_ts_snapshots['max_uu_z_wl'], 'x', label='max_uu_z_wl')
axs[0,2].legend()

axs[1,0].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_min_wg'], '-', label='VTXBUF_UUX_min_wg')
axs[1,0].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_min_wl'], '--', label='VTXBUF_UUX_min_wl')
axs[1,0].plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_x_wg'], 'o', label='min_uu_x_wg')
axs[1,0].plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_x_wl'], 'x', label='min_uu_x_wl')
axs[1,0].legend()

axs[1,1].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUY_min_wg'], '-', label='VTXBUF_UUY_min_wg')
axs[1,1].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUY_min_wl'], '--', label='VTXBUF_UUY_min_wl')
axs[1,1].plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_y_wg'], 'o', label='min_uu_y_wg')
axs[1,1].plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_y_wl'], 'x', label='min_uu_y_wl')
axs[1,1].legend()

axs[1,2].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUZ_min_wg'], '-', label='VTXBUF_UUZ_min_wg')
axs[1,2].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUZ_min_wl'], '--', label='VTXBUF_UUZ_min_wl')
axs[1,2].plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_z_wg'], 'o', label='min_uu_z_wg')
axs[1,2].plot(df_ts_snapshots['time'], df_ts_snapshots['min_uu_z_wl'], 'x', label='min_uu_z_wl')
axs[1,2].legend()

axs[2,0].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_sum_wg'], '-', label='VTXBUF_UUX_sum_wg')
axs[2,0].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUX_sum_wl'], '--', label='VTXBUF_UUX_sum_wl')
axs[2,0].plot(df_ts_snapshots['time'], df_ts_snapshots['sum_uu_x_wg'], 'o', label='sum_uu_x_wg')
axs[2,0].plot(df_ts_snapshots['time'], df_ts_snapshots['sum_uu_x_wl'], 'x', label='sum_uu_x_wl')
axs[2,0].legend()

axs[2,1].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUY_sum_wg'], '-', label='VTXBUF_UUY_sum_wg')
axs[2,1].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUY_sum_wl'], '--', label='VTXBUF_UUY_sum_wl')
axs[2,1].plot(df_ts_snapshots['time'], df_ts_snapshots['sum_uu_y_wg'], 'o', label='sum_uu_y_wg')
axs[2,1].plot(df_ts_snapshots['time'], df_ts_snapshots['sum_uu_y_wl'], 'x', label='sum_uu_y_wl')
axs[2,1].legend()

axs[2,2].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUZ_sum_wg'], '-', label='VTXBUF_UUZ_sum_wg')
axs[2,2].plot(SimulationDiagnostics.ts_dataframe['t_step'] , SimulationDiagnostics.ts_dataframe['VTXBUF_UUZ_sum_wl'], '--', label='VTXBUF_UUZ_sum_wl')
axs[2,2].plot(df_ts_snapshots['time'], df_ts_snapshots['sum_uu_z_wg'], 'o', label='sum_uu_z_wg')
axs[2,2].plot(df_ts_snapshots['time'], df_ts_snapshots['sum_uu_z_wl'], 'x', label='sum_uu_z_wl')
axs[2,2].legend()

plt.tight_layout()

plt.show()



