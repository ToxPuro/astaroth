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
import astar.data as ad
import astar.visual as vis
import pylab as plt 
import numpy as np 
import sys

import os
import pandas as pd

##mesh = ad.read.Mesh(500, fdir="/tiara/home/mvaisala/astaroth-code/astaroth_2.0/build/")
##
##print(np.shape(mesh.uu))
##print(np.shape(mesh.lnrho))
##
##uu_tot = np.sqrt(mesh.uu[0]**2.0 + mesh.uu[1]**2.0 + mesh.uu[2]**2.0)
##vis.slices.plot_3(mesh, uu_tot, title = r'$|u|$', bitmap = True, fname = 'uutot')
##
##vis.slices.plot_3(mesh, mesh.lnrho, title = r'$\ln \rho$', bitmap = True, fname = 'lnrho')
##
##print(mesh.minfo.contents)


AC_unit_density  =  1e-17
AC_unit_velocity = 1e5
AC_unit_length   = 1.496e+13


print("sys.argv", sys.argv)


meshdir  = "$HOME/astaroth/build/"


#Example fixed scaling template
if (meshdir == "$HOME/astaroth/build/"):
    rlnrho  = [- 0.08,   0.08]
    rrho    = [  0.93,   1.08]
    rNcol   = [  500.0,  530.0]
    
    ruu_tot = [ 0.0,  0.3]
    ruu_xyz = [-0.3,  0.3]
    
    raa_tot = [ 0.0,  0.14]
    raa_xyz = [-0.14, 0.14]
    
    rbb_tot = [ 0.0, 0.3] 
    rbb_xyz = [-0.3, 0.3] 


if "xtopbound" in sys.argv: 
    for i in range(0, 171):
        mesh = ad.read.Mesh(i, fdir=meshdir) 
        if mesh.ok:
            np.set_printoptions(precision=4, linewidth=150)
            uu_tot = np.sqrt(mesh.uu[0]**2.0 + mesh.uu[1]**2.0 + mesh.uu[2]**2.0)
            print(mesh.lnrho.shape)
            print(range((mesh.lnrho.shape[0]-7),mesh.lnrho.shape[0]))
            print('lnrho', i, mesh.lnrho[(mesh.lnrho.shape[0]-7):mesh.lnrho.shape[0], 20, 100]) 
            print('uux', i, mesh.uu[0][(mesh.lnrho.shape[0]-7):mesh.lnrho.shape[0], 20, 100]) 
            print('uuy', i, mesh.uu[1][(mesh.lnrho.shape[0]-7):mesh.lnrho.shape[0], 20, 100]) 
            print('uuz', i, mesh.uu[2][(mesh.lnrho.shape[0]-7):mesh.lnrho.shape[0], 20, 100]) 
            print('uu_tot', i, uu_tot[    (mesh.lnrho.shape[0]-7):mesh.lnrho.shape[0], 20, 100]) 
    

if "single" in sys.argv:
    mesh = ad.read.Mesh(1, fdir=meshdir)
    print(mesh.lnrho.shape)
    
    print( mesh.lnrho[1, 50, 100], 0.0)
    print( mesh.lnrho[197, 50, 100], 0.0)
    print( mesh.lnrho[100, 50, 1], 0.0)
    print( mesh.lnrho[100, 50, 197], 0.0)
    print( mesh.lnrho[100, 1, 100], "periodic")
    print( mesh.lnrho[100, 101, 00], "periodic")

    angle = 0.78
    UUXX = -0.25 * np.cos(angle)
    zorig = 4.85965
    zz = [0.0490874*1.0 - zorig,  0.0490874*100.0 - zorig, 0.0490874*197.0 - zorig]
    print (zz) 
    zz = np.array(zz)
    UUZZ = - 0.25*np.sin(angle)*np.tanh(zz/0.2)
    #plt.plot(np.linspace(-5.0, 5.0, num=100),- (0.25*np.sin(angle))*np.tanh(np.linspace(-5.0, 5.0, num=100)/0.2)) 
    #plt.show()
    print("---- UUX")
    print( mesh.uu[0][1, 50, 100], 0.0)
    print( mesh.uu[0][197, 50, 100], UUXX)
    print( mesh.uu[0][100, 50, 1], UUXX)
    print( mesh.uu[0][100, 50, 197], UUXX)
    print( mesh.uu[0][100, 1, 100], "periodic")
    print( mesh.uu[0][100, 101, 00], "periodic")
    print("---- UUY")
    print( mesh.uu[1][1, 50, 100], 0.0)
    print( mesh.uu[1][197, 50, 100], 0.0)
    print( mesh.uu[1][100, 50, 1], 0.0)
    print( mesh.uu[1][100, 50, 197], 0.0)
    print( mesh.uu[1][100, 1, 100], "periodic")
    print( mesh.uu[1][100, 101, 00], "periodic")
    print("---- UUZ")
    print( mesh.uu[2][1, 50, 100], 0.0)
    print( mesh.uu[2][197, 50, 100], UUZZ[1])
    print( mesh.uu[2][100, 50, 1],   UUZZ[0])
    print( mesh.uu[2][100, 50, 197], UUZZ[2])
    print( mesh.uu[2][100, 1, 100], "periodic")
    print( mesh.uu[2][100, 101, 00], "periodic")

if 'xline' in sys.argv:
    mesh = ad.read.Mesh(0, fdir=meshdir)
    plt.figure()
    plt.plot(mesh.uu[0][100, 50, :] , label="z")
    plt.plot(mesh.uu[0][100, :, 100], label="x")
    plt.plot(mesh.uu[0][:, 50, 100] , label="y")
    plt.legend()

    plt.figure()
    plt.plot(mesh.uu[0][197, 50, :] , label="z edge")

    plt.figure()
    plt.plot(mesh.uu[1][100, 50, :] , label="z")
    plt.plot(mesh.uu[1][100, :, 100], label="x")
    plt.plot(mesh.uu[1][:, 50, 100] , label="y")
    plt.legend()

    plt.figure()
    plt.plot(mesh.uu[2][100, 50, :] , label="z")
    plt.plot(mesh.uu[2][100, :, 100], label="x")
    plt.plot(mesh.uu[2][:, 50, 100] , label="y")
    plt.legend()
    plt.show()

if 'check' in sys.argv:
    mesh = ad.read.Mesh(0, fdir=meshdir)
    vis.slices.plot_3(mesh, mesh.lnrho, title = r'$\ln \rho$', bitmap = False, fname = 'lnrho', contourplot = True)
    plt.show()



if 'diff' in sys.argv:
    mesh0 = ad.read.Mesh(1, fdir=meshdir)
    mesh1 = ad.read.Mesh(2, fdir=meshdir)
    vis.slices.plot_3(mesh1, mesh1.lnrho - mesh0.lnrho, title = r'$\ln \rho$', bitmap = True, fname = 'lnrho')
    vis.slices.plot_3(mesh1, mesh1.uu[0] - mesh0.uu[0], title = r'$u_x$',      bitmap = True, fname = 'uux')
    vis.slices.plot_3(mesh1, mesh1.uu[1] - mesh0.uu[1], title = r'$u_y$',      bitmap = True, fname = 'uuy')
    vis.slices.plot_3(mesh1, mesh1.uu[2] - mesh0.uu[2], title = r'$u_z$',      bitmap = True, fname = 'uuz')

if '1d' in sys.argv:
    plt.figure()
    for i in range(0, 100001, 1000):
        mesh = ad.read.Mesh(i, fdir=meshdir) 
        if mesh.ok:

            if 'lnrho' in sys.argv:
                plt.plot(mesh.lnrho[:, 20, 100], label=i)
            elif 'uux' in sys.argv:
                plt.plot(mesh.uu[0][:, 20, 100], label=i)
            elif 'uuy' in sys.argv:
                plt.plot(mesh.uu[1][:, 20, 100], label=i)
            elif 'uuz' in sys.argv:
                plt.plot(mesh.uu[2][:, 20, 100], label=i)
            elif 'uutot' in sys.argv:
                uu_tot = np.sqrt(mesh.uu[0]**2.0 + mesh.uu[1]**2.0 + mesh.uu[2]**2.0)
                plt.plot(uu_tot[:, 20, 100], label=i)
 
            plt.legend()

    plt.show()


if 'sl' in sys.argv:
    mesh_file_numbers = ad.read.parse_directory(meshdir)
    print(mesh_file_numbers)
    maxfiles = np.amax(mesh_file_numbers)
 
    for i in mesh_file_numbers:
        mesh = ad.read.Mesh(i, fdir=meshdir) 
        print(" %i / %i" % (i, maxfiles))
        if mesh.ok:
            uu_tot = np.sqrt(mesh.uu[0]**2.0 + mesh.uu[1]**2.0 + mesh.uu[2]**2.0)
            aa_tot = np.sqrt(mesh.aa[0]**2.0 + mesh.aa[1]**2.0 + mesh.aa[2]**2.0)
            mesh.Bfield()
            bb_tot = np.sqrt(mesh.bb[0]**2.0 + mesh.bb[1]**2.0 + mesh.bb[2]**2.0)

            if 'sym' in sys.argv:
                rlnrho  = [np.amin(mesh.lnrho), np.amax(mesh.lnrho)]
                rrho    = [  np.exp(rlnrho[0]),   np.exp(rlnrho[1])]
                rNcol   = [                0.0,                 1.0]
                ruu_tot = [    np.amin(uu_tot),     np.amax(uu_tot)]
                maxucomp = np.amax([np.amax(np.abs(mesh.uu[0])), np.amax(np.abs(mesh.uu[1])), np.amax(np.abs(mesh.uu[2]))])
                maxacomp = np.amax([np.amax(np.abs(mesh.aa[0])), np.amax(np.abs(mesh.aa[1])), np.amax(np.abs(mesh.aa[2]))])
                maxbcomp = np.amax([np.amax(np.abs(mesh.bb[0])), np.amax(np.abs(mesh.bb[1])), np.amax(np.abs(mesh.bb[2]))])
                ruu_xyz = [-maxucomp, maxucomp]
                raa_tot = [    np.amin(aa_tot),     np.amax(aa_tot)]
                raa_xyz = [-maxacomp, maxacomp]
                rbb_tot = [    np.amin(bb_tot),     np.amax(bb_tot)]
                rbb_xyz = [-maxbcomp, maxbcomp]

            if ('lim' in sys.argv) or ('sym' in sys.argv):
                vis.slices.plot_3(mesh, mesh.lnrho,         title = r'$\ln \rho$', bitmap = True, fname = 'lnrho',      colrange=rlnrho)
                vis.slices.plot_3(mesh, np.exp(mesh.lnrho), title = r'$\rho$', bitmap = True, fname = 'rho',            colrange=rrho)
                vis.slices.plot_3(mesh, np.exp(mesh.lnrho), title = r'$N_\mathrm{col}$', bitmap = True, fname = 'colden', slicetype = 'sum', colrange=rNcol)
                vis.slices.plot_3(mesh, uu_tot,             title = r'$|u|$', bitmap = True, fname = 'uutot',           colrange=ruu_tot)
                vis.slices.plot_3(mesh, mesh.uu[0],         title = r'$u_x$', bitmap = True, fname = 'uux',             colrange=ruu_xyz)
                vis.slices.plot_3(mesh, mesh.uu[1],         title = r'$u_y$', bitmap = True, fname = 'uuy',             colrange=ruu_xyz)
                vis.slices.plot_3(mesh, mesh.uu[2],         title = r'$u_z$', bitmap = True, fname = 'uuz',             colrange=ruu_xyz)
                vis.slices.plot_3(mesh, aa_tot,             title = r'$\|A\|$', bitmap = True, fname = 'aatot',         colrange=raa_tot)
                vis.slices.plot_3(mesh, mesh.aa[0],         title = r'$A_x$', bitmap = True, fname = 'aax',             colrange=raa_xyz)
                vis.slices.plot_3(mesh, mesh.aa[1],         title = r'$A_y$', bitmap = True, fname = 'aay',             colrange=raa_xyz)
                vis.slices.plot_3(mesh, mesh.aa[2],         title = r'$A_z$', bitmap = True, fname = 'aaz',             colrange=raa_xyz)
                #vis.slices.plot_3(mesh, mesh.accretion,     title = r'$Accretion$', bitmap = True, fname = 'accretion', colrange=[0.0,0.000001])
                vis.slices.plot_3(mesh, bb_tot,             title = r'$\|B\|$', bitmap = True, fname = 'bbtot',         colrange=rbb_tot, trimghost=3)
                vis.slices.plot_3(mesh, mesh.bb[0],         title = r'$B_x$', bitmap = True, fname = 'bbx',             colrange=rbb_xyz, trimghost=3)#, bfieldlines=True)
                vis.slices.plot_3(mesh, mesh.bb[1],         title = r'$B_y$', bitmap = True, fname = 'bby',             colrange=rbb_xyz, trimghost=3)#, bfieldlines=True)
                vis.slices.plot_3(mesh, mesh.bb[2],         title = r'$B_z$', bitmap = True, fname = 'bbz',             colrange=rbb_xyz, trimghost=3)#, bfieldlines=True)
            else: 
                vis.slices.plot_3(mesh, mesh.lnrho,         title = r'$\ln \rho$', bitmap = True, fname = 'lnrho')
                vis.slices.plot_3(mesh, np.exp(mesh.lnrho), title = r'$\rho$', bitmap = True, fname = 'rho')
                #vis.slices.plot_3(mesh, mesh.ss, title = r'$s$', bitmap = True, fname = 'ss')
                vis.slices.plot_3(mesh, mesh.uu[0],         title = r'$u_x$', bitmap = True, fname = 'uux')#, velfieldlines=True)
                vis.slices.plot_3(mesh, mesh.uu[1],         title = r'$u_y$', bitmap = True, fname = 'uuy')
                vis.slices.plot_3(mesh, mesh.uu[2],         title = r'$u_z$', bitmap = True, fname = 'uuz')
                vis.slices.plot_3(mesh, np.exp(mesh.lnrho), title = r'$N_\mathrm{col}$', bitmap = True, fname = 'colden', slicetype = 'sum')
                vis.slices.plot_3(mesh, uu_tot,             title = r'$|u|$', bitmap = True, fname = 'uutot')
                #vis.slices.plot_3(mesh, mesh.accretion,     title = r'$Accretion$', bitmap = True, fname = 'accretion')
                vis.slices.plot_3(mesh, aa_tot,             title = r'$\|A\|$', bitmap = True, fname = 'aatot')
                vis.slices.plot_3(mesh, mesh.aa[0],         title = r'$A_x$', bitmap = True, fname = 'aax')
                vis.slices.plot_3(mesh, mesh.aa[1],         title = r'$A_y$', bitmap = True, fname = 'aay')
                vis.slices.plot_3(mesh, mesh.aa[2],         title = r'$A_z$', bitmap = True, fname = 'aaz')
                vis.slices.plot_3(mesh, bb_tot,             title = r'$\|B\|$', bitmap = True, fname = 'bbtot', trimghost=3)
                vis.slices.plot_3(mesh, mesh.bb[0],         title = r'$B_x$', bitmap = True, fname = 'bbx',     trimghost=3)#, bfieldlines=True)
                vis.slices.plot_3(mesh, mesh.bb[1],         title = r'$B_y$', bitmap = True, fname = 'bby',     trimghost=3)#, bfieldlines=True)
                vis.slices.plot_3(mesh, mesh.bb[2],         title = r'$B_z$', bitmap = True, fname = 'bbz',     trimghost=3)#, bfieldlines=True)
                 

if 'ts' in sys.argv:
   ts = ad.read.TimeSeries(fdir=meshdir)
   vis.lineplot.plot_ts(ts, show_all=True)
   #vis.lineplot.plot_ts(ts, isotherm=True)


if 'getvtk' in sys.argv:
    mesh_file_numbers = ad.read.parse_directory(meshdir)
    print(mesh_file_numbers)
    maxfiles = np.amax(mesh_file_numbers)

    if os.path.exists("grouped.csv"):
        df_archive = pd.read_csv("grouped.csv")
        print(df_archive)
        useBeq = True
    else:
        print("reduced.csv missing!")
        useBeq = False
    

    #for i in mesh_file_numbers[-1:]:
    for i in mesh_file_numbers:
        mesh = ad.read.Mesh(i, fdir=meshdir) 
        resolution = mesh.minfo.contents['AC_nx'      ]
        eta        = mesh.minfo.contents['AC_eta']
        relhel     = mesh.minfo.contents['AC_relhel']
        kk         = (mesh.minfo.contents['AC_kmax']+mesh.minfo.contents['AC_kmin'])/2.0

        if i == mesh_file_numbers[0]:
            if useBeq:
                #MV: Do not use unless you know what you are doing. 
                df_archive = df_archive.loc[df_archive['relhel'] == relhel]
                df_archive = df_archive.loc[df_archive['eta'] == eta]
                df_archive = df_archive.loc[df_archive['resolution'] == resolution]
                df_archive = df_archive.loc[df_archive['kk'] == kk]
                df_archive = df_archive.reset_index()
                print(df_archive)
                uu_eq = df_archive['urms_growth'].values[0]
                print(uu_eq)
                myBeq = np.sqrt(1.0*1.0)*uu_eq
                print(myBeq)
            else:
                myBeq = 1.0
        

        print(" %i / %i" % (i, maxfiles))
        if mesh.ok:
            #mesh.Bfield()
            mesh.export_vtk_ascii(Beq = myBeq)
