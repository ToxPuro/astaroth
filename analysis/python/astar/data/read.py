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

# This module is for reading data.

import numpy as np
import os

#Optional YT interface
try:
    import yt
    yt_present = True 
except ImportError:
    yt_present = False

def set_dtype(endian, AcRealSize, print_type = True):
    if endian == 0:
        en = '>'
    elif endian == 1: 
        en = '<'
    type_instruction = en + 'f' + str(AcRealSize)
    if print_type:
        print("type_instruction", type_instruction)
    my_dtype = np.dtype(type_instruction)
    return my_dtype

def read_bin(fname, fdir, fnum, minfo, numtype=np.longdouble, getfilename=True):
    '''Read in a floating point array'''
    filename = fdir + fname + '_' + fnum + '.mesh'
    datas = np.DataSource()
    read_ok = datas.exists(filename)

    my_dtype = set_dtype(minfo.contents['endian'], minfo.contents['AcRealSize'], print_type=getfilename)

    if read_ok:
        if getfilename:
            print(filename)
        array = np.fromfile(filename, dtype=my_dtype)

        timestamp = array[0]

        array = np.reshape(array[1:], (minfo.contents['AC_mx'], 
                                       minfo.contents['AC_my'], 
                                       minfo.contents['AC_mz']), order='F')
    else:
        array = None
        timestamp = None
     
    return array, timestamp, read_ok 

def read_meshtxt(fdir, fname, dbg_output):

    with open(fdir+fname) as f:
        filetext = f.read().splitlines()

    contents = {}  

    for line in filetext:
        line = line.split()
        if line[0] == 'int':
            contents[line[1]] = np.int(line[2])
            if dbg_output:
                print(line[1], contents[line[1]])
        elif line[0] == 'size_t':
            contents[line[1]] = np.int(line[2])
            if dbg_output:
                print(line[1], contents[line[1]])
        elif line[0] == 'int3':
            contents[line[1]] = [np.int(line[2]), np.int(line[3]), np.int(line[4])]
            if dbg_output:
                print(line[1], contents[line[1]])
        elif line[0] == 'real':
            contents[line[1]] = np.float(line[2])
            if dbg_output:
                print(line[1], contents[line[1]])
        elif line[0] == 'real3':
            contents[line[1]] = [np.float(line[2]), np.float(line[3]), np.float(line[4])]
            if dbg_output:
                print(line[1], contents[line[1]])
        else: 
            print(line)
            if dbg_output:
                print('ERROR: ' + line[0] +' not recognized!')

    return contents

def parse_directory(meshdir):
    dirlist = os.listdir(meshdir)
    dirlist = [k for k in dirlist if 'LNRHO' in k]
    for i, item in enumerate(dirlist):
        tmp = item.strip('.mesh')
        tmp = tmp.strip('VTXBUF_LNRHO')
        dirlist[i] = int(tmp)
    dirlist.sort()
    return dirlist

def apply_boundcond(array, btype):

    if btype == "p":
        be = 3
        bi = 6 
        # Edges
        # xx
        array[  : 3,  :  ,  :  ] = array[-6:-3,  :  ,  :  ]
        array[-3:  ,  :  ,  :  ] = array[ 3: 6,  :  ,  :  ]
        # yy
        array[  :  ,  : 3,  :  ] = array[  :  ,-6:-3,  :  ]
        array[  :  ,-3:  ,  :  ] = array[  :  , 3: 6,  :  ]
        # zz 
        array[  :  ,  :  ,  : 3] = array[  :  ,  :  ,-6:-3]
        array[  :  ,  :  ,-3:  ] = array[  :  ,  :  , 3: 6]
        # Corner parts
        # xy
        array[  : 3,  : 3,  :  ] = array[-6:-3,-6:-3,  :  ]
        array[-3:  ,-3:  ,  :  ] = array[ 3: 6, 3: 6,  :  ]
        array[-3:  ,  : 3,  :  ] = array[ 3: 6,-6:-3,  :  ]
        array[  : 3,-3:  ,  :  ] = array[-6:-3, 3: 6,  :  ]
        # xz
        array[  : 3,  :  ,  : 3] = array[-6:-3,  :  ,-6:-3]
        array[-3:  ,  :  ,-3:  ] = array[ 3: 6,  :  , 3: 6]
        array[-3:  ,  :  ,  : 3] = array[ 3: 6,  :  ,-6:-3]
        array[  : 3,  :  ,-3:  ] = array[-6:-3,  :  , 3: 6]
        # yz
        array[  :  ,  : 3,  : 3] = array[  :  ,-6:-3,-6:-3]
        array[  :  ,-3:  ,-3:  ] = array[  :  , 3: 6, 3: 6]
        array[  :  ,-3:  ,  : 3] = array[  :  , 3: 6,-6:-3]
        array[  :  ,  : 3,-3:  ] = array[  :  ,-6:-3, 3: 6]
    else:
        print("Unknown btype", btype)

    return array


def DERX(array, dx):
    output = np.zeros_like(array)
    for i in range(3, array.shape[0]-3): #Keep boundary poits as 0
        output[i,:,:] =( -45.0*array[i-1,:,:] + 45.0*array[i+1,:,:]
                         + 9.0*array[i-2,:,:] -  9.0*array[i+2,:,:]
                         -     array[i-3,:,:] +      array[i+3,:,:] )/(60.0*dx)
    return output

def DERY(array, dy):
    output = np.zeros_like(array)
    for i in range(3,array.shape[1]-3):
        output[:,i,:] =( -45.0*array[:,i-1,:] + 45.0*array[:,i+1,:]
                         + 9.0*array[:,i-2,:] -  9.0*array[:,i+2,:]
                         -     array[:,i-3,:] +      array[:,i+3,:] )/(60.0*dy)
    return output

def DERZ(array, dz):
    output = np.zeros_like(array)
    for i in range(3, array.shape[2]-3):
        output[:,:,i] =( -45.0*array[:,:,i-1] + 45.0*array[:,:,i+1]
                         + 9.0*array[:,:,i-2] -  9.0*array[:,:,i+2]
                         -     array[:,:,i-3] +      array[:,:,i+3] )/(60.0*dz)
    return output

def DER2X(array, dx):
    output = np.zeros_like(array)
    for i in range(3, array.shape[0]-3): #Keep boundary poits as 0
        output[i,:,:] =(    2.0*array[i-1,:,:] +   2.0*array[i+1,:,:]
                         - 27.0*array[i-2,:,:] -  27.0*array[i+2,:,:]
                         +270.0*array[i-3,:,:] + 270.0*array[i+3,:,:] 
                         -490.0*array[i  ,:,:]        )/(180.0*dx*dx)
    return output

def DER2Y(array, dy):
    output = np.zeros_like(array)
    for i in range(3,array.shape[1]-3):
        output[:,i,:] =(    2.0*array[:,i-1,:] +   2.0*array[:,i+1,:]
                         - 27.0*array[:,i-2,:] -  27.0*array[:,i+2,:]
                         +270.0*array[:,i-3,:] + 270.0*array[:,i+3,:] 
                         -490.0*array[:,i  ,:]        )/(180.0*dy*dy)
    return output

def DER2Z(array, dz):
    output = np.zeros_like(array)
    for i in range(3, array.shape[2]-3):
        output[:,:,i] =(    2.0*array[:,:,i-1] +   2.0*array[:,:,i+1]
                         - 27.0*array[:,:,i-2] -  27.0*array[:,:,i+2]
                         +270.0*array[:,:,i-3] + 270.0*array[:,:,i+3] 
                         -490.0*array[:,:,i  ]        )/(180.0*dz*dz)
    return output


def curl(aa, minfo):
    dx = minfo.contents['AC_dsx']
    dy = minfo.contents['AC_dsy']
    dz = minfo.contents['AC_dsz']
    return (DERY(aa[2], dy)-DERZ(aa[1], dz), 
            DERZ(aa[0], dz)-DERX(aa[2], dx), 
            DERX(aa[1], dx)-DERY(aa[0], dy))

def div(array, minfo):
    dx = minfo.contents['AC_dsx']
    dy = minfo.contents['AC_dsy']
    dz = minfo.contents['AC_dsz']
    return (  DERX(array[0], dx)  
            + DERY(array[1], dy)  
            + DERZ(array[2], dz))

def grad(array, minfo):
    dx = minfo.contents['AC_dsx']
    dy = minfo.contents['AC_dsy']
    dz = minfo.contents['AC_dsz']
    return (DERX(array, dx), 
            DERY(array, dy), 
            DERZ(array, dz))

def grad_div(array, minfo):
    scalar = div(array, minfo)
    scalar = apply_boundcond(scalar, "p") 
    vec = grad(scalar, minfo)
    return vec

def laplace_scal(array, minfo):
    dx = minfo.contents['AC_dsx']
    dy = minfo.contents['AC_dsy']
    dz = minfo.contents['AC_dsz']
    return (DER2X(array, dx) + DER2Y(array, dy) + DER2Z(array, dz))

def laplace_vec(array, minfo):
    return (laplace_scal(array[0], minfo), 
            laplace_scal(array[1], minfo), 
            laplace_scal(array[2], minfo))

def curl_of_curl(array, minfo):
    array1 = curl(array, minfo) 
    array2 = (apply_boundcond(array1[0], "p"), apply_boundcond(array1[1], "p"), apply_boundcond(array1[2], "p"))
    return curl(array2, minfo)


class MeshInfo():
    '''Object that contains all mesh info'''

    def __init__(self, fdir, dbg_output=False):
        self.contents = read_meshtxt(fdir, 'mesh_info.list', dbg_output) 

class Mesh:
    '''Class tha contains all 3d mesh data'''

    def __init__(self, fnum, fdir="", only_info = False, pdiag = True):
        fnum = str(fnum)
        self.framenum = fnum.zfill(10)

        self.minfo = MeshInfo(fdir)

        if only_info == False:
            self.lnrho, self.timestamp, self.ok = read_bin('VTXBUF_LNRHO', fdir, fnum, self.minfo, getfilename=pdiag)
        else:
            self.ok = False

        if self.ok:

            self.ss, timestamp, ok = read_bin('VTXBUF_ENTROPY', fdir, fnum, self.minfo, getfilename=pdiag)

            self.accretion, timestamp, ok = read_bin('VTXBUF_ACCRETION', fdir, fnum, self.minfo, getfilename=pdiag)
 
            #TODO Generalize is a dict. Do not hardcode!  
            uux, timestamp, ok = read_bin('VTXBUF_UUX', fdir, fnum, self.minfo, getfilename=pdiag)
            uuy, timestamp, ok = read_bin('VTXBUF_UUY', fdir, fnum, self.minfo, getfilename=pdiag) 
            uuz, timestamp, ok = read_bin('VTXBUF_UUZ', fdir, fnum, self.minfo, getfilename=pdiag)
            self.uu = (uux, uuy, uuz)
            uux = []
            uuy = [] 
            uuz = []
 
            aax, timestamp, ok = read_bin('VTXBUF_AX', fdir, fnum, self.minfo, getfilename=pdiag)
            aay, timestamp, ok = read_bin('VTXBUF_AY', fdir, fnum, self.minfo, getfilename=pdiag) 
            aaz, timestamp, ok = read_bin('VTXBUF_AZ', fdir, fnum, self.minfo, getfilename=pdiag)
            self.aa = (aax, aay, aaz)
            aax = []
            aay = [] 
            aaz = []

            self.xx =  np.arange(self.minfo.contents['AC_mx']) * self.minfo.contents['AC_dsx']
            self.yy =  np.arange(self.minfo.contents['AC_my']) * self.minfo.contents['AC_dsy']
            self.zz =  np.arange(self.minfo.contents['AC_mz']) * self.minfo.contents['AC_dsz']

            self.xmid = int(self.minfo.contents['AC_mx']/2)
            self.ymid = int(self.minfo.contents['AC_my']/2)
            self.zmid = int(self.minfo.contents['AC_mz']/2)
    def Bfield(self, get_jj = False, trim=False):
        self.bb = curl(self.aa, self.minfo) 
        if get_jj:
            self.jj = curl_of_curl(self.aa, self.minfo)
        if trim:
            self.bb = (    self.bb[0][3:-3, 3:-3, 3:-3],self.bb[1][3:-3, 3:-3, 3:-3],self.bb[2][3:-3, 3:-3, 3:-3])
            if get_jj:
                self.jj = (self.jj[0][3:-3, 3:-3, 3:-3],self.jj[1][3:-3, 3:-3, 3:-3],self.jj[2][3:-3, 3:-3, 3:-3])

    def get_jj(self, trim=False):
        self.jj = curl_of_curl(self.aa, minfo, trim=False)
        if trim:
            self.jj =     (self.jj[0][3:-3, 3:-3, 3:-3],self.jj[1][3:-3, 3:-3, 3:-3],self.jj[2][3:-3, 3:-3, 3:-3])

    def vorticity(self, trim=False):
        self.oo = curl(self.uu, self.minfo) 
        if trim:
            self.oo =     (self.oo[0][3:-3, 3:-3, 3:-3],self.oo[1][3:-3, 3:-3, 3:-3],self.oo[2][3:-3, 3:-3, 3:-3])
 
    def rad_vel(self):
        print("Calculating spherical velocity components")
        self.uu_pherical = np.zeros_like(self.uu)
        xx, yy, zz = np.meshgrid(self.xx - self.xmid, self.yy - self.ymid, self.zz - self.zmid)       
        rr                         = np.sqrt(xx**2.0 + yy**2.0 + zz**2.0)
        theta                      = np.arccos(zz/rr)
        phi                        = np.arctan2(yy,xx) 
        sin_theta_sin_phi          = np.sin(theta)*np.sin(phi)
        cos_theta_cos_phi          = np.cos(theta)*np.cos(phi)
        sin_theta_cos_phi          = np.sin(theta)*np.cos(phi)
        cos_theta_sin_phi          = np.cos(theta)*np.sin(phi)
        ux = self.uu[0]; uy = self.uu[1]; uz = self.uu[2];
        vr     = sin_theta_cos_phi*ux + sin_theta_sin_phi*uy + np.cos(theta)*uz
        vtheta = cos_theta_cos_phi*ux + cos_theta_sin_phi*uy - np.sin(theta)*uz 
        vphi   =      -np.sin(phi)*ux +       np.cos(phi)*uy 
        self.uu_pherical[0] = vr
        self.uu_pherical[1] = vtheta
        self.uu_pherical[2] = vphi


    def yt_conversion(self):
        if yt_present:
            self.ytdict = dict(density = (np.exp(self.lnrho)*self.minfo.contents['AC_unit_density'], "g/cm**3"),
                               uux     = (self.uu[0]*self.minfo.contents['AC_unit_velocity'], "cm/s"),
                               uuy     = (self.uu[1]*self.minfo.contents['AC_unit_velocity'], "cm/s"),
                               uuz     = (self.uu[2]*self.minfo.contents['AC_unit_velocity'], "cm/s"),
                               bbx     = (self.bb[0]*self.minfo.contents['AC_unit_magnetic'], "gauss"),
                               bby     = (self.bb[1]*self.minfo.contents['AC_unit_magnetic'], "gauss"),
                               bbz     = (self.bb[2]*self.minfo.contents['AC_unit_magnetic'], "gauss"),
                               )
            bbox = self.minfo.contents['AC_unit_length'] \
                   *np.array([[self.xx.min(), self.xx.max()], [self.yy.min(), self.yy.max()], [self.zz.min(), self.zz.max()]])
            self.ytdata = yt.load_uniform_grid(self.ytdict, self.lnrho.shape, length_unit="cm", bbox=bbox)
        else:
            print("ERROR. No YT support found!")

    def export_csv(self):
        csvfile = open("grid.csv.%s" % self.framenum, "w")
        csvfile.write("xx, yy, zz, rho, uux, uuy, uuz, bbx, bby, bbz\n")
        ul = self.minfo.contents['AC_unit_length']
        uv = self.minfo.contents['AC_unit_velocity']
        ud = self.minfo.contents['AC_unit_density']
        um = self.minfo.contents['AC_unit_magnetic']
        for kk in np.arange(3, self.zz.size-3):
            for jj in np.arange(3, self.yy.size-3):
                for ii in np.arange(3, self.xx.size-3):
                    #print(self.xx.size, self.yy.size, self.zz.size)
                    linestring = "%e, %e, %e, %e, %e, %e, %e, %e, %e, %e\n"% (self.xx[ii]*ul, self.yy[jj]*ul, self.zz[kk]*ul, 
                                                                              np.exp(self.lnrho[ii, jj, kk])*ud,
                                                                              self.uu[0][ii, jj, kk]*uv, self.uu[1][ii, jj, kk]*uv, 
                                                                              self.uu[2][ii, jj, kk]*uv,
                                                                              self.bb[0][ii, jj, kk]*um, self.bb[1][ii, jj, kk]*um, 
                                                                              self.bb[2][ii, jj, kk]*um)
                    csvfile.write(linestring)  
        csvfile.close()

    def export_raw(self):
        uv = self.minfo.contents['AC_unit_velocity']
        ud = self.minfo.contents['AC_unit_density']
        um = self.minfo.contents['AC_unit_magnetic']
        print(self.lnrho.shape, set_dtype(self.minfo.contents['endian'], self.minfo.contents['AcRealSize']))

        f = open("rho%s.raw" % self.framenum, 'w+b')
        binary_format =(np.exp(self.lnrho)*ud).tobytes()
        f.write(binary_format)
        f.close()

        f = open("uux%s.raw" % self.framenum, 'w+b')
        binary_format =(self.uu[0]*uv).tobytes()
        f.write(binary_format)
        f.close()

        f = open("uuy%s.raw" % self.framenum, 'w+b')
        binary_format =(self.uu[1]*uv).tobytes()
        f.write(binary_format)
        f.close()

        f = open("uuz%s.raw" % self.framenum, 'w+b')
        binary_format =(self.uu[2]*uv).tobytes()
        f.write(binary_format)
        f.close()

        f = open("bbx%s.raw" % self.framenum, 'w+b')
        binary_format =(self.bb[0]*um).tobytes()
        f.write(binary_format)
        f.close()

        f = open("bby%s.raw" % self.framenum, 'w+b')
        binary_format =(self.bb[1]*um).tobytes()
        f.write(binary_format)
        f.close()

        f = open("bbz%s.raw" % self.framenum, 'w+b')
        binary_format =(self.bb[2]*um).tobytes()
        f.write(binary_format)
        f.close()

    def export_vtk_ascii(self, Beq = 1.0):
        #BASED ON https://lorensen.github.io/VTKExamples/site/VTKFileFormats/#dataset-attribute-format 
        self.Bfield()

        f = open("GRID%s.vtk" % self.framenum, 'w')

        Ntot = self.minfo.contents['AC_mx']*self.minfo.contents['AC_my']*self.minfo.contents['AC_mz']

        mx = self.minfo.contents['AC_mx']
        my = self.minfo.contents['AC_my']
        mz = self.minfo.contents['AC_mz']

        print("Writing GRID%s.vtk" % self.framenum)

        f.write("# vtk DataFile Version 2.0\n")
        f.write("Astaroth grid for visualization\n")
        f.write("ASCII\n")
        #f.write("DATASET STRUCTURED_GRID\n")
        #f.write("DIMENSIONS %i %i %i \n" % (mx, my, mz))
        #f.write("POINTS %i float \n" % (Ntot))
        #for i in range(mx):
        #    for j in range(my):
        #        for k in range(mz):
        #             f.write("%e %e %e \n" % (i, j, k))
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write("DIMENSIONS %i %i %i \n" % (mx, my, mz))
        f.write("X_COORDINATES %i float \n" % mx)
        for i in range(mx):  
            f.write("%e " % (i))
        f.write("\n")
        f.write("Y_COORDINATES %i float \n" % my)
        for j in range(my):  
            f.write("%e " % (j))
        f.write("\n")
        f.write("Z_COORDINATES %i float \n" % mz)
        for k in range(mz):  
            f.write("%e " % (k))
        f.write("\n")
        f.write("POINT_DATA %i \n" % (Ntot))
        f.write("VECTORS velocity float \n" )
        for i in range(mx):
            if (i % 8) == 0:
                print("i = %i / %i" %(i, mx-1))
            for j in range(my):
                for k in range(mz):
                     f.write("%e %e %e \n" % ( self.uu[0][i, j, k], self.uu[1][i, j, k], self.uu[2][i, j, k]))
        #f.write("POINT_DATA %i \n" % (Ntot))
        f.write("VECTORS bfield float \n")
        eqprint = True
        for i in range(mx):
            if (i % 8) == 0:
                print("i = %i / %i" %(i, mx-1))
                eqprint = True
            for j in range(my):
                for k in range(mz):
                     #Beq is the equipartition magnetic field.
                     while(eqprint):
                         print("normal        B %e %e %e \n" % (self.bb[0][i, j, k],     self.bb[1][i, j, k],     self.bb[2][i, j, k]    ))
                         print("equipartition B %e %e %e \n" % (self.bb[0][i, j, k]/Beq, self.bb[1][i, j, k]/Beq, self.bb[2][i, j, k]/Beq))
                         eqprint = False
                     f.write("%e %e %e \n" % ( self.bb[0][i, j, k]/Beq, self.bb[1][i, j, k]/Beq, self.bb[2][i, j, k]/Beq))
        #ADD DENSITY SCALAR
        f.write("\n")

        print("Done.")

        f.close()

def find_explosion_index(array, criterion = 1e5):
    for i, ar in enumerate(array):
        if (np.abs(array[i])-np.abs(array[i-1])) > criterion:
            return i
    return -1

def mask_bad_values_ts(ts, criterion = 1e5):
    indexmask = np.zeros_like(ts)
    
    ts = np.ma.masked_invalid(ts)
    
    index  = find_explosion_index(ts, criterion = criterion)
    
    if index >= 0:
        indexmask[index:] = 1
    
    ts = np.ma.array(ts, mask=indexmask)
    
    return ts


def parse_ts(fdir, fname, debug = False):
    var = {}  

    tsfile = fdir+fname

    if os.path.exists(tsfile): 

        with open(tsfile) as f:
            filetext = f.read().splitlines()

        line = filetext[0].split()
        for i in range(len(line)):
            line[i] = line[i].replace('VTXBUF_', "")
            line[i] = line[i].replace('UU', "uu")
            line[i] = line[i].replace('_total', "tot")
            line[i] = line[i].replace('ACCRETION', "acc")
            line[i] = line[i].replace('A', "aa")
            line[i] = line[i].replace('LNRHO', "lnrho")
            line[i] = line[i].replace('ENTROPY', "ss")
            line[i] = line[i].replace('BFIELD', "bb")
            line[i] = line[i].replace('X', "x")
            line[i] = line[i].replace('Y', "y")
            line[i] = line[i].replace('Z', "z")
            line[i] = line[i].replace('vaa', "vA")

        #tsdata = np.loadtxt(fdir+fname,skiprows=1)
        tsdata = np.genfromtxt(fdir+fname,skip_header=1, skip_footer=1)

        for i in range(len(line)):
            var[line[i]] = tsdata[:,i]

        var['step'] = np.int64(var['step'])

        var['exist'] = True

    else:
        var['exist'] = False

    if debug:
        print("HERE ARE ALL KEYS FOR TS DATA:")
        print(var.keys())
   
    return var


class TimeSeries:
    '''Class for time series data'''

    def __init__(self, fdir="", fname="timeseries.ts", debug = False):

        self.var = parse_ts(fdir, fname, debug = debug)
