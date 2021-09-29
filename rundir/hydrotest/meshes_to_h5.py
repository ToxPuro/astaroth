import sys
import astar.data
from astar.data import read
import os
from glob import glob
import re
import h5py as h5
import numpy as np

# create h5 file:
# 1D-field with the timesteps timestep-argument
# one field for every meshfile type, like density etc, that has a 3d array for every timestep
# use compression for every timestep
def init_h5(types, nt, nx, ny, nz):

    name = "simul.h5"
        
    with open(name, "w") as f:
        for t in types:
            f.create_dataset(t,(nt,nx,ny,nz), chunks=(1,64,64,64), compression="szip")

class ArrStatAccumulator():

    def __init__(self, shape):

        self.shape = shape
        self.values_squared = np.zeros(shape) 
        self.values_simple = np.zeros(shape)
        self.n = 0

    def add_array(self, arr):

        assert(arr.shape == self.shape)
        self.values_squared += arr**2
        self.values_simple += arr
        self.n += 1

    def get_variance(self):

        assert(self.n >= 1)
        var = np.sum((self.values_squared - self.values_simple**2))/self.n
        return var

    def get_mean(self):

        assert(self.n >= 1)
        mean = np.sum(self.values_simple) / self.n
        return mean

        
        
class UxUyUzStat():

    def __init__(self, shape):

        self.ux_acc = ArrStatAccumulator(shape)
        self.uy_acc = ArrStatAccumulator(shape)
        self.uz_acc = ArrStatAccumulator(shape)
    
    def add_arrays(self, ux, uy, uz):

        self.ux_acc.add_array(ux)
        self.uy_acc.add_array(uy)
        self.uz_acc.add_array(uz)

    def get_rms(self):

        rms = np.sqrt(
             (self.ux_acc.get_mean()
             +self.uy_acc.get_mean()
             +self.uz_acc.get_mean()) / 3)
        return rms

    def get_variance(self):

        var = self.ux_acc.get_variance() + self.uy_acc.get_variance() + self.uz_acc.get_variance()

        return var


def aggregate(dir, at_timestep, stats):

    mesh_info = read.MeshInfo(working_scratch_dir, dbg_output=True)
    meshfiles = glob("*.mesh")
    print(meshfiles)


    filetuples = [] # a filetuple is (kind, timestep) where kind is eg "UUX"
    # parse meshfile names
    for file in meshfiles:
        m = re.match(r"VTXBUF_(?P<kind>[A-Z]+)_(?P<timestep>\d+)\.mesh", file)
        if not m:
            raise ValueError(f"filename {file} looks weird")
        d = m.groupdict()
        timestep = int(d["timestep"])
        kind = d["kind"]
        ftup = (timestep, kind)
        assert(ftup not in filetuples)
        filetuples.append(ftup)


    filetuples.sort()
    time = {} # dict mapping timesteps to physical time
    for timestep,kind in filetuples:
        if timestep != at_timestep:
            continue
        print(timestep,kind)
        data, phystime, read_ok = read.read_bin(f"VTXBUF_{kind}", dir, str(timestep), mesh_info)
        if not read_ok:
            raise ValueError(f"failed to read file {file}")
        
        # check validity of physical time:
        if timestep in time and phystime != time[timestep]:
            raise ValueError(f"folder {dir} has contradictory physical time values for timestep {timestep}: {phystime} vs {time[timestep]}")
        time[timestep] = phystime

        # add data to h5 file


    # add timesteps-series to h5 file
    timestep, phystime = list(zip(*sorted(time.items())))
    print(timestep,"\n",phystime)


if __name__ == "__main__":

    working_scratch_dir = "/scratch/project_2000403/lyapunov/baserun/"
    os.chdir(working_scratch_dir)
    print(os.getcwd())

    stats = UxUyUzStat()
    aggregate(working_scratch_dir)






