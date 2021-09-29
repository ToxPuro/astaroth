from genericpath import isfile
from posixpath import dirname
import pickle
import sys
import shutil
import astar.data
from astar.data import read
import os
import os.path
from glob import glob
import re
import numpy as np
import json
import itertools as it
import csv
from observables import *

ana_dir = "/users/julianlagg/analysis/"
amend = (len(sys.argv) == 3 and sys.argv[2]=="--amend")
# todo: amending is not so easy: it is not all off of the same baserun!!!
# need to think of a way to even define this
if amend:
    raise ValueError("unsupported, fix problems")

targetdir = ana_dir + sys.argv[1]
if not amend:
    assert(len(sys.argv)==2)
    if os.path.exists(targetdir):
        raise ValueError(targetdir + " already exists, dont override")
    os.mkdir(targetdir)
    snapshotdir = targetdir + "/snapshots"
    os.mkdir(snapshotdir)
if amend:
    if not os.path.exists(targetdir):
        raise ValueError(targetdir+ " does not exist, cant amend")
    if not os.path.exists(targetdir+"/welf_x_10000.pickle"):
        raise ValueError(targetdir + " seems to be lacking the necessary pickle files for amending")



"""
def get_pickled_var(name):
    with open(targetdir+"/"+name+".pickle","rb") as f:
        obj = pickle.load(f)
    return obj

def save_var_to_pickle(var, name, t):

    for dim,data in var.items():
        with open(targetdir+"/"+name+"_"+dim+"_"+str(t)+".pickle", "wb") as f:
            pickle.dump(data, f)
"""

vars = {} # timestep -> (naive_variance, welford_variance)
with open("timebounds.json", "r") as f:
    lengths = json.load(f)

locals().update(lengths) # not pretty, gives the _len and _dump_freq

# x+c-x%c is the first number strictly bigger than x that is divisible by c
def next_int(x,c):
    return x+c-x%c

baserun_dump_steps = zip(it.repeat("baserun"), range(0,baserun_len+1, baserun_dump_freq))
perturbation_dump_steps = zip(it.repeat("perturb_run_*"), range(next_int(baserun_len,perturbation_dump_freq), baserun_len+perturbation_len+1, perturbation_dump_freq))
final_dump_steps = zip(it.repeat("final_run_*"), range(next_int(baserun_len+perturbation_len, final_dump_freq), baserun_len+perturbation_len+final_len+1, final_dump_freq))

for phase, t in it.chain(baserun_dump_steps,perturbation_dump_steps,final_dump_steps):

    naive = {}; welf = {}; Re_vals = []; Re_mesh_vals = []; urms_vals = []; mach_vals = []
    real_time = None
    if not amend:
        for d in "XYZ":
            naive[d] = NaiveVariance()
            welf[d] = WelfordVariance()
    if amend:
        for d in "XYZ":
            naive[d] = get_pickled_var("naive_"+d+"_"+t)
            welf[d] = get_pickled_var("welf_"+d+"_"+t)

    meshes = glob(f"{phase}/VTXBUF_UUX_{t}.mesh") # only use x to get the directories, the other dimensions should give the same

    assert(len(meshes) > 0 or print("faulty time is time ", t))
    for mesh in meshes:
        rundir = "/".join(mesh.split("/")[:-1]) + "/"

        mi = read.MeshInfo(rundir, dbg_output=False)
        dataXYZ = {}
        for d in "XYZ":
            assert(os.path.isfile(f"{rundir}/VTXBUF_UU{d}_{t}.mesh"))
            data, phystime, read_ok = read.read_bin("VTXBUF_UU"+d,rundir, str(t),mi)
            assert(read_ok)
            assert(phystime is not None)
            assert(data is not None)
            if real_time is not None and phystime != real_time:
                raise ValueError(
                    f"physical times for timestep t={t} not matching, we have values phystime={phystime} and realtime={real_time}")
            real_time = phystime
            
            data = data[3:-3,3:-3,3:-3]
            if not read_ok:
                raise ValueError(f"failed to read file {mesh}")
            welf[d].add(data)
            naive[d].add(data)
            dataXYZ[d] = data

        urms_new = calc_urms(*[dataXYZ[d] for d in "XYZ"])
        Re_mesh_vals.append(calc_Re_mesh(urms_new, mesh_info=mi))
        Re_vals.append(calc_Re_real(urms_new, mesh_info=mi))
        urms_vals.append(urms_new)
        mach_new = calc_Mach(*[dataXYZ[d] for d in "XYZ"])
        mach_vals.append(mach_new)

    n_simuls = len(urms_vals)
    assert(n_simuls == len(Re_vals))
    assert(n_simuls == len(Re_mesh_vals))
    assert(n_simuls == welf[d].n )
    assert(n_simuls == naive[d].n )

    vars[t] = {
        # old definition:
        #"naive_var": np.mean(sum([naive[d].get_var() for d in "XYZ"]) / 3),
        #"welf_var": np.mean(sum([welf[d].get_var() for d in "XYZ"]) / 3),
        # new definition:
        "naive_var": np.sqrt(np.mean([naive[d].get_var() for d in "XYZ"])),
        "welf_var": np.sqrt(np.mean([welf[d].get_var() for d in "XYZ"])),
        "mesh_Re" : np.mean(Re_mesh_vals),
        "Re" : np.mean(Re_vals),
        "urms" : np.mean(urms_vals),
        "n_simuls" : n_simuls,
        "mach" : np.mean(mach_vals),
        "real_time" : real_time,
    }

    # once restarting feature becomes useful maybe?
    #save_var_to_pickle(welf, "welf", t)
    #save_var_to_pickle(naive, "naive", t)


#save options
with open("baserun/baserun.json", "r") as f:
    optstr = f.read()
    options = json.loads(optstr)

with open(targetdir+"/options.json", "w") as f:
    f.write(json.dumps(options))

#save calculated output
with open(targetdir+"/results.csv", "w") as f:
    fieldnames = ["time"] + list(vars[next(iter(vars))])
    writer = csv.DictWriter(f, fieldnames)

    writer.writeheader()
    for t, valdict in vars.items():
        vals = {**valdict, "time": t}
        writer.writerow(vals)

#save metadata files
os.system(f"mv timebounds.json {targetdir}/")
os.system(f"mv perturb.ts {targetdir}/")
os.system(f"mv final.ts {targetdir}/")

# save snapshot by saving one directory
# todo: do this properly to not fill the disk with so much stuff
"""
os.system(f"cp -r baserun {snapshotdir}/baserun")
os.system(f"cp -r perturb_run_000000 {snapshotdir}/perturb")
os.mkdir(f"{snapshotdir}/final")
os.system(f"cp -r final_run_000000/mesh_info.list {snapshotdir}/final/mesh_info.list")
for t in range(perturbstep+1, finalend, 10):
    meshes = glob(f"final_run_000000/*{t}.mesh")
    assert(len(meshes) > 0 or print(f"faulty time is {t}"))
    for mesh in meshes:
        meshname = mesh.split("/")[1]
        os.system(f"cp {mesh} {snapshotdir}/final/{meshname}")
"""