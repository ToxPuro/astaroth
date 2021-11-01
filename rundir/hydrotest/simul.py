print("starting simul.py")
from enum import unique
import os
import itertools as it
import os.path
from glob import glob
import random
import json
import argparse
import subprocess
from simul_util import *
print("finished imports")

parser = argparse.ArgumentParser()
parser.add_argument("--hel", type=float)
parser.add_argument("--forcing", type=float)
parser.add_argument("--visc", type=float)
parser.add_argument("--fixedseed", type=int, default=123456)
parser.add_argument("--ana_dir_name", type=str)
args = parser.parse_args()

# does not depend on time
random.seed(sum([r*2**(i*8) for i,r in enumerate(os.getrandom(4))]))

#unique_name = "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") for _ in range(30)])

# can use name of analysis directory as name on scratch dir, as that name should be unique anyways
unique_name = args.ana_dir_name
working_scratch_dir = "/scratch/project_2000403/lyapunov/"+unique_name
os.mkdir(working_scratch_dir) # will throw if it already exists
# do everything on scratch, not home
os.chdir(working_scratch_dir)

set_pipe_dir("/users/julianlagg/astar_pipes/"+unique_name)

n_runs = 30
baserun_len = 12000
baserun_dump_freq = baserun_len
perturbation_len = 1
perturbation_dump_freq = 1
final_len = 70
final_dump_freq = 5

core_opts = {
    "save_steps" : 25,
    "viscosity" : args.visc,
    "forcing_magnitude" : args.forcing,
    "relhel" : args.hel,
    "forcing_kmin" : 0.8,
    "forcing_kmax" : 1.2,
    "nx" : 128
}
complete_nx(core_opts)




with open("core_options.json", "w") as f:
    json.dump(core_opts, f)

with open("timebounds.json", "w") as f:
    json.dump( {
        "baserun_len" : baserun_len,
        "perturbation_len" : perturbation_len,
        "final_len" : final_len,
        "baserun_dump_freq" :baserun_dump_freq,
        "perturbation_dump_freq" :perturbation_dump_freq,
        "final_dump_freq" :final_dump_freq,
    }, f)

with open("timesteps.json", "w") as f:

    # x+c-x%c is the first number strictly bigger than x that is divisible by c
    def next_int(x,c):
        return x+c-x%c

    p = {}
    p["num_simuls"] = n_runs
    p["first_perturb"] = next_int(baserun_len, perturbation_dump_freq)
    p["first_final"] = next_int(baserun_len+perturbation_len, final_dump_freq)
    
    # todo: this is actually WRONG for certain combinations of freqs, or is it?
    perturb_steps = range(p["first_perturb"], p["first_perturb"]+perturbation_len, perturbation_dump_freq)

    final_steps = range(p["first_final"], p["first_final"]+final_len, final_dump_freq)

    p["timesteps"] = list(it.chain(perturb_steps, final_steps))

    json.dump(p, f)


fixed_seed = args.fixedseed

base_run_op = {**core_opts,
"start_step": 0, "steps": baserun_len+1, "bin_steps": baserun_len,}
perturb_run_op = {**core_opts,
"start_step": baserun_len, "steps": perturbation_len+1, "bin_steps": perturbation_len}
final_run_op = {**core_opts,
"start_step": baserun_len+perturbation_len, "steps": final_len+1, "bin_steps": final_len+10} # there is no reason to dump final




baserundir = "baserun"

# base run --> comment out this block to skip
os.mkdir(baserundir)
os.chdir(baserundir)
run_ac("baserun", base_run_op, fixed_seed)
os.chdir("..")
###

os.chdir(baserundir)
baserun_meshes = glob(f"*_{baserun_len}.mesh") #globs ALL the meshes (x,y,z,rho)
if len(baserun_meshes) != 4:
    print(f"saw {len(baserun_meshes)} instead of 4, maybe the baserun didnt complete?")
    exit(1)
os.chdir("..")

# glob final files


timestep_dictate_perturb = True # indicates if this the first run (that determines the timesteps)
timestep_dictate_final = True
timestep_file_path_perturb = working_scratch_dir + "/perturb_timesteps.ts"
timestep_file_path_final = working_scratch_dir + "/final_timesteps.ts"

# more options for the analysis server
with open("analysis_options.json", "w") as f:

    target_dir = "/users/julianlagg/analysis/" + args.ana_dir_name
    if os.path.exists(target_dir):
        raise ValueError(f"target directory for analysis data {target_dir} already exists, refusing to overwrite, aborting simulation")
    p = {}
    p["targetdir"] = target_dir

    json.dump(p, f)

    

# start analysis server (dirty for now)
print("starting server")
with open("server_stdout.txt", "w") as f, open("server_stderr.txt", "w") as e:
    server_process = subprocess.Popen(["python3", "/users/julianlagg/astaroth/rundir/hydrotest/variance_live_analyzer.py", "--pipe_dir", get_pipe_dir()], stdout=f, stderr=e)
import time
time.sleep(3)
print("started server")

# other runs
for i in range(n_runs):
    num = "0"*(6-len(str(i))) + str(i)
    perturb_run_dir = f"perturb_run_{num}"
    os.mkdir(perturb_run_dir)
    print(os.getcwd())
    for old_name in baserun_meshes:
        #new_name = re.sub(r"\d+","0", old_name)
        print("creating symlink ", f"ln -s {baserundir}/{old_name} {perturb_run_dir}/{old_name}")
        linktarget = os.path.abspath(f"{baserundir}/{old_name}")
        ossystemnofail(f"ln -s {linktarget} {perturb_run_dir}/{old_name}")

    print("==============now running perturbation")
    run_ac("perturbation", perturb_run_op, random.randrange(0,2**31), analyze_freq=perturbation_dump_freq,
     in_dir=perturb_run_dir, timestep_file_path=os.path.abspath("perturb.ts"), dictate_timestep=timestep_dictate_perturb,
     )
    timestep_dictate_perturb = False
    print("=========perturbation ran somehow")
    perturbed_meshes = glob(f"{perturb_run_dir}/*_{final_run_op['start_step']}.mesh") # all meshes (x,y,z,rho)
    assert(len(perturbed_meshes)==4)
    assert(len(perturbed_meshes) == len(baserun_meshes))
    final_run_dir = f"final_run_{num}"
    print("making dir for final run")
    os.mkdir(final_run_dir)
    for old_name in perturbed_meshes:
        #new_name = re.sub(r"\d+\.mesh", "0.mesh", old_name )
        linktarget = os.path.abspath(old_name)
        linkname = old_name.split("/")[-1] # ughh
        ossystemnofail(f"ln -s {linktarget} {final_run_dir}/{linkname}")

    print("now running final")
    run_ac("final", final_run_op, fixed_seed, analyze_freq=final_dump_freq,
     in_dir=final_run_dir,
    timestep_file_path=os.path.abspath("final.ts"),
     dictate_timestep=timestep_dictate_final)
    timestep_dictate_final = False
    print("final ran somehow")

    os.system(f"rm {perturb_run_dir}/*.mesh")
    os.system(f"rm {final_run_dir}/*.mesh")

print("waiting for server-process to finish")
server_process.wait()

print("server-process finished, doing cleanup")
os.chdir("..")
os.system(f"rm -rf {working_scratch_dir}")

# mark the output files as having completed until here
print("OK")
print("OK", file=sys.stderr)


# os.system("rm perturb_run_*/*.mesh") # should be unnecessary
# os.system("rm final_run_*/*.mesh") # should be unnecessary
# os.system("rm baserun/*.mesh")
# print("removed all .mesh files, finishing")