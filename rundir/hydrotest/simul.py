import os
import itertools as it
import numpy as np
import os.path
import re
from glob import glob
import random
import json
from datetime import datetime
from time import time
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--hel", type=float)
parser.add_argument("--forcing", type=float)
parser.add_argument("--visc", type=float)
parser.add_argument("--fixedseed", type=int, default=123456)
parser.add_argument("--ana_dir_name", type=str)

#todo: make this independent of the scratch and make the pipe_dir variable in some way

args = parser.parse_args()

adj_conf_path = "/users/julianlagg/astaroth/rundir/adjustable.conf"
astar_exec = "/users/julianlagg/astaroth/rundir/hydrotest/ac_run"
working_scratch_dir = "/scratch/project_2000403/lyapunov"
pipe_dir = "/users/julianlagg"


random.seed(sum([r*2**(i*8) for i,r in enumerate(os.getrandom(4))]))


def ossystemnofail(command):
    err = os.system(command)
    if err != 0:
        print("FAILED COMMAND:", command)
        print("executed in directory ", os.getcwd(), ", got nonzero exit code ", err)
    assert(err == 0)

def make_adjusted_conf(opts, outname):

    used_params = {k : False for k in opts.keys()}

    def replace(match):
        param = match.group(1)
        if param in opts:
            used_params[param] = True
            val = opts[param]
            if isinstance(val, float):
                return format(val, ".12f").rstrip("0")
            return str(opts[param])
        elif param == "max_steps":
            used_params["steps"] = True
            return str(opts["steps"] + opts["start_step"])
        else:
            raise ValueError("adjustable conf has unreplacable parameter: "+match.group(1))


    with open(adj_conf_path, "r") as f:
        adj_conf_s = f.read()

    rex = r"@(\w+)@"
    conf = re.sub(rex, replace, adj_conf_s)

    unused_params = [name for name,used in used_params.items() if not used ]
    if len(unused_params) > 0:
        raise ValueError(f"failed find a place to use the following parameters for astaroth configuration: {unused_params}")

    with open(outname, "w") as f:
        f.write(conf)

core_opts = {
    "save_steps" : 25,
    "viscosity" : args.visc,
    "forcing_magnitude" : args.forcing,
    "relhel" : args.hel,
    "forcing_kmin" : 0.8,
    "forcing_kmax" : 1.2,
    "nx" : 128
}


def complete_nx(opts):
    assert("nx" in opts)
    assert(opts["nx"] in [2**i for i in range(20)]) # we need powers of 2 (and not crazy high)
    opts["ny"] = opts["nx"]
    opts["nz"] = opts["nx"]
    dsxyz = 2*np.pi/opts["nx"]
    opts["dsx"] = dsxyz
    opts["dsy"] = dsxyz
    opts["dsz"] = dsxyz

complete_nx(core_opts)

with open("core_options.json", "w") as f:
    json.dump(core_opts, f)

n_runs = 5
baserun_len = 5000
baserun_dump_freq = baserun_len
perturbation_len = 1
perturbation_dump_freq = 1
final_len = 20
final_dump_freq = 5

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
"start_step": baserun_len+perturbation_len, "steps": final_len+1, "bin_steps": final_len}

def run_ac(name, opts, seed, analyze_freq=-1,
 in_dir =None, silent=False, timestep_file_path=None, dictate_timestep=None, pipe_path=pipe_dir):

    if in_dir is not None:
        old_dir = os.getcwd()
        os.chdir(in_dir)
    print("simulating in directory ", os.getcwd())
    info = {}
    info["conf_options"] = opts.copy()
    info["seed"] = seed
    info["started_on"] = str(datetime.now())
    info["finished"] = False
    info["name"] = name

    info_json = f"{name}.json"
    with open(info_json, "w") as f:
        f.write(json.dumps(info, indent=4))


    conf = f"{name}.conf"
    stdout = f"{name}.stdout"
    stderr = f"{name}.stderr"

    start = time()

    # if we have a timestep_file, we need to know its role
    # if we have a role for the timestepfile we need a file
    assert((timestep_file_path==None) == (dictate_timestep==None) )
    role = "write" if dictate_timestep else "read"
    ts_cmd = f" --{role}_timestep_file {timestep_file_path} " if timestep_file_path else ""

    pipe_cmd = " --pipe_dir " + pipe_path

    make_adjusted_conf(opts, conf)
    print(f"running in dir {os.getcwd()} with seed {seed}")
    if silent:
        ret_code = os.system(
            f"bash -c" +
            f" '{astar_exec} --seed {seed} --analyze_steps {analyze_freq}" +
            ts_cmd + pipe_cmd +
            f" -s -c {conf} " + 
            f"> {stdout} 2> {stderr} '"
        )
    else:
        ret_code = os.system(
            f"bash -c" +  
            f" '{astar_exec} --seed {seed} --analyze_steps {analyze_freq}" +
            ts_cmd +  pipe_cmd +
            f" -s -c {conf} " + 
            f"> >(tee {stdout}) 2> >(tee {stderr} >&2)' "
        )
        

    end = time()

    info["ret_code"] = ret_code
    info["stderr_empty"] = (os.system(f"[ -s {stderr} ]")==0)
    info["finished"] = True
    info["time"] = end-start


    with open(info_json, "w") as f:
        f.write(json.dumps(info, indent=4))
    
    if info["ret_code"] != 0:
        print("nonzero exit code from simulation in dir ", os.getcwd())
        exit(ret_code)

    if in_dir is not None:
        os.chdir(old_dir)

# do everything on scratch, not home
os.chdir(working_scratch_dir)


baserundir = "baserun"

# base run --> comment out this block to skip
os.mkdir(baserundir)
os.chdir(baserundir)
run_ac("baserun", base_run_op, fixed_seed)
os.chdir("..")
###

os.chdir(baserundir)
baserun_meshes = glob(f"*{baserun_len}.mesh")
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
    server_process = subprocess.Popen(["python3", "/users/julianlagg/astaroth/rundir/hydrotest/variance_live_analyzer.py", "--pipe_dir", pipe_dir], stdout=f, stderr=e)
from time import sleep
sleep(2)
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
     in_dir=perturb_run_dir, timestep_file_path=os.path.abspath("perturb.ts"), dictate_timestep=timestep_dictate_perturb)
    timestep_dictate_perturb = False
    print("=========perturbation ran somehow")
    perturbed_meshes = glob(f"{perturb_run_dir}/*{perturbation_len}.mesh")
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

print("waiting for server-process to finish")
server_process.wait()
print("server-process finished")