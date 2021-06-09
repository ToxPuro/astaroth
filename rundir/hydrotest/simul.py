import os
import re
from glob import glob
import random
import json
from datetime import datetime
from time import time

adj_conf_path = "/users/julianlagg/astaroth/rundir/adjustable.conf"

random.seed(sum([r*2**(i*8) for i,r in enumerate(os.getrandom(4))]))

def make_adjusted_conf(opts, outname):

    rex = r"@(\w+)@"

    def replace(match):
        if match.group(1) in opts:
            return str(opts[match.group(1)])
        else:
            raise ValueError("options are missing parameter "+match_group(1))


    with open(adj_conf_path, "r") as f:
        adj_conf_s = f.read()

    conf = re.sub(rex, replace, adj_conf_s)

    with open(outname, "w") as f:
        f.write(conf)

core_opts = {
    "save_steps" : 25,
    "viscosity" : 5e-3,
    "relhel" : 1.0,
    "forcing_kmin" : 0.8,
    "forcing_kmax" : 1.2
}

baserun_len = 5000
perturbation_len = 50
final_len = 5000

baserun_start = 0
baserun_fin = baserun_len + 1
perturbation_start = baserun_len
perturbation_fin = baserun_len+perturbation_len + 1
final_start = baserun_len + perturbation_len
final_fin = baserun_len + perturbation_len + final_len + 1

fixed_seed = 123456

base_run_op = {**core_opts,
"start_step": baserun_start, "max_steps": baserun_fin, "bin_steps" : 1000}
perturb_run_op = {**core_opts,
"start_step": perturbation_start, "max_steps": perturbation_fin, "bin_steps" : 50}
final_run_op = {**core_opts,
"start_step": final_start, "max_steps": final_fin, "bin_steps": 50}

def run_ac(name, opts, seed):

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

    make_adjusted_conf(opts, conf)
    ret_code = os.system(
        f"bash -c" +  
        f" './../ac_run --seed {seed} -s -c {conf} " + 
        f"> >(tee {stdout}) 2> >(tee {stderr} >&2)' ")

    end = time()

    info["ret_code"] = ret_code
    info["stderr_empty"] = (os.system(f"[ -s {stderr} ]")==0)
    info["finished"] = True
    info["time"] = end-start


    with open(info_json, "w") as f:
        f.write(json.dumps(info, indent=4))


# base run
baserundir = "baserun"
os.mkdir(baserundir)
os.chdir(baserundir)
run_ac("baserun", base_run_op, fixed_seed)
fin_meshes = glob(f"*{baserun_len}.mesh")
os.chdir("..")

# glob final files


# other runs
for i in range(10):
    num = "0"*(6-len(str(i))) + str(i)
    rundir = f"perturbed_run_{num}"
    os.mkdir(rundir)
    for m in fin_meshes:
        os.system(f"ln {baserundir}/{m} {rundir}/{m}")

    os.chdir(rundir)
    run_ac("perturbation", perturb_run_op, random.randrange(0,2**31))
    run_ac("final", final_run_op, fixed_seed)
    os.chdir("..")

