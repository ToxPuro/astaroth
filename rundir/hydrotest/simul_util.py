import os
import re
from datetime import datetime
import json
from time import time
import numpy as np

adj_conf_path = "/users/julianlagg/astaroth/rundir/adjustable.conf"
astar_exec = "/users/julianlagg/astaroth/rundir/hydrotest/ac_run"
pipe_dir = None

def complete_nx(opts):
    assert("nx" in opts)
    assert(opts["nx"] in [2**i for i in range(20)]) # we need powers of 2 (and not crazy high)
    opts["ny"] = opts["nx"]
    opts["nz"] = opts["nx"]
    dsxyz = 2*np.pi/opts["nx"]
    opts["dsx"] = dsxyz
    opts["dsy"] = dsxyz
    opts["dsz"] = dsxyz

def _create_pipes(pipe_dir):

    os.system(f"rm -rf {pipe_dir}")
    os.mkdir(pipe_dir)

    data_fname = "pipe2py_data"
    os.mkfifo(pipe_dir+"/"+data_fname)
    status_fname = "pipe2py_status"
    os.mkfifo(pipe_dir+"/"+status_fname)
    internal_start_fname = "pipe2py_internal_start"
    os.mkfifo(pipe_dir+"/"+internal_start_fname)
    internal_fin_fname = "pipe2py_internal_fin"
    os.mkfifo(pipe_dir+"/"+internal_fin_fname)

def set_pipe_dir(path):
    global pipe_dir
    if pipe_dir is not None:
        raise ValueError(f"cant set pipedir to {path} as it has already been set to {pipe_dir}")
    pipe_dir = path
    _create_pipes(pipe_dir)

def get_pipe_dir():
    if pipe_dir is None:
        raise ValueError(f"pipe_dir has not been set")
    return pipe_dir

def run_ac(name, opts, seed, analyze_freq=-1,
 in_dir =None, silent=False, timestep_file_path=None,
  dictate_timestep=None, use_gdb=False, use_valgrind=False):

    pipe_path = get_pipe_dir()

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

    gdb_cmd = " gdb -ex=r --args " if use_gdb else ""
    valgrind_cmd = " valgrind " if use_valgrind else ""
    if use_gdb and use_valgrind:
        raise ValueError("Cant use both gdb and valgrind")

    make_adjusted_conf(opts, conf)
    print(f"running in dir {os.getcwd()} with seed {seed}")
    
    if silent:
        ret_code = os.system(
            f"bash -c" +
            f" ' {valgrind_cmd} {gdb_cmd} {astar_exec} --seed {seed} --analyze_steps {analyze_freq}" +
            ts_cmd + pipe_cmd +
            f" -s -c {conf} " + 
            f"> {stdout} 2> {stderr} '"
        )
    else:
        ret_code = os.system(
            f"bash -c" +  
            f" ' {valgrind_cmd} {gdb_cmd} {astar_exec} --seed {seed} --analyze_steps {analyze_freq}" +
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