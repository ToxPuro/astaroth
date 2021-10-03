from collections import defaultdict
import observables as ob
import astaroth_pipe as ap
import numpy as np
import itertools as it
import json
import csv
import os
import logging
import argparse
import sys


try: # wrap all code in a try-catch, because astaroth-pipe breaks exception-handling for some reason
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe_dir", type=str)
    args = parser.parse_args(sys.argv[1:])

    print("doing init")
    ap.init_astaroth_pipe(args.pipe_dir)
    print("done init")


    # todo: send a signal that this process is ready

    # get all the timesteps
    with open("timesteps.json", "r") as f:
        ts_json = json.load(f)
        num_simuls = ts_json["num_simuls"]
        timesteps = ts_json["timesteps"]
        perturb_start = ts_json["first_perturb"]
        final_start = ts_json["first_final"]

    print("get core options")

    # get parameters of the simulation
    with open("core_options.json", "r") as f:
        p = json.load(f)
        nu = p["viscosity"]
        relhel = p["relhel"]
        kf = (p["forcing_kmin"] + p["forcing_kmax"]) / 2
        nx = p["nx"]
        dsxyz = p["dsx"]
        assert(p["dsx"] == p["dsy"] == p["dsz"])

    print("get ana options")

    # get options for doing the analysis
    with open("analysis_options.json", "r") as f:

        p = json.load(f)
        targetdir = p["targetdir"]

    print("making target dir")
    os.mkdir(targetdir)
    print("doing some copies")
    os.system(f"cp analysis_options.json {targetdir}/")
    print("1")
    os.system(f"cp core_options.json {targetdir}/")
    print("2")
    os.system(f"cp timebounds.json {targetdir}/")

    print("did some copies")

    #initialize intermediate results
    obs = ["varX", "varY", "varZ", "mach", "Re_mesh", "Re", "urms"]
    ir = {ts :{ o : ob.WelfordVariance() for o in obs } for ts in timesteps}
    real_time = {} # maps timestep to real physical time

    print("reaching main loop")
    for i in range(num_simuls):

        for ts in timesteps:

            velos = {}

            for dim in "XYZ":
                velos[dim], info = ap.get_array_blocking()
                print(f"for i={i}, ts={ts}, d={dim}, received {info} and an array of shape {velos[dim].shape}", flush=True)
                if ts != info.timestep:
                    raise ValueError(f"expected timestep {ts} but got {info.timestep} from astaroth. Context: i={i}, d={dim}, info={info}, arrshape={velos[dim].shape}")
                assert(info.dim == dim)
                ir[ts]["var"+dim].add(velos[dim])

                # make sure the physical times are what they should be
                rt = real_time.get(ts)
                if not (rt is None or rt == info.phystime):
                    raise ValueError(f"expected real time rt={rt} does not match physical time of the sender info.phystime={info.phystime};"
                     f"situation is i={i}, ts={ts}, d={dim}, info={info}, arrshape={velos[dim].shape}")
                real_time[ts] = info.phystime

                # make sure there was no Nan-shenanigans
                if not np.all(np.isfinite(velos[dim])):
                    print(velos[dim])
                    raise ValueError(f"NaN-Error: for i={i}, dim={dim}, ts={ts}, rt={rt} we received {info} and there are NaNs in the array")


            print(f"doing calculations on i={i}, ts={ts}")
            urms_new = ob.calc_urms(*velos.values())
            mach_new = ob.calc_Mach(*velos.values())
            re_new = ob.calc_Re_real(urms_new, kf, nu)
            re_mesh_new = ob.calc_Re_mesh(urms_new, kf, dsxyz, nu)


            ir[ts]["urms"].add(urms_new)
            ir[ts]["mach"].add(mach_new)
            ir[ts]["Re"].add(re_new)
            ir[ts]["Re_mesh"].add(re_mesh_new)
            print(f"doing calculations on i={i}, ts={ts} successful")

    print("closing pipe")
    ap.close_astaroth_pipe()
    print("pipe closed")

    # postprocess
    res = {ts : {} for ts in timesteps}
    for ts in timesteps:
        for o in ["urms", "mach", "Re", "Re_mesh"]:
            res[ts][o] = ir[ts][o].get_mean()
            res[ts][o+"_var"] = ir[ts][o].get_var()
    for ts in timesteps:
        res[ts]["var"] = np.sqrt(np.mean([ir[ts]["var"+d].get_var() for d in "XYZ"]))
        res[ts]["real_time"] = real_time[ts]
        res[ts]["timestep"] = ts

    #save calculated output -> maybe dont wait till last step? safety
    with open(targetdir+"/results.csv", "w") as f:
        fieldnames = list(res[next(iter(res))])
        writer = csv.DictWriter(f, fieldnames)

        writer.writeheader()
        for t, valdict in res.items():
            writer.writerow(valdict)

    #save metadata files
    os.system(f"cp timebounds.json {targetdir}/")
    os.system(f"cp perturb.ts {targetdir}/")
    os.system(f"cp final.ts {targetdir}/")

except Exception as ex:
    logging.exception("There has been an exception in the analyzer")