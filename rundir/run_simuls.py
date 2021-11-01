import numpy as np
import subprocess as sp
import time
import os


# create a list of parameters
params = []
j = 0
for i in range(15):
    for visc in np.linspace(0.005, 0.1,10):
        for forcing in np.linspace(0.075,0.15,10):
            params.append((visc,forcing,j))
            j+=1

def timestr_from_seconds(s):

    m = s/60
    h = m/60

    return f"{h}:{m%60}:{s%60}"


timestr = "00:14:59"
with open("~/analysis/report.txt", "w") as f:
    f.write("time, returncode")
# run a simul for each parameter-set 
for p in params: # this could be parallel as soon as the pipe-situation is better
    start = time.time()
    ret = sp.run("srun", "--account=Project_2000403",
     "--gres=gpu:v100:4",  "--mem=24000", "-t", timestr,
      "-p", "gpu", "python3", "$SRC/simul.py",
        f"--forcing={p[1]}", f"--visc={p[0]}", "--hel=1.0",
         f"--ana_dir_name=sample_{p[2]}", f"--fixedseed={1230000+p[2]}")
    end = time.time() + 1
    
    with open("~/analysis/report.txt", "a") as f:
        f.write(f"{(timestr(end-start))}, {ret.returncode}")
   
    

