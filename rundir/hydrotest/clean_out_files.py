from glob import glob
import os

#os.chdir("/users/julianlagg/analysis")

outfiles = glob("*.out")
outfiles.extend(glob("*.err"))

def last_line_is_OK(filename):
    # there is optimization potential here using seek and not everything into a string,
    # I dont think it matters
    with open(filename, "r") as f:
        lastline = f.read().splitlines()[-1]
    return lastline=="OK"

goodfiles = []
for f in outfiles:
    if last_line_is_OK:
        goodfiles.append(f)
if len(goodfiles)>0:
    os.system(f"rm {' '.join(goodfiles)}")
    print(f"removed outfiles for {len(goodfiles)} good runs")
else:
    print("clean_out_files invoked without a single good file to remove, damn.")