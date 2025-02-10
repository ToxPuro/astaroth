#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
from pathlib import Path


# %%
# Parse arguments
parser = argparse.ArgumentParser(description='A tool for computing TFM coeffs',
epilog='''EXAMPLES:
    ./visualize.py --input *.mesh          # Visualizes all meshes
    ./visualize.py --input *.profile       # Visualize all profiles
    ./visualize.py --input *01000*.profile # Visualize all profiles at step 1000
    
See Unix globbing for passing files/directories to the script more easily.
    For example:
        ??.sh matches two characters
        *.sh matches any number of characters
        [1-8] matches a character in range 1-8
        {1..16} expands to 1,2,3,...,16
        ?([0-9]) matches zero or one number
        [0-9]?([0-9]) matches one number and an optional second number
        ?[0-9] matches one character and one number
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument('--dims', type=int, default=[32, 32, 32], nargs=3, help='The dimensions of the computational domain in the order [z y x] (note z first). Should be the same as global_nn defined in mhd.ini')
parser.add_argument('--slice-nz', type=int, default=-1, nargs=1, help='Position on the nz axis to slice on. If set to -1, chooses the middle slice.')
parser.add_argument('--nsteps', type=int, default=100, nargs=1, help='Maximum steps to visualize (defined as AC_simulation_nsteps in mhd.ini)')
parser.add_argument('--inputs', type=str, nargs='+', default=["*.mesh", "*.profile"], help='Input files to visualize (supports globbing)')
parser.add_argument('--output-dir', type=str, nargs='+', default="output", help='The output directory')
#parser.add_argument('--show-individual', action='store_true', help='Controls whether to write individual plots')
#parser.add_argument('--show-grouped', action='store_true', help='Controls whether to write grouped plots')

args = parser.parse_args()

# Set slice_nz to middle if not specified
if args.slice_nz < 0:
    args.slize_nz = int(args.dims[0]/2)

# Make directories
print(f"Creating output directory at {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)

# %%
# Read files
files = args.inputs
files = [os.path.join(os.getcwd(), file) for file in files]   # Relative path
files = [glob.glob(file) for file in files]                   # Glob
files = np.concatenate(files)                                 # Flatten
files.sort()
files

# %%
# Visualize individual profiles
profiles = [file for file in files if file.endswith(".profile")] # Filter profiles
profiles.sort()

# if (idiag_alp11/=0) call sum_mn_name(+cz(n)*Eipq(:,:,1,1)+sz(n)*Eipq(:,:,1,i2),idiag_alp11)
# if (idiag_alp21/=0) call sum_mn_name(+cz(n)*Eipq(:,:,2,1)+sz(n)*Eipq(:,:,2,i2),idiag_alp21)
#            if (idiag_alp31/=0) call sum_mn_name(+cz(n)*Eipq(:,:,3,1)+sz(n)*Eipq(:,:,3,i2),idiag_alp31)
#            if (leta_rank2) then
#              if (idiag_eta12/=0) call sum_mn_name(-(-sz(n)*Eipq(:,:,1,i1)+cz(n)*Eipq(:,:,1,i2))*ktestfield1,idiag_eta12)
#              if (idiag_eta22/=0) call sum_mn_name(-(-sz(n)*Eipq(:,:,2,i1)+cz(n)*Eipq(:,:,2,i2))*ktestfield1,idiag_eta22)
emf11x = []
emf12x = []
emf21x = []
emf22x = []
emf11y = []
emf12y = []
emf21y = []
emf22y = []

# Now hardcoded!!!!
kz=1.0
Bampl=1.0

# Defining z and trigonometric arrays
z=np.arange(args.dims[2])/(args.dims[2]-1)
sinz=np.sin(2.*np.pi*kz*z)
cosz=np.cos(2.*np.pi*kz*z)


for profile in profiles:
    data = np.fromfile(
        profile,
        dtype=np.double,
    )
    
    name = Path(profile).stem
    if "ucrossb11mean_x" in name:
        emf11x.append(data)
    
    if "ucrossb12mean_x" in name:
        emf12x.append(data)
    
    if "ucrossb21mean_x" in name:
        emf21x.append(data)

    if "ucrossb22mean_x" in name:
        emf22x.append(data)

    if "ucrossb11mean_y" in name:
        emf11y.append(data)

    if "ucrossb12mean_y" in name:
        emf12y.append(data)

    if "ucrossb21mean_y" in name:
        emf21y.append(data)

    if "ucrossb22mean_y" in name:
        emf22y.append(data)


# Reform to numpy
emf11x = np.asarray(emf11x)
emf12x = np.asarray(emf12x)
emf21x = np.asarray(emf21x)
emf22x = np.asarray(emf22x)

emf11y = np.asarray(emf11y)
emf12y = np.asarray(emf12y)
emf21y = np.asarray(emf21y)
emf22y = np.asarray(emf22y)

# Alpha and eta profiles per each saved data
alp11zt = (cosz*emf11x+sinz*emf12x)/Bampl
alp21zt = (cosz*emf11y+sinz*emf12y)/Bampl
alp12zt = (cosz*emf21x+sinz*emf22x)/Bampl
alp22zt = (cosz*emf21y+sinz*emf22y)/Bampl
#MJKL Before produced the correct profiles; for a longer run produces negative diagonals
#eta12zt = -1.0*(sinz*emf11x-cosz*emf12x)/(kz*Bampl)
#eta11zt = -1.0*(sinz*emf21x-cosz*emf22x)/(kz*Bampl)
#eta22zt =  (sinz*emf11y-cosz*emf12y)/(kz*Bampl)
#eta21zt = -1.0*(sinz*emf21y-cosz*emf22y)/(kz*Bampl)
#MJKL With this change the diagonals are again positive.
eta12zt = -1.0*(sinz*emf11x-cosz*emf12x)/(kz*Bampl)
eta11zt = -1.0*(sinz*emf21x-cosz*emf22x)/(kz*Bampl)
eta22zt =  (sinz*emf11y-cosz*emf12y)/(kz*Bampl)
eta21zt = -1.0*(sinz*emf21y-cosz*emf22y)/(kz*Bampl)


# Average z-profiles over time
alp11z=np.sum(alp11zt,axis=0)
alp12z=np.sum(alp12zt,axis=0)
alp21z=np.sum(alp21zt,axis=0)
alp22z=np.sum(alp22zt,axis=0)
eta11z=np.sum(eta11zt,axis=0)
eta12z=np.sum(eta12zt,axis=0)
eta21z=np.sum(eta21zt,axis=0)
eta22z=np.sum(eta22zt,axis=0)

plt.plot(z,alp11z,label="alp11zt")
plt.plot(z,alp21z,label="alp21zt")
plt.plot(z,alp12z,label="alp12zt")
plt.plot(z,alp22z,label="alp12zt")
#
#plt.plot(z,eta11z,label="eta11zt")
#plt.plot(z,eta12z,label="eta12zt")
#plt.plot(z,eta21z,label="eta21zt")
#plt.plot(z,eta22z,label="eta22zt")

plt.legend()
plt.show()

 #path = os.path.join(args.output_dir,'tfmz.png')
 #   
 #   print(f"Writing {path}")
 #    plt.plot(data)
 #   plt.title(name)
 #   plt.savefig(path)
 #   plt.show()
 #   plt.pause(1.)
 #   plt.close()


