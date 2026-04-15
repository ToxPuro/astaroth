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
profiles = [file for file in files if file.endswith("_z.profile")] # Filter profiles
profiles.sort()

# if (idiag_alp11/=0) call sum_mn_name(+cz(n)*Eipq(:,:,1,1)+sz(n)*Eipq(:,:,1,i2),idiag_alp11)
# if (idiag_alp21/=0) call sum_mn_name(+cz(n)*Eipq(:,:,2,1)+sz(n)*Eipq(:,:,2,i2),idiag_alp21)
#            if (idiag_alp31/=0) call sum_mn_name(+cz(n)*Eipq(:,:,3,1)+sz(n)*Eipq(:,:,3,i2),idiag_alp31)
#            if (leta_rank2) then
#              if (idiag_eta12/=0) call sum_mn_name(-(-sz(n)*Eipq(:,:,1,i1)+cz(n)*Eipq(:,:,1,i2))*ktestfield1,idiag_eta12)
#              if (idiag_eta22/=0) call sum_mn_name(-(-sz(n)*Eipq(:,:,2,i1)+cz(n)*Eipq(:,:,2,i2))*ktestfield1,idiag_eta22)
emf11 = []
emf12 = []
emf21 = []
emf22 = []
kz=1.0
z=np.arange(args.dims[2])/(args.dims[2]-1)
sinz=np.sin(2.*np.pi*kz*z)
cosz=np.cos(2.*np.pi*kz*z)

for profile in profiles:
    data = np.fromfile(
        profile,
        dtype=np.double,
    )
#    data2d.append(data)
    
    name = Path(profile).stem
    if "ucrossb11" in name:
     #   print("Nimessä oli ucrossb11")
        emf11.append(data)
    
    if "ucrossb12" in name:
     #   print("Nimessä oli ucrossb12")
        emf12.append(data)
    
    if "ucrossb21" in name:
     #   print("Nimessä oli ucrossb21")
        emf21.append(data)

    if "ucrossb22" in name:
     #   print("Nimessä oli ucrossb11")
        emf22.append(data)

# Now hardcoded!!!!
kz=1.0
Bampl=1.0

# Defining z and trigonometric arrays
z=np.arange(args.dims[2])/(args.dims[2]-1)
sinz=np.sin(2.*np.pi*kz*z)
cosz=np.cos(2.*np.pi*kz*z)

# Reform to numpy
emf11 = np.asarray(emf11)
emf12 = np.asarray(emf12)
emf21 = np.asarray(emf21)
emf22 = np.asarray(emf22)

# Alpha and eta profiles per each saved data
alp11zt = (cosz*emf11+sinz*emf12)/Bampl
alp21zt = (cosz*emf21+sinz*emf22)/Bampl
eta12zt = -1.0*(-1.0*sinz*emf11+cosz*emf12)/(kz*Bampl)
eta22zt = -1.0*(-1.0*sinz*emf21+cosz*emf22)/(kz*Bampl)

# Average z-profiles over time
alp11z=np.sum(alp11zt,axis=0)
alp21z=np.sum(alp21zt,axis=0)
eta12z=np.sum(eta12zt,axis=0)
eta22z=np.sum(eta22zt,axis=0)

plt.plot(z,alp11z,label="alp11zt")
plt.plot(z,alp21z,label="alp21zt")
plt.plot(z,eta12z,label="eta12zt")
plt.plot(z,eta22z,label="eta22zt")
plt.legend()
plt.show()
pass

# path = os.path.join(args.output_dir,'tfmz.png')
 #   
 #   print(f"Writing {path}")
 #    plt.plot(data)
 #   plt.title(name)
 #   plt.savefig(path)
 #   plt.show()
 #   plt.pause(1.)
 #   plt.close()

# %%
# Visualize profiles (grouped)
#profiles = [file for file in files if file.endswith(".profile")] # Filter profiles
#profiles.sort()
#
#for step in range(args.nsteps):
#    
#    stepstr = str(step).zfill(12)
#    current_profiles = [profile for profile in profiles if stepstr in profile]
#
#    if len(current_profiles) != 9 * 3:
#        continue
#
#
#fig, axs = plt.subplots(2, 2, layout="constrained")
#fig.set_figheight(15)
#fig.set_figwidth(20)
#    for i, file in enumerate(current_profiles):
#        arr = np.fromfile(
#            file,
#            dtype=np.double,
#        )
#        col = i % 3
#        row = i // 3
#
#        name = os.path.basename(file)
#        #print(f'Input {name}')
#        axs[row, col].plot(arr)
#        axs[row, col].set_title(name)
#
#    path = os.path.join(args.output_dir, f'profiles-{stepstr}.png')
#    
#    print(f"Writing {path}")
#    plt.savefig(path)
#    plt.close()
