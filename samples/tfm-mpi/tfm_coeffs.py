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
kf=5.0
eta=1.0e-2


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
#
eta12zt = -1.0*(sinz*emf11x-cosz*emf12x)/(kz*Bampl) #etanew12=eta11
eta11zt =      (sinz*emf21x-cosz*emf22x)/(kz*Bampl) #etanew11=-eta12
eta22zt = -1.0*(sinz*emf11y-cosz*emf12y)/(kz*Bampl) #etanew22=eta21
eta21zt =      (sinz*emf21y-cosz*emf22y)/(kz*Bampl) #etanew21=-eta22

# Average z-profiles over time
nzz=alp11zt.shape[0]

alp11z=np.sum(alp11zt,axis=0)/nzz
alp12z=np.sum(alp12zt,axis=0)/nzz
alp21z=np.sum(alp21zt,axis=0)/nzz
alp22z=np.sum(alp22zt,axis=0)/nzz
eta11z=np.sum(eta11zt,axis=0)/nzz
eta12z=np.sum(eta12zt,axis=0)/nzz
eta21z=np.sum(eta21zt,axis=0)/nzz
eta22z=np.sum(eta22zt,axis=0)/nzz

#Read timeseries
filepath = 'timeseries.csv' # Path to the timeseries (current working directory by default)
df = pd.read_csv(filepath)
#df
df0 = df[df['label'] == 'uu']
urms=df0['rms']
avurms=np.sum(urms,axis=0)/nzz

alp0=-1.0/3.0*avurms
eta0=1.0/3.0*avurms/kf
rm=avurms/eta/kf
print("alp0,eta0,rm",alp0,eta0,rm)

lt.plot(z,alp11z/alp0,label="alp11/alp0")
plt.plot(z,alp21z/alp0,label="alp21/alp0")
plt.plot(z,alp12z/alp0,label="alp12/alp0")
plt.plot(z,alp22z/alp0,label="alp22/alp0")

plt.legend()
plt.show()
#
plt.plot(z,eta11z/eta0,label="eta11/eta0")
plt.plot(z,eta12z/eta0,label="eta12/eta0")
plt.plot(z,eta21z/eta0,label="eta21/eta0")
plt.plot(z,eta22z/eta0,label="eta22/eta0")

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


