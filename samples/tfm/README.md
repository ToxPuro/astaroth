# Astaroth TFM - getting started

## Work-in-progress
- Capability to modify the four parameters
    - Function for initializing $\overline{B}$ and parameters
        - wavenumber $k_z$
        - amplitude
    - TFM reset time to reset the solution
    - Separate $\eta$ for test fields
- Code to output the computed test fields
- Multi-GPU implementation of the pipeline with concurrency. A draft of the desired order of computations exists but has not been coded. The individual components required for implementing the pipeline on multiple nodes (similar to `tfm_pipeline` function in `samples/tfm/tfm_pipeline.c`) already exist, such as `acGridReduceXYAverages`.
- Further testing of the correctness

## Overview

The code is in `astaroth/samples/tfm`.

The primary program is `tfm_pipeline` compiled from `astaroth/samples/tfm_pipeline.c`. This includes benchmarking, basic verification (hydro only), and a basic simulation function.

The DSL code is in `astaroth/samples/tfm/mhd/mhd.ac`. The time integration of each test field is implemented with functions `singlepass_solve_tfm_b`for respective $b$-field (b11, b12, b21, b22).

Test fields are stored in
```
// Test fields
Field TF_b11_x, TF_b11_y, TF_b11_z
Field TF_b12_x, TF_b12_y, TF_b12_z
Field TF_b21_x, TF_b21_y, TF_b21_z
Field TF_b22_x, TF_b22_y, TF_b22_z
```
defined in the beginning of tfm/mhd/mhd.ac.

Profiles are in
```
// Mean-field profiles
Profile PROFILE_Umean_x, PROFILE_Umean_y, PROFILE_Umean_z

Profile PROFILE_ucrossb11mean_x, PROFILE_ucrossb11mean_y, PROFILE_ucrossb11mean_z
Profile PROFILE_ucrossb12mean_x, PROFILE_ucrossb12mean_y, PROFILE_ucrossb12mean_z
Profile PROFILE_ucrossb21mean_x, PROFILE_ucrossb21mean_y, PROFILE_ucrossb21mean_z
Profile PROFILE_ucrossb22mean_x, PROFILE_ucrossb22mean_y, PROFILE_ucrossb22mean_z

Profile PROFILE_B11mean_x, PROFILE_B11mean_y, PROFILE_B11mean_z
Profile PROFILE_B12mean_x, PROFILE_B12mean_y, PROFILE_B12mean_z
Profile PROFILE_B21mean_x, PROFILE_B21mean_y, PROFILE_B21mean_z
Profile PROFILE_B22mean_x, PROFILE_B22mean_y, PROFILE_B22mean_z
```

## Building and running

A separate build script is used to build the `tfm` sample because the DSL file resides in a separate directory. See `astaroth/samples/tfm/build.sh` for details.

Commands:
```Bash
cd astaroth
mkdir build && cd build # Create a build directory and move there
../samples/tfm/build.sh <additional cmake options>
# F.ex. for Nvidia ../samples/tfm/build.sh -DUSE_HIP=OFF
$SRUN_COMMAND ./tfm_pipeline # Run tfm_pipeline
```

### Implementation notes

- $\overline{B}^{pq}$ corresponds to DSL f.ex. `Profile PROFILE_B11mean_x`, which is the $x$ component of the $B$-field $pq = 11$. $B$-fields are currently initialized to $0$ in `samples/tfm/mhd/mhd.ac` function `Kernel init_profiles()`.
    - **Wavenumber** ($k_z$) and **amplitude** do not yet exist as parameters (as of 2024-07-08), but those would be supplied as device constant parameters `real k_z` etc to the initialization function of the averaged $B$ fields.

- Currently (2024-07-08) test fields use the same $\eta$ (**AC_eta**) as hydro, but this can be changed by introducing additional device constant f.ex. `real AC_eta_tfm` and using that.
- Resetting time does not yet exist (2024-07-08), but this can be implemented f.ex. as a device constant toggle `int reset_tfm` or a toggle in the host code sample (tfm_pipeline.c) 