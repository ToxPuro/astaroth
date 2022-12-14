# Pilot run notes and instructions

# Important files:

* `acc-runtime/samples/mhd_modular/mhdsolver.ac`
> **NOTE**: `dsx`, `dsy`, and `dsz` are hardcoded here. If grid dimensions are changed in config file (f.ex. `config/astaroth.conf`), then they must be manually changed here also, otherwise grid spacing in integration kernel will be incorrect.

* `config/samples/subsonic_forced_nonhelical_turbulence/astaroth.conf`
> One of the configs, should be compatible with the mahti 4k snapshot

* `config/astaroth.conf`
> The default config if no `--config <path-to-config>` is passed to `./ac_run_mpi`

# Usage

