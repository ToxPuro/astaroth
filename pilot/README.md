# Pilot run notes and instructions

# Important files:

* `acc-runtime/samples/mhd_modular/mhdsolver.ac`
> **NOTE**: `dsx`, `dsy`, and `dsz` are hardcoded here. If grid dimensions are changed in config file (f.ex. `config/astaroth.conf`), then they must be manually changed here also, otherwise grid spacing in integration kernel will be incorrect.

* `config/samples/subsonic_forced_nonhelical_turbulence/astaroth.conf`
> One of the configs, should be compatible with the mahti 4k snapshot

* `config/astaroth.conf`
> The default config if no `--config <path-to-config>` is passed to `./ac_run_mpi`

# Issues

* Weird issues when writing slices or mesh (device context invalid, etc)
> Try writing without MPI IO by setting `#define USE_POSIX_IO (1)` in `astaroth/src/core/grid.cc` in `acGridWriteMeshToDiskLaunch` or `acGridWriteSlicesToDiskLaunch`, or both.
> You can also try without async IO by commenting out `threads.push_back(std::move(std::thread(write_async, host_buffer, count))); // Async, threaded` at the end of these functions and uncommenting `write_async(host_buffer, count, device->id); // Synchronous, non-threaded` or similar. However, note that this results in worse performance due to IO being blocking.

* Distributed mesh writing and collective slice writing?
> Replace `acGridWriteSlicesToDiskLaunch` calls in `standalone_mpi/main.cc` with `acGridWriteSlicesToDiskCollectiveSynchronous`.

# Usage

