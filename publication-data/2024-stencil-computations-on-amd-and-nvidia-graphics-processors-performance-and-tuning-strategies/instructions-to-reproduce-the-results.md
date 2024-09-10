# Instructions for Reproducing the Results of "Stencil Computations on AMD and Nvidia Graphics Processors: Performance and Tuning Strategies", J. Pekkilä, O. Lappi, F. Robertsén, and M. J. Korpi-Lagg. 2024.


## Implementations used for key benchmarks (best-performing chosen per device)
|Case|A100|MI250X GCD|V100|
|-|-|-|-|
|Cross-correlation $r=1$|implementation1 (implicit), mbimplementation2|implementation2 (explicit), mbimplementation3|implementation1 (implicit), mbimplementation3|
|Cross-correlation $r=1024$|implementation2 (explicit), mbimplementation3|implementation2 (explicit), mbimplementation3|implementation2 (explicit), mbimplementation3|
|Diffusion equation $r=4$|implementation1 (implicit), MAX_THREADS_PER_BLOCK=0 (32)|implementation1 (implicit), MAX_THREADS_PER_BLOCK=0|implementation1 (implicit), MAX_THREADS_PER_BLOCK=0|
|MHD $r=3$|implementation1 (implicit), MAX_THREADS_PER_BLOCK=0|implementation1 (implicit), MAX_THREADS_PER_BLOCK=512|implementation1 (implicit), MAX_THREADS_PER_BLOCK=256|



## Commands

### Setup

```Bash
# Preprocess
$ASTAROTH/scripts/genbenchmarks.py \
--cmakelistdir $ASTAROTH/ \
--task-type preprocess

# Build
$ASTAROTH/scripts/genbenchmarks.py \
--task-type build \
--build-dirs benchmark-data/builds/*
# Done
```

### Benchmarks
```Bash
# Microbenchmarks: CUDA/HIP
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/implementation*_maxthreadsperblock0_* \
--run-scripts benchmark-data/scripts/microbenchmark-cudahip.sh

# Microbenchmarks: cuDNN/MIOpen
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/implementation1_maxthreadsperblock0_mbimplementation1_distributedTrue_doubleprecision0 \
--run-scripts benchmark-data/scripts/microbenchmark-nn.sh

# Heat equation: Pytorch and Tensorflow
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/heat-equation-implementation1_maxthreadsperblock0_distributedTrue_doubleprecision0 \
--run-scripts benchmark-data/scripts/heat-equation-benchmark-python-pytorch-fp32.sh benchmark-data/scripts/heat-equation-benchmark-python-tensorflow-fp32.sh

# MHD: Pytorch and Tensorflow
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/heat-equation-implementation1_maxthreadsperblock0_distributedTrue_doubleprecision0 \
--run-scripts benchmark-data/scripts/nonlinear-mhd-benchmark-python-tensorflow-fp32.sh benchmark-data/scripts/nonlinear-mhd-benchmark-python-pytorch-fp32.sh

# Heat equation: Astaroth
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/heat-equation-implementation*_maxthreadsperblock*_distributedTrue_doubleprecision[0-1] \
--run-scripts benchmark-data/scripts/heat-equation-benchmark-astaroth*.sh

# MHD: Astaroth
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/implementation*_maxthreadsperblock*_mbimplementation1_distributedTrue_doubleprecision[0-1] \
--run-scripts benchmark-data/scripts/nonlinear-mhd-benchmark-astaroth*.sh
# or
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/implementation* \
--run-scripts benchmark-data/scripts/nonlinear-mhd-benchmark-astaroth*.sh

# Microbenchmarks: CUDA/HIP: tuning strategy comparison: Problem size
$ASTAROTH/scripts/genbenchmarks.py \
--task-type run \
--run-dirs benchmark-data/builds/implementation*_maxthreadsperblock0_* \
--run-scripts benchmark-data/scripts/microbenchmark-cudahip-problemsize.sh
# done
```

### Postprocessing
```Bash
# Postprocess
$ASTAROTH/scripts/genbenchmarks.py --task-type postprocess
# done
```