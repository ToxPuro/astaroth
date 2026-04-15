#!/usr/bin/env bash
# Usage: mpirun <args> ./rocprof-wrapper.sh --hip-trace --trace-start off <application>
# Adapted from https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html#using-rocprof
pid="$$"
outdir="pid_${pid}"
outfile="results_${pid}.csv"
eval "rocprof -d ${outdir} -o ${outdir}/${outfile} $*"
