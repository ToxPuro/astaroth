# Plasma Physics Meets AI: Astaroth workshop

## Building the exercises

> `cd astaroth`

> `mkdir build && cd build`

> `cmake -DPROGRAM_MODULE_DIR=../samples/plasma-meets-ai-workshop/<exercise name> .. && make -j`

> `./<exercise name>`

> (Optional) Visualize the output with `../samples/plasma-meets-ai-workshop/animate-snapshots.py <list of .dat files, f.ex. *.dat or UUX*.dat>`. Requires `python`, `python-numpy`, and `python-matplotlib`.

> (Optional) Check `samples/plasma-meets-ai-workshop/getting-started-with-astaroth.md` for more information on getting Astaroth up and running.

## Exercise 1: Blur Filter

In this exercise, we will implement a blur filter. The DSL file to be modified is in `blur/blur.ac` and the main program is in `blur/blur.c`.

## Exercise 2: Simulating Hydrodynamics

See `hydro/hydro.ac` and `hydro/hydro.c`

## Exercise 3: Adding SGS stress to our hydrodynamics simulation

See `hydro-sgs/hydro-sgs.ac` and `hydro-sgs/hydro-sgs.c`.
