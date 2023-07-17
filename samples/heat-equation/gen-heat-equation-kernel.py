#!/usr/bin/env python3
from contextlib import redirect_stdout

max_ord = 8

with open('heat-equation.ac', 'w') as f:
    with redirect_stdout(f):
        print(f'hostdefine STENCIL_ORDER ({max_ord}) // Define the max order used here')
        print('Field temperature')
        print('''
// Initialization
Kernel init() {
    write(temperature, 2.0 * rand_uniform() - 1.0)
}

Kernel randomize() {
    
    // N.B. scale: result in (-AC_rng_scale, AC_rng_scale] range
    AC_rng_scale = 1e-5

    for field in 0:NUM_FIELDS {
        r = 2.0 * rand_uniform() - 1.0
        write(Field(field), AC_rng_scale * r)
    }
}
        ''')

        for dim in range(1,3+1):
            for ord in range(0,max_ord+1, 2):
                r = int(ord/2)
                
                print(f'Stencil heat{dim}d_ord{ord} {{')
                i = j = k = 0
                print(f'\t[{k}][{j}][{i}] = 1')
                if dim >= 1:
                    for i in range(-r, r+1):
                        if i == 0:
                            continue
                        j = 0
                        k = 0
                        print(f'\t,[{k}][{j}][{i}] = 1')
                if dim >= 2:
                    for j in range(-r, r+1):
                        if j == 0:
                            continue
                        i = 0
                        k = 0
                        print(f'\t,[{k}][{j}][{i}] = 1')
                if dim >= 3:
                    for k in range(-r, r+1):
                        if k == 0:
                            continue
                        i = 0
                        j = 0
                        print(f'\t,[{k}][{j}][{i}] = 1')
                print(f'}}')

                print(f'Kernel solve{dim}d_ord{ord}() {{ write(temperature, heat{dim}d_ord{ord}(temperature)) }}')
