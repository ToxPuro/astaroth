Kernel randomize() {

// results in (-AC_rng_scale, AC_rng_scale] range

    AC_rng_scale = 1e-5

    for field in 0:NUM_VTXBUF_HANDLES {
        r = 2.0 * rand_uniform() - 1.0
        write(Field(field), AC_rng_scale * r)
    }
}
