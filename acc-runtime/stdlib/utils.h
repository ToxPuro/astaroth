Kernel randomize() {

// results in (-rng_scale, rng_scale] range

    rng_scale = 1e-5

    for field in 0:NUM_VTXBUF_HANDLES {
        r = 2.0 * rand_uniform() - 1.0
        write(Field(field), rng_scale * r)
    }
}
