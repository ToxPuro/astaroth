// Declare uniforms (i.e. device constants)
uniform Scalar cs2_sound;
uniform Scalar nu_visc;
uniform Scalar cp_sound;
uniform Scalar cv_sound;
uniform Scalar mu0;
uniform Scalar eta;
uniform Scalar gamma;
uniform Scalar zeta;

uniform Scalar dsx;
uniform Scalar dsy;
uniform Scalar dsz;

uniform Scalar lnT0;
uniform Scalar lnrho0;

uniform int nx_min;
uniform int ny_min;
uniform int nz_min;
uniform int nx;
uniform int ny;
uniform int nz;

Vector
value(in Vector uu)
{
    return (Vector){value(uu.x), value(uu.y), value(uu.z)};
}

Matrix
gradients(in Vector uu)
{
    return (Matrix){gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}

Scalar
continuity(in Vector uu, in Scalar lnrho) {
    return -dot(value(uu), gradient(lnrho))
           - divergence(uu);
}

Vector
momentum(in Vector uu, in Scalar lnrho) {
  Vector mom;

  const Matrix S = stress_tensor(uu);

  // Isothermal: we have constant speed of sound

  mom = -mul(gradients(uu), value(uu)) -
    cs2_sound * gradient(lnrho) +
    nu_visc *
    (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
      Scalar(2.) * mul(S, gradient(lnrho))) + zeta * gradient_of_divergence(uu);

  return mom;
}


Vector
induction(in Vector uu, in Vector aa) {
  // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
  // x A)) in order to avoid taking the first derivative twice (did the math,
  // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
  // u cross B - ETA * mu0 * (mu0^-1 * [- laplace A + grad div A ])
  const Vector B = curl(aa);
  const Vector grad_div = gradient_of_divergence(aa);
  const Vector lap = laplace_vec(aa);

  // Note, mu0 is cancelled out
  const Vector ind = cross(value(uu), B) - eta * (grad_div - lap);

  return ind;
}



// Declare input and output arrays using locations specified in the
// array enum in astaroth.h
in Scalar lnrho = VTXBUF_LNRHO;
out Scalar out_lnrho = VTXBUF_LNRHO;

in Vector uu = (int3) {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
out Vector out_uu = (int3) {VTXBUF_UUX,VTXBUF_UUY,VTXBUF_UUZ};

#if LMAGNETIC
in Vector aa = (int3) {VTXBUF_AX,VTXBUF_AY,VTXBUF_AZ};
out Vector out_aa = (int3) {VTXBUF_AX,VTXBUF_AY,VTXBUF_AZ};
#endif

Kernel void
solve(Scalar dt) {
    out_lnrho = rk3(out_lnrho, lnrho, continuity(uu, lnrho), dt);

    #if LMAGNETIC
    out_aa = rk3(out_aa, aa, induction(uu, aa), dt);
    #endif

    out_uu = rk3(out_uu, uu, momentum(uu, lnrho), dt);

}
