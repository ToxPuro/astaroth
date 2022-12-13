heat_conduction_kramers() {

// Calculates diffusive term in entropy equation for Kramers conductivity.

      cv1 = 1./AC_cv_sound
      rho1 = exp(-value(VTXBUF_LNRHO))
      lnTT = AC_lnT0+cv1*value(VTXBUF_ENTROPY)+(AC_gamma-1)*(value(VTXBUF_LNRHO)-AC_lnrho0)
      glnrho = gradient(VTXBUF_LNRHO)

      glnTT  = cv1*gradient(VTXBUF_ENTROPY) + (AC_gamma-1)*glnrho
      del2lnTT = cv1*laplace(VTXBUF_ENTROPY) + (AC_gamma-1)*laplace(VTXBUF_LNRHO)

      Krho1 = AC_hcond0_kramers * pow(rho1,(2.*AC_n_kramers+1.)) * pow(exp(lnTT),(6.5*AC_n_kramers))   // = K/rho

      g2=dot(-2.*AC_n_kramers*glnrho+(6.5*AC_n_kramers+1)*glnTT,glnTT)
      diffusion = Krho1*(del2lnTT)+g2

// Cooling in a surface layer; step profile
      prof = step(z,AC_step_pos,AC_step_width)
      return diffusion - AC_cool*prof*(AC_gamma*(AC_cp_sound-AC_cv_sound)*exp(lnTT)/AC_cs2cool-1.)
}
step(x,x0,width){   //AcReal x, AcReal x0, AcReal width){
      return 0.5*(1.+tanh((x-x0)/width))
}
