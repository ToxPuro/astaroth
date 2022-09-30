heat_conduction_kramers() {

// Calculates diffusive term in entropy equation for Kramers conductivity.

      cv1 = 1./AC_cv
      rho1 = exp(-value(VTXBUF_LNRHO))
      lnTT = AC_lnTT0+cv1*value(VTXBUF_ENTROPY)+(AC_gamma-1)*(value(VTXBUF_LNRHO)-AC_lnrho0)
      glnrho = gradient(VTXBUF_LNRHO)

      glnTT  = cv1*gradient(VTXBUF_ENTROPY) + (AC_gamma-1)*glnrho
      del2lnTT = cv1*laplace(VTXBUF_ENTROPY) + (AC_gamma-1)*laplace(VTXBUF_LNRHO)

      Krho1 = AC_hcond0_kramers * pow(rho1,(2.*AC_nkramers+1.)) * pow(exp(lnTT),(6.5*AC_nkramers))   // = K/rho

      g2=dot(-2.*AC_nkramers*glnrho+(6.5*AC_nkramers+1)*glnTT,glnTT)
      return Krho1*(del2lnTT)+g2
}
