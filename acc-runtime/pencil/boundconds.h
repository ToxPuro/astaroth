//TP: PC boundconds translated to DSL
//In mhd_modular/mhdsolver.ac see an example how to use them in BoundConds
//Unlike before now the syntax is 
//boundcond(BOUNDARY,params...)
//
//i.e. you do not have to give the field as input since the compiler can figure out which fields are written to and read from at least for these bcs (and it is not that crucial to get them 100% correct for the current use case since the interdependency makes the RHS calc still pretty much depend on all of the bcs and vice versa)
//
//TP: for top bot you can either use simply integers or enums like I did in mhd_modular/mhdsolver.ac
//
bc_steady_z(topbot,VtxBuffer j)
{
  int i;
  if(topbot == AC_bot) {
    if (j[vertexIdx.x][vertexIdx.y][n1-1] <= 0.0) {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][n1-i-1]=j[vertexIdx.x][vertexIdx.y][n1-1];
      }
    }
    else {
      if (j[vertexIdx.x][vertexIdx.y][n1-1] > j[vertexIdx.x][vertexIdx.y][n1+1-1]) {
        j[vertexIdx.x][vertexIdx.y][n1-1-1]=0.5*(j[vertexIdx.x][vertexIdx.y][n1-1]    +j[vertexIdx.x][vertexIdx.y][1+n1-1]);
      }
      else {
        j[vertexIdx.x][vertexIdx.y][n1-1-1]=2.0* j[vertexIdx.x][vertexIdx.y][n1-1]    -j[vertexIdx.x][vertexIdx.y][1+n1-1];
      }
      for i in 2:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][n1-i-1]=2.0* j[vertexIdx.x][vertexIdx.y][n1-i+1-1]-j[vertexIdx.x][vertexIdx.y][n1-i+2-1];
      }
    }
  }
  else if(topbot == AC_top) {
    if (j[vertexIdx.x][vertexIdx.y][n2-1] >= 0.0) {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][n2+i-1]=j[vertexIdx.x][vertexIdx.y][n2-1];
      }
    }
    else {
      if (j[vertexIdx.x][vertexIdx.y][n2-1] < j[vertexIdx.x][vertexIdx.y][n2-1-1]) {
        j[vertexIdx.x][vertexIdx.y][1+n2-1]=0.5*(j[vertexIdx.x][vertexIdx.y][n2-1]    +j[vertexIdx.x][vertexIdx.y][n2-1-1]);
      }
      else {
        j[vertexIdx.x][vertexIdx.y][1+n2-1]=2.0* j[vertexIdx.x][vertexIdx.y][n2-1]    -j[vertexIdx.x][vertexIdx.y][n2-1-1];
      }
      for i in 2:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][n2+i-1]=2.0* j[vertexIdx.x][vertexIdx.y][n2+i-1-1]-j[vertexIdx.x][vertexIdx.y][n2+i-2-1];
      }
    }
  }
  else {
  }
}



//TP: note in Fortran this is (-nghost:nghost) so in C this will then be  of length 3*2+1
//Also one has to index into it index+NGHOST_VAL
real AC_dz2_bound[7]
bc_ss_flux(topbot)
{
  real tmp_xy;
  real cs2_xy;
  real rho_xy;
  int i;
  if(topbot == AC_bot) {
    if (AC_pretend_lntt) {
      tmp_xy=-fbotkbot/exp(AC_iss[vertexIdx.x][vertexIdx.y][AC_n1-1]);
      for i in 1:NGHOST_VAL+1 {
        AC_iss[vertexIdx.x][vertexIdx.y][AC_n1-i-1]=AC_iss[vertexIdx.x][vertexIdx.y][AC_n1+i-1]-AC_dz2_bound[-i+NGHOST_VAL]*tmp_xy;
      }
    }
    else {
      if (AC_ldensity_nolog) {
          rho_xy=AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-1];
      }
      else {
        rho_xy=exp(AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-1]);
      }
      cs2_xy = AC_iss[vertexIdx.x][vertexIdx.y][AC_n1-1];
      if (AC_ldensity_nolog) {
        cs2_xy=AC_cs20*exp(AC_gamma_m1*(log(rho_xy)-AC_lnrho0)+AC_cv1*cs2_xy);
      }
      else {
        cs2_xy=AC_cs20*exp(AC_gamma_m1*(AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-1]-AC_lnrho0)+AC_cv1*cs2_xy);
      }
      if (lheatc_chiconst) {
        tmp_xy=fbot/(rho_xy*AC_chi*cs2_xy);
      }
      else if (lheatc_kramers) {
        tmp_xy=fbot*pow(rho_xy,(2*nkramers))*pow((AC_cp*AC_gamma_m1),(6.5*nkramers))  /(hcond0_kramers*pow(cs2_xy,(6.5*nkramers+1.)));
      }
      else {
        tmp_xy=fbotkbot/cs2_xy;
      }
      for i in 1:NGHOST_VAL+1 {
        rho_xy = AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1+i-1]-AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-i-1];
        if (AC_ldensity_nolog) {
            rho_xy = rho_xy/AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-1];
        }
        AC_iss[vertexIdx.x][vertexIdx.y][AC_n1-i-1]=AC_iss[vertexIdx.x][vertexIdx.y][AC_n1+i-1]+AC_cp*(AC_cp-AC_cv)*(rho_xy+AC_dz2_bound[-i+NGHOST_VAL]*tmp_xy);
      }
    }
  }
  else if(topbot == AC_top) {
    if (AC_pretend_lntt) {
      tmp_xy=-ftopktop/exp(AC_iss[vertexIdx.x][vertexIdx.y][AC_n2-1]);
      for i in 1:NGHOST_VAL+1 {
        AC_iss[vertexIdx.x][vertexIdx.y][AC_n2-i-1]=AC_iss[vertexIdx.x][vertexIdx.y][AC_n2+i-1]-AC_dz2_bound[i+NGHOST_VAL]*tmp_xy;
      }
    }
    else {
      if (AC_ldensity_nolog) {
          rho_xy=AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2-1];
      }
      else {
        rho_xy=exp(AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2-1]);
      }
      cs2_xy = AC_iss[vertexIdx.x][vertexIdx.y][AC_n2-1];
      if (AC_ldensity_nolog) {
        cs2_xy=AC_cs20*exp(AC_gamma_m1*(log(rho_xy)-AC_lnrho0)+AC_cv1*cs2_xy);
      }
      else {
        cs2_xy=AC_cs20*exp(AC_gamma_m1*(AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2-1]-AC_lnrho0)+AC_cv1*cs2_xy);
      }
      if (lheatc_chiconst) {
        tmp_xy=ftop/(rho_xy*AC_chi*cs2_xy);
      }
      else if (lheatc_kramers) {
        tmp_xy=ftop*pow(rho_xy,(2*nkramers))*pow((AC_cp*AC_gamma_m1),(6.5*nkramers))  /(hcond0_kramers*pow(cs2_xy,(6.5*nkramers+1.)));
      }
      else {
        tmp_xy=ftopktop/cs2_xy;
      }
      for i in 1:NGHOST_VAL+1 {
        rho_xy = AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2+i-1]-AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2-i-1];
        if (AC_ldensity_nolog) {
            rho_xy = rho_xy/AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2-1];
        }
        AC_iss[vertexIdx.x][vertexIdx.y][AC_n2+i-1]=AC_iss[vertexIdx.x][vertexIdx.y][AC_n2-i-1]+AC_cp*(AC_cp-AC_cv)*(-rho_xy-AC_dz2_bound[i+NGHOST_VAL]*tmp_xy);
      }
    }
  }
  else {
  }
}



bc_ism(topbot,VtxBuffer j)
{
  int k;
  real density_scale1;
  real density_scale;
  if (AC_density_scale_factor==AC_impossible) {
    density_scale=AC_density_scale_cgs/AC_unit_length;
  }
  else {
    density_scale=AC_density_scale_factor;
  }
  density_scale1=1./density_scale;
  if(topbot == AC_bot) {
    for k in 1:NGHOST_VAL+1 {
      if (j==AC_irho  ||  j==AC_ilnrho) {
        if (AC_ldensity_nolog) {
          j[vertexIdx.x][vertexIdx.y][k-1]=j[vertexIdx.x][vertexIdx.y][AC_n1-1]*exp(-(AC_z[AC_n1]-AC_z[k])*density_scale1);
        }
        else {
          j[vertexIdx.x][vertexIdx.y][k-1]=j[vertexIdx.x][vertexIdx.y][AC_n1-1] - (AC_z[AC_n1]-AC_z[k])*density_scale1;
        }
      }
      else if (j==AC_iss) {
        if (AC_ldensity_nolog) {
          j[vertexIdx.x][vertexIdx.y][AC_n1-k-1]=j[vertexIdx.x][vertexIdx.y][AC_n1-1]+(AC_cp-AC_cv)*(log(AC_irho[vertexIdx.x][vertexIdx.y][AC_n1-1])-log(AC_irho[vertexIdx.x][vertexIdx.y][AC_n1-k-1])) +  AC_cv*log((AC_z[AC_n1]-AC_z[AC_n1-k])*density_scale+1.);
        }
        else {
          j[vertexIdx.x][vertexIdx.y][AC_n1-k-1]=j[vertexIdx.x][vertexIdx.y][AC_n1-1]+(AC_cp-AC_cv)*(AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-1]-AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n1-k-1])+  AC_cv*log((AC_z[AC_n1]-AC_z[AC_n1-k])*density_scale+1.);
        }
      }
      else {
      }
    }
  }
  else if(topbot == AC_top) {
    for k in 1:NGHOST_VAL+1 {
      if (j==AC_irho  ||  j==AC_ilnrho) {
        if (AC_ldensity_nolog) {
          j[vertexIdx.x][vertexIdx.y][AC_n2+k-1]=j[vertexIdx.x][vertexIdx.y][AC_n2-1]*exp(-(AC_z[AC_n2+k]-AC_z[AC_n2])*density_scale1);
        }
        else {
          j[vertexIdx.x][vertexIdx.y][AC_n2+k-1]=j[vertexIdx.x][vertexIdx.y][AC_n2-1] - (AC_z[AC_n2+k]-AC_z[AC_n2])*density_scale1;
        }
      }
      else if (j==AC_iss) {
        if (AC_ldensity_nolog) {
          j[vertexIdx.x][vertexIdx.y][AC_n2+k-1]=j[vertexIdx.x][vertexIdx.y][AC_n2-1]+(AC_cp-AC_cv)*(log(AC_irho[vertexIdx.x][vertexIdx.y][AC_n2-1])-log(AC_irho[vertexIdx.x][vertexIdx.y][AC_n2+k-1]))+  AC_cv*log((AC_z[AC_n2+k]-AC_z[AC_n2])*density_scale+1.);
        }
        else {
          j[vertexIdx.x][vertexIdx.y][AC_n2+k-1]=j[vertexIdx.x][vertexIdx.y][AC_n2-1]+(AC_cp-AC_cv)*(AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2-1]-AC_ilnrho[vertexIdx.x][vertexIdx.y][AC_n2+k-1])+  AC_cv*log((AC_z[AC_n2+k]-AC_z[AC_n2])*density_scale+1.);
        }
      }
      else {
      }
    }
  }
  else {
  }
}

bc_sym_x(sgn,topbot,VtxBuffer j,rel)
{
  int i;
  if(topbot == AC_bot) {
    if (rel) {
      for i in 1:NGHOST_VAL+1 {
        j[AC_l1-i-1][vertexIdx.y][vertexIdx.z]=2*j[AC_l1-1][vertexIdx.y][vertexIdx.z]+sgn*j[AC_l1+i-1][vertexIdx.y][vertexIdx.z];
      }
    }
    else {
      for i in 1:NGHOST_VAL+1 {
        j[AC_l1-i-1][vertexIdx.y][vertexIdx.z]=              sgn*j[AC_l1+i-1][vertexIdx.y][vertexIdx.z];
      }
      if (sgn<0) {
        j[AC_l1-1][vertexIdx.y][vertexIdx.z] = 0.;
      }
    }
  }
  else if(topbot == AC_top) {
    if (rel) {
      for i in 1:NGHOST_VAL+1 {
        j[AC_l2+i-1][vertexIdx.y][vertexIdx.z]=2*j[AC_l2-1][vertexIdx.y][vertexIdx.z]+sgn*j[AC_l2-i-1][vertexIdx.y][vertexIdx.z];
      }
    }
    else {
      for i in 1:NGHOST_VAL+1 {
        j[AC_l2+i-1][vertexIdx.y][vertexIdx.z]=              sgn*j[AC_l2-i-1][vertexIdx.y][vertexIdx.z];
      }
      if (sgn<0) {
        j[AC_l2-1][vertexIdx.y][vertexIdx.z] = 0.;
      }
    }
  }
  else {
  }
}


bc_sym_y(sgn,topbot,VtxBuffer j,rel)
{
  int i;
  if(topbot == AC_bot) {
    if (rel) {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][AC_m1-i-1][vertexIdx.z]=2*j[vertexIdx.x][AC_m1-1][vertexIdx.z]+sgn*j[vertexIdx.x][AC_m1+i-1][vertexIdx.z];
      }
    }
    else {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][AC_m1-i-1][vertexIdx.z]=              sgn*j[vertexIdx.x][AC_m1+i-1][vertexIdx.z];
      }
      if (sgn<0) {
        j[vertexIdx.x][AC_m1-1][vertexIdx.z] = 0.;
      }
    }
  }
  else if(topbot == AC_top) {
    if (rel) {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][AC_m2+i-1][vertexIdx.z]=2*j[vertexIdx.x][AC_m2-1][vertexIdx.z]+sgn*j[vertexIdx.x][AC_m2-i-1][vertexIdx.z];
      }
    }
    else {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][AC_m2+i-1][vertexIdx.z]=              sgn*j[vertexIdx.x][AC_m2-i-1][vertexIdx.z];
      }
      if (sgn<0) {
        j[vertexIdx.x][AC_m2-1][vertexIdx.z] = 0.;
      }
    }
  }
  else {
  }
}


bc_sym_z(sgn,topbot,VtxBuffer j,rel)
{
  int i;
  if(topbot == AC_bot) {
    if (rel) {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][AC_n1-i-1]=2*j[vertexIdx.x][vertexIdx.y][AC_n1-1]+sgn*j[vertexIdx.x][vertexIdx.y][AC_n1+i-1];
      }
    }
    else {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][AC_n1-i-1]=              sgn*j[vertexIdx.x][vertexIdx.y][AC_n1+i-1];
      }
      if (sgn<0) {
        j[vertexIdx.x][vertexIdx.y][AC_n1-1] = 0.;
      }
    }
  }
  else if(topbot == AC_top) {
    if (rel) {
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][AC_n2+i-1]=j[vertexIdx.x][vertexIdx.y][AC_n2-1]+(j[vertexIdx.x][vertexIdx.y][AC_n2-1]+sgn*j[vertexIdx.x][vertexIdx.y][AC_n2-i-1]);
      }
    }
    else { 
      for i in 1:NGHOST_VAL+1 {
        j[vertexIdx.x][vertexIdx.y][AC_n2+i-1]=              sgn*j[vertexIdx.x][vertexIdx.y][AC_n2-i-1];
      }
      if (sgn<0) {
        j[vertexIdx.x][vertexIdx.y][AC_n2-1] = 0.;
      }
    }
  }
  else {
  }
}   


bc_set_der_x(topbot,VtxBuffer j,val)
{
  int i;
  if(topbot == AC_bot) {
    for i in 1:NGHOST_VAL+1 {
      j[AC_l1-i-1][vertexIdx.y][vertexIdx.z] = j[AC_l1+i-1][vertexIdx.y][vertexIdx.z] - AC_dx2_bound[-i+NGHOST_VAL+1-1]*val;
    }
  }
  else if(topbot == AC_top) {
    for i in 1:NGHOST_VAL+1 {
      j[AC_l2+i-1][vertexIdx.y][vertexIdx.z] = j[AC_l2-i-1][vertexIdx.y][vertexIdx.z] + AC_dx2_bound[i+1+NGHOST_VAL-1]*val;
    }
  }
  else {
  }
}

bc_set_der_y(topbot,VtxBuffer j,val)
{
  int i;
  if(topbot == AC_bot) {
    for i in 1:NGHOST_VAL+1 {
      j[vertexIdx.x][AC_m1-i-1][vertexIdx.z] = j[vertexIdx.x][AC_m1+i-1][vertexIdx.z] - AC_dy2_bound[-i+NGHOST_VAL+1-1]*val;
    }
  }
  else if(topbot == AC_top) {
    for i in 1:NGHOST_VAL+1 {
      j[vertexIdx.x][AC_m2+i-1][vertexIdx.z] = j[vertexIdx.x][AC_m2-i-1][vertexIdx.z] + AC_dy2_bound[i+1+NGHOST_VAL-1]*val;
    }
  }
  else {
  }
}



bc_set_der_z(topbot,VtxBuffer j,val)
{
  int i;
  if(topbot == AC_bot) {
    for i in 1:NGHOST_VAL+1 {
      j[vertexIdx.x][vertexIdx.y][AC_n1-i-1] = j[vertexIdx.x][vertexIdx.y][AC_n1+i-1] - AC_dz2_bound[-i+NGHOST_VAL+1-1]*val;
    }
  }
  else if(topbot == AC_top) {
    for i in 1:NGHOST_VAL+1 {
      j[vertexIdx.x][vertexIdx.y][AC_n2+i-1] = j[vertexIdx.x][vertexIdx.y][AC_n2-i-1] + AC_dz2_bound[i+1+NGHOST_VAL-1]*val;
    }
  }
  else {
  }
}

