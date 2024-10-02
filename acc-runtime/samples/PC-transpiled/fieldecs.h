Field F_UX, F_UY, F_UZ
Field F_PHIUU
Field F_U0X, F_U0Y, F_U0Z
Field F_RHO
#define F_LNRHO F_RHO
Field F_LORENTZ
Field F_GUIJ11,F_GUIJ21,F_GUIJ31,F_GUIJ12,F_GUIJ22,F_GUIJ32,F_GUIJ13,F_GUIJ23,F_GUIJ33
Field F_OX, F_OY, F_OZ
Field F_UU_SPHX, F_UU_SPHY, F_UU_SPHZ
Field F_BB_SPHX, F_BB_SPHY, F_BB_SPHZ
Field F_HLESS
Field F_EOSVAR2
Field F_GLOBAL_CS2
Field F_PP
Field F_SS
Field F_SS_B
Field F_PP
Field F_RHO_B
Field F_ETH
Field F_GLOBAL_LNRHO0
Field F_GLOBAL_SS0
Field F_HYPVIS
Field F_NUSMAG
Field F_AX, F_AY, F_AZ
Field F_BX, F_BY, F_BZ
Field F_GLOBAL_EXT_BX, F_GLOBAL_EXT_BY, F_GLOBAL_EXT_BZ
Field F_GLOBAL_EXT_AX, F_GLOBAL_EXT_AY, F_GLOBAL_EXT_AZ

Field F_GLOBAL_EEXT1X,F_GLOBAL_EEXT1Y,F_GLOBAL_EEXT1Z
Field F_GLOBAL_EEXT2X,F_GLOBAL_EEXT2Y,F_GLOBAL_EEXT2Z
Field F_GLOBAL_EEXT3X,F_GLOBAL_EEXT3Y,F_GLOBAL_EEXT3Z

Field F_GLOBAL_JEXT1X,F_GLOBAL_JEXT1Y,F_GLOBAL_JEXT1Z
Field F_GLOBAL_JEXT2X,F_GLOBAL_JEXT2Y,F_GLOBAL_JEXT2Z
Field F_GLOBAL_JEXT3X,F_GLOBAL_JEXT3Y,F_GLOBAL_JEXT3Z

Field F_JX,F_JY,F_JZ
Field F_EDOTX,F_EDOTY,F_EDOTZ
Field F_LAM
Field F_TT
Field F_GLOBAL_HCOND
Field F_SS_RUN_AVER
Field F_ADV_DERX
Field F_ADV_DERY
Field F_ADV_DERZ
Field F_GLOBAL_GLNTX,F_GLOBAL_GLNTY,F_GLOBAL_GLNTZ
Field F_GLOBAL_GLHX,F_GLOBAL_GLHY,F_GLOBAL_GLHZ
Field F_AX,F_AY,F_AZ
Field F_HYPREX, F_HYPREY, F_HYPREZ
Field F_ETAT



const Field3 F_GLOBAL_GLNTVEC = Field3(F_GLOBAL_GLNTX,F_GLOBAL_GLNTY,F_GLOBAL_GLNTZ)
const Field3 F_AVEC    = Field3(F_AX, F_AY, F_AZ)
const Field3 F_UVEC    = Field3(F_UX,F_UY,F_UZ)
const Field3 F_UU      = Field3(F_UX,F_UY,F_UZ)
const Field3 F_U0VEC   = Field3(F_U0X, F_U0Y, F_U0Z)
const Field3 F_OVEC    = Field3(F_OX, F_OY, F_OZ)
const Field3 F_UU_SPH_VEC  = Field3(F_UU_SPHX, F_UU_SPHY, F_UU_SPHZ)
const Field3 F_UU_SPHVEC   = Field3(F_UU_SPHX, F_UU_SPHY, F_UU_SPHZ)
const Field3 F_BB_SPHVEC   = Field3(F_BB_SPHX, F_BB_SPHY, F_BB_SPHZ)
const Field3 F_BVEC        = Field3(F_BX,F_BY,F_BZ)
const Field3 F_GLOBAL_GLHVEC = Field3(F_GLOBAL_GLHX,F_GLOBAL_GLHY,F_GLOBAL_GLHZ)

const Field3 F_GLOBAL_EEXT1VEC = Field3(F_GLOBAL_EEXT1X,F_GLOBAL_EEXT1Y,F_GLOBAL_EEXT1Z)
const Field3 F_GLOBAL_EEXT2VEC = Field3(F_GLOBAL_EEXT2X,F_GLOBAL_EEXT2Y,F_GLOBAL_EEXT2Z)
const Field3 F_GLOBAL_EEXT3VEC = Field3(F_GLOBAL_EEXT3X,F_GLOBAL_EEXT3Y,F_GLOBAL_EEXT3Z)

const Field3 F_GLOBAL_JEXT1VEC = Field3(F_GLOBAL_JEXT1X,F_GLOBAL_JEXT1Y,F_GLOBAL_JEXT1Z)
const Field3 F_GLOBAL_JEXT2VEC = Field3(F_GLOBAL_JEXT2X,F_GLOBAL_JEXT2Y,F_GLOBAL_JEXT2Z)
const Field3 F_GLOBAL_JEXT3VEC = Field3(F_GLOBAL_JEXT3X,F_GLOBAL_JEXT3Y,F_GLOBAL_JEXT3Z)

const Field3 F_JVEC            = Field3(F_JX,F_JY,F_JZ)
const Field3 F_EDOTVEC         = Field3(F_EDOTX,F_EDOTY,F_EDOTZ)
const Field3 F__ADV_DERVEC     = Field3(F_ADV_DERX,F_ADV_DERY,F_ADV_DERZ)
const Field3 F_HYPREVEC        = Field3(F_HYPREX, F_HYPREY, F_HYPREZ)
const Field3 F_GLOBAL_EXT_AVEC = Field3(F_GLOBAL_EXT_AX, F_GLOBAL_EXT_AY, F_GLOBAL_EXT_AZ)


not_implemented(message)
{
    print("NOT IMPLEMENTED: %s\n",message)
}
real AC_t__mod__cdata
run_const real3 AC_xyz1__mod__cdata
run_const real3 AC_xyz0__mod__cdata
const real AC_tini__mod__cparam = AC_REAL_MIN*5.0


tini_sqrt_div(real numerator, real a, real b)
{
  return 
     (abs(a) <= AC_tini__mod__cparam || abs(b) <= AC_tini__mod__cparam)
     ? 0.0
     : numerator/(sqrt(a*b)) 
}
tini_sqrt_div_separate(real numerator, real a, real b)
{
  return 
     (abs(a) <= AC_tini__mod__cparam || abs(b) <= AC_tini__mod__cparam)
     ? 0.0
     : numerator/(sqrt(a)*sqrt(b)) 
}


#define AC_maux__mod__cparam maux__mod__cparam
const int AC_dimensionality__mod__cparam  = 3
#define iphF_UU F_PHIUU
const real AC_pi__mod__cparam = AC_REAL_PI
const real AC_pi_1__mod__cparam=1./AC_pi__mod__cparam
const real AC_pi4_1__mod__cparam=AC_pi_1__mod__cparam*AC_pi_1__mod__cparam*AC_pi_1__mod__cparam*AC_pi_1__mod__cparam
const real AC_pi5_1__mod__cparam=AC_pi_1__mod__cparam*AC_pi_1__mod__cparam*AC_pi_1__mod__cparam*AC_pi_1__mod__cparam*AC_pi_1__mod__cparam
const real AC_sqrt2pi__mod__cparam =AC_sqrt2__mod__cparam*AC_sqrtpi__mod__cparam
const real AC_four_pi_over_three__mod__cparam =4.0/3.0*AC_pi__mod__cparam
const real AC_dtor__mod__cparam  = AC_pi__mod__cparam/180.0
const int AC_n1__mod__cparam = NGHOST_VAL+1
const int AC_m1__mod__cparam = NGHOST_VAL+1
const int AC_l1__mod__cparam = NGHOST_VAL+1

#define AC_n2__mod__cparam AC_nx+NGHOST_VAL+1 
#define AC_m2__mod__cparam AC_ny+NGHOST_VAL+1 
#define AC_l2__mod__cparam AC_nz+NGHOST_VAL+1 

run_const int AC_nx__mod__cparam
run_const int AC_ny__mod__cparam
run_const int AC_nz__mod__cparam

run_const int AC_mx__mod__cparam
run_const int AC_my__mod__cparam
run_const int AC_mz__mod__cparam

run_const int AC_nxyz_max


gmem real AC_hcond_prof__mod__energy[AC_nxyz_max]
gmem real AC_dlnhcond_prof__mod__energy[AC_nxyz_max]
gmem real AC_chit_prof_stored__mod__energy[AC_nxyz_max]
gmem real AC_dchit_prof_stored__mod__energy[AC_nxyz_max]
gmem real AC_chit_prof_fluct_stored__mod__energy[AC_nxyz_max]
gmem real AC_dchit_prof_fluct_stored__mod__energy[AC_nxyz_max]

#define AC_NGHOST_VAL__mod__cparam NGHOST_VAL
