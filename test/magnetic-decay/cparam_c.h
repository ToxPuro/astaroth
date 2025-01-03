#define MIN(a,b) (a<b ? a : b)
#define MAX(a,b) (a>b ? a : b)

#if IN_DSL
 #include "../../../headers_c.h"
#else
  
#include <float.h>
#include <limits.h>
 #include "headers_c.h"
#endif
#define y0 y0_
;
#if IN_DSL
  #include "../../../cparam.local_c.h"
#else
  #include "cparam.local_c.h"
#endif
  const int  nx=nxgrid/nprocx,ny=nygrid/nprocy,nz=nzgrid/nprocz,nyz=ny*nz;
  const int  nxygrid=nxgrid*nygrid,nxzgrid=nxgrid*nzgrid,nyzgrid=nygrid*nzgrid;
  const int  nprocxy=nprocx*nprocy;
  const int  nprocyz=nprocy*nprocz;
  const int  nprocxz=nprocx*nprocz;
  const int  n_forcing_cont_max=2;
  const int  ndustspec0=8;
  const int  dimensionality=MIN(nxgrid-1,1)+MIN(nygrid-1,1)+MIN(nzgrid-1,1);
#if IN_DSL
  #include "../../../cparam.inc_c.h"
#else
  #include "cparam.inc_c.h"
#endif
  const bool  lenergy=lentropy | ltemperature | lthermal_energy;
  const int  penc_name_len=16;
  const int  mfarray=mvar+maux+mglobal+mscratch;
  const int  mcom=mvar+maux_com;
  const int  mparray=mpvar+mpaux;
  const int  mpcom=mpvar+mpaux;
  const int  mqarray=mqvar+mqaux;
  const long long  nw=nx*ny*nz;
  const int  mx=nx+2*nghost,l1=1+nghost;
  const int  my=ny+2*nghost,m1=1+nghost;
  const int  mz=nz+2*nghost,n1=1+nghost;
  const int  mxgrid=nxgrid+2*nghost;
  const int  mygrid=nygrid+2*nghost;
  const int  mzgrid=nzgrid+2*nghost;
  const int  mw=mx*my*mz;
  const long long  nwgrid=(long long)nxgrid* 
                                            (long long)nygrid* 
                                            (long long)nzgrid;
  const int  l1i=l1+nghost-1;
  const int  m1i=m1+nghost-1;
  const int  n1i=n1+nghost-1;


  const int  nrcyl=nxgrid/2;
  const int  nrcylrun=MAX(nx/20,1);
  const int  nbin_angular=19*2;
  const int  mreduce=6;
  const int  ninit=5;
  const int  fnlen=135,intlen=21,bclen=3,labellen=40,linelen=256;
  const int  datelen=30,max_col_width=30,nscbc_len=24,fmtlen=30;
  const int  mseed=256;
  const long  int_sgl=0;
  const int  max_int=INT_MAX;
  const real  huge_real=REAL_MAX;
  const double  zero_double=0., huge_double=HUGE_VAL;
  const real  max_real=huge_real/10.;
  const real  one_real=1.0;
  const real  huge1=0.2*huge_real;
  const real  impossible=3.9085e37;
  const int  impossible_int=-max_int/100;
  const int  root=0;
  const int  ilabel_max=-1,ilabel_sum=1,ilabel_save=0;
  const int  ilabel_max_sqrt=-2,ilabel_sum_sqrt=2;
  const int  ilabel_sum_log10=10, ilabel_sum_masked=11;
  const int  ilabel_max_dt=-3,ilabel_max_neg=-4;
  const int  ilabel_max_reciprocal=-5;
  const int  ilabel_integrate=3,ilabel_integrate_sqrt=30, ilabel_integrate_log10=40;
  const int  ilabel_surf=4;
  const int  ilabel_sum_par=5,ilabel_sum_sqrt_par=6, ilabel_sum_log10_par=20, ilabel_sum_plain=21;
  const int  ilabel_sum_weighted=7,ilabel_sum_weighted_sqrt=8;
  const int  ilabel_sum_lim=9,ilabel_complex=100;
  const real  lntwo=0.69314718055995E0;
  const real  k1bessel0=2.4048255577, k1bessel1=3.8317060;
  const real  k2bessel0=5.5200781;
  const real  pi=3.14159265358979323846264338327950E0;
  const real  pi_1=1./pi,pi4_1=(1.0)/(pi*pi*pi*pi),pi5_1=1.0/(pi*pi*pi*pi*pi);
  const real  sqrtpi=1.77245385090551602729816748334115E0;
  const real  sqrt2=1.41421356237309504880168872420970E0;
  const real  sqrt2pi=sqrt2*sqrtpi;
  const real  four_pi_over_three=4.0/3.0*pi;
  const real  onethird=1./3., twothird=2./3., fourthird=4./3., onesixth=1./6.;
  const real  one_over_sqrt3=0.577350269189625764509148780501958E0;
  const real  twopi = 6.2831853071795864769252867665590E0;
  const real  dtor = pi/180.E0;
  const double  hbar_cgs=1.054571596E-27;
  const double  k_b_cgs=1.3806505E-16;
  const double  m_u_cgs=1.66053886E-24;
  const double  mu0_cgs=4*pi;
  const double  r_cgs=8.3144E7;
  const double  m_p_cgs=1.67262158E-24;
  const double  m_e_cgs=9.10938188E-28;
  const double  m_h_cgs=m_e_cgs+m_p_cgs;
  const double  ev_cgs=1.602176462E-12;
  const double  sigmasb_cgs=5.670400E-5;
  const double  sigmah_cgs=4.E-17;
  const double  kappa_es_cgs=3.4E-1;
  const double  c_light_cgs=2.99792458E10;
  const double  g_newton_cgs=6.6742E-8;
  const double  density_scale_cgs=1.2435E21;
  const double  n_avogadro_cgs=6.022E23;
  const bool  always_false=false;
  const int  ibc_x_top=1;
  const int  ibc_x_bot=-1;
  const int  ibc_y_top=2;
  const int  ibc_y_bot=-2;
  const int  ibc_z_top=3;
  const int  ibc_z_bot=-3;
  const int  bot=1, top=2, both=3;
  const int  iref_rho=1, iref_grho=2, iref_d2rho=3, iref_d6rho=4, 
                        iref_gp=5, iref_s=6, iref_gs=7, iref_d2s=8, iref_d6s=9;
  const int  nref_vars=9;
  const int  bilin=1, biquad=2, bicub=3, quadspline=4, biquin=5;
  const int  xplus=1, yplus=2, xminus=3, yminus=4, zplus=5, zminus=6;
  const int  max_threads_possible = 200;
  const int  perf_diags=1, perf_wsnap=2, perf_powersnap=3, perf_wsnap_down=4;
  const int  n_helperflags=4;
  const int  n_xy_specs_max=10,nk_max=10, nz_max=10;
  const int  mname=100;
  const int  mname_half=20;

