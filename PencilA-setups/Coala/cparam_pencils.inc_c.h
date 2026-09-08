#pragma once

const int  npencils=239;

const int  i_acc=1;
const int  i_ssat=2;
const int  i_ttc=3;
const int  i_ywater=4;
const int  i_lambda=5;
const int  i_chem_conc=6;
const int  i_nucl_rmin=7;
const int  i_nucl_rate=8;
const int  i_conc_satm=9;
const int  i_ff_cond=10;
const int  i_latent_heat=11;
const int  i_lnrho=12;
const int  i_rho=13;
const int  i_rho1=14;
const int  i_glnrho=15;
const int  i_grho=16;
const int  i_uglnrho=17;
const int  i_ugrho=18;
const int  i_glnrho2=19;
const int  i_del2lnrho=20;
const int  i_del2rho=21;
const int  i_del6lnrho=22;
const int  i_del6rho=23;
const int  i_hlnrho=24;
const int  i_sglnrho=25;
const int  i_uij5glnrho=26;
const int  i_transprho=27;
const int  i_ekin=28;
const int  i_uuadvec_glnrho=29;
const int  i_uuadvec_grho=30;
const int  i_rhos1=31;
const int  i_glnrhos=32;
const int  i_totenergy_rel=33;
const int  i_divss=34;
const int  i_ext_force=35;
const int  i_glnnd=36;
const int  i_gmi=37;
const int  i_gmd=38;
const int  i_gnd=39;
const int  i_grhod=40;
const int  i_ad=41;
const int  i_md=42;
const int  i_mi=43;
const int  i_nd=44;
const int  i_rhod=45;
const int  i_rhod1=46;
const int  i_epsd=47;
const int  i_udgmi=48;
const int  i_udgmd=49;
const int  i_udglnnd=50;
const int  i_udgnd=51;
const int  i_glnnd2=52;
const int  i_sdglnnd=53;
const int  i_del2nd=54;
const int  i_del2rhod=55;
const int  i_del6nd=56;
const int  i_del2md=57;
const int  i_del2mi=58;
const int  i_del6lnnd=59;
const int  i_gndglnrho=60;
const int  i_glnndglnrho=61;
const int  i_udrop=62;
const int  i_udropgnd=63;
const int  i_fcloud=64;
const int  i_ccondens=65;
const int  i_ppwater=66;
const int  i_ppsat=67;
const int  i_ppsf=68;
const int  i_mu1=69;
const int  i_udropav=70;
const int  i_glnrhod=71;
const int  i_rhodsum=72;
const int  i_rhodsum1=73;
const int  i_grhodsum=74;
const int  i_glnrhodsum=75;
const int  i_old_uud=76;
const int  i_divud=77;
const int  i_ood=78;
const int  i_od2=79;
const int  i_oud=80;
const int  i_ud2=81;
const int  i_udij=82;
const int  i_sdij=83;
const int  i_udgud=84;
const int  i_uud=85;
const int  i_del2ud=86;
const int  i_del6ud=87;
const int  i_graddivud=88;
const int  i_advec_uud=89;
const int  i_ma2=90;
const int  i_fpres=91;
const int  i_tcond=92;
const int  i_sglntt=93;
const int  i_uglntt=94;
const int  i_advec_cs2=95;
const int  i_ss=96;
const int  i_gss=97;
const int  i_ee=98;
const int  i_pp=99;
const int  i_lntt=100;
const int  i_cs2=101;
const int  i_cv1=102;
const int  i_cp1=103;
const int  i_cp1tilde=104;
const int  i_glntt=105;
const int  i_tt=106;
const int  i_tt1=107;
const int  i_cp=108;
const int  i_cv=109;
const int  i_gtt=110;
const int  i_glnmu=111;
const int  i_gmu1=112;
const int  i_yh=113;
const int  i_hss=114;
const int  i_hlntt=115;
const int  i_del2ss=116;
const int  i_del6ss=117;
const int  i_del2tt=118;
const int  i_del2lntt=119;
const int  i_del6tt=120;
const int  i_del6lntt=121;
const int  i_glnmumol=122;
const int  i_csvap2=123;
const int  i_rho_anel=124;
const int  i_rho1gpp=125;
const int  i_fcont=126;
const int  i_curlfcont=127;
const int  i_gg=128;
const int  i_x_mn=129;
const int  i_y_mn=130;
const int  i_z_mn=131;
const int  i_r_mn=132;
const int  i_r_mn1=133;
const int  i_phix=134;
const int  i_phiy=135;
const int  i_pomx=136;
const int  i_pomy=137;
const int  i_rcyl_mn=138;
const int  i_rcyl_mn1=139;
const int  i_phi_mn=140;
const int  i_evr=141;
const int  i_rr=142;
const int  i_evth=143;
const int  i_divu=144;
const int  i_oo=145;
const int  i_o2=146;
const int  i_ou=147;
const int  i_oxu2=148;
const int  i_oxu=149;
const int  i_u2=150;
const int  i_uij=151;
const int  i_uu=152;
const int  i_curlo=153;
const int  i_sij=154;
const int  i_sij2=155;
const int  i_uij5=156;
const int  i_ugu=157;
const int  i_ugu2=158;
const int  i_oij=159;
const int  i_d2uidxj=160;
const int  i_uijk=161;
const int  i_ogu=162;
const int  i_u3u21=163;
const int  i_u1u32=164;
const int  i_u2u13=165;
const int  i_del2u=166;
const int  i_del4u=167;
const int  i_del6u=168;
const int  i_u2u31=169;
const int  i_u3u12=170;
const int  i_u1u23=171;
const int  i_graddivu=172;
const int  i_del6u_bulk=173;
const int  i_grad5divu=174;
const int  i_rhougu=175;
const int  i_der6u=176;
const int  i_transpurho=177;
const int  i_divu0=178;
const int  i_u0ij=179;
const int  i_uu0=180;
const int  i_uu_advec=181;
const int  i_uuadvec_guu=182;
const int  i_del6u_strict=183;
const int  i_del4graddivu=184;
const int  i_uu_sph=185;
const int  i_der6u_res=186;
const int  i_lorentz=187;
const int  i_hless=188;
const int  i_lorentz_gamma2=189;
const int  i_lorentz_gamma=190;
const int  i_ss_rel2=191;
const int  i_ss_rel=192;
const int  i_ss_rel_ij=193;
const int  i_ss_rel_factor=194;
const int  i_divss_rel=195;
const int  i_heat=196;
const int  i_cool=197;
const int  i_heatcool=198;
const int  i_bb=199;
const int  i_bbb=200;
const int  i_bij=201;
const int  i_jxbr=202;
const int  i_ss12=203;
const int  i_b2=204;
const int  i_uxb=205;
const int  i_jj=206;
const int  i_aa=207;
const int  i_diva=208;
const int  i_del2a=209;
const int  i_aij=210;
const int  i_bunit=211;
const int  i_va2=212;
const int  i_j2=213;
const int  i_el=214;
const int  i_e2=215;
const int  i_uun=216;
const int  i_divun=217;
const int  i_snij=218;
const int  i_rhop=219;
const int  i_grhop=220;
const int  i_peh=221;
const int  i_tauascalar=222;
const int  i_condensationrate=223;
const int  i_watermixingratio=224;
const int  i_part_heatcap=225;
const int  i_cc=226;
const int  i_cc1=227;
const int  i_gcc=228;
const int  i_sgs_heat=229;
const int  i_shock=230;
const int  i_gshock=231;
const int  i_shock_perp=232;
const int  i_gshock_perp=233;
const int  i_fvisc=234;
const int  i_diffus_total=235;
const int  i_visc_heat=236;
const int  i_nu=237;
const int  i_gradnu=238;
const int  i_nu_smag=239;
   /** acc             ', 'ssat            ', 'ttc             ', 'ywater          '  
   , 'lambda          ', 'chem_conc       ', 'nucl_rmin       ', 'nucl_rate       ', 'conc_satm       '  
   , 'ff_cond         ', 'latent_heat     ', 'lnrho           ', 'rho             ', 'rho1            '  
   , 'glnrho          ', 'grho            ', 'uglnrho         ', 'ugrho           ', 'glnrho2         '  
   , 'del2lnrho       ', 'del2rho         ', 'del6lnrho       ', 'del6rho         ', 'hlnrho          '  
   , 'sglnrho         ', 'uij5glnrho      ', 'transprho       ', 'ekin            ', 'uuadvec_glnrho  '  
   , 'uuadvec_grho    ', 'rhos1           ', 'glnrhos         ', 'totenergy_rel   ', 'divss           '  
   , 'ext_force       ', 'glnnd           ', 'gmi             ', 'gmd             ', 'gnd             '  
   , 'grhod           ', 'ad              ', 'md              ', 'mi              ', 'nd              '  
   , 'rhod            ', 'rhod1           ', 'epsd            ', 'udgmi           ', 'udgmd           '  
   , 'udglnnd         ', 'udgnd           ', 'glnnd2          ', 'sdglnnd         ', 'del2nd          '  
   , 'del2rhod        ', 'del6nd          ', 'del2md          ', 'del2mi          ', 'del6lnnd        '  
   , 'gndglnrho       ', 'glnndglnrho     ', 'udrop           ', 'udropgnd        ', 'fcloud          '  
   , 'ccondens        ', 'ppwater         ', 'ppsat           ', 'ppsf            ', 'mu1             '  
   , 'udropav         ', 'glnrhod         ', 'rhodsum         ', 'rhodsum1        ', 'grhodsum        '  
   , 'glnrhodsum      ', 'old_uud         ', 'divud           ', 'ood             ', 'od2             '  
   , 'oud             ', 'ud2             ', 'udij            ', 'sdij            ', 'udgud           '  
   , 'uud             ', 'del2ud          ', 'del6ud          ', 'graddivud       ', 'advec_uud       '  
   , 'ma2             ', 'fpres           ', 'tcond           ', 'sglntt          ', 'uglntt          '  
   , 'advec_cs2       ', 'ss              ', 'gss             ', 'ee              ', 'pp              '  
   , 'lntt            ', 'cs2             ', 'cv1             ', 'cp1             ', 'cp1tilde        '  
   , 'glntt           ', 'tt              ', 'tt1             ', 'cp              ', 'cv              '  
   , 'gtt             ', 'glnmu           ', 'gmu1            ', 'yh              ', 'hss             '  
   , 'hlntt           ', 'del2ss          ', 'del6ss          ', 'del2tt          ', 'del2lntt        '  
   , 'del6tt          ', 'del6lntt        ', 'glnmumol        ', 'csvap2          ', 'rho_anel        '  
   , 'rho1gpp         ', 'fcont           ', 'curlfcont       ', 'gg              ', 'x_mn            '  
   , 'y_mn            ', 'z_mn            ', 'r_mn            ', 'r_mn1           ', 'phix            '  
   , 'phiy            ', 'pomx            ', 'pomy            ', 'rcyl_mn         ', 'rcyl_mn1        '  
   , 'phi_mn          ', 'evr             ', 'rr              ', 'evth            ', 'divu            '  
   , 'oo              ', 'o2              ', 'ou              ', 'oxu2            ', 'oxu             '  
   , 'u2              ', 'uij             ', 'uu              ', 'curlo           ', 'sij             '  
   , 'sij2            ', 'uij5            ', 'ugu             ', 'ugu2            ', 'oij             '  
   , 'd2uidxj         ', 'uijk            ', 'ogu             ', 'u3u21           ', 'u1u32           '  
   , 'u2u13           ', 'del2u           ', 'del4u           ', 'del6u           ', 'u2u31           '  
   , 'u3u12           ', 'u1u23           ', 'graddivu        ', 'del6u_bulk      ', 'grad5divu       '  
   , 'rhougu          ', 'der6u           ', 'transpurho      ', 'divu0           ', 'u0ij            '  
   , 'uu0             ', 'uu_advec        ', 'uuadvec_guu     ', 'del6u_strict    ', 'del4graddivu    '  
   , 'uu_sph          ', 'der6u_res       ', 'lorentz         ', 'hless           ', 'lorentz_gamma2  '  
   , 'lorentz_gamma   ', 'ss_rel2         ', 'ss_rel          ', 'ss_rel_ij       ', 'ss_rel_factor   '  
   , 'divss_rel       ', 'heat            ', 'cool            ', 'heatcool        ', 'bb              '  
   , 'bbb             ', 'bij             ', 'jxbr            ', 'ss12            ', 'b2              '  
   , 'uxb             ', 'jj              ', 'aa              ', 'diva            ', 'del2a           '  
   , 'aij             ', 'bunit           ', 'va2             ', 'j2              ', 'el              '  
   , 'e2              ', 'uun             ', 'divun           ', 'snij            ', 'rhop            '  
   , 'grhop           ', 'peh             ', 'tauascalar      ', 'condensationrate', 'watermixingratio'  
   , 'part_heatcap    ', 'cc              ', 'cc1             ', 'gcc             ', 'sgs_heat        '  
   , 'shock           ', 'gshock          ', 'shock_perp      ', 'gshock_perp     ', 'fvisc           '  
   , 'diffus_total    ', 'visc_heat       ', 'nu              ', 'gradnu          ', 'nu_smag         '  
**/;
