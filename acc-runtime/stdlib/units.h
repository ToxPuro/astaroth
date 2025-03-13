// Physical constants
run_const real G_CONST_CGS 6.674e-8   // cm^3/(g*s^2) GGS definition //TODO define in a separate module
run_const real M_sun = 1.989e33       // g solar mass
run_const real mu0_cgs = 4.0 * M_PI

// Physical units
run_const real AC_unit_density
run_const real AC_unit_velocity
run_const real AC_unit_length
// To force mu0 = 1 from pencil code lfix_unit_std 3.5449077018110318=sqrt(4*pi)
run_const real AC_unit_magnetic = 3.5449077018110318*sqrt(AC_unit_density)*AC_unit_velocity 
run_const real AC_unit_mass = (AC_unit_length*AC_unit_length*AC_unit_length)*AC_unit_density;
// Derived units
// Should give AC_mu0 = 1 with the above definition of AC_unit_magnetic
run_const real AC_mu0 = mu0_cgs * AC_unit_density * ((AC_unit_velocity/AC_unit_magnetic) * (AC_unit_velocity/AC_unit_magnetic))
run_const real AC_G_const = G_CONST_CGS / ((AC_unit_velocity * AC_unit_velocity) / (AC_unit_density * AC_unit_length * AC_unit_length))
