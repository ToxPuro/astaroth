/**
 * This file contains handy units that can be used when defining the values of other variables
 * Also derived units (only mass at the moment) are contained in here.
 */
run_const real AC_unit_density
run_const real AC_unit_velocity
run_const real AC_unit_length
run_const real AC_unit_magnetic
run_const real AC_unit_mass = (AC_unit_length*AC_unit_length*AC_unit_length)*AC_unit_density;
