int AC_bc_type_bot_x
int AC_bc_type_bot_y
int AC_bc_type_bot_z

int AC_bc_type_top_x
int AC_bc_type_top_y
int AC_bc_type_top_z
int AC_init_type

input real AC_dt
input real AC_current_time
global output real AC_dt_min

// Additional params needed by standalone & standalone_mpi
// diagnostics period
int AC_save_steps

// snapshot period
int AC_bin_steps
real AC_bin_save_t

// slices output period
int AC_slice_steps
real AC_slice_save_t

// maximun number of time snapshots during runtime 
// Set AC_num_snapshots < 0 for unlimited snapshots
int AC_num_snapshots

// max simulation time
int AC_max_steps
real AC_max_time

// Forcing parameter generation period (if forcing is on)
int AC_forcing_period_steps
real AC_forcing_period_t

// Initial time step index, default should be 0
int AC_start_step

run_const bool AC_additive_timestep = true
global output real ALFVEN_SPEED_MAX
global output real UU_MAX_ADVEC
global output real AD_ONE_FLUID_MAX_ADVEC
global output real AC_MAX_SHOCK
run_const bool AC_timestep_calc_with_rhs = true
