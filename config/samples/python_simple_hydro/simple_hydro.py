import ctypes 
import pathlib

libname = pathlib.Path().absolute() / "libastaroth.so"
c_lib = ctypes.CDLL(libname)

#MV: In Python interace we cannot use any preprocessor macro stuff. Therefore
#MV: NUM_INT_PARAMS etc. need to be set manually for now.


#MV: Note mandatory double precision.

class py_int3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int),
                ("z", ctypes.c_int)
               ] 

class py_real3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double),
                ("z", ctypes.c_double)
               ] 

class py_vertex_buffer(ctypes.Structure):
    _fields_ = [(,)
               ]

'''
AcMeshInfo mesh_info
typedef struct {
  int int_params[NUM_INT_PARAMS];
  int3 int3_params[NUM_INT3_PARAMS];
  AcReal real_params[NUM_REAL_PARAMS];
  AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;
'''
class py_AcMeshInfo(ctypes.Structure):
    _fields_ = [("int_params",  NUM_INT_PARAMS   * ctypes.c_int),
                ("int_params",  NUM_INT3_PARAMS  * py_int3),
                ("real_params", NUM_REAL_PARAMS  * ctypes.c_double),
                ("rea3_params", NUM_REAL3_PARAMS * py_real3)
               ]
'''
typedef struct {
    AcReal* vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;
'''
class py_AcMesh(ctypes.Structure):
    _fields_ = [("vertex_buffer", NUM_VTXBUF_HANDLES * py_vertex_buffer),
                ("info", py_AcMeshInfo)
               ]

mesh_info = py_load_config(config_path)

mesh      = py_acmesh_create(mesh_info)

mesh      = py_initialize_mesh("random_fields", mesh)

t_step = 0.0 
device_number = 0

device = py_acDeviceCreate(device_number, mesh_info)
py_acDevicePrintInfo(device)
mesh = py_acDeviceLoadMesh(device, mesh)

#
# ...
#

py_acDeviceDestroy(device)

###########################################################
# Adapt what is down from this into Pythonic owrkable form.
# #$$ is a mark of code initial pythonic draft. 
# #!! is a mark of code to be adapted. 
###########################################################

#$$ AcMeshInfo mesh_info;
#$$ load_config(config_path, &mesh_info);
#$$ 
#$$ AcMesh* mesh = acmesh_create(mesh_info);
#$$ acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, mesh);
#$$ 
#$$ AcReal t_step        = 0.0;
#!!
#!! Device device;
#!! acDeviceCreate(0, mesh_info, &device);
#!! acDevicePrintInfo(device);
#!! printf("Loading mesh to GPU.\n");
#!! acDeviceLoadMesh(device, STREAM_DEFAULT, *mesh);
#!! 
#!!     printf("Mesh loaded to GPU(s).\n");
#!! 
#!!     FILE* diag_file;
#!!     int found_nan = 0, found_stop = 0; // Nan or inf finder to give an error signal
#!!     diag_file = fopen("timeseries.ts", "a");
#!! 
#!!     // Generate the title row.
#!!     if (start_step == 0) {
#!!         fprintf(diag_file, "step  t_step  dt  uu_total_min  uu_total_rms  uu_total_max  ");
#!!         for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
#!!             fprintf(diag_file, "%s_min  %s_rms  %s_max  ", vtxbuf_names[i], vtxbuf_names[i],
#!!                     vtxbuf_names[i]);
#!!         }
#!!     }
#!!     fprintf(diag_file, "\n");
#!! 
#!!     write_mesh_info(&mesh_info);
#!! 
#!!     printf("Mesh info written to file.\n");
#!! 
#!!     if (start_step == 0) {
#!!         print_diagnostics(0, AcReal(.0), t_step, diag_file, -1.0, -1.0, &found_nan);
#!!     }
#!! 
#!!     // acBoundcondStep();
#!!     acBoundcondStepGBC(mesh_info);
#!!     acStore(mesh);
#!!     if (start_step == 0) {
#!!         save_mesh(*mesh, 0, t_step);
#!!     }
#!! 
#!!     const int max_steps      = mesh_info.int_params[AC_max_steps];
#!!     const int save_steps     = mesh_info.int_params[AC_save_steps];
#!!     const int bin_save_steps = mesh_info.int_params[AC_bin_steps];
#!! 
#!!     const AcReal max_time   = mesh_info.real_params[AC_max_time];
#!!     const AcReal bin_save_t = mesh_info.real_params[AC_bin_save_t];
#!!     AcReal bin_crit_t       = bin_save_t;
#!! 
#!!     /* initialize random seed: */
#!!     srand(312256655);
#!! 
#!!     /* Step the simulation */
#!!     AcReal dt_typical    = 0.0;
#!!     int dtcounter        = 0;
#!! 
#!!     printf("Starting simulation...\n");
#!!     for (int i = start_step + 1; i < max_steps; ++i) {
#!!         const AcReal shock_max = 0.0;
#!!         const AcReal umax      = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
#!! 
#!!         const AcReal dt        = host_timestep(umax, 0.0l, shock_max, mesh_info);
#!! 
#!! #if LFORCING
#!!         const ForcingParams forcing_params = generateForcingParams(mesh_info);
#!!         loadForcingParamsToDevice(forcing_params);
#!! #endif
#!! 
#!!         /* Uses now flexible bokundary conditions */
#!!         // acIntegrate(dt);
#!!         acIntegrateGBC(mesh_info, dt);
#!! 
#!!         t_step += dt;
#!! 
#!!         /* Get the sense of a typical timestep */
#!!         if (i < start_step + 100) {
#!!             dt_typical = dt;
#!!         }
#!! 
#!!         /* Save the simulation state and print diagnostics */
#!!         if ((i % save_steps) == 0) {
#!! 
#!!             /*
#!!                 print_diagnostics() writes out both std.out printout from the
#!!                 results and saves the diagnostics into a table for ascii file
#!!                 timeseries.ts.
#!!             */
#!!             print_diagnostics(i, dt, t_step, diag_file, sink_mass, accreted_mass, &found_nan);
#!!             /*
#!!                 We would also might want an XY-average calculating funtion,
#!!                 which can be very useful when observing behaviour of turbulent
#!!                 simulations. (TODO)
#!!             */
#!!         }
#!! 
#!!         /* Save the simulation state and print diagnostics */
#!!         if ((i % bin_save_steps) == 0 || t_step >= bin_crit_t) {
#!! 
#!!             /*
#!!                 This loop saves the data into simple C binaries which can be
#!!                 used for analysing the data snapshots closely.
#!! 
#!!                 The updated mesh will be located on the GPU. Also all calls
#!!                 to the astaroth interface (functions beginning with ac*) are
#!!                 assumed to be asynchronous, so the meshes must be also synchronized
#!!                 before transferring the data to the CPU. Like so:
#!! 
#!!                 acBoundcondStep();
#!!                 acStore(mesh);
#!!             */
#!!             // acBoundcondStep();
#!!             acBoundcondStepGBC(mesh_info);
#!!             acStore(mesh);
#!!             save_mesh(*mesh, i, t_step);
#!! 
#!!             bin_crit_t += bin_save_t;
#!!         }
#!! 
#!!         // End loop if max time reached.
#!!         if (max_time > AcReal(0.0)) {
#!!             if (t_step >= max_time) {
#!!                 printf("Time limit reached! at t = %e \n", double(t_step));
#!!                 break;
#!!             }
#!!         }
#!! 
#!!         // End loop if dt is too low
#!!         if (dt < dt_typical / AcReal(1e5)) {
#!!             if (dtcounter > 10) {
#!!                 printf("dt = %e TOO LOW! Ending run at t = %#e \n", double(dt), double(t_step));
#!!                 // acBoundcondStep();
#!!                 acBoundcondStepGBC(mesh_info);
#!!                 acStore(mesh);
#!!                 save_mesh(*mesh, i, t_step);
#!!                 break;
#!!             }
#!!             else {
#!!                 dtcounter += 1;
#!!             }
#!!         }
#!!         else {
#!!             dtcounter = 0;
#!!         }
#!! 
#!!         // End loop if nan is found
#!!         if (found_nan > 0) {
#!!             printf("Found nan at t = %e \n", double(t_step));
#!!             // acBoundcondStep();
#!!             acBoundcondStepGBC(mesh_info);
#!!             acStore(mesh);
#!!             save_mesh(*mesh, i, t_step);
#!!             break;
#!!         }
#!! 
#!!         // End loop if STOP file is found
#!!         if (access("STOP", F_OK) != -1) {
#!!             found_stop = 1;
#!!         }
#!!         else {
#!!             found_stop = 0;
#!!         }
#!! 
#!!         if (found_stop == 1) {
#!!             printf("Found STOP file at t = %e \n", double(t_step));
#!!             // acBoundcondStep();
#!!             acBoundcondStepGBC(mesh_info);
#!!             acStore(mesh);
#!!             save_mesh(*mesh, i, t_step);
#!!             break;
#!!         }
#!!     }
#!! 
#!!     acDeviceDestroy(device);
#!!     acmesh_destroy(mesh);

