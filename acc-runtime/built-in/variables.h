/**
 *  First point the computational subdomain/domain (thus also the ghost zone sizes)
 */
int3 AC_nmin = (int3)
                {
                        AC_dimension_inactive.x ? 0 : NGHOST,
                        AC_dimension_inactive.y ? 0 : NGHOST,
                        AC_dimension_inactive.z ? 0 : NGHOST
                };
/**
 * Domain size (not incl. halos)
 * (Could be run_const but that gives really bad performance, most likely because they are interpreted as unsigned integers which messes the index calculations)
 */
int3 AC_ngrid;

/**
 * Specifies are the corresponding dimensions inactive. If a dimension is not active ghost zones in that dimension won't be allocated. Dimension is inactive by default if the grid has one grid point in that dimension.
 */
run_const bool3 AC_dimension_inactive = (bool3){
					AC_ngrid.x == 1,
					AC_ngrid.y == 1,
					TWO_D || AC_ngrid.z == 1
				};
/**
 * Domain size (incl. halos)
 */
int3 AC_mgrid  = AC_ngrid + 2*AC_nmin;

/**
 *  How the domain is decomposed to multiple processes.
 *  By default uses AC_decompose_strategy to calculate the domain decomposition (e.g. morton decomposition).
 */
run_const int3  AC_domain_decomposition = ac_get_process_decomposition();

/**
 * Subdomain size (not incl. halos)
 */
int3 AC_nlocal = (int3)
		{
			AC_ngrid.x/AC_domain_decomposition.x,
			AC_ngrid.y/AC_domain_decomposition.y,
			AC_ngrid.z/AC_domain_decomposition.z
		};
/**
 * Subdomain size (incl. halos)
 */
int3 AC_mlocal = AC_nlocal + 2*AC_nmin;
/**
 * Upper bound of the computational subdomain i.e. computational subdomain are the vertices:
 * AC_nmin <= vertex  < AC_nlocal_max
 */
int3 AC_nlocal_max = AC_nlocal + AC_nmin;
/**
 * Maximum extent in the computational subdomain
 */
int AC_nlocal_max_dim = max(AC_nlocal);
/**
 * Maximum extent in the subdomain (with halos included)
 */
int AC_mlocal_max_dim = max(AC_mlocal);
/**
 * Upper bound of the computational grid i.e. the computational grid are the vertices:
 * AC_nmin <= vertex < AC_ngrid_max
 */
int3 AC_ngrid_max = AC_ngrid + AC_nmin;

/**
 * Inverses of AC_nlocal
 */
run_const real3 AC_nlocal_inv = (real3) {1.0/AC_nlocal.x, 1.0/AC_nlocal.y, 1.0/AC_nlocal.z};

/**
 * Inverses of AC_ngrid
 */
run_const real3 AC_ngrid_inv = (real3){1.0/AC_ngrid.x, 1.0/AC_ngrid.y, 1.0/AC_ngrid.z};

/**
 * Products of the combinations of AC_nlocal
 */
AcDimProducts AC_nlocal_products = ac_get_dim_products(AC_nlocal);

/**
 * Inverses of AC_nlocal_products
 */
run_const AcDimProductsInv AC_nlocal_products_inv = ac_get_dim_products_inv(AC_nlocal_products);

/**
 * Products of the combinations of AC_ngrid
 */
AcDimProducts AC_ngrid_products = ac_get_dim_products(AC_ngrid);
/**
 * Inverses of AC_ngrid_products_inv
 */
run_const AcDimProductsInv AC_ngrid_products_inv = ac_get_dim_products_inv(AC_ngrid_products);

/**
 * Products of the combinations of AC_mlocal
 */
AcDimProducts AC_mlocal_products = ac_get_dim_products(AC_mlocal);
/**
 * Inverses of AC_mlocal_products_inv
 */
run_const AcDimProductsInv AC_mlocal_products_inv = ac_get_dim_products_inv(AC_mlocal_products);

/**
 * Products of the combinations of AC_mgrid
 */
AcDimProducts AC_mgrid_products = ac_get_dim_products(AC_mgrid);
/**
 * Inverses of AC_mgrid_products
 */
run_const AcDimProductsInv AC_mgrid_products_inv = ac_get_dim_products_inv(AC_mgrid_products);

/**
 * Is the grid periodic in the given dimensions
 */
run_const bool3 AC_periodic_grid;
/**
 * Overrides the normal safety feature of not allowing periodic bcs if the grid is not specified as periodic. Might be sometimes needed if one needs e.g. impose artificial vanishing Dirichlet bcs as a step of calculating line integrals along the domain.
 */
run_const bool  AC_allow_non_periodic_bcs_with_periodic_grid = false;
/**
 * Helper variable to know if all dimensions are periodic
 */
run_const bool AC_fully_periodic_grid = AC_periodic_grid.x && AC_periodic_grid.y && AC_periodic_grid.z;

/**
 * Grid spacings. If is not given and AC_len, AC_periodic_grid and AC_ngrid are, then by default calculated from them.
 * Naturally these only make sense for equidistant Cartesian grids
 */

run_const real3 AC_ds = ac_is_loaded(AC_len) && ac_is_loaded(AC_periodic_grid) && ac_is_loaded(AC_ngrid) ? 
						real3(
							AC_periodic_grid.x ? AC_len.x/AC_ngrid.x : AC_len.x/(AC_ngrid.x-1),	
							AC_periodic_grid.y ? AC_len.y/AC_ngrid.y : AC_len.y/(AC_ngrid.y-1),	
							AC_periodic_grid.z ? AC_len.z/AC_ngrid.z : AC_len.z/(AC_ngrid.z-1)
						) : 
						real3(1.0,1.0,1.0);

/**
 * Smallest grid spacing. Only correct for equidistant Cartesian grids.
 */
run_const real AC_dsmin = AC_dimension_inactive.z ? min(AC_ds.x,AC_ds.y) : min(AC_ds);
/**
 * AC_dsmin^2. Only correct for equidistant Cartesian grids.
 */
run_const real AC_dsmin_2 = AC_dsmin*AC_dsmin;
/**
 * AC_dsmin^3. Only correct for equidistant Cartesian grids.
 */
run_const real AC_dsmin_3 = AC_dsmin_2*AC_dsmin;
/**
 * AC_dsmin^4. Only correct for equidistant Cartesian grids.
 */
run_const real AC_dsmin_4 = AC_dsmin_2*AC_dsmin_2;
/**
 * AC_dsmin^5. Only correct for equidistant Cartesian grids.
 */
run_const real AC_dsmin_5 = AC_dsmin_3*AC_dsmin_2;
/**
 * AC_dsmin^6. Only correct for equidistant Cartesian grids.
 */
run_const real AC_dsmin_6 = AC_dsmin_3*AC_dsmin_3;


/**
 * Inverse of AC_ds. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_inv_ds = (real3)
		{
			AC_dimension_inactive.x  ? 1.0 : 1.0/AC_ds.x,	
			AC_dimension_inactive.y  ? 1.0 : 1.0/AC_ds.y,
			AC_dimension_inactive.z  ? 1.0 : 1.0/AC_ds.z
		}
/**
 * Inverse of AC_ds_2. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_inv_ds_2 = AC_inv_ds*AC_inv_ds;
/**
 * Inverse of AC_ds_3. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_inv_ds_3 = AC_inv_ds_2*AC_inv_ds;
/**
 * Inverse of AC_ds_4. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_inv_ds_4 = AC_inv_ds_2*AC_inv_ds_2;
/**
 * Inverse of AC_ds_5. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_inv_ds_5 = AC_inv_ds_3*AC_inv_ds_2;
/**
 * Inverse of AC_ds_6. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_inv_ds_6 = AC_inv_ds_3*AC_inv_ds_3;

/**
 * AC_ds^2. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_ds_2 = AC_ds*AC_ds;
/**
 * AC_ds^3. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_ds_3 = AC_ds_2*AC_ds;
/**
 * AC_ds^4. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_ds_4 = AC_ds_2*AC_ds_2;
/**
 * AC_ds^5. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_ds_5 = AC_ds_3*AC_ds_2;
/**
 * AC_ds^6. Makes sense only for equidistant Cartesian grids.
 */
run_const real3 AC_ds_6 = AC_ds_3*AC_ds_3;

/**
 * How processes should be mapped to process grid. See AcProcMappingStrategy for possible options.
 */
run_const AcProcMappingStrategy AC_proc_mapping_strategy = AC_PROC_MAPPING_STRATEGY_MORTON;
/**
 * How the grid should be decomposed to subdomains. See AcDecomposeStrategy for possible options.
 */
run_const AcDecomposeStrategy   AC_decompose_strategy    = AC_DECOMPOSE_STRATEGY_MORTON;
/**
 * How to construct the MPI communicator of Astaroth. See AcMPICommStrategy for possible options.
 */
run_const AcMPICommStrategy     AC_MPI_comm_strategy     = AC_MPI_COMM_STRATEGY_DUP_WORLD;

/**
 * Length of the computational grid. If not given then by default calculated from AC_ds 
 */
run_const real3 AC_len = (AC_ngrid + AC_periodic_grid - 1)*AC_ds;

/**
 * Helper variable to get the spacing in the Fourier space
 */
run_const real3 AC_frequency_spacing = real3(
				AC_dimension_inactive.x ? (2.0*AC_REAL_PI) : (2.0*AC_REAL_PI)/AC_len.x,
				AC_dimension_inactive.y ? (2.0*AC_REAL_PI) : (2.0*AC_REAL_PI)/AC_len.y,
				AC_dimension_inactive.z ? (2.0*AC_REAL_PI) : (2.0*AC_REAL_PI)/AC_len.z
				);
run_const real3 AC_first_gridpoint =  (real3){
					AC_periodic_grid.x*AC_ds.x*0.5,
					AC_periodic_grid.y*AC_ds.y*0.5,
					AC_periodic_grid.z*AC_ds.z*0.5
				       };


#if AC_LAGRANGIAN_GRID 
#if TWO_D == 0
/**
 * Field3 to store the coordinates of the gridpoints for Lagrangian grids.
 */
Field3 AC_COORDS;
#else
/**
 * Field2 to store the coordinates of the gridpoints for Lagrangian grids.
 * (Used if -D2D=ON)
 */
Field2 AC_COORDS;
#endif
#endif

/**
 * Boolean specifying does the host store arrays in row-major order (C-like order).
 */
run_const bool  AC_host_has_row_memory_order = false;

			

/**
 * Whether the grid is Lagrangian (i.e. reference frame where one follows the points)
 */
run_const bool  AC_lagrangian_grid = AC_LAGRANGIAN_GRID;
/**
 * Variable specifying how the 3d volumes should be decomposed for reduction-only Kernels.
 * At the moment x value has to be always 1 to ease warp-reductions.
 */
int3 AC_thread_block_loop_factors = (int3){1,1,1};
/**
 * Variable limiting threads per block for reduction-only kernels. Knowing this beforehand allows to safe some memory allocated for safety.
 */
int3 AC_max_tpb_for_reduce_kernels = (int3){-1,8,8};
/**
 * Size of the reduction tile that is slided across the 3d volumes. Calculated at initialization.
 */
int3 AC_reduction_tile_dimensions;

/**
 * Coordinates of the process in the process grid.
 * Calculated based on the AC_proc_mapping_strategy.
 */
int3 AC_domain_coordinates;
/**
 * Offset to the first point on the GPU/process in the global grid coordinates.
 * AC_multigpu_offset = AC_domain_coordinates*AC_nlocal
 */
int3 AC_multigpu_offset = (int3){0,0,0};

/**
 * Global option to always include the 3d corner cubicles in halo exchanges.
 */
run_const bool AC_include_3d_halo_corners = false;
/**
 * Option to handle the single gpu case with the same code as the multi-GPU case.
 */
run_const bool AC_skip_single_gpu_optim = false;

/**
 * Specifies the coordinate system. See AcCoordinateSystem for possible options.
 */
run_const AcCoordinateSystem AC_coordinate_system = AC_CARTESIAN_COORDINATES;
/**
 * Specifies are the given dimensions equidistant or not.
 */
run_const bool3 AC_nonequidistant_grid = (bool3){false,false,false};

/**
 * Option to make the autotuning faster by considering less possible configurations.
 */
bool AC_sparse_autotuning=false;

/**
 * Reduction output variable used by Device-layer API functions.
 */
output real AC_default_real_output;

/**
 * Whether the box is a shearing box or not.
 * Used to know should the bc in x be shear-periodic or not.
 */
run_const bool AC_shear = false;
/**
 * Whether the box is a shearing box or not.
 * Option to use GPUDirect RDMA for direct GPU-GPU communication instead of routing communication through host memory
 */
run_const bool AC_use_cuda_aware_mpi = true;
/**
 * Background shear flow stored on the host.
 * Used in the shear-periodic boundary conditions in x.
 */
input real AC_shear_delta_y;
/**
 * How to decompose the full 3d cube to smaller cubes for raytracing kernels.
 * Needed for large subdomains since raytracing uses co-operative kernel launches,
 * which require the full kernel launch to fit on the SMs at the same time.
 */
run_const int3 AC_raytracing_block_factors = (int3){1,1,1};
/**
 * Configuration option to tune the cache size for raytracing in x-direction.
 */
run_const int AC_x_ray_shared_mem_block_size = 32;
/**
 * Configuration option to tune the cache size for raytracing in z-direction.
 * Has an effect only if using shared mem caching in z-direction which is by default off.
 */
run_const int AC_z_ray_shared_mem_block_size = 1;

/**
 * Whether to use the bidiagonal scheme or the naive scheme for cross derivatives.
 * To read about the bidiagonal scheme see the Pencil Code manual G.3
 */
run_const bool AC_bidiagonal_derij = true;

/**
 * Specifies how many extra grid points the extended grid has to the left of the normal grid (left meaning they come before in indexing).
 * By default no extended grid so zero points
 */
dconst int3 AC_left_extended_halo = (int3){0,0,0}
/**
 * Specifies how many extra grid points the extended grid has to the right of the normal grid (left meaning they come after in indexing).
 * By default no extended grid so zero points
 */
dconst int3 AC_right_extended_halo = (int3){0,0,0}

/**
 * Size of Fields on the extended grid with halos included
 */
dconst int3 AC_extended_mlocal = AC_mlocal + AC_left_extended_halo + AC_right_extended_halo

/**
 * First grid point on the extended grid
 */
run_const real3 AC_first_gridpoint_extended = AC_first_gridpoint

/**
 * Length of the extended grid
 */
run_const real3 AC_len_extended = AC_len

/**
 * Number of points in the computational domain of the extended grid
 */
run_const int3 AC_ngrid_extended  = AC_ngrid + AC_left_extended_halo  + AC_right_extended_halo
/**
 * Size of Fields on the extended grid without halos
 */
dconst int3 AC_extended_nlocal = AC_nlocal + AC_left_extended_halo + AC_right_extended_halo

dconst bool AC_autotuning_at_work = false
