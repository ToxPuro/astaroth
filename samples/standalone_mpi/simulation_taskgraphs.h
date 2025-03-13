#include <astaroth.h>

#include <astaroth_debug.h>

// TODO: allow selecting single our doublepass here?
enum class Simulation { MHD , Shock_Singlepass_Solve, Hydro_Heatduct_Solve, Collapse_Singlepass_Solve, Bound_Test_Solve, Default = MHD};

void
log_simulation_choice(int pid, Simulation sim)
{
    const char* sim_label;
    switch (sim) {
    case Simulation::MHD:
        sim_label = "MHD";
        break;
    case Simulation::Shock_Singlepass_Solve:
        sim_label = "Shock with singlepass solve";
        break;
    case Simulation::Collapse_Singlepass_Solve:
        sim_label = "Collapse model with singlepass solve and shock viscosity";
        break;
    case Simulation::Hydro_Heatduct_Solve:
        sim_label = "Heat duct with doublepass solve";
        break;
    case Simulation::Bound_Test_Solve:
        sim_label = "Boundary test with doublepass solve";
        break;
    default:
        sim_label = "WARNING: No label exists for simulation";
        break;
    }
    acLogFromRootProc(pid, "Simulation program: %s \n", sim_label);
}

static std::map<Simulation, AcTaskGraph*> task_graphs;

AcTaskGraph*
get_simulation_graph(int pid, Simulation sim, AcMeshInfo info)
{

    auto make_graph = [pid, &info](Simulation sim_in) -> AcTaskGraph* {
        acLogFromRootProc(pid, "Creating task graph for simulation\n");

        auto intermediate_loader = [](ParamLoadingInfo l){
                l.params -> twopass_solve_intermediate.step_num = l.step_number;
                l.params -> twopass_solve_intermediate.dt= 
                acDeviceGetInput(l.device,AC_dt);
        };
        auto final_loader = [](ParamLoadingInfo l){
                l.params -> twopass_solve_final.step_num = l.step_number;
                l.params -> twopass_solve_final.current_time= 
            	   acDeviceGetInput(l.device,AC_current_time);
        };

        switch (sim_in) {
        case Simulation::Shock_Singlepass_Solve: {
#if LSHOCK
            // This still has to be behind a preprocessor feature, because e.g., VTXBUF_SHOCK is not
            // defined in general
            VertexBufferHandle all_fields[] =
                {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                 VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ, // VTXBUF_ENTROPY,
                 VTXBUF_SHOCK, BFIELDX,    BFIELDY,    BFIELDZ};
	    auto single_loader = [&info](ParamLoadingInfo p){
		    acKernelInputParams* params = acDeviceGetKernelInputParamsObject(p.device);
		    params->singlepass_solve.time_params= {
	    							acDeviceGetInput(p.device,AC_dt);
	    							acDeviceGetInput(p.device,AC_current_time);
		    					   };
		    params->singlepass_solve.step_num = p.step_number;
	    };
            VertexBufferHandle shock_field[] = {VTXBUF_SHOCK};
            AcTaskDefinition shock_ops[] =
                {acHaloExchange(all_fields),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
                 acCompute(KERNEL_shock_1_divu, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
                 acCompute(KERNEL_shock_2_max, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
                 acCompute(KERNEL_shock_3_smooth, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, shock_field),
                 acComputeWithParams(KERNEL_singlepass_solve, all_fields, single_loader),
		};
            acLogFromRootProc(pid, "Creating shock singlepass solve task graph\n");
            return acGridBuildTaskGraph(shock_ops);
#endif
        }
        case Simulation::Collapse_Singlepass_Solve: {
#if (LSHOCK && LSINK)
            // This still has to be behind a preprocessor feature, because e.g., VTXBUF_SHOCK is not
            // defined in general
            VertexBufferHandle all_fields[] =
                {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                 VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ, // VTXBUF_ENTROPY,
                 VTXBUF_SHOCK, BFIELDX,    BFIELDY,    BFIELDZ, VTXBUF_ACCRETION};

            VertexBufferHandle shock_field[]   = {VTXBUF_SHOCK};
            VertexBufferHandle density_field[] = {VTXBUF_LNRHO};
            VertexBufferHandle   aa_fields[]   = {VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ};
            VertexBufferHandle   uu_fields[]   = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
            VertexBufferHandle    ux_field[]   = {VTXBUF_UUX};
            VertexBufferHandle    uy_field[]   = {VTXBUF_UUY};
            VertexBufferHandle    uz_field[]   = {VTXBUF_UUZ};

            //AcRealParam bbz_bound[1] = {AC_B0_selfsim_numerical_unit};

            AcTaskDefinition shock_ops[] =
                {acHaloExchange(all_fields),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, uu_fields), //Set default condition
                 //Use a2 boundary condition for density
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_A2,        density_field),
                 //Set specific AY vectorpotential field component derivative to a magnetic field value 
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_A2, aa_fields),
                 acBoundaryCondition(BOUNDARY_X,   BOUNDCOND_OUTFLOW,   ux_field),    //Set special condition 
                 acBoundaryCondition(BOUNDARY_Y,   BOUNDCOND_OUTFLOW,   uy_field),    //Set special condition
                 acBoundaryCondition(BOUNDARY_Z,   BOUNDCOND_OUTFLOW,   uz_field),    //Set special condition
                 acHaloExchange(all_fields),
                 acCompute(KERNEL_shock_1_divu, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, shock_field),
                 acCompute(KERNEL_shock_2_max, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, shock_field),
                 acCompute(KERNEL_shock_3_smooth, shock_field),
                 acHaloExchange(shock_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, shock_field),
                 acCompute(KERNEL_singlepass_solve, all_fields)};
            acLogFromRootProc(pid, "Creating shock singlepass solve task graph\n");
            return acGridBuildTaskGraph(shock_ops);
#endif
        }


        case Simulation::Hydro_Heatduct_Solve: {
#if LENTROPY
	    //TP: deprecated for now because of the special boundary condition
            //// This is an example of having multiple types of boundary conditions
	    //std::vector<VertexBufferHandle> all_fields{VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
            //                                      VTXBUF_ENTROPY};
	    //std::vector<VertexBufferHandle> lnrho_field{VTXBUF_LNRHO};
            //// VertexBufferHandle entropy_field[] = {VTXBUF_ENTROPY}; // Unused
	    //std::vector<VertexBufferHandle> scalar_fields{VTXBUF_LNRHO, VTXBUF_ENTROPY};
	    //std::vector<VertexBufferHandle> uux_field{VTXBUF_UUX};
            //std::vector<VertexBufferHandle> uuy_field{VTXBUF_UUY};
            //std::vector<VertexBufferHandle> uuz_field{VTXBUF_UUZ};
            //std::vector<VertexBufferHandle> uuxy_fields{VTXBUF_UUX, VTXBUF_UUY};
            //std::vector<VertexBufferHandle> uuxz_fields{VTXBUF_UUX, VTXBUF_UUZ};
            //std::vector<VertexBufferHandle> uuyz_fields{VTXBUF_UUY, VTXBUF_UUZ};

            //// AcRealParam const_lnrho_bound[1] = {AC_lnrho0}; // Unused
            //AcRealParam const_heat_flux[1]   = {AC_hflux};
	    //auto intermediate_loader = [&info](ParamLoadingInfo p){
	    //        acKernelInputParams* params = acDeviceGetKernelInputParamsObject(p.device);
	    //        params->twopass_solve_intermediate.dt = info[AC_dt];
	    //        params->twopass_solve_intermediate.step_num= p.step_number;
	    //};
	    //auto final_loader = [&info](ParamLoadingInfo p){
	    //        acKernelInputParams* params = acDeviceGetKernelInputParamsObject(p.device);
	    //        params->twopass_solve_final.current_time = info[AC_current_time];
	    //        params->twopass_solve_final.step_num = p.step_number;
	    //};
            //AcTaskDefinition heatduct_ops[] =
            //    {acHaloExchange(all_fields),
            //     acBoundaryCondition(BOUNDARY_XZ, BOUNDCOND_SYMMETRIC, scalar_fields),

            //     acBoundaryCondition(BOUNDARY_X, BOUNDCOND_ANTISYMMETRIC, uux_field),
            //     acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, uuyz_fields),
            //     acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, uuxz_fields),
            //     acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, uuxy_fields),
            //     acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_ANTISYMMETRIC, uuz_field),

            //     acBoundaryCondition(BOUNDARY_Y_BOT, BOUNDCOND_INFLOW, uuy_field),
            //     acBoundaryCondition(BOUNDARY_Y_TOP, BOUNDCOND_OUTFLOW, uuy_field),
            //     acBoundaryCondition(BOUNDARY_Y_BOT, BOUNDCOND_A2, lnrho_field),
            //     acBoundaryCondition(BOUNDARY_Y_TOP, BOUNDCOND_A2, scalar_fields),

            //     acSpecialMHDBoundaryCondition(BOUNDARY_Y_BOT,
            //                                   SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX,
            //                                   const_heat_flux),

	    //     acComputeWithParams(KERNEL_twopass_solve_intermediate, all_fields, intermediate_loader),
            //     acComputeWithParams(KERNEL_twopass_solve_final, all_fields, final_loader)
	    //    };
            //acLogFromRootProc(pid, "Creating heat duct task graph\n");
            //AcTaskGraph* my_taskgraph = acGridBuildTaskGraph(heatduct_ops);
            //acGraphPrintDependencies(my_taskgraph);
            //return my_taskgraph;
#endif
        }
#if LBFIELD
        case Simulation::Bound_Test_Solve: {
    		std::vector<Field> all_fields{};
    		for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        		all_fields.push_back(Field(i));
    		}
		std::vector<VertexBufferHandle> scalar_fields{VTXBUF_LNRHO};
#if LENTROPY
		scalar_fields.push_back(VTXBUF_ENTROPY);
#endif

            AcTaskDefinition boundtest_ops[] =
                {acHaloExchange(all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_ANTISYMMETRIC, all_fields),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_A2, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_OUTFLOW, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_INFLOW, all_fields),
                 
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, scalar_fields),

		 /**
                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_OUTFLOW,   {VTXBUF_UUX}),
                 //acBoundaryCondition(BOUNDARY_X, BOUNDCOND_INFLOW,   uux_field),
                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, {VTXBUF_UUY}),
                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, {VTXBUF_UUZ}),

                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, {VTXBUF_UUX}),
                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_OUTFLOW,   {VTXBUF_UUY}),
                 //acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_INFLOW,   uuy_field),
                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, {VTXBUF_UUZ}),

                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, {VTXBUF_UUX}),
                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, {VTXBUF_UUY}),
                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_OUTFLOW,   {VTXBUF_UUZ}),
                 ////acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_INFLOW,   uuz_field),
		 **/

                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, {VTXBUF_AX}),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, {VTXBUF_AY}),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, {VTXBUF_AZ}),

                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, {VTXBUF_UUX}),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, {VTXBUF_UUY}),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, {VTXBUF_UUZ}),

                 acCompute(twopass_solve_intermediate, all_fields,intermediate_loader),
                 acCompute(twopass_solve_final, all_fields,final_loader)};
            acLogFromRootProc(pid, "Creating Boundary test task graph\n");
            AcTaskGraph* my_taskgraph = acGridBuildTaskGraph(boundtest_ops);
            return my_taskgraph;
        }
#endif
        default:
            acLogFromRootProc(pid, "ERROR: no custom task graph exists for selected simulation. "
                                   "This is probably a fatal error.\n");
            return nullptr;
        }
    };

    if (sim == Simulation::MHD) {
        return acGridGetDefaultTaskGraph();
    }

    if (task_graphs.count(sim) == 0) {
        task_graphs[sim] = make_graph(sim);
    }
    return task_graphs[sim];
}

void
free_simulation_graphs(int pid)
{
    for (auto& [sim, graph] : task_graphs) {
        acLogFromRootProc(pid, "Destroying custom task graph\n");
        acGridDestroyTaskGraph(graph);
    }
    task_graphs.clear();
}
