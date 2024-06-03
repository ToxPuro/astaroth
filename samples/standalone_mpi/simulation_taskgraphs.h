#include <astaroth.h>

#include <astaroth_debug.h>

// TODO: allow selecting single our doublepass here?
enum class Simulation { Solve, Shock_Singlepass_Solve, Hydro_Heatduct_Solve, Bound_Test_Solve, Default = Solve };

void
log_simulation_choice(int pid, Simulation sim)
{
    const char* sim_label;
    switch (sim) {
    case Simulation::Solve:
        sim_label = "Solve";
        break;
    case Simulation::Shock_Singlepass_Solve:
        sim_label = "Shock with singlepass solve";
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
			    					info.real_params[AC_dt],
			    					info.real_params[AC_current_time]
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
        case Simulation::Hydro_Heatduct_Solve: {
#if LENTROPY
            // This is an exmaple of having multiple types of boundary conditions
            VertexBufferHandle all_fields[]    = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                                  VTXBUF_ENTROPY};
            VertexBufferHandle lnrho_field[]   = {VTXBUF_LNRHO};
	    //not used at the moment
            //VertexBufferHandle entropy_field[] = {VTXBUF_ENTROPY};
            VertexBufferHandle scalar_fields[] = {VTXBUF_LNRHO, VTXBUF_ENTROPY};
            VertexBufferHandle uux_field[]     = {VTXBUF_UUX};
            VertexBufferHandle uuy_field[]     = {VTXBUF_UUY};
            VertexBufferHandle uuz_field[]     = {VTXBUF_UUZ};
            VertexBufferHandle uuxy_fields[]   = {VTXBUF_UUX, VTXBUF_UUY};
            VertexBufferHandle uuxz_fields[]   = {VTXBUF_UUX, VTXBUF_UUZ};
            VertexBufferHandle uuyz_fields[]   = {VTXBUF_UUY, VTXBUF_UUZ};

	    //not used at the moment
            //AcRealParam const_lnrho_bound[1] = {AC_lnrho0};
            AcRealParam const_heat_flux[1]   = {AC_hflux};
	    auto intermediate_loader = [&info](ParamLoadingInfo p){
		    acKernelInputParams* params = acDeviceGetKernelInputParamsObject(p.device);
		    params->twopass_solve_intermediate.dt = info.real_params[AC_dt];
		    params->twopass_solve_intermediate.step_num= p.step_number;
	    };
	    auto final_loader = [&info](ParamLoadingInfo p){
		    acKernelInputParams* params = acDeviceGetKernelInputParamsObject(p.device);
		    params->twopass_solve_final.current_time = info.real_params[AC_current_time];
		    params->twopass_solve_final.step_num = p.step_number;
	    };
            AcTaskDefinition heatduct_ops[] =
                {acHaloExchange(all_fields),
                 acBoundaryCondition(BOUNDARY_XZ, BOUNDCOND_SYMMETRIC, scalar_fields),

                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_ANTISYMMETRIC, uux_field),
                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, uuyz_fields),
                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, uuxz_fields),
                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, uuxy_fields),
                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_ANTISYMMETRIC, uuz_field),

                 acBoundaryCondition(BOUNDARY_Y_BOT, BOUNDCOND_INFLOW, uuy_field),
                 acBoundaryCondition(BOUNDARY_Y_TOP, BOUNDCOND_OUTFLOW, uuy_field),
                 acBoundaryCondition(BOUNDARY_Y_BOT, BOUNDCOND_A2, lnrho_field),
                 acBoundaryCondition(BOUNDARY_Y_TOP, BOUNDCOND_A2, scalar_fields),

                 acSpecialMHDBoundaryCondition(BOUNDARY_Y_BOT,
                                               SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX,
                                               const_heat_flux),

		 acComputeWithParams(KERNEL_twopass_solve_intermediate, all_fields, intermediate_loader),
                 acComputeWithParams(KERNEL_twopass_solve_final, all_fields, final_loader)
		};
            acLogFromRootProc(pid, "Creating heat duct task graph\n");
            AcTaskGraph* my_taskgraph = acGridBuildTaskGraph(heatduct_ops,1);
            acGraphPrintDependencies(my_taskgraph);
            return my_taskgraph;
#endif
        }
#if LBFIELD
        case Simulation::Bound_Test_Solve: {
            VertexBufferHandle all_fields[] =
                {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                 VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ, //VTXBUF_ENTROPY,
                 BFIELDX,    BFIELDY,    BFIELDZ};
            VertexBufferHandle scalar_fields[] = {VTXBUF_LNRHO};//, VTXBUF_ENTROPY};
            VertexBufferHandle uux_field[]     = {VTXBUF_UUX};
            VertexBufferHandle uuy_field[]     = {VTXBUF_UUY};
            VertexBufferHandle uuz_field[]     = {VTXBUF_UUZ};
            VertexBufferHandle aax_field[]     = {VTXBUF_AX};
            VertexBufferHandle aay_field[]     = {VTXBUF_AY};
            VertexBufferHandle aaz_field[]     = {VTXBUF_AZ};

            AcTaskDefinition boundtest_ops[] =
                {acHaloExchange(all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_ANTISYMMETRIC, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_A2, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_OUTFLOW, all_fields),
                 //acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_INFLOW, all_fields),
                 
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, scalar_fields),

                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_OUTFLOW,   uux_field),
                 //acBoundaryCondition(BOUNDARY_X, BOUNDCOND_INFLOW,   uux_field),
                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, uuy_field),
                 acBoundaryCondition(BOUNDARY_X, BOUNDCOND_SYMMETRIC, uuz_field),

                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, uux_field),
                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_OUTFLOW,   uuy_field),
                 //acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_INFLOW,   uuy_field),
                 acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_SYMMETRIC, uuz_field),

                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, uux_field),
                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, uuy_field),
                 acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_OUTFLOW,   uuz_field),
                 //acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_INFLOW,   uuz_field),

                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, aax_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, aay_field),
                 acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, aaz_field),

                 acCompute(KERNEL_twopass_solve_intermediate, all_fields),
                 acCompute(KERNEL_twopass_solve_final, all_fields)};
            acLogFromRootProc(pid, "Creating Boundary test task graph\n");
            AcTaskGraph* my_taskgraph = acGridBuildTaskGraph(boundtest_ops);
            acGraphPrintDependencies(my_taskgraph);
            return my_taskgraph;
        }
#endif
        default:
            acLogFromRootProc(pid, "ERROR: no custom task graph exists for selected simulation. "
                                   "This is probably a fatal error.\n");
            return nullptr;
        }
    };

    if (sim == Simulation::Default) {
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
