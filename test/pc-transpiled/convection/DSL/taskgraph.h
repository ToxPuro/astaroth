BoundConds boundconds{
  periodic(BOUNDARY_X)
  periodic(BOUNDARY_Y)
  periodic(BOUNDARY_Z)
}

ComputeSteps rhs(boundconds)
{
        twopass_solve_intermediate()
}
