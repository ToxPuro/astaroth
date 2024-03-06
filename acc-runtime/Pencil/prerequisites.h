// Provides all declarations and functions needed for the formulation of the PDEs' rhss by DSL code
// and finally for the definition of the solve kernel.

#include "fieldecs.h"
#include "../stdlib/operators.h"
#include "../stdlib/integrators.h"
#include "../stdlib/units.h"
#include "../stdlib/utils.h"
#include "../../../PC_moduleflags.h"
#include "../../../DSL/phys_consts.h"
#include "PC_modulepardecs.h"
#include "equations.h"
#if LFORCING
#include "../../../DSL/forcing/pcstyleforcing.h"
#endif
