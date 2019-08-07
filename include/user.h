// clang-format off
#ifdef PENCIL_ASTAROTH

  #include "../cparam.inc_c.h"
  #define STENCIL_ORDER (2*NGHOST)

  #include "PC_moduleflags.h"

  #define CONFIG_PATH
  #define AC_MULTIGPU_ENABLED (false)

  #define USER_PROVIDED_DEFINES

#endif
// clang-format on
