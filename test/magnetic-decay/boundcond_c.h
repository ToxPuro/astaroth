# pragma once
# include "headers_c.h"
# define boundconds_x_c boundconds_x_c$boundcond_
# define boundconds_y_c boundconds_y_c$boundcond_
# define boundconds_z_c boundconds_z_c$boundcond_
extern "C" void *boundconds_x_c(REAL *f, int *ivar1, int *ivar2);
extern "C" void *boundconds_y_c(REAL *f, int *ivar1, int *ivar2);
extern "C" void *boundconds_z_c(REAL *f, int *ivar1, int *ivar2);
