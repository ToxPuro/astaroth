# pragma once
# include "headers_c.h"
# define set_dt set_dt$sub_
# define get_dxyzs get_dxyzs$sub_
extern "C" void *set_dt(REAL &dt1_);
extern "C" AcReal3 get_dxyzs(void);
