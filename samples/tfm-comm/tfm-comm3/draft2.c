#include "errchk.h"

#include <stddef.h>

#define create_nalloc(T_)                                                                          \
    inline size_t* nalloc_##T_(const size_t count) { return ac_calloc(count, sizeof(T_)); }
#define create_ndealloc(T_)                                                                        \
    inline void ndealloc_##T_(T_** ptr) { ac_free((void**)ptr); }

create_nalloc(size_t);
create_ndealloc(size_t);

// And once the pointer has been created, then generics can be used
#define ndealloc(ptr) _Generic(((*ptr)[0]), size_t: ndealloc_size_t)(ptr)

size_t* ptr = nalloc_size_t(10);
ndealloc(&ptr);
