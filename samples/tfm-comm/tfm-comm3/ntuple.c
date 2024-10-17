#include "ntuple.h"

#include "alloc.h"
#include "errchk.h"
#include "misc.h"
#include "type_conversion.h"

#include <inttypes.h>

static uint64_t
min_uint64_t(const uint64_t a, const uint64_t b)
{
    return a < b ? a : b;
}

Ntuple
make_ntuple(const size_t nelems)
{
    WARNCHKK(nelems <= NTUPLE_MAX_NELEMS,
             "Increase NTUPLE_MAX_NELEMS to enable larger ntuples than defined in ntuple.h");
    Ntuple ntuple = (Ntuple){
        .nelems = min_uint64_t(nelems, NTUPLE_MAX_NELEMS),
    };
    ntuple_fill(0, &ntuple);
    return ntuple;
}

Ntuple
make_ntuple_ones(const size_t nelems)
{
    Ntuple ntuple = make_ntuple(nelems);
    ntuple_fill(1, &ntuple);
    return ntuple;
}

Ntuple
make_ntuple_with_elems(const size_t nelems, const uint64_t* elems)
{
    Ntuple ntuple = make_ntuple(nelems);
    ac_copy(ntuple.nelems, sizeof(ntuple.elems[0]), elems, ntuple.elems);
    return ntuple;
}

Ntuple
make_ntuple_from_mpi(const int nelems, const int* elems)
{
    // Set nelems
    Ntuple ntuple = make_ntuple(as_size_t(nelems));

    // Convert type
    for (size_t i = 0; i < ntuple.nelems; ++i)
        ntuple.elems[i] = as_uint64_t(elems[i]);

    // Reverse element order
    ac_reverse(ntuple.nelems, sizeof(ntuple.elems[0]), ntuple.elems);

    return ntuple;
}

void
ntuple_fill(const uint64_t value, Ntuple* ntuple)
{
    for (size_t i = 0; i < ntuple->nelems; ++i)
        ntuple->elems[i] = value;
}

Ntuple
ntuple_add(const Ntuple a, const Ntuple b)
{
    ERRCHK(a.nelems == b.nelems);

    Ntuple c = make_ntuple(a.nelems);
    for (size_t i = 0; i < a.nelems; ++i)
        c.elems[i] = a.elems[i] + b.elems[i];

    return c;
}

Ntuple
ntuple_sub(const Ntuple a, const Ntuple b)
{
    ERRCHK(a.nelems == b.nelems);

    Ntuple c = make_ntuple(a.nelems);
    for (size_t i = 0; i < a.nelems; ++i)
        c.elems[i] = a.elems[i] - b.elems[i];

    return c;
}

Ntuple
ntuple_mul(const Ntuple a, const Ntuple b)
{
    Ntuple c = make_ntuple(b.nelems);
    for (size_t i = 0; i < b.nelems; ++i)
        c.elems[i] = a.elems[i] * b.elems[i];

    return c;
}

Ntuple
ntuple_scal_mul(const uint64_t a, const Ntuple b)
{
    Ntuple c = make_ntuple(b.nelems);
    for (size_t i = 0; i < b.nelems; ++i)
        c.elems[i] = a * b.elems[i];

    return c;
}

#include <stdio.h>
void
print_ntuple(const char* label, const Ntuple ntuple)
{
    printf("%s:\n", label);
    printf("\tnelems: %zu\n", ntuple.nelems);
    printf("\telems[]: {");
    for (size_t i = 0; i < ntuple.nelems; ++i)
        printf("%" PRIu64 "%s", ntuple.elems[i], i + 1 < ntuple.nelems ? ", " : "");
    printf("}\n");
}

#define PRINTD_NTUPLE(var) print_ntuple(#var, (var))

int
test_ntuple(void)
{
    Ntuple nt0 = make_ntuple(5);
    PRINTD_NTUPLE(nt0);

    Ntuple nt1 = make_ntuple_with_elems(3, (uint64_t[]){1, 2, 3});
    PRINTD_NTUPLE(nt1);
    nt1 = ntuple_add(nt1, nt1);
    PRINTD_NTUPLE(nt1);
    nt1 = ntuple_scal_mul(2, nt1);
    PRINTD_NTUPLE(nt1);
    nt1 = ntuple_sub(nt1, nt1);
    PRINTD_NTUPLE(nt1);

    Ntuple a = make_ntuple_with_elems(3, (uint64_t[]){2, 2, 2});
    PRINTD_NTUPLE(ntuple_mul(a, a));

    return 0;
}
