#include "ntuple.h"

#include "errchk.h"

Ntuple::Ntuple(const size_t _nelems, const uint64_t* _elems)
{
    ERRCHK(_nelems > 0);
    ERRCHK(_nelems <= NTUPLE_MAX_NELEMS);

    nelems = std::min(_nelems, NTUPLE_MAX_NELEMS);
    if (_elems)
        for (size_t i = 0; i < nelems; ++i)
            elems[i] = _elems[i];
    else
        std::fill_n(elems, NTUPLE_MAX_NELEMS, 0);
}

std::ostream&
operator<<(std::ostream& os, const Ntuple& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.nelems; ++i)
        os << obj.elems[i] << (i + 1 < obj.nelems ? ", " : "}");
    return os;
}
