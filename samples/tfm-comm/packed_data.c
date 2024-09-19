#include "packed_data.h"

#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"

PackedData
acCreatePackedData(const size_t ndims, const size_t* dims, const size_t* offset,
                   const size_t nfields)
{
    PackedData data = (PackedData){
        .ndims   = ndims,
        .dims    = malloc(sizeof(data.dims[0]) * ndims),
        .offset  = malloc(sizeof(data.offset[0]) * ndims),
        .nfields = nfields,
        .buffer  = acBufferCreate(nfields * prod(ndims, dims)),
    };
    ERRCHK(data.dims);
    ERRCHK(data.offset);
    copy(ndims, dims, data.dims);
    copy(ndims, offset, data.offset);
    return data;
}

void
acDestroyPackedData(PackedData* data)
{
    acBufferDestroy(&data->buffer);
    data->nfields = 0;
    free(data->offset);
    free(data->dims);
    data->ndims = 0;
}
