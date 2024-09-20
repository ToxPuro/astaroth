#include "packed_data.h"

#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"
#include "print.h"

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
acPackedDataPrint(const char* label, const PackedData packed_data)
{
    printf("PackedData %s:\n", label);
    print("\tndims", packed_data.ndims);
    print_array("\tdims", packed_data.ndims, packed_data.dims);
    print_array("\toffset", packed_data.ndims, packed_data.offset);
    print("\tfields", packed_data.nfields);
    acBufferPrint("\tbuffer", packed_data.buffer);
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
