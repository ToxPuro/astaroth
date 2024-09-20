#include "comm_data.h"

#include "errchk.h"
#include "math.h"
#include "math_utils.h"
#include "print.h"

CommData
acCommDataCreate(const size_t ndims, const size_t nfields)
{
    // Determine the number of halo partitions
    const size_t npackets = powzu(3, ndims) - 1; // The neighbor count
    print("Creating npackets", npackets);

    // Create CommData
    CommData comm_data = (CommData){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(comm_data.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(comm_data.remote_packets[0]) * npackets),
    };
    ERRCHK(comm_data.local_packets);
    ERRCHK(comm_data.remote_packets);

    for (size_t i = 0; i < npackets; ++i) {
        const size_t dims[]         = {3, 3, 3};
        const size_t offset[]       = {0, 0, 0};
        comm_data.local_packets[i]  = acCreatePackedData(ndims, dims, offset, nfields);
        comm_data.remote_packets[i] = acCreatePackedData(ndims, dims, offset, nfields);
    }

    return comm_data;
}

void
acCommDataPrint(const char* label, const CommData comm_data)
{
    printf("CommData %s:\n", label);

    const size_t buflen = 128;
    char buf[buflen];

    for (size_t i = 0; i < comm_data.npackets; ++i) {
        snprintf(buf, buflen, "local_packets[%zu]", i);
        acPackedDataPrint(buf, comm_data.local_packets[i]);
    }

    for (size_t i = 0; i < comm_data.npackets; ++i) {
        snprintf(buf, buflen, "remote_packets[%zu]", i);
        acPackedDataPrint(buf, comm_data.remote_packets[i]);
    }
}

void
acCommDataDestroy(CommData* comm_data)
{
    for (size_t i = 0; i < comm_data->npackets; ++i) {
        acDestroyPackedData(&comm_data->local_packets[i]);
        acDestroyPackedData(&comm_data->remote_packets[i]);
    }
    free(comm_data->remote_packets);
    free(comm_data->local_packets);
    comm_data->npackets = 0;
}
