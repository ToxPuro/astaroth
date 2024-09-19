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
    print("npackets", npackets);

    // Create CommData
    CommData comm_data = (CommData){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(comm_data.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(comm_data.remote_packets[0]) * npackets),
    };
    WARNING("TODO setup local and remote packets");

    return comm_data;
}

void
acCommDataDestroy(CommData* comm_data)
{
    free(comm_data->remote_packets);
    free(comm_data->local_packets);
    comm_data->npackets = 0;
}
