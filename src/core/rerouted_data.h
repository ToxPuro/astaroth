#include "astaroth_decomp.h"

typedef struct ReroutedData{
    PackedData packedData;
    int3 haloCoords;
    int3 offset;

    ReroutedData(PackedData p_data, int3 h_data, int3 os){
        packedData = p_data;
        haloCoords = h_data;
        offset = os;
    }

    //Move constructor would save copies
    //But needs move constructors in PackedData and int3
    //
    //ReroutedData(PackedData&& p_data, int3&& h_id){
    //    packedData = std::move(p_data);
    //    halo_id = std::move(h_id);
    //}

} ReroutedData;

