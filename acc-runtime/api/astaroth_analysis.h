#pragma once
typedef struct
{
      int read_fields[NUM_ALL_FIELDS];
      int field_has_stencil_op[NUM_ALL_FIELDS];
      int stencils_accessed[NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS];
      int written_fields[NUM_ALL_FIELDS];
      int read_profiles[NUM_PROFILES+1];
      int reduced_profiles[NUM_PROFILES+1];
      int written_profiles[NUM_PROFILES+1];
      int profile_has_stencil_op[NUM_PROFILES+1];
      int ray_accessed[NUM_ALL_FIELDS][NUM_RAYS+1];
      KernelReduceOutput reduce_inputs[NUM_OUTPUTS+1];
      KernelReduceOutput reduce_outputs[NUM_OUTPUTS+1];
      size_t n_reduce_inputs;
      size_t n_reduce_outputs;
} KernelAnalysisInfo;

typedef struct
{
	bool larger_input;
	bool larger_output;
} acAnalysisBCInfo;

#ifdef __cplusplus
#include <string.h>
UNUSED static bool
operator==(const KernelAnalysisInfo& a, const KernelAnalysisInfo& b)
{
       if(memcmp(a.read_fields,b.read_fields,sizeof(int)*NUM_FIELDS)) return false;
       if(memcmp(a.field_has_stencil_op,b.field_has_stencil_op,sizeof(int)*NUM_FIELDS)) return false;
       if(memcmp(a.written_fields,b.written_fields,sizeof(int)*NUM_FIELDS)) return false;
       if(memcmp(a.read_profiles,b.read_profiles,sizeof(int)*NUM_PROFILES)) return false;
       if(memcmp(a.reduced_profiles,b.reduced_profiles,sizeof(int)*NUM_PROFILES)) return false;
       if(memcmp(a.written_profiles,b.written_profiles,sizeof(int)*NUM_PROFILES)) return false;
       if(memcmp(a.profile_has_stencil_op,b.profile_has_stencil_op,sizeof(int)*NUM_PROFILES)) return false;
       if(memcmp(a.reduce_inputs,b.reduce_inputs,sizeof(int)*NUM_OUTPUTS)) return false;
       if(memcmp(a.reduce_outputs,b.reduce_outputs,sizeof(int)*NUM_OUTPUTS)) return false;
       if(a.n_reduce_inputs  != b.n_reduce_inputs)  return false;
       if(a.n_reduce_outputs != b.n_reduce_outputs) return false;
       for(int field = 0; field < NUM_FIELDS; ++field)
       {
               for(int stencil = 0; stencil < NUM_STENCILS; ++stencil)
               {
                       if(a.stencils_accessed[field][stencil] != b.stencils_accessed[field][stencil]) return false;
               }
               for(int ray = 0; ray < NUM_RAYS; ++ray)
               {
                       if(a.ray_accessed[field][ray] != b.ray_accessed[field][ray]) return false;
               }
       }
       return true;
}
#endif

