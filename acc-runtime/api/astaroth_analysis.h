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
