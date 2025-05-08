#if AC_USE_HIP
	const char* shuffle_instruction = "rocprim__warp_shuffle(val,target_tid)";
#else
	const char* shuffle_instruction = "__shfl_sync(AC_INTERNAL_active_threads,val,target_tid)";
#endif

void
print_shuffle_down(FILE* stream, const int offset)
{
#if AC_USE_HIP
	fprintf(stream,"rocprim__warp_shuffle_down(val,%d)",offset);
#else
	fprintf(stream,"__shfl_down_sync(AC_INTERNAL_active_threads,val,%d)",offset);
#endif
}

void
print_shuffle_iteration(FILE* stream, const char* tmp_type, const int offset, const char* op_instruction, const bool check_activeness)
{
	char shuffle_down[1000];
	memset(shuffle_down,0,sizeof(shuffle_down));
	{
  		FILE* instruction_stream = fmemopen(shuffle_down, sizeof(shuffle_down), "w");
		print_shuffle_down(instruction_stream,offset);
		fclose(instruction_stream);
	}
	char guard[1000];
	memset(guard,0,sizeof(guard));
	{
		FILE* guard_stream = fmemopen(guard, sizeof(guard), "w");
		fprintf(guard_stream,"if(AC_INTERNAL_active_threads >> (lane_id + %d))",offset);
		fclose(guard_stream);
	}
	fprintf(stream,
			"%s shuffle_tmp = %s;"
			"%s %s;"
		,tmp_type
	        ,shuffle_down
		,check_activeness ? guard : ""
		,op_instruction
	      );
}

void
print_warp_reduction(FILE* stream, const int warp_size,const char* op_instruction, const bool check_activeness)
{
	print_shuffle_iteration(stream,"auto",warp_size/2,op_instruction,check_activeness);
	for(int offset = warp_size/4;  offset >= 1; offset /= 2)
		print_shuffle_iteration(stream,"",offset,op_instruction,check_activeness);
}
const char*
get_op_instruction(const ReduceOp op)
{
	return
		op == REDUCE_SUM ? "val += shuffle_tmp" :
		op == REDUCE_MIN ? "val = val > shuffle_tmp ? shuffle_tmp : val" :
		op == REDUCE_MAX ? "val = val > shuffle_tmp ? val : shuffle_tmp" :
		NULL;
}
const char*
reduce_op_to_name(const ReduceOp op)
{
	return
		op == REDUCE_SUM ? "sum" :
		op == REDUCE_MIN ? "min" :
		op == REDUCE_MAX ? "max" :
		NULL;
}
