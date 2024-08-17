#include "astaroth.h"
int main()
{
	AcReal real_arr[4];
	int int_arr[2];
	bool bool_arr[2] = {false,true};
	for(int i = 0; i < 4; ++i)
		real_arr[i] = -i;
	for(int i = 0; i < 2; ++i)
		int_arr[i] = i;
	AcCompInfo info = acInitCompInfo();
	acLoadCompInfo(AC_lspherical_coords,true,&info);
	acLoadCompInfo(AC_runtime_int,1,&info);
	acLoadCompInfo(AC_runtime_real,0.12345,&info);
	acLoadCompInfo(AC_runtime_real3,{0.12345,0.12345,0.12345},&info);
	acLoadCompInfo(AC_runtime_int3,{0,1,2},&info);
	acLoadCompInfo(AC_runtime_real_arr,real_arr,&info);
	acLoadCompInfo(AC_runtime_int_arr,int_arr,&info);
	acLoadCompInfo(AC_runtime_bool_arr,bool_arr,&info);
	const char* build_str = "-DUSE_HIP=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON";
	acCompile(build_str,info);
	return 0;
}
