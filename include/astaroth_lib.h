#ifndef AC_LIB_H
#define AC_LIB_H

#include "func_define.h"

//#ifdef __cplusplus
//  class AcLibHandle
//  {
//	  private:
//		  void* handle;
//	  public:
//		  AcLibHandle(void* _handle) : handle(_handle){}
//		  ~AcLibHandle() {dlclose(handle); }
//  };
//#else
//  typedef void* AcLibHandle;
//#endif

AC_BEGIN_C_DECLARATIONS

typedef void* AcLibHandle;


static AcLibHandle UNUSED astarothLibHandle=NULL, kernelsLibHandle=NULL, utilsLibHandle=NULL;

AC_END_C_DECLARATIONS

#endif
