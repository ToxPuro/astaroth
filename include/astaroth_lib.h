#pragma once

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
typedef void* AcLibHandle;
static AcLibHandle UNUSED astarothLibHandle=NULL, kernelsLibHandle=NULL, utilsLibHandle=NULL;
