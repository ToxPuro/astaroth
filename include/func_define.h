#ifndef __FUNC_DEFINE_ALREADY_INCLUDED__
#define __FUNC_DEFINE_ALREADY_INCLUDED__

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus

#define AC_BEGIN_C_DECLARATIONS extern "C" {
#define AC_END_C_DECLARATIONS }

#else

#define AC_BEGIN_C_DECLARATIONS
#define AC_END_C_DECLARATIONS

#endif

#if AC_RUNTIME_COMPILATION

static void
ac_library_not_yet_loaded()
{
    fprintf(stderr,
            "This function needs Astaroth to be loaded via acLoadLibrary before calling it!\n");
    abort();
}

#if __cplusplus
#define BASE_FUNC_NAME(func_name) func_name##_BASE
#else
#define BASE_FUNC_NAME(func_name) func_name
#endif

#else

#define BASE_FUNC_NAME(func_name) func_name

#endif /* AC_RUNTIME_COMPILATION */

#endif /* __FUNC_DEFINE_ALREADY_INCLUDED__ */

#undef FUNC_DEFINE
#undef OVERLOADED_FUNC_DEFINE

#if AC_RUNTIME_COMPILATION

#ifdef __FUNC_DEFINE_MAIN_STORAGE__

#ifdef __cplusplus
#define FUNC_DEFINE(return_type, func_name, ...) return_type (*func_name) __VA_ARGS__ = (return_type (*) __VA_ARGS__ ) ac_library_not_yet_loaded
#else
#define FUNC_DEFINE(return_type, func_name, ...) extern return_type (*func_name) __VA_ARGS__
#endif

#define OVERLOADED_FUNC_DEFINE(return_type, func_name, ...) return_type (*BASE_FUNC_NAME(func_name)) __VA_ARGS__ = (return_type (*) __VA_ARGS__ ) ac_library_not_yet_loaded

#else

#define FUNC_DEFINE(return_type, func_name, ...) extern return_type (*func_name) __VA_ARGS__

#define OVERLOADED_FUNC_DEFINE(return_type, func_name, ...) extern return_type (*BASE_FUNC_NAME(func_name)) __VA_ARGS__

#endif

#else

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) return_type func_name __VA_ARGS__
#endif

#ifndef OVERLOADED_FUNC_DEFINE
#define OVERLOADED_FUNC_DEFINE FUNC_DEFINE
#endif

#endif
