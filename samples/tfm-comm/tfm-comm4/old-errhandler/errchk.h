#pragma once
#include <stdarg.h>

/**
 * Macros for internal error checking.
 *
 * Aborts the program if an assertion fails.
 * This should be handled by catching SIGABRT on upper levels if needed.
 *
 * Should only be used for catching programming errors.
 * Proper error handling and the error.h module should be used instead
 * for checking unexpected or user errors.
 *
 * C++ exceptions are not used in this design because
 *  a) Abort and signal handler is as simple as it gets
 *  b) To maintain compatibility with both C and C++ modules
 *  c) Proper exception handling has to be done with care and for our case,
 *     there is no clear incentive to recover from a failure that has resulted
 *     in a corrupted simulation state. Therefore the drawbacks outweigh the benefits.
 */
#define ERROR_DESC(...) (errchk_print_error(__func__, __FILE__, __LINE__, "", __VA_ARGS__))
#define WARNING_DESC(...) (errchk_print_warning(__func__, __FILE__, __LINE__, "", __VA_ARGS__))

#define ERROR_EXPR(expr) (errchk_print_error(__func__, __FILE__, __LINE__, #expr, ""))
#define WARNING_EXPR(expr) (errchk_print_warning(__func__, __FILE__, __LINE__, #expr, ""))

#define ERROR(expr, ...) (errchk_print_error(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__))
#define WARNING(expr, ...) (errchk_print_warning(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__))

#ifdef __cplusplus
extern "C" {
#endif

void errchk_print_error(const char* function, const char* file, const long line,
                        const char* expression, const char* fmt, ...);

void errchk_print_warning(const char* function, const char* file, const long line,
                          const char* expression, const char* fmt, ...);

#ifdef __cplusplus
}
#endif
