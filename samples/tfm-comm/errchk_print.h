#pragma once
#include <stdarg.h>

/**
 * Functions and macros for printing error messages.
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

void errchk_print_stacktrace(void);

#ifdef __cplusplus
}
#endif
