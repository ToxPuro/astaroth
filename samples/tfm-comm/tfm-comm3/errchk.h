#pragma once
#include <stdarg.h>

/**
 * Macros for internal error checking.
 * Should only be used for catching programming errors.
 * Proper error handling and the error.h module should be used instead
 * for checking unexpected or user errors.
 */
#define ERROR(...) (errchk_print_error(__func__, __FILE__, __LINE__, "", __VA_ARGS__))
#define WARNING(...) (errchk_print_warning(__func__, __FILE__, __LINE__, "", __VA_ARGS__))

#define ERROR_EXPR(expr) (errchk_print_error(__func__, __FILE__, __LINE__, #expr, ""))
#define WARNING_EXPR(expr) (errchk_print_warning(__func__, __FILE__, __LINE__, #expr, ""))

#define ERRCHK(expr)                                                                               \
    ((expr) ? (expr) : (errchk_print_error(__func__, __FILE__, __LINE__, #expr, ""), (expr)))
#define WARNCHK(expr)                                                                              \
    ((expr) ? (expr) : (errchk_print_warning(__func__, __FILE__, __LINE__, #expr, ""), (expr)))

#define ERRCHKK(expr, ...)                                                                         \
    ((expr) ? (expr)                                                                               \
            : (errchk_print_error(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__), (expr)))
#define WARNCHKK(expr, ...)                                                                        \
    ((expr) ? (expr)                                                                               \
            : (errchk_print_warning(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__), (expr)))

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
