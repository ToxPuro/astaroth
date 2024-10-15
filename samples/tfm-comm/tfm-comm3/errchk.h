#pragma once
#include <stdarg.h>

/**
 * Macros for internal error checking.
 * Should only be used for catching programming errors.
 * Proper error handling and the error.h module should be used instead
 * for checking unexpected or user errors.
 */
#define ERROR_EXPR_VA(expr, ...) errchk_print_err(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__)
#define ERROR_EXPR(expr) errchk_print_err(__func__, __FILE__, __LINE__, #expr, "")
#define ERROR_VA(...) errchk_print_err(__func__, __FILE__, __LINE__, "", __VA_ARGS__)

#define ERRCHK_VA(expr, ...) ((expr) ? (expr) : (ERROR_EXPR_VA((expr), __VA_ARGS__), (expr)))
#define ERRCHK(expr) ERRCHK_VA((expr), "")

#define WARNING_EXPR_VA(expr, ...)                                                                 \
    errchk_print_warn(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__)
#define WARNING_EXPR(expr) errchk_print_warn(__func__, __FILE__, __LINE__, #expr, "")
#define WARNING_VA(...) errchk_print_warn(__func__, __FILE__, __LINE__, "", __VA_ARGS__)

#define WARNCHK_VA(expr, ...) ((expr) ? (expr) : (WARNING_EXPR_VA((expr), __VA_ARGS__), (expr)))
#define WARNCHK(expr) WARNCHK_VA((expr), "")

#ifdef __cplusplus
extern "C" {
#endif

void errchk_print_err(const char* function, const char* file, const long line,
                      const char* expression, const char* fmt, ...);

void errchk_print_warn(const char* function, const char* file, const long line,
                       const char* expression, const char* fmt, ...);

#ifdef __cplusplus
}
#endif
