#pragma once
#include <stdarg.h>

/**
 * Macros for internal error handling.
 * Adapted from the original C version. Can be revised later for C++ style.
 */
#define ERROR(...)                                                                                 \
    (errchk_print_error(__func__, __FILE__, __LINE__, "", __VA_ARGS__), errchk_raise_error())
#define WARNING(...) (errchk_print_warning(__func__, __FILE__, __LINE__, "", __VA_ARGS__))

#define ERROR_EXPR(expr)                                                                           \
    (errchk_print_error(__func__, __FILE__, __LINE__, #expr, ""), errchk_raise_error())
#define WARNING_EXPR(expr) (errchk_print_warning(__func__, __FILE__, __LINE__, #expr, ""))

#define ERRCHK(expr)                                                                               \
    ((expr) ? 0                                                                                    \
            : ((errchk_print_error(__func__, __FILE__, __LINE__, #expr, "")),                      \
               errchk_raise_error(), -1))
#define WARNCHK(expr)                                                                              \
    ((expr) ? 0 : ((errchk_print_warning(__func__, __FILE__, __LINE__, #expr, "")), -1))

#define ERRCHKK(expr, ...)                                                                         \
    ((expr) ? 0                                                                                    \
            : ((errchk_print_error(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__)),             \
               errchk_raise_error(), -1))
#define WARNCHKK(expr, ...)                                                                        \
    ((expr) ? 0 : ((errchk_print_warning(__func__, __FILE__, __LINE__, #expr, __VA_ARGS__)), -1))

#ifdef __cplusplus
extern "C" {
#endif

void errchk_print_error(const char* function, const char* file, const long line,
                        const char* expression, const char* fmt, ...);

void errchk_raise_error(void);

void errchk_print_warning(const char* function, const char* file, const long line,
                          const char* expression, const char* fmt, ...);

#ifdef __cplusplus
}
#endif
