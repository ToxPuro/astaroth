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

/** Log verbosity levels */
#define ACM_LOG_LEVEL_ERROR (0)
#define ACM_LOG_LEVEL_WARNING (1)
#define ACM_LOG_LEVEL_INFO (2)
#define ACM_LOG_LEVEL_DEBUG (3)
#define ACM_LOG_LEVEL_TRACE (4)
#define ACM_LOG_LEVEL (ACM_LOG_LEVEL_DEBUG)

#define PRINT_LOG_ERROR(...) (errchk_print_log(__func__, __LINE__, __VA_ARGS__))

#if defined(ACM_LOG_LEVEL)
#if ACM_LOG_LEVEL >= ACM_LOG_LEVEL_WARNING
#define PRINT_LOG_WARNING(...) (errchk_print_log(__func__, __LINE__, __VA_ARGS__))
#else
#define PRINT_LOG_WARNING(...)
#define PRINT_LOG_INFO(...)
#define PRINT_LOG_DEBUG(...)
#define PRINT_LOG_TRACE(...)
#endif
#if ACM_LOG_LEVEL >= ACM_LOG_LEVEL_INFO
#define PRINT_LOG_INFO(...) (errchk_print_log(__func__, __LINE__, __VA_ARGS__))
#else
#define PRINT_LOG_INFO(...)
#define PRINT_LOG_DEBUG(...)
#define PRINT_LOG_TRACE(...)
#endif
#if ACM_LOG_LEVEL >= ACM_LOG_LEVEL_DEBUG
#define PRINT_LOG_DEBUG(...) (errchk_print_log(__func__, __LINE__, __VA_ARGS__))
#else
#define PRINT_LOG_DEBUG(...)
#define PRINT_LOG_TRACE(...)
#endif
#if ACM_LOG_LEVEL >= ACM_LOG_LEVEL_TRACE
#define PRINT_LOG_TRACE(...) (errchk_print_log(__func__, __LINE__, __VA_ARGS__))
#else
#define PRINT_LOG_TRACE(...)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

void errchk_print_error(const char* function, const char* file, const long line,
                        const char* expression, const char* fmt, ...);

void errchk_print_warning(const char* function, const char* file, const long line,
                          const char* expression, const char* fmt, ...);

void errchk_print_log(const char* function, const long line, const char* fmt, ...);

void errchk_print_stacktrace(void);

#ifdef __cplusplus
}
#endif
