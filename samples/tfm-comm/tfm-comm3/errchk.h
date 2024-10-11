#pragma once
/**
 * Macros for internal error checking.
 * Should only be used for catching programming errors.
 * Proper error handling and the error.h module should be used instead
 * for checking unexpected or user errors.
 */

#define ERROR(description)                                                                         \
    do {                                                                                           \
        errchk_print_error(__func__, __FILE__, __LINE__, NULL, (description));                     \
    } while (0)

#define WARNING(description)                                                                       \
    do {                                                                                           \
        errchk_print_warning(__func__, __FILE__, __LINE__, NULL, (description));                   \
    } while (0)

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(expression)                                                                         \
    do {                                                                                           \
        if (!(expression))                                                                         \
            errchk_print_error(__func__, __FILE__, __LINE__, #expression, NULL);                   \
    } while (0)

#define WARNCHK(expression)                                                                        \
    do {                                                                                           \
        if (!(expression))                                                                         \
            errchk_print_warning(__func__, __FILE__, __LINE__, #expression, NULL);                 \
    } while (0)

#define ERRCHKK(expression, description)                                                           \
    do {                                                                                           \
        if (!(expression))                                                                         \
            errchk_print_error(__func__, __FILE__, __LINE__, #expression, (description));          \
    } while (0)

#define WARNCHKK(expression, description)                                                          \
    do {                                                                                           \
        if (!(expression))                                                                         \
            errchk_print_warning(__func__, __FILE__, __LINE__, #expression, (description));        \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

void errchk_print_error(const char* function, const char* file, const long line,
                        const char* expression, const char* description);

void errchk_print_warning(const char* function, const char* file, const long line,
                          const char* expression, const char* description);

#ifdef __cplusplus
}
#endif
