#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline void
fprintc_multiple(FILE* stream, const char c, const size_t count)
{
    for (size_t i = 0; i < count; ++i)
        fprintf(stream, "%c", c);
    fprintf(stream, "\n");
}

static inline void
print_error(const char* function, const char* file, const size_t line, const char* expression,
            const char* description)
{
    fflush(stdout);
    fflush(stderr);

    time_t terr;
    time(&terr);

    fprintc_multiple(stderr, '\n', 3);
    fprintf(stderr, "┌──────────────────────── ERROR ───────────────────────────┐\n");
    fprintc_multiple(stderr, '\n', 2);
    fprintf(stderr, "%s\n", ctime(&terr));
    if (expression)
        fprintf(stderr, "Expression '%s' evaluated false\n", expression);
    if (description)
        fprintf(stderr, "Description: '%s'\n", description);
    fprintf(stderr, "Function '%s',\n", function);
    fprintf(stderr, "File '%s',\n", file);
    fprintf(stderr, "On line '%zu'\n", line);
    fprintc_multiple(stderr, '\n', 2);
    fprintf(stderr, "└──────────────────────── ERROR ───────────────────────────┘\n");
    fprintc_multiple(stderr, '\n', 3);
    fflush(stdout);
    fflush(stderr);
}

static inline void
print_warning(const char* function, const char* file, const size_t line, const char* expression,
              const char* description)
{
    time_t terr;
    time(&terr);

    fprintf(stderr, "\n");
    fprintf(stderr, "──────────────────────── Warning ─────────────────────────────\n");
    fprintf(stderr, "%s\n", ctime(&terr));
    if (expression)
        fprintf(stderr, "Expression '%s' evaluated false\n", expression);
    if (description)
        fprintf(stderr, "Description: '%s.'\n", description);
    fprintf(stderr, "in function '%s',\n", function);
    fprintf(stderr, "file '%s',\n", file);
    fprintf(stderr, "on line '%zu.'\n", line);
    fprintf(stderr, "───────────────────────── Warning ────────────────────────────\n");
    fprintf(stderr, "\n");
    fflush(stdout);
    fflush(stderr);
}

#define ERROR(description)                                                                         \
    do {                                                                                           \
        print_error(__func__, __FILE__, __LINE__, NULL, description);                              \
    } while (0)

#define WARNING(description)                                                                       \
    do {                                                                                           \
        print_warning(__func__, __FILE__, __LINE__, NULL, description);                            \
    } while (0)

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(expression)                                                                         \
    do {                                                                                           \
        if ((expression) == false)                                                                 \
            print_error(__func__, __FILE__, __LINE__, #expression, NULL);                          \
    } while (0)

#define WARNCHK(expression)                                                                        \
    do {                                                                                           \
        if ((expression) == false)                                                                 \
            print_warning(__func__, __FILE__, __LINE__, #expression, NULL);                        \
    } while (0)

#define ERRCHKK(expression, description)                                                           \
    do {                                                                                           \
        if ((expression) == false)                                                                 \
            print_error(__func__, __FILE__, __LINE__, #expression, description);                   \
    } while (0)

#define WARNCHKK(expression, description)                                                          \
    do {                                                                                           \
        if ((expression) == false)                                                                 \
            print_warning(__func__, __FILE__, __LINE__, #expression, description);                 \
    } while (0)
