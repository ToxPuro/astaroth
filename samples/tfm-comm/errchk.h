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
}

static inline void
print_error(const char* function, const char* file, const size_t line, const char* expression,
            const char* description)
{
    fflush(stdout);
    fflush(stderr);
    time_t terr;
    time(&terr);

    fprintc_multiple(stderr, '\n', 4);
    fprintf(stderr, "┌──────────────────────── ERROR ───────────────────────────┐\n");
    fprintc_multiple(stderr, '\n', 3);
    fprintf(stderr, "%s\n", ctime(&terr));
    if (expression)
        fprintf(stderr, "Expression '%s' evaluated false\n", expression);
    if (description)
        fprintf(stderr, "Description: '%s'\n", description);
    fprintf(stderr, "Function '%s',\n", function);
    fprintf(stderr, "File '%s',\n", file);
    fprintf(stderr, "On line '%zu'\n", line);
    fprintc_multiple(stderr, '\n', 3);
    fprintf(stderr, "└──────────────────────── ERROR ───────────────────────────┘\n");
    fprintc_multiple(stderr, '\n', 4);
    fflush(stdout);
    fflush(stderr);
}

static inline void
print_warning(const char* function, const char* file, const size_t line, const char* expression,
              const char* description)
{
    time_t terr;
    time(&terr);

    fprintc_multiple(stderr, '\n', 1);
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
    fprintc_multiple(stderr, '\n', 1);
    fflush(stdout);
    fflush(stderr);
}

#define ERROR(description)                                                                         \
    {                                                                                              \
        print_error(__func__, __FILE__, __LINE__, NULL, description);                              \
    }

#define WARNING(description)                                                                       \
    {                                                                                              \
        print_warning(__func__, __FILE__, __LINE__, NULL, description);                            \
    }

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(expression)                                                                         \
    {                                                                                              \
        if ((expression) == false)                                                                 \
            print_error(__func__, __FILE__, __LINE__, #expression, NULL);                          \
    }

#define WARNCHK(expression)                                                                        \
    {                                                                                              \
        if ((expression) == false)                                                                 \
            print_warning(__func__, __FILE__, __LINE__, #expression, NULL);                        \
    }

#define ERRCHKK(expression, description)                                                           \
    {                                                                                              \
        if ((expression) == false)                                                                 \
            print_error(__func__, __FILE__, __LINE__, #expression, description);                   \
    }

#define WARNCHKK(expression, description)                                                          \
    {                                                                                              \
        if ((expression) == false)                                                                 \
            print_warning(__func__, __FILE__, __LINE__, #expression, description);                 \
    }
