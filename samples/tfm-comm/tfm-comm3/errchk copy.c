#include "errchk.h"

#include <stdio.h>
#include <time.h>

static void
fprintc_multiple(FILE* stream, const char c, const size_t count)
{
    for (size_t i = 0; i < count; ++i)
        fprintf(stream, "%c", c);
    fprintf(stream, "\n");
}

void
errchk_print_error(const char* function, const char* file, const long line, const char* expression,
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
    fprintf(stderr, "On line '%ld'\n", line);
    fprintc_multiple(stderr, '\n', 2);
    fprintf(stderr, "└──────────────────────── ERROR ───────────────────────────┘\n");
    fprintc_multiple(stderr, '\n', 3);
    fflush(stdout);
    fflush(stderr);
}

void
errchk_print_warning(const char* function, const char* file, const long line,
                     const char* expression, const char* description)
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
    fprintf(stderr, "on line '%ld.'\n", line);
    fprintf(stderr, "───────────────────────── Warning ────────────────────────────\n");
    fprintf(stderr, "\n");
    fflush(stdout);
    fflush(stderr);
}

bool
errchk_check_ok(const char* function, const char* file, const long line, const char* expression,
                const char* description)
{
    return true;
}
