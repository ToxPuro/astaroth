#include "errchk_print.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <execinfo.h> // Stacktrace
#include <unistd.h>

static void
fprintc_multiple(FILE* stream, const char c, const size_t count)
{
    for (size_t i = 0; i < count; ++i)
        fprintf(stream, "%c", c);
    fprintf(stream, "\n");
}

__attribute__((__format__(__printf__, 5, 6))) void
errchk_print_error(const char* function, const char* file, const long line, const char* expression,
                   const char* fmt, ...)
{
    fflush(stdout);
    fflush(stderr);

    time_t terr;
    time(&terr);

    fprintc_multiple(stderr, '\n', 3);
    fprintf(stderr, "┌──────────────────────── ERROR ───────────────────────────┐\n");
    fprintc_multiple(stderr, '\n', 2);
    fprintf(stderr, "%s\n", ctime(&terr));
    if (expression && expression[0] != '\0')
        fprintf(stderr, "Expression '%s' evaluated false\n", expression);
    if (fmt && fmt[0] != '\0') {
        fprintf(stderr, "Description: ");
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Function '%s',\n", function);
    fprintf(stderr, "File '%s',\n", file);
    fprintf(stderr, "On line '%ld'\n", line);
    fprintc_multiple(stderr, '\n', 2);
    fprintf(stderr, "└──────────────────────── ERROR ───────────────────────────┘\n");
    fprintc_multiple(stderr, '\n', 3);
    fflush(stdout);
    fflush(stderr);
}

__attribute__((__format__(__printf__, 5, 6))) void
errchk_print_warning(const char* function, const char* file, const long line,
                     const char* expression, const char* fmt, ...)
{
    time_t terr;
    time(&terr);

    fprintf(stderr, "\n");
    fprintf(stderr, "──────────────────────── Warning ─────────────────────────────\n");
    fprintf(stderr, "%s\n", ctime(&terr));
    if (expression && expression[0] != '\0')
        fprintf(stderr, "Expression '%s' evaluated false\n", expression);
    if (fmt && fmt[0] != '\0') {
        fprintf(stderr, "Description: ");
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "in function '%s',\n", function);
    fprintf(stderr, "file '%s',\n", file);
    fprintf(stderr, "on line '%ld.'\n", line);
    fprintf(stderr, "───────────────────────── Warning ────────────────────────────\n");
    fprintf(stderr, "\n");
    fflush(stdout);
    fflush(stderr);
}

__attribute__((__format__(__printf__, 3, 4))) void
errchk_print_log(const char* function, const long line, const char* fmt, ...)
{
    time_t now          = time(NULL);
    struct tm* timeinfo = localtime(&now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%H:%M:%S", timeinfo);
    printf("[%s] %s:%ld", timestamp, function, line);
    if (fmt && fmt[0] != '\0') {
        printf(", ");
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
    }
    printf("\n");
}

void
errchk_print_stacktrace(void)
{
    void* array[20];
    const int count = backtrace(array, 20);
    char** strings  = backtrace_symbols(array, count);
    backtrace_symbols_fd(array, count, STDERR_FILENO);
    free(strings);
}
