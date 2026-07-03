/*
    Copyright (C) 2026, Ondřej Míchal.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "source_manager.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ast.h"
#include "string_vec.h"

#define ACC_SOURCE_RESOURCE_INIT_CAPACITY 4
#define ACC_SOURCES_MANAGER_INIT_CAPACITY 32

#define ACC_CHECK_INSTANCE(T)                                                  \
  if (self == NULL) {                                                          \
    fprintf(stderr, "Fatal error(%s:%d): Invalid instance of type " #T "\n",   \
            __FILE__, __LINE__);                                               \
    exit(EXIT_FAILURE);                                                        \
  }
#define ACC_CHECK_SNPRINTF(E)                                                  \
  do {                                                                         \
    int _snprintf_retval = (E);                                                \
    if (_snprintf_retval < 0 || _snprintf_retval >= BUFFER_SIZE) {             \
      fprintf(                                                                 \
          stderr,                                                              \
          "Fatal error(%s:%d): snprintf truncated or failed with code %d\n",   \
          __FILE__, __LINE__, _snprintf_retval);                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define ACC_FPRINTF(S, F, ...)                                                 \
  do {                                                                         \
    if (S != NULL)                                                             \
      fprintf(S, F, ##__VA_ARGS__);                                            \
  } while (0)

void
acc_gen_dlsym(FILE* fp, const char* func_name)
{
  fprintf(fp, "LOAD_DSYM(%s, stream)\n", func_name);
}

struct _AccSourceDeclaration {
  char* expr;
  AccSourceDeclarationFlags flags;
};

static const char*
get_declaration_definition_qualifier(AccSourceDeclarationFlags flags)
{
  switch ((int)flags) {
  case ACC_SRC_DECL_PUBLIC | ACC_SRC_DECL_DEVICE | ACC_SRC_DECL_CONSTANT:
  case ACC_SRC_DECL_DEVICE | ACC_SRC_DECL_CONSTANT:
    return "__device__ __constant__";
  case ACC_SRC_DECL_PUBLIC | ACC_SRC_DECL_DEVICE:
  case ACC_SRC_DECL_DEVICE:
    return "__device__";
  case ACC_SRC_DECL_PUBLIC | ACC_SRC_DECL_CONSTANT:
    return "const";
  case ACC_SRC_DECL_CONSTANT:
    return "static const";
  case ACC_SRC_DECL_PUBLIC:
    return "";
  default:
    fprintf(stderr,
            "Unexpected source declaration flags! Expect the unexpected!\n");
    return "";
  }
}

static const char*
get_declaration_reference_qualifier(AccSourceDeclarationFlags flags)
{
  // The reference qualifier is always used with a public symbol.
  if (flags & ACC_SRC_DECL_DEVICE) {
    if (flags & ACC_SRC_DECL_CONSTANT)
      return "extern __device__ __constant__";
    else
      return "extern __device__";
  }
  else {
    if (flags & ACC_SRC_DECL_CONSTANT)
      return "extern const";
    else
      return "extern";
  }
}

static void
acc_source_declaration_invalidate(struct _AccSourceDeclaration* self)
{
  if (self == NULL)
    return;

  #define _FREE(P) if (P != NULL) { free(P); P = NULL; }
  _FREE(self->expr);
  #undef _FREE
}

static void
acc_source_declaration_flush(struct _AccSourceDeclaration* self, FILE* fp_decls,
                             FILE* fp_defs)
{
  if (self->expr == NULL)
    return;

  if (!(self->flags & ACC_SRC_DECL_UNMANAGED)) {
    // Add appropriate qualifiers (e.g., const, __constant__) based on where
    // the data resides (host, device).
    char decl_definition[BUFFER_SIZE] = {0};
    ACC_CHECK_SNPRINTF(snprintf(
        decl_definition, BUFFER_SIZE, "%s %s",
        get_declaration_definition_qualifier(self->flags), self->expr));

    ACC_FPRINTF(fp_defs, "%s\n", decl_definition);

    // If public, then define the variable in a source file and reference it
    // as external in a header file.
    if (self->flags & ACC_SRC_DECL_PUBLIC) {
      char decl_reference[BUFFER_SIZE] = {0};
      ACC_CHECK_SNPRINTF(snprintf(
          decl_reference, BUFFER_SIZE, "%s %s",
          get_declaration_reference_qualifier(self->flags), self->expr));

      ACC_FPRINTF(fp_decls, "%s\n", decl_reference);
    }
  }
  else {
    if (self->flags & ACC_SRC_DECL_PUBLIC)
      ACC_FPRINTF(fp_decls, "%s\n", self->expr);
    else
      ACC_FPRINTF(fp_defs, "%s\n", self->expr);
  }

  acc_source_declaration_invalidate(self);
}

struct _AccSourceFunction {
  char name[BUFFER_SIZE];
  char qualifiers[BUFFER_SIZE];
  char params[BUFFER_SIZE];

  char* implementation;
  unsigned long implementation_n;

  enum AccSourceFunctionFlags flags;
};

static void
acc_source_function_invalidate(AccSourceFunction* self)
{
  if (self == NULL)
    return;

  #define _FREE(P) if (P != NULL) { free(P); P = NULL; }
  _FREE(self->implementation);
  #undef _FREE
}

static void
print_func_implementation(FILE* stream, char* implementation)
{
  char* tmp  = NULL;
  char* line = strtok_r(implementation, "\n", &tmp);
  while (line != NULL) {
    fprintf(stream, "  %s\n", line);
    line = strtok_r(NULL, "\n", &tmp);
  }
  fprintf(stream, "}\n\n");
}

static void
acc_source_function_flush(AccSourceFunction* self, bool header_only,
                          FILE* fp_decls, FILE* fp_defs, FILE* fp_loads)
{
  if (!header_only) {
    // Declarations
    ACC_FPRINTF(fp_decls, "FUNC_DEFINE(%s, %s, %s);\n", self->qualifiers,
                self->name, self->params);

    if (self->implementation != NULL) {
      // Implementations
      ACC_FPRINTF(fp_defs, "%s %s%s\n{\n", self->qualifiers, self->name,
                  self->params);

      print_func_implementation(fp_defs, self->implementation);

      // Loaders
      acc_gen_dlsym(fp_loads, self->name);
    }
  }
  else {
    if (self->implementation != NULL) {
      ACC_FPRINTF(fp_decls, "%s %s%s\n{\n", self->qualifiers, self->name,
                  self->params);
      print_func_implementation(fp_decls, self->implementation);
    }
    else {
      ACC_FPRINTF(fp_decls, "%s %s%s;\n", self->qualifiers, self->name,
                  self->params);
    }
  }

  acc_source_function_invalidate(self);
}

void
acc_source_function_add_impl(AccSourceFunction* self, const char* format, ...)
{
  ACC_CHECK_INSTANCE(AccSourceFunction)

  va_list args;
  va_start(args, format);
  char* added_impl           = NULL;
  unsigned long added_impl_n = vasprintf(&added_impl, format, args) /
                               sizeof(char);
  va_end(args);

  if (self->implementation != NULL) {
    char* new_impl = calloc(added_impl_n + self->implementation_n, 1);

    // We know the size of the old and added implementation, so we can just
    // use memset directly.
    memcpy(new_impl, self->implementation, self->implementation_n);
    // Separate the two parts of concatenated implementation by a newline.
    new_impl[self->implementation_n - 1] = '\n';
    memcpy(new_impl + self->implementation_n, added_impl, added_impl_n);

    free(added_impl);
    free(self->implementation);

    self->implementation = new_impl;
    self->implementation_n += added_impl_n;
  }
  else {
    self->implementation   = added_impl;
    self->implementation_n = added_impl_n;
  }
}

void
acc_source_function_set_params(AccSourceFunction* self, const char* format, ...)
{
  ACC_CHECK_INSTANCE(AccSourceFunction)

  va_list args;
  va_start(args, format);
  ACC_CHECK_SNPRINTF(vsnprintf(self->params, BUFFER_SIZE, format, args));
  va_end(args);
}

void
acc_source_function_set_qualifiers(AccSourceFunction* self, const char* format,
                                   ...)
{
  ACC_CHECK_INSTANCE(AccSourceFunction)

  va_list args;
  va_start(args, format);
  ACC_CHECK_SNPRINTF(vsnprintf(self->qualifiers, BUFFER_SIZE, format, args));
  va_end(args);
}

struct _AccSource {
  char name[BUFFER_SIZE];
  AccSourceFlags flags;

  FILE* fp_decls;
  FILE* fp_defs;
  FILE* fp_loads;

  string_vec includes;
  int includes_public;

  struct _AccSourceDeclaration* decls;
  int decls_n;
  int decls_capacity;

  AccSourceFunction* funcs;
  int funcs_n;
  int funcs_capacity;
};

void
acc_source_add_declaration(AccSource* self, AccSourceDeclarationFlags flags,
                           const char* format, ...)
{
  ACC_CHECK_INSTANCE(AccSource)

  va_list args;
  va_start(args, format);
  char* expr = NULL;
  vasprintf(&expr, format, args);
  va_end(args);

  if (self->decls_n >= self->decls_capacity) {
    self->decls_capacity *= 2;
    self->decls = reallocarray(self->decls, self->decls_capacity,
                               sizeof(struct _AccSourceDeclaration));
  }

  struct _AccSourceDeclaration* decl = &self->decls[self->decls_n++];
  decl->expr                         = expr;
  decl->flags                        = flags;
}

void
acc_source_add_include(AccSource* self, bool private, bool system,
                       const char* include, const char* condition)
{
  ACC_CHECK_INSTANCE(AccSource)

  if (str_vec_contains(self->includes, include))
    return;

  self->includes_public |= (private ? 0 : 1 << self->includes.size);
  push(&self->includes, include);
}

AccSourceFunction*
acc_source_get_function(AccSource* self, AccSourceFunctionFlags flags,
                        const char* format, ...)
{
  ACC_CHECK_INSTANCE(AccSource)

  // All functions in a CPP source are CPP functions.
  if (self->flags & ACC_SRC_CPP)
    flags |= ACC_SRC_FUNC_CPP;

  if (self->funcs_n >= self->funcs_capacity) {
    self->funcs_capacity *= 2;
    self->funcs = reallocarray(self->funcs, self->funcs_capacity,
                               sizeof(AccSourceFunction));
  }

  AccSourceFunction* func = &self->funcs[self->funcs_n++];
  va_list args;
  va_start(args, format);
  ACC_CHECK_SNPRINTF(vsnprintf(func->name, BUFFER_SIZE, format, args));
  va_end(args);
  func->flags = flags;
  memset(func->qualifiers, 0, BUFFER_SIZE * sizeof(char));
  memset(func->params, 0, BUFFER_SIZE * sizeof(char));
  func->implementation = NULL;

  return func;
}

static void
acc_source_invalidate(AccSource* self)
{
  #define _FCLOSE(F) if (F != NULL) { fclose(F); F = NULL; }
  _FCLOSE(self->fp_decls);
  _FCLOSE(self->fp_defs);
  _FCLOSE(self->fp_loads);
  #undef _FCLOSE

  for (int i = 0; i < self->funcs_n; ++i)
    acc_source_function_invalidate(&self->funcs[i]);

  for (int i = 0; i < self->decls_n; ++i)
    acc_source_declaration_invalidate(&self->decls[i]);

  #define _FREE(P) if (P != NULL) { free(P); P = NULL; }
  _FREE(self->decls);
  _FREE(self->funcs);
  #undef _FREE

  free_str_vec(&self->includes);
  self->includes_public = 0;
}

static void
print_system_include(FILE* stream, const char* included_file)
{
  fprintf(stream, "#include <%s>\n", included_file);
}

static void
print_app_include(FILE* stream, const char* included_file)
{
  fprintf(stream, "#include \"%s\"\n", included_file);
}

static void
print_includes(AccSource* self, const char* decls_filename)
{
  ACC_FPRINTF(self->fp_decls, "#pragma once\n\n");

  if (!(self->flags & ACC_SRC_HEADER_ONLY)) {
    print_app_include(self->fp_defs, decls_filename);
    ACC_FPRINTF(self->fp_defs, "\n");
  }

  int public_includes  = 0;
  int private_includes = 0;
  for (size_t i = 0; i < self->includes.size; ++i) {
    const char* include = self->includes.data[i];

    print_app_include(self->includes_public & (1 << i) ? self->fp_decls
                                                       : self->fp_defs,
                      include);

    self->includes_public & (1 << i) ? ++public_includes : ++private_includes;
  }

  // Formatting
  if (public_includes)
    ACC_FPRINTF(self->fp_decls, "\n");
  if (private_includes)
    ACC_FPRINTF(self->fp_defs, "\n");
}

static void
print_start_global_guards(AccSource* self)
{
  if (self->flags & ACC_SRC_CPP) {
    ACC_FPRINTF(self->fp_decls, "#ifdef __cplusplus\n\n");
    ACC_FPRINTF(self->fp_defs, "#ifdef __cplusplus\n\n");
    ACC_FPRINTF(self->fp_loads, "#ifdef __cplusplus\n\n");
  }
}

static void
print_declarations(AccSource* self, bool epilogue)
{
  int public_decls  = 0;
  int private_decls = 0;

  for (int i = 0; i < self->decls_n; ++i) {
    struct _AccSourceDeclaration* decl = &self->decls[i];

    if (!epilogue && decl->flags & ACC_SRC_DECL_EPILOGUE)
      continue;

    acc_source_declaration_flush(decl, self->fp_decls, self->fp_defs);

    (decl->flags & ACC_SRC_DECL_PUBLIC) ? ++public_decls : ++private_decls;
  }

  // Spacing
  if (public_decls > 0)
    ACC_FPRINTF(self->fp_decls, "\n");
  if (private_decls > 0)
    ACC_FPRINTF(self->fp_defs, "\n");
}

static void
print_start_c_declarations(AccSource* self)
{
  ACC_FPRINTF(self->fp_decls, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
  ACC_FPRINTF(self->fp_defs, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
  ACC_FPRINTF(self->fp_loads, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
}

static void
print_end_c_declarations(AccSource* self)
{
  ACC_FPRINTF(self->fp_decls, "\n#ifdef __cplusplus\n}\n#endif\n");
  ACC_FPRINTF(self->fp_defs, "\n#ifdef __cplusplus\n}\n#endif\n");
  ACC_FPRINTF(self->fp_loads, "\n#ifdef __cplusplus\n}\n#endif\n");
}

static void
print_c_sources(AccSource* self)
{
  print_start_c_declarations(self);

  int c_funcs = 0;
  for (int i = 0; i < self->funcs_n; ++i) {
    AccSourceFunction* func = &self->funcs[i];

    if (func->flags & ACC_SRC_FUNC_CPP)
      continue;

    acc_source_function_flush(func, self->flags & ACC_SRC_HEADER_ONLY,
                              self->fp_decls, self->fp_defs, self->fp_loads);
    ++c_funcs;
  }

  print_end_c_declarations(self);

  // Formatting
  if (c_funcs > 0) {
    ACC_FPRINTF(self->fp_decls, "\n");
    ACC_FPRINTF(self->fp_defs, "\n");
    ACC_FPRINTF(self->fp_loads, "\n");
  }
}

static void
print_cpp_sources(AccSource* self)
{
  if (!(self->flags & ACC_SRC_CPP)) {
    ACC_FPRINTF(self->fp_decls, "#ifdef __cplusplus\n");
    ACC_FPRINTF(self->fp_defs, "#ifdef __cplusplus\n");
    ACC_FPRINTF(self->fp_loads, "#ifdef __cplusplus\n");
  }

  int cpp_funcs = 0;
  for (int i = 0; i < self->funcs_n; ++i) {
    AccSourceFunction* func = &self->funcs[i];

    if (!(func->flags & ACC_SRC_FUNC_CPP))
      continue;

    acc_source_function_flush(func, self->flags & ACC_SRC_HEADER_ONLY,
                              self->fp_decls, self->fp_defs, self->fp_loads);

    ++cpp_funcs;
  }

  // Formatting
  if (cpp_funcs > 0) {
    ACC_FPRINTF(self->fp_decls, "\n");
    ACC_FPRINTF(self->fp_defs, "\n");
    ACC_FPRINTF(self->fp_loads, "\n");
  }
}

static void
print_end_guards(AccSource* self)
{
  ACC_FPRINTF(self->fp_decls, "#endif\n\n");
  ACC_FPRINTF(self->fp_defs, "#endif\n\n");
  ACC_FPRINTF(self->fp_loads, "#endif\n\n");
}

static const char*
get_source_file_extension(AccSourceFlags flags)
{
  switch ((int)flags) {
  case ACC_SRC_DEVICE:
    return "cu";
  case ACC_SRC_CPP:
    return "cpp";
  default:
    return "c";
  }
}

void
acc_source_flush(AccSource* self)
{
  ACC_CHECK_INSTANCE(AccSource)

  if (self->decls == NULL || self->funcs == NULL)
    return;

  char decls_filename[BUFFER_SIZE] = {0};
  ACC_CHECK_SNPRINTF(
      snprintf(decls_filename, BUFFER_SIZE, "%s%s.h", self->name,
               (self->flags & ACC_SRC_HEADER_ONLY) ? "" : "_decls"));
  self->fp_decls = fopen(decls_filename, "w");

  if (!(self->flags & ACC_SRC_HEADER_ONLY)) {
    char filename[BUFFER_SIZE] = {0};

    ACC_CHECK_SNPRINTF(snprintf(filename, BUFFER_SIZE, "%s.%s", self->name,
                                get_source_file_extension(self->flags)));
    self->fp_defs = fopen(filename, "w");

    ACC_CHECK_SNPRINTF(
        snprintf(filename, BUFFER_SIZE, "%s_loads.inc", self->name));
    self->fp_loads = fopen(filename, "w");
  }

  print_includes(self, decls_filename);
  print_start_global_guards(self);
  print_declarations(self, false);
  print_c_sources(self);
  print_cpp_sources(self);
  print_end_guards(self);
  print_declarations(self, true);

  acc_source_invalidate(self);
}

struct _AccSourcesManager {
  AccSource* sources;
  int sources_n;
  int capacity;
};

static AccSourcesManager sources_manager;

AccSourcesManager*
acc_sources_manager_singleton()
{
  if (sources_manager.capacity == 0) {
    sources_manager.sources_n = 0;
    sources_manager.capacity  = ACC_SOURCES_MANAGER_INIT_CAPACITY;
    sources_manager.sources   = calloc(ACC_SOURCES_MANAGER_INIT_CAPACITY,
                                       sizeof(AccSource));
  }

  return &sources_manager;
}

void
acc_sources_manager_flush(AccSourcesManager* self)
{
  ACC_CHECK_INSTANCE(AccSourcesManager)

  for (int i = 0; i < self->sources_n; ++i) {
    acc_source_flush(&self->sources[i]);
  }

  // Set the manager to its initial state in case we'd like to use it again
  // after flushing.
  free(self->sources);
  self->sources   = NULL;
  self->sources_n = 0;
  self->capacity  = 0;
}

AccSource*
acc_sources_manager_get_source(AccSourcesManager* self, const char* name,
                               AccSourceFlags flags)
{
  ACC_CHECK_INSTANCE(AccSourcesManager)

  for (int i = 0; i < self->sources_n; ++i) {
    AccSource* source = &self->sources[i];

    if (strcmp(source->name, name) == 0)
      return source;
  }

  if (self->sources_n >= self->capacity) {
    self->capacity *= 2;
    self->sources = reallocarray(self->sources, self->capacity,
                                 sizeof(AccSource));
  }

  AccSource* source = &self->sources[self->sources_n++];

  strncpy(source->name, name, BUFFER_SIZE - 1);
  source->flags = flags;

  source->fp_decls = NULL;
  source->fp_defs  = NULL;
  source->fp_loads = NULL;

  source->includes_public = 0;

  source->decls_capacity = ACC_SOURCE_RESOURCE_INIT_CAPACITY;
  source->decls_n        = 0;
  source->decls          = calloc(ACC_SOURCE_RESOURCE_INIT_CAPACITY,
                                  sizeof(struct _AccSourceDeclaration));

  source->funcs_capacity = ACC_SOURCE_RESOURCE_INIT_CAPACITY;
  source->funcs_n        = 0;
  source->funcs          = calloc(ACC_SOURCE_RESOURCE_INIT_CAPACITY,
                                  sizeof(AccSourceFunction));

  return source;
}

void
acc_sources_manager_invalidate_source(AccSourcesManager* self, const char* name)
{
  ACC_CHECK_INSTANCE(AccSourcesManager)

  for (int i = 0; i < self->sources_n; ++i) {
    AccSource* source = &self->sources[i];

    if (strcmp(source->name, name) != 0)
      continue;

    // Invalidate the whole source and its content and shift all subsequent
    // sources to the left unless the invalidated source is the last one.
    acc_source_invalidate(source);

    for (int j = i + 1; j < self->sources_n; ++j) {
      AccSource* prev = &self->sources[i-1];
      AccSource* curr = &self->sources[i];

      memcpy(prev, curr, sizeof(AccSource));
    }

    self->sources_n -= 1;
    return;
  }
}
