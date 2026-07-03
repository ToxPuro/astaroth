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

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

void acc_gen_dlsym(FILE* fp, const char* func_name);

typedef struct _AccSourceFunction AccSourceFunction;

void acc_source_function_add_impl(AccSourceFunction* self, const char* format,
                                  ...);
void acc_source_function_set_params(AccSourceFunction* self, const char* format,
                                    ...);
void acc_source_function_set_qualifiers(AccSourceFunction* self,
                                        const char* format, ...);

typedef enum AccSourceFlags {
  ACC_SRC_CPP         = (1 << 0),
  ACC_SRC_HEADER_ONLY = (1 << 1),
  ACC_SRC_DEVICE      = (1 << 2),
} AccSourceFlags;

typedef enum AccSourceFunctionFlags {
  ACC_SRC_FUNC_CPP     = (1 << 0),
  ACC_SRC_FUNC_PRIVATE = (1 << 1),
} AccSourceFunctionFlags;

typedef enum AccSourceDeclarationFlags {
  ACC_SRC_DECL_UNMANAGED = (1 << 0),
  ACC_SRC_DECL_CONSTANT  = (1 << 1),
  ACC_SRC_DECL_PUBLIC    = (1 << 2),
  ACC_SRC_DECL_DEVICE    = (1 << 3),
  ACC_SRC_DECL_EPILOGUE  = (1 << 4),
} AccSourceDeclarationFlags;

typedef struct _AccSource AccSource;

void acc_source_add_declaration(AccSource* self,
                                AccSourceDeclarationFlags flags,
                                const char* format, ...);
void acc_source_add_include(AccSource* self, bool private, bool system,
                            const char* include, const char* condition);
AccSourceFunction* acc_source_get_function(AccSource* self,
                                           AccSourceFunctionFlags flags,
                                           const char* format, ...);
void acc_source_flush(AccSource* self);

typedef struct _AccSourcesManager AccSourcesManager;

AccSourcesManager* acc_sources_manager_singleton();

void acc_sources_manager_flush(AccSourcesManager* self);
AccSource* acc_sources_manager_get_source(AccSourcesManager* self,
                                          const char* name,
                                          AccSourceFlags flags);
void acc_sources_manager_invalidate_source(AccSourcesManager* self,
                                           const char* name);
