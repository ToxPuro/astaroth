/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_USERS_JULIANLAGG_ASTAROTH_RUNDIR_HYDROTEST_ACC_SRC_ACC_TAB_H_INCLUDED
# define YY_YY_USERS_JULIANLAGG_ASTAROTH_RUNDIR_HYDROTEST_ACC_SRC_ACC_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    CONSTANT = 258,
    IN = 259,
    OUT = 260,
    UNIFORM = 261,
    IDENTIFIER = 262,
    NUMBER = 263,
    REAL_NUMBER = 264,
    DOUBLE_NUMBER = 265,
    RETURN = 266,
    SCALAR = 267,
    VECTOR = 268,
    MATRIX = 269,
    SCALARFIELD = 270,
    SCALARARRAY = 271,
    VOID = 272,
    INT = 273,
    INT3 = 274,
    COMPLEX = 275,
    IF = 276,
    ELSE = 277,
    FOR = 278,
    WHILE = 279,
    ELIF = 280,
    LAND = 281,
    LOR = 282,
    BINARY_OP = 283,
    KERNEL = 284,
    DEVICE = 285,
    PREPROCESSED = 286,
    INPLACE_INC = 287,
    INPLACE_DEC = 288
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_USERS_JULIANLAGG_ASTAROTH_RUNDIR_HYDROTEST_ACC_SRC_ACC_TAB_H_INCLUDED  */
