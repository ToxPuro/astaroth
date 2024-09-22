#include "stdarg.h"

#define VEC_INITIALIZER {.size = 0,.capacity  = 0,.data = NULL}
typedef enum ReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} ReduceOp;

typedef struct string_vec
{
	//char* data[256];
	char** data;
	size_t size;
	int capacity;

} string_vec;
typedef struct int_vec 
{
	int* data;
	size_t size;
	int capacity;

} int_vec;

typedef struct op_vec 
{
	ReduceOp* data;
	size_t size;
	int capacity;

} op_vec;

static inline void
init_str_vec(string_vec* vec)
{
	vec -> size = 0;
	vec -> capacity = 1;
	vec -> data = malloc(sizeof(char*)*vec ->capacity);
}

static inline void
free_str_vec(string_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 0;
	vec -> data = NULL;
	//vec -> data = malloc(sizeof(char*)*vec ->capacity);
}
static inline void
free_int_vec(int_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 0;
	vec -> data = NULL;
}

static inline void
free_op_vec(op_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 0;
	vec -> data = NULL;
	vec = NULL;
	//vec -> data = malloc(sizeof(char*)*vec ->capacity);
}
static inline void
init_int_vec(int_vec* vec)
{
	vec -> size = 0;
	vec -> capacity = 1;
	vec -> data = malloc(sizeof(int)*vec ->capacity);
}
static inline int
int_vec_get_index(const int_vec vec, const int val)
{
	for(size_t i = 0; i <  vec.size; ++i)
		if(vec.data[i] == val) return i;
	return -1;
}
static inline bool
int_vec_contains(const int_vec vec, const int val)
{
	return int_vec_get_index(vec,val) >= 0;
}

static inline bool
int_vec_contains_arr(int_vec vec,const int* elems,const int n_elems)
{
	for(int i = 0; i < n_elems; ++i)
		if(int_vec_get_index(vec,elems[i]) >= 0) return true;
	return false;
}

#define int_vec_contains(F, ...) int_vec_contains_arr( \
	F, (const int[]){ __VA_ARGS__ }, \
	sizeof (const int[]){ __VA_ARGS__ } / sizeof (const int) \
)
static inline void
init_op_vec(op_vec* vec)
{
	vec -> size = 0;
	vec -> capacity = 1;
	vec -> data = malloc(sizeof(ReduceOp)*vec ->capacity);
}
static inline int
str_vec_get_index(string_vec vec, const char* str)
{
	for(size_t i = 0; i <  vec.size; ++i)
		if(!strcmp(vec.data[i],str)) return i;
	return -1;
}
static inline bool
str_vec_contains_arr(string_vec vec,const char* const* elems,const int n_elems)
{
	for(int i = 0; i < n_elems; ++i)
		if(str_vec_get_index(vec,elems[i]) >= 0) return true;
	return false;
}

#define str_vec_contains(F, ...) str_vec_contains_arr( \
	F, (const char*[]){ __VA_ARGS__ }, \
	sizeof (const char*[]){ __VA_ARGS__ } / sizeof (const char*) \
)
static inline int 
str_cmps_arr(const char* string_to_test, const char* const* elems, const int n_elems)
{
	for(int i = 0; i < n_elems; ++i)
		if(!strcmp(string_to_test,elems[i])) return 0;
	return 1;
}
#define strcmps(F, ...) str_cmps_arr( \
	F, (const char*[]){ __VA_ARGS__ }, \
	sizeof (const char*[]){ __VA_ARGS__ } / sizeof (const char*) \
)

static char*
strdupnullok(const char* src)
{
	if(src == NULL) return NULL;
	return strdup(src);
}
static inline int
push(string_vec* dst, const char* src)
{
	if(dst->capacity == 0)
	{
		dst->capacity++;
		dst->data = malloc(sizeof(char*)*dst->capacity);
	}
	dst->data[dst->size] = strdupnullok(src);
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		char** tmp = malloc(sizeof(char*)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}

static inline
string_vec
str_vec_copy(string_vec vec)
{
	string_vec res = VEC_INITIALIZER;
	for(size_t i = 0; i < vec.size; ++i)
		push(&res,vec.data[i]);
	return res;
}
static inline
void
str_vec_remove(string_vec* vec, const char* elem_to_remove)
{
	string_vec res = VEC_INITIALIZER;
	for(size_t i = 0; i < vec->size; ++i)
	{
		if(strcmp(vec->data[i],elem_to_remove))
			push(&res,vec->data[i]);
	}
	free_str_vec(vec);
	*vec = res;
}

static inline int
push_int(int_vec* dst, int src)
{
	if(dst->capacity == 0)
	{
		dst->capacity++;
		dst->data = malloc(sizeof(int)*dst->capacity);
	}
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		int* tmp = malloc(sizeof(int)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}


static inline int
push_op(op_vec* dst, ReduceOp src)
{
	if(dst->capacity == 0)
	{
		dst->capacity++;
		dst->data = malloc(sizeof(ReduceOp)*dst->capacity);
	}
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		ReduceOp* tmp = malloc(sizeof(ReduceOp)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}

static inline char* remove_substring(char *str, const char *sub) {
	int len = strlen(sub);
	char *found = strstr(str, sub); // Find the first occurrence of the substring

	while (found) {
		memmove(found, found + len, strlen(found + len) + 1); // Shift characters to overwrite the substring
		found = strstr(found, sub); // Find the next occurrence of the substring
	}
	return str;
}


static inline int replacestr(char *line, const char *search, const char *replace)
{
   int count;
   char *sp; // start of pattern

   //printf("replacestr(%s, %s, %s)\n", line, search, replace);
   if ((sp = strstr(line, search)) == NULL) {
      return(0);
   }
   count = 1;
   int sLen = strlen(search);
   int rLen = strlen(replace);
   if (sLen > rLen) {
      // move from right to left
      char *src = sp + sLen;
      char *dst = sp + rLen;
      while((*dst = *src) != '\0') { dst++; src++; }
   } else if (sLen < rLen) {
      // move from left to right
      int tLen = strlen(sp) - sLen;
      char *stop = sp + rLen;
      char *src = sp + sLen + tLen;
      char *dst = sp + rLen + tLen;
      while(dst >= stop) { *dst = *src; dst--; src--; }
   }
   memcpy(sp, replace, rLen);

   count += replacestr(sp + rLen, search, replace);

   return(count);
}

static inline void 
replace_substring(char** str, const char* sub, const char* replace)
{
	replacestr(*str,sub,replace);
}
//TP: if the user has enabled GNU_SOURCE then we already have vasprintf and asprintf
#ifndef _GNU_SOURCE
static int
vasprintf(char **strp, const char *fmt, va_list ap)
{
    va_list ap1;
    int len;
    char *buffer;
    int res;

    va_copy(ap1, ap);
    len = vsnprintf(NULL, 0, fmt, ap1);

    if (len < 0)
        return len;

    va_end(ap1);
    buffer = malloc(len + 1);

    if (!buffer)
        return -1;

    res = vsnprintf(buffer, len + 1, fmt, ap);

    if (res < 0)
        free(buffer);
    else
        *strp = buffer;

    return res;
}

static int
asprintf(char **strp, const char *fmt, ...)
{
    int error;
    va_list ap;

    va_start(ap, fmt);
    error = vasprintf(strp, fmt, ap);
    va_end(ap);

    return error;
}
#endif
