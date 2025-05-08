#include "stdarg.h"

#include "string_vec.h"
typedef enum ReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} ReduceOp;

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
	vec -> data = (int*)malloc(sizeof(int)*vec ->capacity);
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
	vec -> data = (ReduceOp*)malloc(sizeof(ReduceOp)*vec ->capacity);
}


static __attribute__((unused)) char* 
strdupnullok(const char* src)
{
	if(src == NULL) return NULL;
	return strdup(src);
}


static inline int
push_int(int_vec* dst, int src)
{
	if(dst->capacity == 0)
	{
		dst->capacity++;
		dst->data = (int*)malloc(sizeof(int)*dst->capacity);
	}
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		int* tmp = (int*)malloc(sizeof(int)*dst->capacity);
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
		dst->data = (ReduceOp*)malloc(sizeof(ReduceOp)*dst->capacity);
	}
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		ReduceOp* tmp = (ReduceOp*)malloc(sizeof(ReduceOp)*dst->capacity);
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

static __attribute__((unused)) int
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
