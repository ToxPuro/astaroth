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

static inline char* get_replaced_substring(const char *str, const char *sub, const char *replace) {
    const char *pos; 
    char *temp;
    int len_sub = strlen(sub);
    int len_replace = strlen(replace);
    int len_str = strlen(str);

    // Count occurrences of the substring
    int count = 0;
    pos = str;
    while ((pos = strstr(pos, sub)) != NULL) {
        count++;
        pos += len_sub;
    }

    // Allocate memory for the new string
    temp = (char*)malloc(len_str + (len_replace - len_sub) * count + 1);
    if (!temp) {
        return NULL; // Memory allocation failed
    }

    char *current_pos = temp;
    pos = str;
    while ((pos = strstr(pos, sub)) != NULL) {
        // Copy the part before the substring
        int len_before_sub = pos - str;
        memcpy(current_pos, str, len_before_sub);
        current_pos += len_before_sub;

        // Copy the replacement substring
        memcpy(current_pos, replace, len_replace);
        current_pos += len_replace;

        // Move past the substring in the original string
        str = pos + len_sub;
        pos = str;
    }
    // Copy the remaining part of the original string
    strcpy(current_pos, str);

    return temp;
}
static inline void 
replace_substring(char** str, const char* sub, const char* replace)
{
	char* new_str  = get_replaced_substring(*str,sub,replace);
	free(*str);
	*str = new_str;
}
