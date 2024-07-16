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
	//for(size_t i = 0; i < vec->size; ++i)
	//	free(vec->data[i]);
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
	//vec -> data = malloc(sizeof(char*)*vec ->capacity);
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
str_vec_contains(string_vec vec, const char* str)
{
	return str_vec_get_index(vec,str) >= 0;
}
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
static inline string_vec
str_vec_copy(string_vec vec)
{
        string_vec copy;
	init_str_vec(&copy);
        for(size_t i = 0; i<vec.size; ++i)
                push(&copy,vec.data[i]);
        return copy;
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
#define VEC_INITIALIZER {.size = 0,.capacity  = 0,.data = NULL}

static inline char* remove_substring(char *str, const char *sub) {
	int len = strlen(sub);
	char *found = strstr(str, sub); // Find the first occurrence of the substring

	while (found) {
		memmove(found, found + len, strlen(found + len) + 1); // Shift characters to overwrite the substring
		found = strstr(found, sub); // Find the next occurrence of the substring
	}
	return str;
}
