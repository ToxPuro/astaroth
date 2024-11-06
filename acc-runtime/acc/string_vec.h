#include "stdarg.h"
#define VEC_INITIALIZER {.size = 0,.capacity  = 0,.data = NULL}
typedef struct string_vec
{
	const char** data;
	size_t size;
	int capacity;

} string_vec;
static inline void
init_str_vec(string_vec* vec)
{
	vec -> size = 0;
	vec -> capacity = 1;
	vec -> data = (const char**)malloc(sizeof(char*)*vec ->capacity);
}
static inline void
free_str_vec(string_vec* vec)
{
	free(vec->data);
	vec -> size = 0;
	vec -> capacity = 0;
	vec -> data = NULL;
}
static inline int
str_vec_get_index(string_vec vec, const char* str)
{
	for(size_t i = 0; i <  vec.size; ++i)
		if(vec.data[i] == str) return i;
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

static inline int
push(string_vec* dst, const char* src)
{
	if(dst->capacity == 0)
	{
		dst->capacity++;
		dst->data = (const char**)malloc(sizeof(char*)*dst->capacity);
	}
	/**
	if(src != intern(src))
	{
		printf("WRONG: %s\n",src);
		void* NULL_PTR= NULL;
		printf("HMM :%s\n",((char*) NULL_PTR)[10]);
	}
	**/
	dst->data[dst->size] = src;
	++(dst->size);
	if(dst->size == (size_t)dst->capacity)
	{
		dst->capacity = dst->capacity*2;
		const char** tmp = (const char**)malloc(sizeof(char*)*dst->capacity);
		for(size_t i = 0; i < dst->size; ++i)
			tmp[i] = dst->data[i];
		free(dst->data);
		dst->data = tmp;
	}
	return dst->size-1;
}
static string_vec
get_csv_entry(const char* line)
{
      string_vec res;
      res.data = NULL;
      res.size = 0;
      res.capacity = 0;
      char* line_copy = strdup(line);
      char* token;
      token = strtok(line_copy,",");
      while(token != NULL)
      {
	      push(&res,strdup(token));
              token = strtok(NULL,",");
      }
      free(line_copy);
      return res;
}
static int
get_csv_entries(string_vec* dst, FILE* file)
{
	if(file == NULL) return 0;
	char line[10000];
	int counter = 0;
	while(fgets(line,sizeof(line),file) != NULL)
	{
		dst[counter] = get_csv_entry(line);
		++counter;
	}
	return counter;
}
static inline
string_vec
str_vec_copy(string_vec vec)
{
        string_vec res;
        res.data = NULL;
        res.size = 0;
        res.capacity = 0;
	for(size_t i = 0; i < vec.size; ++i)
		push(&res,vec.data[i]);
	return res;
}
static inline
void
str_vec_remove(string_vec* vec, const char* elem_to_remove)
{
        string_vec res;
        res.data = NULL;
        res.size = 0;
        res.capacity = 0;
	for(size_t i = 0; i < vec->size; ++i)
	{
		if(strcmp(vec->data[i],elem_to_remove))
			push(&res,vec->data[i]);
	}
	free_str_vec(vec);
	*vec = res;
}
static inline bool
str_vec_eq(const string_vec a, const string_vec b)
{
	if(a.size != b.size) return false;
	for(size_t i = 0; i < a.size; ++i)
		if(a.data[i] != b.data[i]) return false;
	return true;
}
static inline bool
str_vec_in(const string_vec* elems, const int n_elems, const string_vec b)
{
	for(int i = 0; i < n_elems; ++i)
		if(str_vec_eq(elems[i],b)) return true;
	return false;
}
