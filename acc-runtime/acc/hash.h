#include <hashtable.h>
static inline const char*
intern(const char* buffer)
{
  if(!buffer)
	  return NULL;
  const char* str_from_map = (const char*)hashmap_get(&string_intern_hashmap, buffer, strlen(buffer));
  if(str_from_map)
	  return str_from_map;
  const char* dup = strdup(buffer);
  hashmap_put(&string_intern_hashmap,dup, strlen(dup),(void*)dup);
  return dup;
}
