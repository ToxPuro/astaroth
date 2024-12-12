#include <string>



template <typename P>
auto
get_default_value()
{
#include"get_default_value.h"
}

template <typename V>
std::string
get_datatype(){return{};}

template <>
std::string
get_datatype<int>()     {return "int";};

template <>
std::string
get_datatype<bool>()    {return "bool";};


template <>
std::string
get_datatype<AcReal>()  {return "real";};

template <>
std::string
get_datatype<long>()  {return "long";};

template <>
std::string
get_datatype<long long>()  {return "long long";};


std::string
to_str(const int value)
{
	return std::to_string(value);
}

std::string
to_str(const long value)
{
	return std::to_string(value);
}

std::string
to_str(const long long value)
{
	return std::to_string(value);
}

std::string
to_str(const AcReal value)
{
	char* tmp;
	asprintf(&tmp,"%.17g",value);
	std::string res = tmp;
	free(tmp);
	return res;
}
#if AC_DOUBLE_PRECISION
std::string
to_str(const float value)
{
	return to_str((AcReal)value);
}
#endif

std::string
to_str(const bool value)
{
	char* tmp;
	asprintf(&tmp,"%s",(value) ? "true" : "false");
	std::string res = tmp;
	free(tmp);
	return res;
}



template <typename V>
std::string
to_str(const V value, const char* name, const char* prefix, const bool output_datatype)
{
	const std::string val_str = to_str(value);
	const std::string name_str = name;
	const std::string prefix_str = !strcmp(prefix,"") ? prefix : std::string(prefix) + " ";
	const auto datatype_str = output_datatype ? get_datatype<V>() + " " : "";
	return prefix_str +  datatype_str  +  name_str + " = " + val_str + "\n";
}

#include "to_str_funcs.h"

template <typename V>
std::string
get_value_type(V value)
{
	(void)value;
	return get_datatype<V>();
}
template <typename P>
struct load_comp_arrays
{
	void operator()(const AcCompInfo info, FILE* fp, const char* prefix, const bool output_datatype)
	{
		const auto default_value = get_default_value<P>();
		const std::string type = output_datatype ? get_value_type(default_value) : "";
		for(P array : get_params<P>())
		{
			const int n_dims = get_array_n_dims(array);
			const char* name = get_array_name(array);
			const bool is_loaded = info.is_loaded[array];
			auto* loaded_val = info.config[array];
			const auto dims = get_array_dim_sizes(array,{});
			if(n_dims == 1)
			{
				fprintf(fp,"%s %s %s = [",prefix,type.c_str(),name);
				for(size_t j = 0; j < dims[0]; ++j)
				{
					auto val = is_loaded ? loaded_val[j] : default_value;
					std::string val_string = to_str(val);
					fprintf(fp,"%s",val_string.c_str());
					if(j < dims[0]-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"%s","]\n");
			}
			else if(n_dims == 2)
			{
				fprintf(fp,"%s %s %s = [",prefix,type.c_str(), name);
				for(size_t y = 0; y < dims[1]; ++y)
				{
					fprintf(fp,"%s","[");
					for(size_t x = 0; x < dims[0]; ++x)
					{
						auto val = is_loaded ? loaded_val[x + y*dims[0]] : default_value;
						std::string val_string = to_str(val);
						fprintf(fp,"%s",val_string.c_str());
						if(x < dims[0]-1) fprintf(fp,"%s",",");
					}
					fprintf(fp,"%s","]");
					if(y < dims[1]-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"]\n");
			}
		}
	}
};

template <typename P>
struct load_comp_scalars
{
	void operator()(const AcCompInfo info, FILE* fp, const char* prefix, const bool output_datatype)
	{
		for(P var : get_params<P>())
		{
			auto val =  info.is_loaded[var] ? info.config[var] : get_default_value<P>();
			std::string res = to_str(val,get_param_name(var),prefix,output_datatype);
			fprintf(fp,"%s",res.c_str());
		}
	}
};

template <typename P>
struct load_arrays
{
	void operator()(const AcMeshInfoArrays info, FILE* fp, const char* prefix, const bool output_datatype)
	{
		const auto default_value = get_default_value<P>();
		const std::string type = output_datatype ? get_value_type(default_value) : "";
		for(P array : get_params<P>())
		{
			const int n_dims = get_array_n_dims(array);
			const char* name = get_array_name(array);
			auto* loaded_val = info[array];
			if(loaded_val == NULL) continue;
			const auto dims = get_array_dim_sizes(array,{});
			if(n_dims == 1)
			{
				fprintf(fp,"%s %s %s = [",prefix,type.c_str(),name);
				for(size_t j = 0; j < dims[0]; ++j)
				{
					auto val = loaded_val[j];
					std::string val_string = to_str(val);
					fprintf(fp,"%s",val_string.c_str());
					if(j < dims[0]-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"%s","]\n");
			}
			else if(n_dims == 2)
			{
				fprintf(fp,"%s %s %s = [",prefix,type.c_str(), name);
				for(size_t y = 0; y < dims[1]; ++y)
				{
					fprintf(fp,"%s","[");
					for(size_t x = 0; x < dims[0]; ++x)
					{
						auto val = loaded_val[x + y*dims[0]];
						std::string val_string = to_str(val);
						fprintf(fp,"%s",val_string.c_str());
						if(x < dims[0]-1) fprintf(fp,"%s",",");
					}
					fprintf(fp,"%s","]");
					if(y < dims[1]-1) fprintf(fp,"%s",",");
				}
				fprintf(fp,"]\n");
			}
		}
	}
};


template <typename P>
struct load_scalars
{
	void operator()(const AcMeshInfoScalars info, FILE* fp, const char* prefix, const bool output_datatype)
	{
		for(P var : get_params<P>())
		{
			auto val =  info[var];
			std::string res = to_str(val,get_param_name(var),prefix,output_datatype);
			fprintf(fp,"%s",res.c_str());
		}
	}
};
