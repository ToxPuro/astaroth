array_val(field_array)
{
	real res[size(field_array)]
	for i in 0:size(field_array)
		res[i] = value(Field(field_array[i]))
	return res
}
