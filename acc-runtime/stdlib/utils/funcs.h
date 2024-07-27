array_val(field_array, size)
{
	real res[size]
	for i in 0:size
	{
		res[i] = value(Field(field_array[i]))
	}
	return res
}
