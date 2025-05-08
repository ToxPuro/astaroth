array_val(field_array)
{
	real res[size(field_array)]
	for i in 0:size(field_array)
		res[i] = value(Field(field_array[i]))
	return res
}

not_implemented(message)
{
    print("NOT IMPLEMENTED: %s\n",message)
}

tini_sqrt_div(real numerator, real a, real b)
{
  return
     (abs(a) <= AC_REAL_MIN*5.0 || abs(b) <= AC_REAL_MIN*5.0)
     ? 0.0
     : numerator/(sqrt(a*b))
}

tini_sqrt_div_separate(real numerator, real a, real b)
{
  return
     (abs(a) <= AC_REAL_MIN*5.0|| abs(b) <= AC_REAL_MIN*5.0)
     ? 0.0
     : numerator/(sqrt(a)*sqrt(b))
}
