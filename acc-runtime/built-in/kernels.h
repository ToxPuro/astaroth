Kernel AC_BUILTIN_RESET()
{
	for field in 0:NUM_FIELDS {
		write(Field(field), 0.0)
	}
}
