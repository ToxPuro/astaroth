utility Kernel AC_BUILTIN_RESET()
{
	for field in 0:NUM_VTXBUF_HANDLES{
		write(Field(field), 0.0)
	}
}
utility Kernel AC_PERIODIC()
{
}
