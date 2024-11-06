utility Kernel AC_NULL_KERNEL(){}
utility Kernel AC_BUILTIN_RESET()
{
	for field in 0:NUM_VTXBUF_HANDLES{
		write(Field(field), 0.0)
	}
}
utility Kernel BOUNDCOND_PERIODIC()
{
}
