# Astaroth Communications Module

## Error handling

```c
int dostuff(void* input)
{
    ERRCHK(input != NULL);

    if (input == NULL)
        return 1;
    else
        return 0;


    // ERRCHK actually isn't an error handling module: it is an error printing module
    if (check_ok(input))
        return 0;
    else
        return 1;
}
```
