/* misc.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void real_error(const char *msg, int err)
{
    fprintf(stderr, "%s: %s\n", msg, err ? strerror(err) : "Unknown error");
    exit(1);
}

void real_pexit(const char *msg, ...)
{
    fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(1);
}
