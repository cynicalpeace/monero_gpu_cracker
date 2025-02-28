// slow_hash_extra.h
#ifndef SLOW_HASH_EXTRA_H
#define SLOW_HASH_EXTRA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*
 * extra_hashes is an array of four function pointers.
 * Each function takes (const void *data, size_t length, char *hash)
 * and computes one of the extra hashes (for example, Blake, Groestl, JH, Skein).
 *
 * In the CPU code (slow_hash_plug.c), remove the static qualifier so that this symbol is exported.
 */
extern void (*extra_hashes[4])(const void *, size_t, char *);

#ifdef __cplusplus
}
#endif

#endif // SLOW_HASH_EXTRA_H
