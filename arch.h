/* arch.h */
#ifndef _JOHN_ARCH_H
#define _JOHN_ARCH_H

#define ARCH_WORD           long
#define ARCH_SIZE           8
#define ARCH_BITS           64
#define ARCH_LITTLE_ENDIAN  1
#define ARCH_INT_GT_32      1
#define HAVE_SSE2           1
#define HAVE_OPENMP         1
#define ARCH_INDEX(x)       ((unsigned int)(unsigned char)(x))

#endif /* _JOHN_ARCH_H */
