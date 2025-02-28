/* memory.h */
#ifndef _JOHN_MEMORY_H
#define _JOHN_MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARCH_SIZE           8
#define MEM_ALIGN_NONE      1
#define MEM_ALIGN_WORD      ARCH_SIZE
#define MEM_ALIGN_CACHE     64
#define MEM_ALIGN_PAGE      0x1000
#define MEM_ALIGN_SIMD      16
#define MEM_ALLOC_SIZE      0x10000
#define MEM_ALLOC_MAX_WASTE 0xff

extern unsigned int mem_saving_level;

extern void *mem_alloc(size_t size);
extern void *mem_calloc(size_t nmemb, size_t size);
extern void *mem_realloc(void *old_ptr, size_t size);
extern void *mem_alloc_align(size_t size, size_t align);
extern void *mem_calloc_align(size_t count, size_t size, size_t align);
extern char *xstrdup(const char *str);
extern void mem_free(void *ptr);
extern void *mem_alloc_tiny(size_t size, size_t align);
extern void *mem_calloc_tiny(size_t size, size_t align);
extern void *mem_alloc_copy(const void *src, size_t size, size_t align);
extern char *str_alloc_copy(const char *src);
extern void cleanup_tiny_memory();

typedef struct {
    void *base, *aligned;
    size_t base_size, aligned_size;
} region_t;

extern void *alloc_region(region_t *region, size_t size);
extern void init_region(region_t *region);
extern int free_region(region_t *region);

#endif /* _JOHN_MEMORY_H */
