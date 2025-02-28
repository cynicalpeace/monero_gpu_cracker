/* memory.c */
#include "memory.h"

void *mem_alloc(size_t size) {
    void *res = malloc(size);
    if (!res && size) {
        fprintf(stderr, "mem_alloc: out of memory\n");
        exit(1);
    }
    return res;
}

void *mem_calloc(size_t nmemb, size_t size) {
    void *res = calloc(nmemb, size);
    if (!res && nmemb && size) {
        fprintf(stderr, "mem_calloc: out of memory\n");
        exit(1);
    }
    return res;
}

void *mem_alloc_tiny(size_t size, size_t align) {
    return mem_alloc(size);  
}

void mem_free(void *ptr) {
    free(ptr);
}

void cleanup_tiny_memory() {
    /* No-op for minimal version */
}

void *alloc_region(region_t *region, size_t size) {
    void *base = mem_alloc(size);
    region->base = base;
    region->aligned = base;
    region->base_size = size;
    region->aligned_size = size;
    return base;
}

void init_region(region_t *region) {
    region->base = region->aligned = NULL;
    region->base_size = region->aligned_size = 0;
}

int free_region(region_t *region) {
    if (region->base) {
        mem_free(region->base);
    }
    init_region(region);
    return 0;
}
