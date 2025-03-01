/*
 * cn_cuda.cu - CUDA Implementation for Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 * GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
 *
 * This file provides CUDA-accelerated CryptoNight slow hash computation for the
 * Monero GPU Cracker, a tool to crack Monero wallet passwords using a hash from
 * monero2john.py. It is part of a port of the slow hash computation from John the
 * Ripper Bleeding Jumbo (https://github.com/openwall/john) to leverage GPU parallelism.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Original John the Ripper code is licensed under a relaxed BSD-style license
 * or public domain where applicable (see https://github.com/openwall/john).
 */

#include "cn_cuda.h"
#include "aes_cuda.h"
#include "keccak_cuda.h"
#include "slow_hash_extra.h"
#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#define MEMORY         (1 << 21) // 2 MiB
#define ITER           (1 << 20) // 1,048,576 iterations (524,288 in loop due to /2)
#define AES_BLOCK_SIZE 16
#define AES_KEY_SIZE   32
#define INIT_SIZE_BLK  8
#define INIT_SIZE_BYTE (INIT_SIZE_BLK * AES_BLOCK_SIZE) // 128 bytes
#define STATE_SIZE     200
#define HASH_SIZE      32
#define SHARED_SBOX_SIZE (256) // 16x16 bytes

__device__ void copy_block_cuda(uint64_t *dst, const uint64_t *src) {
    dst[0] = src[0];
    dst[1] = src[1];
}

__device__ void xor_blocks_cuda(union cn_slow_hash_block *a, const union cn_slow_hash_block *b) {
    a->dwords[0] ^= b->dwords[0];
    a->dwords[1] ^= b->dwords[1];
}

__device__ void swap_blocks_cuda(union cn_slow_hash_block *a, union cn_slow_hash_block *b) {
    union cn_slow_hash_block temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void mul128_ptx(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi) {
    asm volatile (
        "mul.lo.u64 %0, %2, %3;"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(*lo), "=l"(*hi)
        : "l"(a), "l"(b)
    );
}

__device__ void mul_cuda(const uint8_t *a, const uint8_t *b, uint8_t *res) {
    uint64_t a0 = ((uint64_t*)a)[0];
    uint64_t b0 = ((uint64_t*)b)[0];
    uint64_t hi, lo;
    mul128_ptx(a0, b0, &lo, &hi);
    ((uint64_t*)res)[0] = hi;
    ((uint64_t*)res)[1] = lo;
}

__device__ void sum_half_blocks_cuda(uint8_t *a, const uint8_t *b) {
    uint64_t a0 = ((uint64_t*)a)[0];
    uint64_t a1 = ((uint64_t*)a)[1];
    uint64_t b0 = ((uint64_t*)b)[0];
    uint64_t b1 = ((uint64_t*)b)[1];
    a0 += b0;
    a1 += b1;
    ((uint64_t*)a)[0] = a0;
    ((uint64_t*)a)[1] = a1;
}

__device__ void aes_encrypt_round_software(union cn_slow_hash_block *block, const union cn_slow_hash_block *round_key, uint8_t *shared_sbox) {
    oaes_sub_bytes_cuda(block->bytes, shared_sbox);
    oaes_shift_rows_cuda(block->bytes);
    oaes_mix_columns_cuda(block->bytes);
    oaes_add_round_key_cuda(round_key->words, block->bytes);
}

__global__ void cryptonight_kernel(
    const char **passwords,
    size_t *lengths,
    uint8_t *states,
    size_t num_passwords,
    uint8_t *scratchpads
) {
    extern __shared__ uint8_t shared_sbox[];
    
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    int total_entries = 256;

    for (int index = tid; index < total_entries; index += threads_per_block) {
        int row = index / 16;
        int col = index % 16;
        shared_sbox[index] = oaes_sub_byte_value[row][col];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + tid;
    if (idx >= num_passwords) return;

    const char *password = passwords[idx];
    size_t length = lengths[idx];
    cn_slow_hash_block *scratchpad = (cn_slow_hash_block*)(scratchpads + idx * MEMORY);

    uint8_t state[STATE_SIZE];
    union cn_slow_hash_block a, b, c, d;
    uint32_t expanded_key[60];

    keccak1600_cuda(password, length, state);

    uint8_t aes_key[AES_KEY_SIZE];
    uint8_t initial_state[STATE_SIZE];
    memcpy(aes_key, state, AES_KEY_SIZE);
    memcpy(initial_state, state, STATE_SIZE);
    oaes_key_expand_cuda_10(expanded_key, aes_key);

    union cn_slow_hash_block text[INIT_SIZE_BLK];
    memcpy(text, &state[64], INIT_SIZE_BYTE);
    for (size_t i = 0; i < MEMORY / INIT_SIZE_BYTE; i++) {
        for (size_t j = 0; j < INIT_SIZE_BLK; j++) {
            oaes_pseudo_encrypt_ecb_cuda_10rounds(expanded_key, text[j].bytes, shared_sbox);
            scratchpad[i * INIT_SIZE_BLK + j] = text[j];
        }
    }

    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        a.bytes[i] = state[i] ^ state[32 + i];
        b.bytes[i] = state[16 + i] ^ state[48 + i];
    }

    for (size_t i = 0; i < ITER / 2; i++) {
        uint64_t a_val = a.dwords[0];
        size_t j = (a_val >> 4) & ((MEMORY / AES_BLOCK_SIZE) - 1);
        c = scratchpad[j];
        aes_encrypt_round_software(&c, &a, shared_sbox);
        xor_blocks_cuda(&b, &c);
        swap_blocks_cuda(&b, &c);
        scratchpad[j] = c;

        swap_blocks_cuda(&a, &b);
        a_val = a.dwords[0];
        j = (a_val / AES_BLOCK_SIZE) & ((MEMORY / AES_BLOCK_SIZE) - 1);
        c = scratchpad[j];
        mul_cuda(a.bytes, c.bytes, d.bytes);
        sum_half_blocks_cuda(b.bytes, d.bytes);
        swap_blocks_cuda(&b, &c);
        xor_blocks_cuda(&b, &c);
        scratchpad[j] = c;
        swap_blocks_cuda(&a, &b);
    }

    uint8_t key2[AES_KEY_SIZE];
    memcpy(key2, initial_state + 32, AES_KEY_SIZE);
    oaes_key_expand_cuda_10(expanded_key, key2);

    memcpy(text, initial_state + 64, INIT_SIZE_BYTE);
    for (size_t i = 0; i < MEMORY / INIT_SIZE_BYTE; i++) {
        for (size_t j = 0; j < INIT_SIZE_BLK; j++) {
            union cn_slow_hash_block *text_block = &text[j];
            union cn_slow_hash_block *scratch_block = &scratchpad[i * INIT_SIZE_BLK + j];
            xor_blocks_cuda(text_block, scratch_block);
            oaes_pseudo_encrypt_ecb_cuda_10rounds(expanded_key, text_block->bytes, shared_sbox);
        }
    }

    memcpy(state + 64, text, INIT_SIZE_BYTE);
    keccakf_cuda((uint64_t*)state, 24);
    memcpy(&states[idx * STATE_SIZE], state, STATE_SIZE);
}

int compute_cn_slow_hash_cuda(const char **passwords, size_t num_passwords, unsigned char *hashes, int threads_per_block) {
    // Ensure threads_per_block is a multiple of 32 for warp efficiency
    if (threads_per_block % 32 != 0) {
        fprintf(stderr, "Error: threads_per_block must be a multiple of 32 for optimal performance.\n");
        return -1;
    }

    size_t *d_lengths = NULL;
    char *d_password_buffer = NULL;
    const char **d_passwords = NULL;
    uint8_t *d_states = NULL;
    uint8_t *d_scratchpads = NULL;
    const char **h_d_passwords = NULL;
    size_t *lengths = NULL;
    char *h_password_buffer = NULL;
    uint8_t *h_states = NULL;
    size_t total_password_size = 0;
    size_t offset = 0;
    dim3 grid((num_passwords + threads_per_block - 1) / threads_per_block);
    dim3 block(threads_per_block);
    size_t total_scratchpad_size = num_passwords * MEMORY;

    if (cudaDeviceSetLimit(cudaLimitStackSize, 4096) != cudaSuccess) {
        fprintf(stderr, "Failed to set stack size limit\n");
        goto cleanup;
    }

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // Removed printf to prevent repeated memory info output
    if (total_scratchpad_size > free_mem * 0.9) {
        fprintf(stderr, "Insufficient GPU memory for %zu passwords\n", num_passwords);
        goto cleanup;
    }

    lengths = (size_t*)malloc(num_passwords * sizeof(size_t));
    if (!lengths) {
        fprintf(stderr, "Host memory allocation failed for lengths\n");
        goto cleanup;
    }

    for (size_t i = 0; i < num_passwords; i++) {
        lengths[i] = strlen(passwords[i]);
        total_password_size += lengths[i] + 1; // Include null terminator
    }

    if (cudaHostAlloc((void**)&h_password_buffer, total_password_size, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory for passwords\n");
        goto cleanup;
    }

    if (cudaMalloc(&d_password_buffer, total_password_size) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for passwords\n");
        goto cleanup;
    }
    if (cudaMalloc(&d_passwords, num_passwords * sizeof(const char*)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_passwords\n");
        goto cleanup;
    }
    if (cudaMalloc(&d_lengths, num_passwords * sizeof(size_t)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_lengths\n");
        goto cleanup;
    }
    if (cudaMalloc(&d_states, num_passwords * STATE_SIZE) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_states\n");
        goto cleanup;
    }
    if (cudaMalloc(&d_scratchpads, total_scratchpad_size) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_scratchpads\n");
        goto cleanup;
    }

    h_d_passwords = (const char**)malloc(num_passwords * sizeof(const char*));
    if (!h_d_passwords) {
        fprintf(stderr, "Host memory allocation failed for h_d_passwords\n");
        goto cleanup;
    }

    offset = 0;
    for (size_t i = 0; i < num_passwords; i++) {
        strcpy(h_password_buffer + offset, passwords[i]);
        h_d_passwords[i] = d_password_buffer + offset;
        offset += lengths[i] + 1;
    }

    if (cudaMemcpy(d_password_buffer, h_password_buffer, total_password_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy password buffer to device\n");
        goto cleanup;
    }
    if (cudaMemcpy(d_passwords, h_d_passwords, num_passwords * sizeof(const char*), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy d_passwords\n");
        goto cleanup;
    }
    if (cudaMemcpy(d_lengths, lengths, num_passwords * sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy d_lengths\n");
        goto cleanup;
    }

    cryptonight_kernel<<<grid, block, SHARED_SBOX_SIZE>>>(d_passwords, d_lengths, d_states, num_passwords, d_scratchpads);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed\n");
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Kernel synchronization failed\n");
        goto cleanup;
    }

    if (cudaHostAlloc((void**)&h_states, num_passwords * STATE_SIZE, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory for states\n");
        goto cleanup;
    }
    if (cudaMemcpy(h_states, d_states, num_passwords * STATE_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy states back to host\n");
        goto cleanup;
    }

    for (size_t i = 0; i < num_passwords; i++) {
        uint8_t *state = h_states + i * STATE_SIZE;
        int index = state[0] & 3;
        extra_hashes[index](state, STATE_SIZE, (char*)(hashes + i * HASH_SIZE));
    }

    cudaFreeHost(h_password_buffer);
    free(h_d_passwords);
    free(lengths);
    cudaFreeHost(h_states);
    cudaFree(d_password_buffer);
    cudaFree(d_passwords);
    cudaFree(d_lengths);
    cudaFree(d_states);
    cudaFree(d_scratchpads);
    return 0;

cleanup:
    if (h_password_buffer) cudaFreeHost(h_password_buffer);
    if (h_d_passwords) free(h_d_passwords);
    if (lengths) free(lengths);
    if (h_states) cudaFreeHost(h_states);
    if (d_password_buffer) cudaFree(d_password_buffer);
    if (d_passwords) cudaFree(d_passwords);
    if (d_lengths) cudaFree(d_lengths);
    if (d_states) cudaFree(d_states);
    if (d_scratchpads) cudaFree(d_scratchpads);
    return -1;
}