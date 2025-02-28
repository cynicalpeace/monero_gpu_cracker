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

// AES S-box for shared memory initialization
static __device__ uint8_t oaes_sub_byte_value[16][16] = {
    {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
    {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
    {0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15},
    {0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75},
    {0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84},
    {0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf},
    {0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8},
    {0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2},
    {0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73},
    {0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb},
    {0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79},
    {0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08},
    {0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a},
    {0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e},
    {0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf},
    {0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16}
};

// GF(2^8) multiplication tables for MixColumns (precomputed as in original)
__device__ uint8_t gf_mul_2[256] = {
    0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e,0x10,0x12,0x14,0x16,0x18,0x1a,0x1c,0x1e,
    0x20,0x22,0x24,0x26,0x28,0x2a,0x2c,0x2e,0x30,0x32,0x34,0x36,0x38,0x3a,0x3c,0x3e,
    0x40,0x42,0x44,0x46,0x48,0x4a,0x4c,0x4e,0x50,0x52,0x54,0x56,0x58,0x5a,0x5c,0x5e,
    0x60,0x62,0x64,0x66,0x68,0x6a,0x6c,0x6e,0x70,0x72,0x74,0x76,0x78,0x7a,0x7c,0x7e,
    0x80,0x82,0x84,0x86,0x88,0x8a,0x8c,0x8e,0x90,0x92,0x94,0x96,0x98,0x9a,0x9c,0x9e,
    0xa0,0xa2,0xa4,0xa6,0xa8,0xaa,0xac,0xae,0xb0,0xb2,0xb4,0xb6,0xb8,0xba,0xbc,0xbe,
    0xc0,0xc2,0xc4,0xc6,0xc8,0xca,0xcc,0xce,0xd0,0xd2,0xd4,0xd6,0xd8,0xda,0xdc,0xde,
    0xe0,0xe2,0xe4,0xe6,0xe8,0xea,0xec,0xee,0xf0,0xf2,0xf4,0xf6,0xf8,0xfa,0xfc,0xfe,
    0x1b,0x19,0x1f,0x1d,0x13,0x11,0x17,0x15,0x0b,0x09,0x0f,0x0d,0x03,0x01,0x07,0x05,
    0x3b,0x39,0x3f,0x3d,0x33,0x31,0x37,0x35,0x2b,0x29,0x2f,0x2d,0x23,0x21,0x27,0x25,
    0x5b,0x59,0x5f,0x5d,0x53,0x51,0x57,0x55,0x4b,0x49,0x4f,0x4d,0x43,0x41,0x47,0x45,
    0x7b,0x79,0x7f,0x7d,0x73,0x71,0x77,0x75,0x6b,0x69,0x6f,0x6d,0x63,0x61,0x67,0x65,
    0x9b,0x99,0x9f,0x9d,0x93,0x91,0x97,0x95,0x8b,0x89,0x8f,0x8d,0x83,0x81,0x87,0x85,
    0xbb,0xb9,0xbf,0xbd,0xb3,0xb1,0xb7,0xb5,0xab,0xa9,0xaf,0xad,0xa3,0xa1,0xa7,0xa5,
    0xdb,0xd9,0xdf,0xdd,0xd3,0xd1,0xd7,0xd5,0xcb,0xc9,0xcf,0xcd,0xc3,0xc1,0xc7,0xc5,
    0xfb,0xf9,0xff,0xfd,0xf3,0xf1,0xf7,0xf5,0xeb,0xe9,0xef,0xed,0xe3,0xe1,0xe7,0xe5
};

__device__ uint8_t gf_mul_3[256] = {
    0x00,0x03,0x06,0x05,0x0c,0x0f,0x0a,0x09,0x18,0x1b,0x1e,0x1d,0x14,0x17,0x12,0x11,
    0x30,0x33,0x36,0x35,0x3c,0x3f,0x3a,0x39,0x28,0x2b,0x2e,0x2d,0x24,0x27,0x22,0x21,
    0x60,0x63,0x66,0x65,0x6c,0x6f,0x6a,0x69,0x78,0x7b,0x7e,0x7d,0x74,0x77,0x72,0x71,
    0x50,0x53,0x56,0x55,0x5c,0x5f,0x5a,0x59,0x48,0x4b,0x4e,0x4d,0x44,0x47,0x42,0x41,
    0xc0,0xc3,0xc6,0xc5,0xcc,0xcf,0xca,0xc9,0xd8,0xdb,0xde,0xdd,0xd4,0xd7,0xd2,0xd1,
    0xf0,0xf3,0xf6,0xf5,0xfc,0xff,0xfa,0xf9,0xe8,0xeb,0xee,0xed,0xe4,0xe7,0xe2,0xe1,
    0xa0,0xa3,0xa6,0xa5,0xac,0xaf,0xaa,0xa9,0xb8,0xbb,0xbe,0xbd,0xb4,0xb7,0xb2,0xb1,
    0x90,0x93,0x96,0x95,0x9c,0x9f,0x9a,0x99,0x88,0x8b,0x8e,0x8d,0x84,0x87,0x82,0x81,
    0x9b,0x98,0x9d,0x9e,0x97,0x94,0x91,0x92,0x83,0x80,0x85,0x86,0x8f,0x8c,0x89,0x8a,
    0xab,0xa8,0xad,0xae,0xa7,0xa4,0xa1,0xa2,0xb3,0xb0,0xb5,0xb6,0xbf,0xbc,0xb9,0xba,
    0xfb,0xf8,0xfd,0xfe,0xf7,0xf4,0xf1,0xf2,0xe3,0xe0,0xe5,0xe6,0xef,0xec,0xe9,0xea,
    0xcb,0xc8,0xcd,0xce,0xc7,0xc4,0xc1,0xc2,0xd3,0xd0,0xd5,0xd6,0xdf,0xdc,0xd9,0xda,
    0x5b,0x58,0x5d,0x5e,0x57,0x54,0x51,0x52,0x43,0x40,0x45,0x46,0x4f,0x4c,0x49,0x4a,
    0x6b,0x68,0x6d,0x6e,0x67,0x64,0x61,0x62,0x73,0x70,0x75,0x76,0x7f,0x7c,0x79,0x7a,
    0x3b,0x38,0x3d,0x3e,0x37,0x34,0x31,0x32,0x23,0x20,0x25,0x26,0x2f,0x2c,0x29,0x2a,
    0x0b,0x08,0x0d,0x0e,0x07,0x04,0x01,0x02,0x13,0x10,0x15,0x16,0x1f,0x1c,0x19,0x1a
};

// Helper device functions for 64-bit operations
__device__ void copy_block_cuda(uint64_t *dst, const uint64_t *src) {
    dst[0] = src[0];
    dst[1] = src[1];
}

__device__ void xor_blocks_cuda(uint64_t *a, const uint64_t *b) {
    a[0] ^= b[0];
    a[1] ^= b[1];
}

__device__ void swap_blocks_cuda(uint64_t *a, uint64_t *b) {
    uint64_t temp[2];
    temp[0] = a[0];
    temp[1] = a[1];
    a[0] = b[0];
    a[1] = b[1];
    b[0] = temp[0];
    b[1] = temp[1];
}

__device__ uint64_t mul128_device(uint64_t multiplier, uint64_t multiplicand, uint64_t *product_hi) {
    uint64_t a = multiplier >> 32;
    uint64_t b = multiplier & 0xFFFFFFFF;
    uint64_t c = multiplicand >> 32;
    uint64_t d = multiplicand & 0xFFFFFFFF;

    uint64_t ac = a * c;
    uint64_t bc = b * c;
    uint64_t ad = a * d;
    uint64_t bd = b * d;

    uint64_t adbc = ad + bc;
    uint64_t adbc_carry = adbc < ad ? 1 : 0;

    uint64_t product_lo = bd + (adbc << 32);
    uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
    *product_hi = ac + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;
    return product_lo;
}

__device__ void mul_cuda(const uint8_t *a, const uint8_t *b, uint8_t *res) {
    uint64_t a0 = ((uint64_t*)a)[0];
    uint64_t b0 = ((uint64_t*)b)[0];
    uint64_t hi, lo;
    lo = mul128_device(a0, b0, &hi);
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

// AES round function using shared memory
__device__ void cn_aes_round_cuda(const uint8_t *key, uint8_t *block) {
    extern __shared__ uint8_t s_sbox[];
    uint8_t (*s_sbox_2d)[16] = (uint8_t(*)[16])s_sbox;
    uint8_t *s_gf_mul_2 = &s_sbox[256];
    uint8_t *s_gf_mul_3 = &s_sbox[512];

    uint8_t temp[AES_BLOCK_SIZE];
    // SubBytes
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        uint8_t x = block[i] & 0x0f, y = (block[i] & 0xf0) >> 4;
        temp[i] = s_sbox_2d[y][x];
    }
    // ShiftRows
    block[0x00] = temp[0x00]; block[0x01] = temp[0x05]; block[0x02] = temp[0x0a]; block[0x03] = temp[0x0f];
    block[0x04] = temp[0x04]; block[0x05] = temp[0x09]; block[0x06] = temp[0x0e]; block[0x07] = temp[0x03];
    block[0x08] = temp[0x08]; block[0x09] = temp[0x0d]; block[0x0a] = temp[0x02]; block[0x0b] = temp[0x07];
    block[0x0c] = temp[0x0c]; block[0x0d] = temp[0x01]; block[0x0e] = temp[0x06]; block[0x0f] = temp[0x0b];
    // MixColumns
    for (int i = 0; i < 4; i++) {
        uint8_t a0 = block[i*4], a1 = block[i*4+1], a2 = block[i*4+2], a3 = block[i*4+3];
        temp[i*4]   = s_gf_mul_2[a0] ^ s_gf_mul_3[a1] ^ a2 ^ a3;
        temp[i*4+1] = a0 ^ s_gf_mul_2[a1] ^ s_gf_mul_3[a2] ^ a3;
        temp[i*4+2] = a0 ^ a1 ^ s_gf_mul_2[a2] ^ s_gf_mul_3[a3];
        temp[i*4+3] = s_gf_mul_3[a0] ^ a1 ^ a2 ^ s_gf_mul_2[a3];
    }
    memcpy(block, temp, AES_BLOCK_SIZE);
    // AddRoundKey
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        block[i] ^= key[i];
    }
}

__global__ void cryptonight_kernel(
    const char **passwords,
    size_t *lengths,
    uint8_t *states,
    size_t num_passwords,
    uint8_t *scratchpads
) {
    size_t idx = blockIdx.x;
    if (idx >= num_passwords) return;

    const char *password = passwords[idx];
    size_t length = lengths[idx];
    uint8_t *scratchpad = scratchpads + idx * MEMORY;

    // Shared memory for lookup tables (768 bytes total)
    extern __shared__ uint8_t s_sbox[];
    uint8_t (*s_sbox_2d)[16] = (uint8_t(*)[16])s_sbox;
    uint8_t *s_gf_mul_2 = &s_sbox[256];
    uint8_t *s_gf_mul_3 = &s_sbox[512];

    // Initialize S-box rows (only 16 rows needed)
    if (threadIdx.x < 16) {
        for (int j = 0; j < 16; j++) {
            s_sbox_2d[threadIdx.x][j] = oaes_sub_byte_value[threadIdx.x][j];
        }
    }

    // Initialize GF multiplication tables
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_gf_mul_2[i] = gf_mul_2[i];
        s_gf_mul_3[i] = gf_mul_3[i];
    }
    __syncthreads();

    __shared__ uint8_t state[STATE_SIZE];
    __shared__ uint8_t b[AES_BLOCK_SIZE];
    __shared__ uint8_t c[AES_BLOCK_SIZE];
    __shared__ uint8_t d[AES_BLOCK_SIZE];
    __shared__ uint32_t expanded_key[60];
    uint8_t a[AES_BLOCK_SIZE];

    if (threadIdx.x == 0) {
        keccak1600_cuda(password, length, state);
    }
    __syncthreads();

    __shared__ uint8_t aes_key[AES_KEY_SIZE];
    __shared__ uint8_t initial_state[STATE_SIZE];
    if (threadIdx.x == 0) {
        memcpy(aes_key, state, AES_KEY_SIZE);
        memcpy(initial_state, state, STATE_SIZE);
        oaes_key_expand_cuda_10(expanded_key, aes_key);
    }
    __syncthreads();

    __shared__ uint8_t text[INIT_SIZE_BYTE];
    if (threadIdx.x == 0) {
        memcpy(text, &state[64], INIT_SIZE_BYTE);
        for (size_t i = 0; i < MEMORY / INIT_SIZE_BYTE; i++) {
            for (size_t j = 0; j < INIT_SIZE_BLK; j++) {
                oaes_pseudo_encrypt_ecb_cuda_10rounds(expanded_key, &text[j * AES_BLOCK_SIZE]);
            }
            memcpy(&scratchpad[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < AES_BLOCK_SIZE; i++) {
            a[i] = state[i] ^ state[32 + i];
            b[i] = state[16 + i] ^ state[48 + i];
        }

        for (size_t i = 0; i < ITER / 2; i++) {
            uint64_t a_val = ((uint64_t*)a)[0];
            size_t j = (a_val / AES_BLOCK_SIZE) & ((MEMORY / AES_BLOCK_SIZE) - 1);
            copy_block_cuda((uint64_t*)c, (uint64_t*)&scratchpad[j * AES_BLOCK_SIZE]);
            cn_aes_round_cuda(a, c);
            xor_blocks_cuda((uint64_t*)b, (uint64_t*)c);
            swap_blocks_cuda((uint64_t*)b, (uint64_t*)c);
            copy_block_cuda((uint64_t*)&scratchpad[j * AES_BLOCK_SIZE], (uint64_t*)c);

            swap_blocks_cuda((uint64_t*)a, (uint64_t*)b);
            a_val = ((uint64_t*)a)[0];
            j = (a_val / AES_BLOCK_SIZE) & ((MEMORY / AES_BLOCK_SIZE) - 1);
            copy_block_cuda((uint64_t*)c, (uint64_t*)&scratchpad[j * AES_BLOCK_SIZE]);
            mul_cuda(a, c, d);
            sum_half_blocks_cuda(b, d);
            swap_blocks_cuda((uint64_t*)b, (uint64_t*)c);
            xor_blocks_cuda((uint64_t*)b, (uint64_t*)c);
            copy_block_cuda((uint64_t*)&scratchpad[j * AES_BLOCK_SIZE], (uint64_t*)c);
            swap_blocks_cuda((uint64_t*)a, (uint64_t*)b);
        }

        uint8_t key2[AES_KEY_SIZE];
        memcpy(key2, initial_state + 32, AES_KEY_SIZE);
        oaes_key_expand_cuda_10(expanded_key, key2);

        memcpy(text, initial_state + 64, INIT_SIZE_BYTE);
        for (size_t i = 0; i < MEMORY / INIT_SIZE_BYTE; i++) {
            for (size_t j = 0; j < INIT_SIZE_BLK; j++) {
                xor_blocks_cuda((uint64_t*)&text[j * AES_BLOCK_SIZE], (uint64_t*)&scratchpad[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
                oaes_pseudo_encrypt_ecb_cuda_10rounds(expanded_key, &text[j * AES_BLOCK_SIZE]);
            }
        }

        memcpy(state + 64, text, INIT_SIZE_BYTE);
        keccakf_cuda((uint64_t*)state, 24);
        memcpy(&states[idx * STATE_SIZE], state, STATE_SIZE);
    }
}

int compute_cn_slow_hash_cuda(const char **passwords, size_t num_passwords, unsigned char *hashes, int threads_per_block) {
    size_t *d_lengths = NULL;
    const char **d_passwords = NULL;
    uint8_t *d_states = NULL;
    uint8_t *d_scratchpads = NULL;
    const char **h_d_passwords = NULL;
    size_t *lengths = NULL;
    uint8_t *h_states = NULL;
    dim3 grid(1);  // Default initialization to satisfy C++ rules
    dim3 block(1); // Default initialization to satisfy C++ rules

    // Set stack size
    if (cudaDeviceSetLimit(cudaLimitStackSize, 4096) != cudaSuccess) {
        goto cleanup;
    }

    // Allocate GPU memory
    if (cudaMalloc(&d_lengths, num_passwords * sizeof(size_t)) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMalloc(&d_passwords, num_passwords * sizeof(const char *)) != cudaSuccess) {
        cudaFree(d_lengths);
        goto cleanup;
    }
    if (cudaMalloc(&d_states, num_passwords * STATE_SIZE) != cudaSuccess) {
        cudaFree(d_lengths);
        cudaFree(d_passwords);
        goto cleanup;
    }
    if (cudaMalloc(&d_scratchpads, num_passwords * MEMORY) != cudaSuccess) {
        cudaFree(d_lengths);
        cudaFree(d_passwords);
        cudaFree(d_states);
        goto cleanup;
    }

    // Allocate host memory
    h_d_passwords = (const char**)malloc(num_passwords * sizeof(const char *));
    lengths = (size_t*)malloc(num_passwords * sizeof(size_t));
    if (!h_d_passwords || !lengths) {
        fprintf(stderr, "Host memory allocation failed\n");
        goto cleanup;
    }

    // Copy passwords to GPU
    for (size_t i = 0; i < num_passwords; i++) {
        lengths[i] = strlen(passwords[i]);
        const char *temp;
        if (cudaMalloc((void**)&temp, lengths[i] + 1) != cudaSuccess) {
            for (size_t j = 0; j < i; j++) {
                cudaFree((void*)h_d_passwords[j]);
            }
            free(h_d_passwords);
            free(lengths);
            goto cleanup;
        }
        if (cudaMemcpy((void*)temp, passwords[i], lengths[i] + 1, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree((void*)temp);
            for (size_t j = 0; j < i; j++) {
                cudaFree((void*)h_d_passwords[j]);
            }
            free(h_d_passwords);
            free(lengths);
            goto cleanup;
        }
        h_d_passwords[i] = temp;
    }
    if (cudaMemcpy(d_passwords, h_d_passwords, num_passwords * sizeof(const char *), cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaMemcpy(d_lengths, lengths, num_passwords * sizeof(size_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        goto cleanup;
    }

    // Initialize grid and block before kernel launch
    grid = dim3(num_passwords);
    block = dim3(threads_per_block);

    // Launch kernel
    cryptonight_kernel<<<grid, block, 768>>>(d_passwords, d_lengths, d_states, num_passwords, d_scratchpads);
    if (cudaGetLastError() != cudaSuccess) {
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        goto cleanup;
    }

    // Copy states back to host
    h_states = (uint8_t*)malloc(num_passwords * STATE_SIZE);
    if (!h_states) {
        fprintf(stderr, "Host memory allocation failed for states\n");
        goto cleanup;
    }
    if (cudaMemcpy(h_states, d_states, num_passwords * STATE_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(h_states);
        goto cleanup;
    }

    // Compute extra hashes
    for (size_t i = 0; i < num_passwords; i++) {
        uint8_t *state = h_states + i * STATE_SIZE;
        int index = state[0] & 3;
        extra_hashes[index](state, STATE_SIZE, (char*)(hashes + i * HASH_SIZE));
    }

    // Cleanup
    for (size_t i = 0; i < num_passwords; i++) {
        cudaFree((void*)h_d_passwords[i]);
    }
    free(h_d_passwords);
    free(lengths);
    free(h_states);
    cudaFree(d_passwords);
    cudaFree(d_lengths);
    cudaFree(d_states);
    cudaFree(d_scratchpads);
    return 0;

cleanup:
    if (d_lengths) cudaFree(d_lengths);
    if (d_passwords) cudaFree(d_passwords);
    if (d_states) cudaFree(d_states);
    if (d_scratchpads) cudaFree(d_scratchpads);
    if (h_d_passwords) {
        for (size_t i = 0; i < num_passwords; i++) {
            if (h_d_passwords[i]) cudaFree((void*)h_d_passwords[i]);
        }
        free(h_d_passwords);
    }
    if (lengths) free(lengths);
    if (h_states) free(h_states);
    return -1;
}