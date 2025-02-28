/*
 * aes_cuda.h - Header for CUDA AES Implementation in Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 * GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
 *
 * This header defines functions and structures for CUDA-accelerated AES encryption
 * used in the Monero GPU Cracker, a tool to crack Monero wallet passwords. It is
 * part of a port of the slow hash computation from John the Ripper Bleeding Jumbo
 * (https://github.com/openwall/john) to leverage GPU parallelism.
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

#ifndef AES_CUDA_H
#define AES_CUDA_H

#include <stdint.h>

// Define constants and types
#define AES_BLOCK_SIZE 16
#define AES_KEY_SIZE 32
#define OAES_RET int
#define OAES_RET_SUCCESS 0

// AES transformation functions
__device__ void oaes_sub_bytes_cuda(uint8_t *block);
__device__ void oaes_shift_rows_cuda(uint8_t block[AES_BLOCK_SIZE]);
__device__ void oaes_mix_columns_cuda(uint8_t block[AES_BLOCK_SIZE]);
__device__ void oaes_add_round_key_cuda(const uint32_t *round_key, uint8_t *block);

// AES round and encryption functions
__device__ OAES_RET oaes_encryption_round_cuda_full(const uint32_t *round_key, uint8_t *block);
__device__ void oaes_key_expand_cuda_10(uint32_t *expanded_key, const uint8_t *key);
__device__ OAES_RET oaes_pseudo_encrypt_ecb_cuda_10rounds(uint32_t *expanded_key, uint8_t *block);
__device__ void oaes_key_expand_cuda_14(uint32_t *expanded_key, const uint8_t *key);
__device__ OAES_RET oaes_pseudo_encrypt_ecb_cuda_14rounds(uint32_t *expanded_key, uint8_t *block);

// Key import function
__device__ void oaes_key_import_data_cuda(uint32_t *key, const uint8_t *data, size_t length);

#endif // AES_CUDA_H