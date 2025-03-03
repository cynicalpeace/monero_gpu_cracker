/*
 * keccak_cuda.h - Header for CUDA Keccak Implementation in Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 *
 * This header defines functions and structures for CUDA-accelerated Keccak hash
 * computation used in the Monero GPU Cracker, a tool to crack Monero wallet
 * passwords. It is part of a port of the slow hash computation from John the
 * Ripper Bleeding Jumbo (https://github.com/openwall/john) to leverage GPU
 * parallelism.
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

#ifndef KECCAK_CUDA_H
#define KECCAK_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Device-side function that performs the Keccak-f[1600] permutation.
 * 'state' points to an array of 25 uint64_t words.
 * 'rounds' specifies the number of rounds.
 */
__device__ void keccakf_cuda(uint64_t *state, int rounds);

/*
 * Device-side function that computes the Keccak-1600 hash.
 * 'data' is the input buffer of 'length' bytes.
 * 'hash' must point to a buffer of at least STATE_SIZE bytes.
 */
__device__ void keccak1600_cuda(const void *data, size_t length, uint8_t *hash);

#ifdef __cplusplus
}
#endif

#endif // KECCAK_CUDA_H
