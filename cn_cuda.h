/*
 * cn_cuda.h - Header for CUDA CryptoNight Implementation in Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 * GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
 *
 * This header defines functions and structures for CUDA-accelerated CryptoNight
 * slow hash computation used in the Monero GPU Cracker, a tool to crack Monero
 * wallet passwords. It is part of a port of the slow hash computation from John
 * the Ripper Bleeding Jumbo (https://github.com/openwall/john) to leverage GPU
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

 #ifndef CN_CUDA_H
 #define CN_CUDA_H
 
 #include <stddef.h>
 #include <stdint.h>
 
 /* Define HASH_SIZE if not already defined */
 #ifndef HASH_SIZE
 #define HASH_SIZE 32
 #endif
 
 /* Union for efficient block handling, aligned to 16 bytes for CUDA efficiency */
 union cn_slow_hash_block {
     uint8_t bytes[16];
     uint32_t words[4];
     uint64_t dwords[2];
 } __attribute__((aligned(16)));
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /*
  * Host-side wrapper function to compute the CryptoNight slow hash on the GPU.
  * [passwords] is an array of C strings.
  * [num_passwords] is the number of candidate passwords.
  * [hashes] must point to a contiguous buffer of (num_passwords * HASH_SIZE) bytes.
  * [threads_per_block] specifies the number of threads per block for the CUDA kernel.
  * Returns 0 on success, -1 on failure.
  */
 int compute_cn_slow_hash_cuda(const char **passwords, size_t num_passwords, unsigned char *hashes, int threads_per_block);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // CN_CUDA_H