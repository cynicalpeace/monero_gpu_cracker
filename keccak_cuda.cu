/*
 * keccak_cuda.cu - CUDA Implementation for Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 * GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
 *
 * This file provides CUDA-accelerated Keccak hash computation for the Monero GPU
 * Cracker, a tool to crack Monero wallet passwords using a hash from monero2john.py.
 * It is part of a port of the slow hash computation from John the Ripper Bleeding
 * Jumbo (https://github.com/openwall/john) to leverage GPU parallelism.
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

#include "keccak_cuda.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// Use these definitions to match the CPU implementation.
#define RATE 136              // 1088 bits
#define STATE_SIZE 200        // 1600 bits

#ifdef __CUDACC__
__constant__
#endif
static const uint64_t keccakf_rndc_cuda[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

#define ROL64(x, offset) (((x) << (offset)) | ((x) >> (64 - (offset))))

__device__ void keccakf_cuda(uint64_t *st, int rounds) {
    int round;
    uint64_t t, bc[5];

    for (round = 0; round < rounds; ++round) {
        // Theta step
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                st[i + j] ^= t;
            }
        }

        // Rho-Pi steps
        t = st[1];
        st[1] = ROL64(st[6], 44);
        st[6] = ROL64(st[9], 20);
        st[9] = ROL64(st[22], 61);
        st[22] = ROL64(st[14], 39);
        st[14] = ROL64(st[20], 18);
        st[20] = ROL64(st[2], 62);
        st[2] = ROL64(st[12], 43);
        st[12] = ROL64(st[13], 25);
        st[13] = ROL64(st[19], 8);
        st[19] = ROL64(st[23], 56);
        st[23] = ROL64(st[15], 41);
        st[15] = ROL64(st[4], 27);
        st[4] = ROL64(st[24], 14);
        st[24] = ROL64(st[21], 2);
        st[21] = ROL64(st[8], 55);
        st[8] = ROL64(st[16], 45);
        st[16] = ROL64(st[5], 36);
        st[5] = ROL64(st[3], 28);
        st[3] = ROL64(st[18], 21);
        st[18] = ROL64(st[17], 15);
        st[17] = ROL64(st[11], 10);
        st[11] = ROL64(st[7], 6);
        st[7] = ROL64(st[10], 3);
        st[10] = ROL64(t, 1);

        // Chi step
        for (int j = 0; j < 25; j += 5) {
            uint64_t st0 = st[j], st1 = st[j + 1], st2 = st[j + 2], st3 = st[j + 3], st4 = st[j + 4];
            st[j]     ^= (~st1) & st2;
            st[j + 1] ^= (~st2) & st3;
            st[j + 2] ^= (~st3) & st4;
            st[j + 3] ^= (~st4) & st0;
            st[j + 4] ^= (~st0) & st1;
        }

        // Iota step
        st[0] ^= keccakf_rndc_cuda[round];
    }
}

__device__ void keccak1600_cuda(const void *data, size_t length, uint8_t *hash) {
    uint64_t st[25];
    memset(st, 0, 25 * sizeof(uint64_t));

    // Absorb input (up to RATE bytes)
    size_t copy_len = (length < RATE) ? length : RATE;
    memcpy(st, data, copy_len);

    // Apply CryptoNight padding (0x01 followed by zeros, then 0x80 at RATE-1)
    // Note: This matches John the Ripper's keccak1600 for CryptoNight
    ((uint8_t*)st)[copy_len] ^= 0x01;  // Multi-rate padding start
    ((uint8_t*)st)[RATE - 1] ^= 0x80;  // Multi-rate padding end

    // Apply Keccak-f[1600] permutation with 24 rounds
    keccakf_cuda(st, 24);

    // Output full 200-byte state
    memcpy(hash, st, STATE_SIZE);
}