/*
 * aes_cuda.cu - CUDA Implementation for Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 * GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
 *
 * This file provides CUDA-accelerated AES encryption for the Monero GPU Cracker,
 * a tool to crack Monero wallet passwords using a hash from monero2john.py. It is
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

#include "aes_cuda.h"
#include <stdint.h>
#include <string.h>

// Ensure constants are defined
#ifndef AES_BLOCK_SIZE
#define AES_BLOCK_SIZE 16
#endif
#ifndef AES_KEY_SIZE
#define AES_KEY_SIZE 32
#endif
#ifndef OAES_RET
#define OAES_RET int
#endif
#ifndef OAES_RET_SUCCESS
#define OAES_RET_SUCCESS 0
#endif

// AES S-box in device memory
__device__ uint8_t oaes_sub_byte_value[16][16] = {
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

// Galois Field multiplication tables
__device__ uint8_t oaes_gf_mul_2[16][16] = {
    {0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e},
    {0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e},
    {0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e},
    {0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e},
    {0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e},
    {0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe},
    {0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde},
    {0xe0, 0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfc, 0xfe},
    {0x1b, 0x19, 0x1f, 0x1d, 0x13, 0x11, 0x17, 0x15, 0x0b, 0x09, 0x0f, 0x0d, 0x03, 0x01, 0x07, 0x05},
    {0x3b, 0x39, 0x3f, 0x3d, 0x33, 0x31, 0x37, 0x35, 0x2b, 0x29, 0x2f, 0x2d, 0x23, 0x21, 0x27, 0x25},
    {0x5b, 0x59, 0x5f, 0x5d, 0x53, 0x51, 0x57, 0x55, 0x4b, 0x49, 0x4f, 0x4d, 0x43, 0x41, 0x47, 0x45},
    {0x7b, 0x79, 0x7f, 0x7d, 0x73, 0x71, 0x77, 0x75, 0x6b, 0x69, 0x6f, 0x6d, 0x63, 0x61, 0x67, 0x65},
    {0x9b, 0x99, 0x9f, 0x9d, 0x93, 0x91, 0x97, 0x95, 0x8b, 0x89, 0x8f, 0x8d, 0x83, 0x81, 0x87, 0x85},
    {0xbb, 0xb9, 0xbf, 0xbd, 0xb3, 0xb1, 0xb7, 0xb5, 0xab, 0xa9, 0xaf, 0xad, 0xa3, 0xa1, 0xa7, 0xa5},
    {0xdb, 0xd9, 0xdf, 0xdd, 0xd3, 0xd1, 0xd7, 0xd5, 0xcb, 0xc9, 0xcf, 0xcd, 0xc3, 0xc1, 0xc7, 0xc5},
    {0xfb, 0xf9, 0xff, 0xfd, 0xf3, 0xf1, 0xf7, 0xf5, 0xeb, 0xe9, 0xef, 0xed, 0xe3, 0xe1, 0xe7, 0xe5}
};

__device__ uint8_t oaes_gf_mul_3[16][16] = {
    {0x00, 0x03, 0x06, 0x05, 0x0c, 0x0f, 0x0a, 0x09, 0x18, 0x1b, 0x1e, 0x1d, 0x14, 0x17, 0x12, 0x11},
    {0x30, 0x33, 0x36, 0x35, 0x3c, 0x3f, 0x3a, 0x39, 0x28, 0x2b, 0x2e, 0x2d, 0x24, 0x27, 0x22, 0x21},
    {0x60, 0x63, 0x66, 0x65, 0x6c, 0x6f, 0x6a, 0x69, 0x78, 0x7b, 0x7e, 0x7d, 0x74, 0x77, 0x72, 0x71},
    {0x50, 0x53, 0x56, 0x55, 0x5c, 0x5f, 0x5a, 0x59, 0x48, 0x4b, 0x4e, 0x4d, 0x44, 0x47, 0x42, 0x41},
    {0xc0, 0xc3, 0xc6, 0xc5, 0xcc, 0xcf, 0xca, 0xc9, 0xd8, 0xdb, 0xde, 0xdd, 0xd4, 0xd7, 0xd2, 0xd1},
    {0xf0, 0xf3, 0xf6, 0xf5, 0xfc, 0xff, 0xfa, 0xf9, 0xe8, 0xeb, 0xee, 0xed, 0xe4, 0xe7, 0xe2, 0xe1},
    {0xa0, 0xa3, 0xa6, 0xa5, 0xac, 0xaf, 0xaa, 0xa9, 0xb8, 0xbb, 0xbe, 0xbd, 0xb4, 0xb7, 0xb2, 0xb1},
    {0x90, 0x93, 0x96, 0x95, 0x9c, 0x9f, 0x9a, 0x99, 0x88, 0x8b, 0x8e, 0x8d, 0x84, 0x87, 0x82, 0x81},
    {0x9b, 0x98, 0x9d, 0x9e, 0x97, 0x94, 0x91, 0x92, 0x83, 0x80, 0x85, 0x86, 0x8f, 0x8c, 0x89, 0x8a},
    {0xab, 0xa8, 0xad, 0xae, 0xa7, 0xa4, 0xa1, 0xa2, 0xb3, 0xb0, 0xb5, 0xb6, 0xbf, 0xbc, 0xb9, 0xba},
    {0xfb, 0xf8, 0xfd, 0xfe, 0xf7, 0xf4, 0xf1, 0xf2, 0xe3, 0xe0, 0xe5, 0xe6, 0xef, 0xec, 0xe9, 0xea},
    {0xcb, 0xc8, 0xcd, 0xce, 0xc7, 0xc4, 0xc1, 0xc2, 0xd3, 0xd0, 0xd5, 0xd6, 0xdf, 0xdc, 0xd9, 0xda},
    {0x5b, 0x58, 0x5d, 0x5e, 0x57, 0x54, 0x51, 0x52, 0x43, 0x40, 0x45, 0x46, 0x4f, 0x4c, 0x49, 0x4a},
    {0x6b, 0x68, 0x6d, 0x6e, 0x67, 0x64, 0x61, 0x62, 0x73, 0x70, 0x75, 0x76, 0x7f, 0x7c, 0x79, 0x7a},
    {0x3b, 0x38, 0x3d, 0x3e, 0x37, 0x34, 0x31, 0x32, 0x23, 0x20, 0x25, 0x26, 0x2f, 0x2c, 0x29, 0x2a},
    {0x0b, 0x08, 0x0d, 0x0e, 0x07, 0x04, 0x01, 0x02, 0x13, 0x10, 0x15, 0x16, 0x1f, 0x1c, 0x19, 0x1a}
};

// Galois Field multiplication helper
__device__ uint8_t oaes_gf_mul(uint8_t left, uint8_t right) {
    size_t _x = left & 0x0f, _y = (left & 0xf0) >> 4;
    switch (right) {
        case 0x02: return oaes_gf_mul_2[_y][_x];
        case 0x03: return oaes_gf_mul_3[_y][_x];
        default: return left;
    }
}

// AES transformation functions
__device__ void oaes_sub_bytes_cuda(uint8_t *block) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        uint8_t x = block[i] & 0x0f, y = (block[i] & 0xf0) >> 4;
        block[i] = oaes_sub_byte_value[y][x];
    }
}

__device__ void oaes_shift_rows_cuda(uint8_t block[AES_BLOCK_SIZE]) {
    uint8_t temp[AES_BLOCK_SIZE];
    temp[0x00] = block[0x00]; temp[0x01] = block[0x05]; temp[0x02] = block[0x0a]; temp[0x03] = block[0x0f];
    temp[0x04] = block[0x04]; temp[0x05] = block[0x09]; temp[0x06] = block[0x0e]; temp[0x07] = block[0x03];
    temp[0x08] = block[0x08]; temp[0x09] = block[0x0d]; temp[0x0a] = block[0x02]; temp[0x0b] = block[0x07];
    temp[0x0c] = block[0x0c]; temp[0x0d] = block[0x01]; temp[0x0e] = block[0x06]; temp[0x0f] = block[0x0b];
    memcpy(block, temp, AES_BLOCK_SIZE);
}

__device__ void oaes_mix_columns_cuda(uint8_t block[AES_BLOCK_SIZE]) {
    uint8_t temp[AES_BLOCK_SIZE];
    for (int col = 0; col < 4; col++) {
        uint8_t s0 = block[col * 4], s1 = block[col * 4 + 1], s2 = block[col * 4 + 2], s3 = block[col * 4 + 3];
        temp[col * 4]     = oaes_gf_mul(s0, 0x02) ^ oaes_gf_mul(s1, 0x03) ^ s2 ^ s3;
        temp[col * 4 + 1] = s0 ^ oaes_gf_mul(s1, 0x02) ^ oaes_gf_mul(s2, 0x03) ^ s3;
        temp[col * 4 + 2] = s0 ^ s1 ^ oaes_gf_mul(s2, 0x02) ^ oaes_gf_mul(s3, 0x03);
        temp[col * 4 + 3] = oaes_gf_mul(s0, 0x03) ^ s1 ^ s2 ^ oaes_gf_mul(s3, 0x02);
    }
    memcpy(block, temp, AES_BLOCK_SIZE);
}

__device__ void oaes_add_round_key_cuda(const uint32_t *round_key, uint8_t *block) {
    for (int i = 0; i < 4; i++) {
        uint32_t rk = round_key[i];
        block[i * 4 + 0] ^= (rk & 0xff);         // LSB
        block[i * 4 + 1] ^= (rk >> 8) & 0xff;
        block[i * 4 + 2] ^= (rk >> 16) & 0xff;
        block[i * 4 + 3] ^= (rk >> 24) & 0xff;   // MSB
    }
}

// Key expansion for 10 rounds (scratchpad initialization)
__device__ void oaes_key_expand_cuda_10(uint32_t *expanded_key, const uint8_t *key) {
    const int Nk = 8;  // AES-256: 8 32-bit words
    const int Nr = 10; // 10 rounds
    const int Nb = 4;  // 4 words per block
    uint32_t temp;
    uint8_t rcon[11] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
    int rcon_idx = 1;

    // Load initial key in little-endian order
    for (int i = 0; i < Nk; i++) {
        expanded_key[i] = ((uint32_t)key[i * 4 + 0]) |
                          ((uint32_t)key[i * 4 + 1] << 8) |
                          ((uint32_t)key[i * 4 + 2] << 16) |
                          ((uint32_t)key[i * 4 + 3] << 24);
    }

    // Expand key
    for (int i = Nk; i < Nb * (Nr + 1); i++) {
        temp = expanded_key[i - 1];
        if (i % Nk == 0) {
            temp = (temp >> 8) | ((temp & 0xff) << 24); // RotWord
            temp = (oaes_sub_byte_value[(temp & 0xff) >> 4][temp & 0x0f]) |
                   (oaes_sub_byte_value[((temp >> 8) & 0xff) >> 4][(temp >> 8) & 0x0f] << 8) |
                   (oaes_sub_byte_value[((temp >> 16) & 0xff) >> 4][(temp >> 16) & 0x0f] << 16) |
                   (oaes_sub_byte_value[((temp >> 24) & 0xff) >> 4][(temp >> 24) & 0x0f] << 24); // SubWord
            temp ^= ((uint32_t)rcon[rcon_idx++]);
        } else if (Nk > 6 && i % Nk == 4) {
            temp = (oaes_sub_byte_value[(temp & 0xff) >> 4][temp & 0x0f]) |
                   (oaes_sub_byte_value[((temp >> 8) & 0xff) >> 4][(temp >> 8) & 0x0f] << 8) |
                   (oaes_sub_byte_value[((temp >> 16) & 0xff) >> 4][(temp >> 16) & 0x0f] << 16) |
                   (oaes_sub_byte_value[((temp >> 24) & 0xff) >> 4][(temp >> 24) & 0x0f] << 24); // SubWord
        }
        expanded_key[i] = expanded_key[i - Nk] ^ temp;
    }
}

// Key expansion for 14 rounds (finalization)
__device__ void oaes_key_expand_cuda_14(uint32_t *expanded_key, const uint8_t *key) {
    const int Nk = 8;  // AES-256: 8 32-bit words
    const int Nr = 14; // 14 rounds
    const int Nb = 4;  // 4 words per block
    uint32_t temp;
    uint8_t rcon[15] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d};
    int rcon_idx = 1;

    // Load initial key in little-endian order
    for (int i = 0; i < Nk; i++) {
        expanded_key[i] = ((uint32_t)key[i * 4 + 0]) |
                          ((uint32_t)key[i * 4 + 1] << 8) |
                          ((uint32_t)key[i * 4 + 2] << 16) |
                          ((uint32_t)key[i * 4 + 3] << 24);
    }

    // Expand key
    for (int i = Nk; i < Nb * (Nr + 1); i++) {
        temp = expanded_key[i - 1];
        if (i % Nk == 0) {
            temp = (temp >> 8) | ((temp & 0xff) << 24); // RotWord
            temp = (oaes_sub_byte_value[(temp & 0xff) >> 4][temp & 0x0f]) |
                   (oaes_sub_byte_value[((temp >> 8) & 0xff) >> 4][(temp >> 8) & 0x0f] << 8) |
                   (oaes_sub_byte_value[((temp >> 16) & 0xff) >> 4][(temp >> 16) & 0x0f] << 16) |
                   (oaes_sub_byte_value[((temp >> 24) & 0xff) >> 4][(temp >> 24) & 0x0f] << 24); // SubWord
            temp ^= ((uint32_t)rcon[rcon_idx++]);
        } else if (Nk > 6 && i % Nk == 4) {
            temp = (oaes_sub_byte_value[(temp & 0xff) >> 4][temp & 0x0f]) |
                   (oaes_sub_byte_value[((temp >> 8) & 0xff) >> 4][(temp >> 8) & 0x0f] << 8) |
                   (oaes_sub_byte_value[((temp >> 16) & 0xff) >> 4][(temp >> 16) & 0x0f] << 16) |
                   (oaes_sub_byte_value[((temp >> 24) & 0xff) >> 4][(temp >> 24) & 0x0f] << 24); // SubWord
        }
        expanded_key[i] = expanded_key[i - Nk] ^ temp;
    }
}

// Pseudo-encryption for scratchpad initialization (10 rounds)
__device__ OAES_RET oaes_pseudo_encrypt_ecb_cuda_10rounds(uint32_t *expanded_key, uint8_t *block) {
    for (int round = 0; round < 10; round++) {
        oaes_sub_bytes_cuda(block);
        oaes_shift_rows_cuda(block);
        oaes_mix_columns_cuda(block);
        oaes_add_round_key_cuda(expanded_key + round * 4, block);
    }
    return OAES_RET_SUCCESS;
}

// Pseudo-encryption for finalization (14 rounds)
__device__ OAES_RET oaes_pseudo_encrypt_ecb_cuda_14rounds(uint32_t *expanded_key, uint8_t *block) {
    for (int round = 0; round < 13; round++) {
        oaes_sub_bytes_cuda(block);
        oaes_shift_rows_cuda(block);
        oaes_mix_columns_cuda(block);
        oaes_add_round_key_cuda(expanded_key + round * 4, block);
    }
    oaes_sub_bytes_cuda(block);
    oaes_shift_rows_cuda(block);
    oaes_add_round_key_cuda(expanded_key + 13 * 4, block);
    return OAES_RET_SUCCESS;
}

// Key import function
__device__ void oaes_key_import_data_cuda(uint32_t *key, const uint8_t *data, size_t length) {
    if (length != AES_KEY_SIZE) return;
    oaes_key_expand_cuda_10(key, data);  // Default to 10 rounds for scratchpad
}