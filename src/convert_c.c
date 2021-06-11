/*
 * Copyright 2021 BBC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

 #include <stdint.h>
 #include <string.h>


size_t size_10p2(size_t n) {
    return n * sizeof(uint16_t);
}

size_t size_8p2(size_t n) {
    return n * sizeof(uint8_t);
}

size_t size_pef(size_t n) {
    return ((n + 63) / 64)*80;
}

void convert_c_10p2_8p2(uint8_t * dst, const uint16_t * src, size_t n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (src[i] >> 2) & 0xFF;
    }
}

void convert_c_10p2_pef10(uint8_t * dst, const uint16_t * src, size_t n) {
    for (int i = 0; i < ((n + 63)/64); i++) {
        for (int j = 0; j < 16; j ++) {
            uint8_t tmp = 0x00;
            for (int k = 0; k < 4; k++) {
                tmp <<= 2;
                tmp |= (src[i*64 + j*4 + k] & 0x03);
            }
            dst[i*16 + j] = tmp;
        }
    }
    convert_c_10p2_8p2(&dst[size_pef(n) - size_8p2(n)], src, n);
}

void convert_c_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n) {
    const size_t offs = size_pef(n) - size_8p2(n);
    const uint8_t *src8 = &src[offs];

    for (int i = 0; i < ((n + 63)/64); i++) {
        for (int j = 0; j < 16; j++) {
            uint8_t tmp = src[i*16 + j];
            for (int k = 0; k < 4; k++) {
                dst[i*64 + j*4 + k] = (src8[i*64 + j*4 + k] << 2) | ((tmp >> 6) & 0x3);
                tmp <<= 2;
            }
        }
    }
}




