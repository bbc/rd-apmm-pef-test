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

 #include "convert_c.h"
 #include <x86intrin.h>


void convert_simd_10p2_pef10(uint8_t * dst, const uint16_t * src, size_t n) {
}

/*
 *   This routine. Per 64 sample block:
 *     5 x load
 *     8 x mullo
 *     4 x slli
 *     8 x bsrli
 *     8 x unpack
 *     8 x srli
 *     8 x store
 *
 *     == 36 non-memory operations per 5 loads/8 stores
 */

void convert_simd_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n) {
    const size_t offs = ((n + 63)/64)*16;
    const uint8_t *src8 = &src[offs];

    const __m128i MUL_SHIFT = _mm_set_epi16(
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001
    );

    const __m128i SHUF_CTRL = _mm_set_epi16(
        0x03FF, 0x03FF,
        0x02FF, 0x02FF,
        0x01FF, 0x01FF,
        0x00FF, 0x00FF
    );

    for (int i = 0; i < (n + 63)/64; i++) {
        __m128i low_order_bits = _mm_load_si128((__m128i *)&src[i*16]);

        for (int j = 0; j < 4; j++) {
            __m128i high_order_bits = _mm_load_si128((__m128i *)&src8[i*64 + j*16]);

            __m128i lob = _mm_shuffle_epi8(low_order_bits, SHUF_CTRL);
            __m128i lob_even = _mm_mullo_epi16(lob, MUL_SHIFT);
            __m128i lob_odd = _mm_slli_epi16(lob, 2);
            lob_odd = _mm_mullo_epi16(lob_odd, MUL_SHIFT);
            lob_even = _mm_bsrli_si128(lob_even, 1);
            lob = _mm_or_si128(lob_odd, lob_even);
            low_order_bits = _mm_bsrli_si128(low_order_bits, 4);

            __m128i first8  = _mm_unpacklo_epi8(lob, high_order_bits);
            first8  = _mm_srli_epi16(first8, 6);
            _mm_store_si128((__m128i *)&dst[i*64 + j*16 + 0], first8);

            __m128i second8 = _mm_unpackhi_epi8(lob, high_order_bits);
            second8 = _mm_srli_epi16(second8, 6);
            _mm_store_si128((__m128i *)&dst[i*64 + j*16 + 8], second8);
        }
    }
}
