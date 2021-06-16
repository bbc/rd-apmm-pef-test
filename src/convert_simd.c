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

#define OP(C) (C);ops++


/*
 *   This routine. Per 64 sample block:
 *     8 x load
 *     8 x and
 *     8 x pack
 *     4 x shuffle
 *     8 x srl
 *     8 x unpack
 *     3 x sll
 *     3 x or
 *     5 x store
 *
 *     == 42 non-memory operations per 8 loads/5 stores
 */
int convert_simd_10p2_pef10(uint8_t * dst, const uint16_t * src, size_t n) {
    int ops = 0;
    const size_t offs = ((n + 63)/64)*16;
    const uint8_t *dst8 = &dst[offs];

    const __m128i LSB_MASK = _mm_set_epi16(
        0x0003, 0x0003, 0x0003, 0x0003,
        0x0003, 0x0003, 0x0003, 0x0003
    );

    const __m128i SHUF_MASK = _mm_set_epi8(
        0x0F, 0x0B, 0x07, 0x03, 0x0D, 0x09, 0x05, 0x01,
        0x0E, 0x0A, 0x06, 0x02, 0x0C, 0x08, 0x04, 0x00
    );

    for (int i = 0; i < (n + 63)/64; i++) {
        {
            int j = 0;
            __m128i lob0, lob1, lob2, lob3;
            {
                __m128i first_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 0]);
                __m128i second_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 8]);

                __m128i lob01 = OP(_mm_and_si128(first_eight, LSB_MASK));
                __m128i lob89 = OP(_mm_and_si128(second_eight, LSB_MASK));

                lob0 = OP(_mm_packus_epi16(lob01, lob89));
                lob0 = OP(_mm_shuffle_epi8(lob0, SHUF_MASK));

                __m128i first_eight_hob = OP(_mm_srli_epi16(first_eight, 2));
                __m128i second_eight_hob = OP(_mm_srli_epi16(second_eight, 2));
                __m128i hob = OP(_mm_packus_epi16(first_eight_hob, second_eight_hob));

                _mm_store_si128((__m128i *)&dst8[i*64 + j*16], hob);
            }

            j++;

            {
                __m128i first_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 0]);
                __m128i second_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 8]);

                __m128i lob01 = OP(_mm_and_si128(first_eight, LSB_MASK));
                __m128i lob89 = OP(_mm_and_si128(second_eight, LSB_MASK));

                lob1 = OP(_mm_packus_epi16(lob01, lob89));
                lob1 = OP(_mm_shuffle_epi8(lob1, SHUF_MASK));

                __m128i first_eight_hob = OP(_mm_srli_epi16(first_eight, 2));
                __m128i second_eight_hob = OP(_mm_srli_epi16(second_eight, 2));
                __m128i hob = OP(_mm_packus_epi16(first_eight_hob, second_eight_hob));

                _mm_store_si128((__m128i *)&dst8[i*64 + j*16], hob);
            }

            __m128i lobA = OP(_mm_unpacklo_epi32(lob0, lob1));
            __m128i lobB = OP(_mm_unpackhi_epi32(lob0, lob1));

            j++;

            {
                __m128i first_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 0]);
                __m128i second_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 8]);

                __m128i lob01 = OP(_mm_and_si128(first_eight, LSB_MASK));
                __m128i lob89 = OP(_mm_and_si128(second_eight, LSB_MASK));

                lob2 = OP(_mm_packus_epi16(lob01, lob89));
                lob2 = OP(_mm_shuffle_epi8(lob2, SHUF_MASK));

                __m128i first_eight_hob = OP(_mm_srli_epi16(first_eight, 2));
                __m128i second_eight_hob = OP(_mm_srli_epi16(second_eight, 2));
                __m128i hob = OP(_mm_packus_epi16(first_eight_hob, second_eight_hob));

                _mm_store_si128((__m128i *)&dst8[i*64 + j*16], hob);
            }

            j++;

            {
                __m128i first_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 0]);
                __m128i second_eight = _mm_load_si128((__m128i *)&src[i*64 + j*16 + 8]);

                __m128i lob01 = OP(_mm_and_si128(first_eight, LSB_MASK));
                __m128i lob89 = OP(_mm_and_si128(second_eight, LSB_MASK));

                lob3 = OP(_mm_packus_epi16(lob01, lob89));
                lob3 = OP(_mm_shuffle_epi8(lob3, SHUF_MASK));

                __m128i first_eight_hob = OP(_mm_srli_epi16(first_eight, 2));
                __m128i second_eight_hob = OP(_mm_srli_epi16(second_eight, 2));
                __m128i hob = OP(_mm_packus_epi16(first_eight_hob, second_eight_hob));

                _mm_store_si128((__m128i *)&dst8[i*64 + j*16], hob);
            }

            __m128i lobC = OP(_mm_unpacklo_epi32(lob2, lob3));
            __m128i lobD = OP(_mm_unpackhi_epi32(lob2, lob3));

            lob0 = OP(_mm_unpacklo_epi64(lobA, lobC));
            lob1 = OP(_mm_unpacklo_epi64(lobB, lobD));
            lob2 = OP(_mm_unpackhi_epi64(lobA, lobC));
            lob3 = OP(_mm_unpackhi_epi64(lobB, lobD));

            lob0 = OP(_mm_slli_epi16(lob0, 6));
            lob1 = OP(_mm_slli_epi16(lob1, 4));
            lob2 = OP(_mm_slli_epi16(lob2, 2));

            lobA = OP(_mm_or_si128(lob0, lob1));
            lobB = OP(_mm_or_si128(lob2, lob3));
            __m128i lob = OP(_mm_or_si128(lobA, lobB));
            _mm_store_si128((__m128i *)&dst[i*16], lob);
        }
    }

    return ops;
}

/*
 *   This routine. Per 64 sample block:
 *     5 x load
 *     8 x mullo
 *     4 x slli
 *     8 x bsrli
 *     4 x or
 *     8 x unpack
 *     8 x srli
 *     8 x store
 *
 *     == 40 non-memory operations per 5 loads/8 stores
 */

int convert_simd_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n) {
    int ops = 0;
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

            __m128i lob = OP(_mm_shuffle_epi8(low_order_bits, SHUF_CTRL));
            __m128i lob_even = OP(_mm_mullo_epi16(lob, MUL_SHIFT));
            __m128i lob_odd = OP(_mm_slli_epi16(lob, 2));
            lob_odd = OP(_mm_mullo_epi16(lob_odd, MUL_SHIFT));
            lob_even = OP(_mm_bsrli_si128(lob_even, 1));
            lob = OP(_mm_or_si128(lob_odd, lob_even));
            low_order_bits = OP(_mm_bsrli_si128(low_order_bits, 4));

            __m128i first8  = OP(_mm_unpacklo_epi8(lob, high_order_bits));
            first8  = OP(_mm_srli_epi16(first8, 6));
            _mm_store_si128((__m128i *)&dst[i*64 + j*16 + 0], first8);

            __m128i second8 = OP(_mm_unpackhi_epi8(lob, high_order_bits));
            second8 = OP(_mm_srli_epi16(second8, 6));
            _mm_store_si128((__m128i *)&dst[i*64 + j*16 + 8], second8);
        }
    }

    return ops;
}


/*
 *   This routine. Per 64 sample block:
 *     3 x load
 *     1 x broadcast
 *     2 x shuffle
 *     4 x mullo
 *     2 x slli
 *     2 x bsrli
 *     2 x or
 *     4 x unpack
 *     6 x srli
 *     4 x permute
 *     4 x store
 *
 *     == 27 non-memory operations per 3 loads/4 stores
 */

int convert_simd256_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n) {
    int ops = 0;
    const size_t offs = ((n + 63)/64)*16;
    const uint8_t *src8 = &src[offs];

    const __m256i MUL_SHIFT = _mm256_set_epi16(
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001,
        0x0010, 0x0001
    );

    const __m256i SHUF_CTRL = _mm256_set_epi16(
        0x07FF, 0x07FF,
        0x06FF, 0x06FF,
        0x05FF, 0x05FF,
        0x04FF, 0x04FF,
        0x03FF, 0x03FF,
        0x02FF, 0x02FF,
        0x01FF, 0x01FF,
        0x00FF, 0x00FF
    );

    for (int i = 0; i < (n + 63)/64; i++) {
        __m256i low_order_bits = OP(_mm256_broadcastsi128_si256(_mm_load_si128((__m128i *)&src[i*16])));

        for (int j = 0; j < 4; j+=2) {
            __m256i high_order_bits = _mm256_load_si256((__m256i *)&src8[i*64 + j*16]);

            __m256i lob = OP(_mm256_shuffle_epi8(low_order_bits, SHUF_CTRL));
            __m256i lob_even = OP(_mm256_mullo_epi16(lob, MUL_SHIFT));
            __m256i lob_odd = OP(_mm256_slli_epi16(lob, 2));
            lob_odd = OP(_mm256_mullo_epi16(lob_odd, MUL_SHIFT));
            lob_even = OP(_mm256_srli_si256(lob_even, 1));
            lob = OP(_mm256_or_si256(lob_odd, lob_even));
            low_order_bits = OP(_mm256_bsrli_epi128(low_order_bits, 8));

            __m256i first8  = OP(_mm256_unpacklo_epi8(lob, high_order_bits));
            first8  = OP(_mm256_srli_epi16(first8, 6));
            __m256i second8 = OP(_mm256_unpackhi_epi8(lob, high_order_bits));
            second8 = OP(_mm256_srli_epi16(second8, 6));

            __m256i tmp = OP(_mm256_permute2f128_si256(first8, second8, 0x20));
            _mm256_storeu_si256((__m256i *)&dst[i*64 + j*16 + 0], tmp);

            tmp = OP(_mm256_permute2f128_si256(first8, second8, 0x31));
            _mm256_storeu_si256((__m256i *)&dst[i*64 + j*16 + 16], tmp);
        }
    }

    return ops;
}

/*
 *   This routine. Per 64 sample block:
 *     2 x load
 *     1 x broadcast
 *     1 x shuffle
 *     2 x mullo
 *     1 x slli
 *     1 x bsrli
 *     1 x or
 *     2 x unpack
 *     2 x srli
 *     2 x permute
 *     2 x store
 *
 *     == 13 non-memory operations per 2 loads/2 stores
 */

int convert_simd512_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n) {
    int ops = 0;
    const size_t offs = ((n + 63)/64)*16;
    const uint8_t *src8 = &src[offs];

    const __m512i MUL_SHIFT = _mm512_set_epi32(
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001,
        0x00100001
    );

    const __m512i SHUF_CTRL = _mm512_set_epi32(
        0x0FFF0FFF,
        0x0EFF0EFF,
        0x0DFF0DFF,
        0x0CFF0CFF,
        0x0BFF0BFF,
        0x0AFF0AFF,
        0x09FF09FF,
        0x08FF08FF,
        0x07FF07FF,
        0x06FF06FF,
        0x05FF05FF,
        0x04FF04FF,
        0x03FF03FF,
        0x02FF02FF,
        0x01FF01FF,
        0x00FF00FF
    );

    const __m512i PERM_MASK1 = _mm512_set_epi64(
        0xB, 0xA,
        0x3, 0x2,
        0x9, 0x8,
        0x1, 0x0
    );

    const __m512i PERM_MASK2 = _mm512_set_epi64(
        0xF, 0xE,
        0x7, 0x6,
        0xD, 0xC,
        0x5, 0x4
    );

    for (int i = 0; i < (n + 63)/64; i++) {
        __m512i low_order_bits = OP(_mm512_broadcast_i32x4(_mm_load_si128((__m128i *)&src[i*16])));
        __m512i high_order_bits = _mm512_load_si512((__m512i *)&src8[i*64]);

        __m512i lob = OP(_mm512_shuffle_epi8(low_order_bits, SHUF_CTRL));
        __m512i lob_even = OP(_mm512_mullo_epi16(lob, MUL_SHIFT));
        __m512i lob_odd = OP(_mm512_slli_epi16(lob, 2));
        lob_odd = OP(_mm512_mullo_epi16(lob_odd, MUL_SHIFT));
        lob_even = OP(_mm512_bsrli_epi128(lob_even, 1));
        lob = OP(_mm512_or_si512(lob_odd, lob_even));

        __m512i first8  = OP(_mm512_unpacklo_epi8(lob, high_order_bits));
        first8  = OP(_mm512_srli_epi16(first8, 6));
        __m512i second8 = OP(_mm512_unpackhi_epi8(lob, high_order_bits));
        second8 = OP(_mm512_srli_epi16(second8, 6));

        __m512i tmp;
        tmp = OP(_mm512_permutex2var_epi64(first8, PERM_MASK1, second8));
        _mm512_storeu_si512((__m512i *)&dst[i*64 + 0], tmp);

        tmp = OP(_mm512_permutex2var_epi64(first8, PERM_MASK2, second8));
        _mm512_storeu_si512((__m512i *)&dst[i*64 + 32], tmp);
    }

    return ops;
}

/*
 *   This routine. Per 64 sample block:
 *     4 x load
 *     5 x store
 *
 *     == 0 non-memory operations per 4 loads/5 stores
 */
int convert_simd_8p2_pef10(uint8_t * dst, const uint8_t * src, size_t n) {
    int ops = 0;
    const size_t offs = ((n + 63)/64)*16;
    const uint8_t *dst8 = &dst[offs];

    const __m128i DUMMY_LSBs = _mm_set_epi8(
        0xAA, 0xAA, 0xAA, 0xAA,
        0xAA, 0xAA, 0xAA, 0xAA,
        0xAA, 0xAA, 0xAA, 0xAA,
        0xAA, 0xAA, 0xAA, 0xAA
    );

    for (int i = 0; i < (n + 63)/64; i++) {
        _mm_store_si128((__m128i *)&dst[i*16], DUMMY_LSBs);
        for (int j = 0; j < 4; j++) {
            __m128i data = _mm_load_si128((__m128i *)&src[i*64 + j*16]);
            _mm_store_si128((__m128i *)&dst8[i*64 + j*16], data);
        }
    }

    return ops;
}

/*
 *   This routine. Per 64 sample block:
 *     4 x load
 *     4 x store
 *
 *     == 0 non-memory operations per 4 loads/4 stores
 */
int convert_simd_pef10_8p2(uint8_t * dst, const uint8_t * src, size_t n) {
    int ops = 0;
    const size_t offs = ((n + 63)/64)*16;
    const uint8_t *src8 = &src[offs];

    memcpy(dst, src8, n);
    return ops;
}
