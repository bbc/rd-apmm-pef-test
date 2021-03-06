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

int convert_simd_10p2_pef10(uint8_t * dst, const uint16_t * src, size_t n);
int convert_simd_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n);

int convert_simd256_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n);
int convert_simd512_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n);

int convert_simd_8p2_pef10(uint8_t * dst, const uint8_t * src, size_t n);
int convert_simd_pef10_8p2(uint8_t * dst, const uint8_t * src, size_t n);
