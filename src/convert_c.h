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

size_t size_10p2(size_t n);
size_t size_8p2(size_t n);
size_t size_pef(size_t n);

void convert_c_10p2_8p2(uint8_t * dst, const uint16_t * src, size_t n);
void convert_c_10p2_pef10(uint8_t * dst, const uint16_t * src, size_t n);
void convert_c_pef10_10p2(uint16_t * dst, const uint8_t * src, size_t n);
void convert_c_8p2_pef10(uint8_t * dst, const uint8_t * src, size_t n);
void convert_c_pef10_8p2(uint8_t * dst, const uint8_t * src, size_t n);
