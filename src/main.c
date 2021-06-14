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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "convert_c.h"
#include "convert_simd.h"


int main(int argc, char** argv) {
    const int width = 1920;

    uint8_t *DATA_LINE_PEF = malloc(size_pef(width));
    uint16_t *result_10p2 = malloc(size_10p2(width));

    convert_c_10p2_pef10(DATA_LINE_PEF, DATA_LINE_10P2, width);
    convert_simd_pef10_10p2(result_10p2, DATA_LINE_PEF, width);

    if (memcmp(DATA_LINE_10P2, result_10p2, size_10p2(width)) == 0) {
        printf("[OK]    Conversion matches expected value\n");
    } else {
        printf("[ERROR] Conversion does not match expected value!\n");
        printf("expected: ");
        for (int i=0; i < 16; i++) {
            printf("0x%03x ", DATA_LINE_10P2[i]);
        }
        printf("\n");
        printf("actual  : ");
        for (int i=0; i < 16; i++) {
            printf("0x%03x ", result_10p2[i]);
        }
        printf("\n");
        return 1;
    }

    free(result_10p2);
    free(DATA_LINE_PEF);
    return 0;
}