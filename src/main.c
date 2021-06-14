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

int check_reuslt_10(const uint16_t *A, const uint16_t *B, size_t size, const char *name) {
    if (memcmp(A, B, size) == 0) {
        printf("[OK]    %s: Conversion matches expected value\n", name);
        return 1;
    } else {
        printf("[ERROR] %s: Conversion does not match expected value!\n", name);
        printf("expected: ");
        for (int i=0; i < 16; i++) {
            printf("0x%03x ", A[i]);
        }
        printf("\n");
        printf("actual  : ");
        for (int i=0; i < 16; i++) {
            printf("0x%03x ", B[i]);
        }
        printf("\n");
        return 0;
    }
}


int main(int argc, char** argv) {
    const int width = 1920;
    int rval = 0;

    {
        uint8_t *DATA_LINE_PEF = calloc(size_pef(width), 1);
        uint16_t *result_10p2 = calloc(size_10p2(width), 1);

        convert_c_10p2_pef10(DATA_LINE_PEF, DATA_LINE_10P2, width);  // Perform a pure C conversion 10p2 -> PEF10
        convert_simd_pef10_10p2(result_10p2, DATA_LINE_PEF, width);  // Convert the PEF10 back to 10P2 using unit under test

        if (!check_reuslt_10(DATA_LINE_10P2, result_10p2, size_10p2(width), "convert_simd_pef10_10p2"))
            rval = 1;

        free(result_10p2);
        free(DATA_LINE_PEF);
    }

    {
        uint8_t *tmp_pef = calloc(size_pef(width), 1);
        uint16_t *result_10p2 = calloc(size_10p2(width), 1);

        convert_simd_10p2_pef10(tmp_pef, DATA_LINE_10P2, width);     // Convert 10p2 -> PEF10 using unit under test
        convert_c_pef10_10p2(result_10p2, tmp_pef, width);           // Convert back to 10p2 using pure C implementation

        if (!check_reuslt_10(DATA_LINE_10P2, result_10p2, size_10p2(width), "convert_simd_10p2_pef10"))
            rval = 1;

        free(result_10p2);
        free(tmp_pef);
    }

    return rval;
}