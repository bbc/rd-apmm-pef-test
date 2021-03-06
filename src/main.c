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

int check_reuslt_10(const uint16_t *A, const uint16_t *B, size_t size, const char *name, float ops) {
    if (memcmp(A, B, size) == 0) {
        printf("[OK]    %s: Conversion in %.2f simd ops per sample matches expected value\n", name, ops);
        return 1;
    } else {
        printf("[ERROR] %s: Conversion does not match expected value!\n", name);
        printf("expected: ");
        for (int i=0; i < 32; i++) {
            printf("0x%03x ", A[i]);
        }
        printf("\n");
        printf("actual  : ");
        for (int i=0; i < 32; i++) {
            printf("0x%03x ", B[i]);
        }
        printf("\n");
        return 0;
    }
}

int check_reuslt_8(const uint8_t *A, const uint8_t *B, size_t size, const char *name, float ops) {
    if (memcmp(A, B, size) == 0) {
        printf("[OK]    %s: Conversion in %.2f simd ops per sample matches expected value\n", name, ops);
        return 1;
    } else {
        printf("[ERROR] %s: Conversion does not match expected value!\n", name);
        printf("expected: ");
        for (int i=0; i < 16; i++) {
            printf("0x%02x ", A[i]);
        }
        printf("\n");
        printf("actual  : ");
        for (int i=0; i < 16; i++) {
            printf("0x%02x ", B[i]);
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
        int ops = convert_simd_pef10_10p2(result_10p2, DATA_LINE_PEF, width);  // Convert the PEF10 back to 10P2 using unit under test

        if (!check_reuslt_10(DATA_LINE_10P2, result_10p2, size_10p2(width), "convert_simd_pef10_10p2", ((float)ops)/width))
            rval = 1;

        free(result_10p2);
        free(DATA_LINE_PEF);
    }

    {
        uint8_t *DATA_LINE_PEF = calloc(size_pef(width), 1);
        uint16_t *result_10p2 = calloc(size_10p2(width), 1);

        convert_c_10p2_pef10(DATA_LINE_PEF, DATA_LINE_10P2, width);  // Perform a pure C conversion 10p2 -> PEF10
        int ops = convert_simd256_pef10_10p2(result_10p2, DATA_LINE_PEF, width);  // Convert the PEF10 back to 10P2 using unit under test

        if (!check_reuslt_10(DATA_LINE_10P2, result_10p2, size_10p2(width), "convert_simd256_pef10_10p2", ((float)ops)/width))
            rval = 1;

        free(result_10p2);
        free(DATA_LINE_PEF);
    }

    {
        uint8_t *DATA_LINE_PEF = calloc(size_pef(width), 1);
        uint16_t *result_10p2 = calloc(size_10p2(width), 1);

        convert_c_10p2_pef10(DATA_LINE_PEF, DATA_LINE_10P2, width);  // Perform a pure C conversion 10p2 -> PEF10
        int ops = convert_simd512_pef10_10p2(result_10p2, DATA_LINE_PEF, width);  // Convert the PEF10 back to 10P2 using unit under test

        if (!check_reuslt_10(DATA_LINE_10P2, result_10p2, size_10p2(width), "convert_simd512_pef10_10p2", ((float)ops)/width))
            rval = 1;

        free(result_10p2);
        free(DATA_LINE_PEF);
    }

    {
        uint8_t *tmp_pef = calloc(size_pef(width), 1);
        uint16_t *result_10p2 = calloc(size_10p2(width), 1);

        int ops = convert_simd_10p2_pef10(tmp_pef, DATA_LINE_10P2, width);     // Convert 10p2 -> PEF10 using unit under test
        convert_c_pef10_10p2(result_10p2, tmp_pef, width);           // Convert back to 10p2 using pure C implementation

        if (!check_reuslt_10(DATA_LINE_10P2, result_10p2, size_10p2(width), "convert_simd_10p2_pef10", ((float)ops)/width))
            rval = 1;

        free(result_10p2);
        free(tmp_pef);
    }

    uint8_t *DATA_LINE_8P2 = calloc(size_8p2(width), 1);
    convert_c_10p2_8p2(DATA_LINE_8P2, DATA_LINE_10P2, width);

    {
        uint8_t *DATA_LINE_PEF = calloc(size_pef(width), 1);
        uint8_t *result_8p2 = calloc(size_8p2(width), 1);

        convert_c_8p2_pef10(DATA_LINE_PEF, DATA_LINE_8P2, width);  // Perform a pure C conversion 8p2 -> PEF10
        int ops = convert_simd_pef10_8p2(result_8p2, DATA_LINE_PEF, width);  // Convert the PEF10 back to 8P2 using unit under test

        if (!check_reuslt_8(DATA_LINE_8P2, result_8p2, size_8p2(width), "convert_simd_pef10_8p2", ((float)ops)/width))
            rval = 1;

        free(result_8p2);
        free(DATA_LINE_PEF);
    }

    {
        uint8_t *tmp_pef = calloc(size_pef(width), 1);
        uint8_t *result_8p2 = calloc(size_8p2(width), 1);

        int ops = convert_simd_8p2_pef10(tmp_pef, DATA_LINE_8P2, width);     // Convert 8p2 -> PEF10 using unit under test
        convert_c_pef10_8p2(result_8p2, tmp_pef, width);           // Convert back to 8p2 using pure C implementation

        if (!check_reuslt_8(DATA_LINE_8P2, result_8p2, size_8p2(width), "convert_simd_8p2_pef10", ((float)ops)/width))
            rval = 1;

        free(result_8p2);
        free(tmp_pef);
    }

    free(DATA_LINE_8P2);

    return rval;
}