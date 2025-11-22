#include <stdio.h>
#include <stdlib.h>
#include <mysofa.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <KEMAR.sofa>\n", argv[0]);
        return 1;
    }

    const char *sofa_file = argv[1];
    int err;
    struct MYSOFA_HRTF *hrtf = mysofa_load(sofa_file, &err);
    if (!hrtf || err != MYSOFA_OK) {
        fprintf(stderr, "Failed to load SOFA file. Error: %d\n", err);
        return 1;
    }

    // 获取维度信息
    size_t n_measurements = hrtf->M;
    size_t fir_length = hrtf->N;
    printf("HRTF: %zu measurements, FIR length = %zu\n", n_measurements, fir_length);

    // 打开输出文件
    FILE *param_file = fopen("hrtf_params.txt", "w");
    FILE *left_file = fopen("hrtf_left.f32", "wb");
    FILE *right_file = fopen("hrtf_right.f32", "wb");

    if (!param_file || !left_file || !right_file) {
        perror("Failed to create output files");
        return 1;
    }

    // 遍历所有测量点
    for (size_t i = 0; i < n_measurements; i++) {
        float az = hrtf->SourcePosition.values[i * 3 + 0];     // azimuth
        float el = hrtf->SourcePosition.values[i * 3 + 1];     // elevation
        // distance = hrtf->SourcePosition.values[i * 3 + 2];  // usually 1.0

        // 获取 FIR 指针（每个是长度为 N 的 float 数组）
        float *left_fir = hrtf->DataIR.values + i * fir_length * 2 + 0;   // [L, R, L, R, ...]
        float *right_fir = hrtf->DataIR.values + i * fir_length * 2 + fir_length;

        // 写入参数文件
        fprintf(param_file, "%zu,%.2f,%.2f\n", i, az, el);

        // 写入二进制（float32）
        fwrite(left_fir, sizeof(float), fir_length, left_file);
        fwrite(right_fir, sizeof(float), fir_length, right_file);

        if (i % 50 == 0) {
            printf("Exported %zu / %zu\n", i, n_measurements);
        }
    }

    // 清理
    fclose(param_file);
    fclose(left_file);
    fclose(right_file);
    mysofa_free(hrtf);

    printf("✅ Export completed:\n");
    printf("   hrtf_params.txt\n");
    printf("   hrtf_left.f32 (%zu floats)\n", n_measurements * fir_length);
    printf("   hrtf_right.f32\n");
    return 0;
}