#include <iostream>
#include "litenet.h"
#include "input_image_packed.h" // 您的原始 int8 数据
#include <iomanip> // 记得包含这个头文件用于格式化输出

// 声明打包后的输入和输出数组
// 使用 ap_int<32> 类型以匹配 HLS 接口定义
ap_int<32> output_packed_arr[3];

int main() {

    // 2. 调用 HLS 顶层函数
    std::cout << "Running HLS kernel..." << std::endl;
    litenet(input_packed_arr, output_packed_arr);


    printf("=== Output Packed Data (All 3) ===\n");
    printf("Index |  Hex (LE)  | Byte3 Byte2 Byte1 Byte0\n");
    printf("---------------------------------------------\n");

    for (int i = 0; i < 3; i++) {
        unsigned int val = (unsigned int)output_packed_arr[i];
        
        unsigned char b0 = (val >> 0)  & 0xFF;
        unsigned char b1 = (val >> 8)  & 0xFF;
        unsigned char b2 = (val >> 16) & 0xFF;
        unsigned char b3 = (val >> 24) & 0xFF;

        printf("%5d | 0x%08X |  %02X   %02X   %02X   %02X\n", i, val, b3, b2, b1, b0);
    }
    printf("\n");



    // 3. 解析输出 (解包) 并打印结果
    std::cout << "Result:" << std::endl;
    data_t final_output[12]; // 存放最终的 int8 结果

    for (int i = 0; i < 3; i++) {
        ap_int<32> val = output_packed_arr[i];
        final_output[i*4 + 0] = val.range(7, 0);
        final_output[i*4 + 1] = val.range(15, 8);
        final_output[i*4 + 2] = val.range(23, 16);
        final_output[i*4 + 3] = val.range(31, 24);
    }

    // 打印分类得分
    for (int i = 0; i < 12; i++) {
        std::cout << "Class " << i << ": " << (int)final_output[i] << std::endl;
    }

    return 0;
}