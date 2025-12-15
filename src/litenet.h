#ifndef _LITENET_H_
#define _LITENET_H_

#include <ap_int.h>

// 定义数据位宽
typedef ap_int<8>  data_t;   // int8 用于特征图、权重
typedef ap_int<32> acc_t;    // int32 用于累加器
typedef ap_int<32> bias_t;   // int32 Bias
typedef ap_int<32> mult_t;   // 量化乘数
typedef ap_int<32> shift_t;  // 量化移位

// 顶层函数声明
void litenet(ap_int<32> input_packed[3*128*128/4], ap_int<32> output_packed[12/4]);

#endif