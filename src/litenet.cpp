#include "litenet.h"
#include "parameters.h"

// 【注意】请填入 image2int8.py 运行输出的 Input Zero Point
// 如果不确定，可以去 fpga_params/block1_depthwise/quant_info.txt 里看上一层(QuantStub)的信息，
// 或者暂时设为 0 调试
#define INPUT_ZP 57  

// 辅助函数：重量化 (Re-quantization)
// 执行: (acc + bias) * M >> shift + zp_out
// 即使 bias 是 0，这个逻辑也通用
data_t requantize(acc_t acc, bias_t bias, mult_t mult, shift_t shift, data_t zp_out) {
    //控制函数内联展开以提高性能
    #pragma HLS INLINE

    // 1. 加 Bias
    acc_t acc_biased = acc + bias;

    // 2. 定点乘法
    ap_int<64> product = (ap_int<64>)acc_biased * (ap_int<64>)mult;
    
    // 3. 移位, 带四舍五入
    if (shift > 0) {
        product += (1LL << (shift - 1)); 
    }
    acc_t scaled = (acc_t)(product >> shift); 

    // 4. 加输出零点 (注意：zp_out 也应该被视为无符号，但在加法中 int32 足够容纳)
    // 这里我们需要把 zp_out 当作无符号数读取
    acc_t result_zp = scaled + (acc_t)(ap_uint<8>)zp_out;

    // 5. 截断/Clamp 到 uint8 范围 [0, 255]
    // PyTorch FBGEMM 激活通常是 uint8
    if (result_zp > 255) return (data_t)255; // 强制转换回 data_t (位模式不变)
    if (result_zp < 0) return (data_t)0;
    
    // 将无符号结果强制转换为 data_t 存储 (例如 200 会变成 -56，但这没关系，只要读取时转回 uint8 即可)
    return (data_t)(ap_uint<8>)result_zp;
}

// Depthwise Convolution (无 Bias 的情况通过传入全0数组处理)
void depthwise_conv(
    data_t* input, 
    data_t* output,
    const int8_t* weight, 
    const int32_t* bias,  // 这里传入上面定义的 zeros 数组
    const int32_t* mult, 
    const int32_t* shift, 
    data_t zp_out,
    int in_h, int in_w, int c, int k, 
    data_t zp_in,
    int8_t weight_zp
) {
    int out_h = in_h - k + 1; // 128 - 9 + 1 = 120
    int out_w = in_w - k + 1;

    for (int ch = 0; ch < c; ch++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                
                acc_t acc = 0;
                
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        // NCHW 地址计算
                        int idx_in = (ch * in_h * in_w) + ((h + kh) * in_w) + (w + kw);
                        int idx_w  = (ch * k * k) + (kh * k + kw);
                        
                        acc_t in_val = (acc_t)(ap_uint<8>)input[idx_in] - (acc_t)(ap_uint<8>)zp_in;
                        acc_t w_val  = (acc_t)(weight[idx_w]) - (acc_t)(weight_zp);
                        acc += in_val * w_val;
                    }
                }
                
                // Requantize (含加 Bias，此处 bias[ch] 为 0)
                int idx_out = (ch * out_h * out_w) + (h * out_w + w);
                output[idx_out] = requantize(acc, bias[ch], mult[ch], shift[ch], zp_out);
            }
        }
    }
}

// Pointwise Convolution (1x1)
void pointwise_conv(
    data_t* input, 
    data_t* output,
    const int8_t* weight, 
    const int32_t* bias, 
    const int32_t* mult, 
    const int32_t* shift, 
    data_t zp_out,
    int h, int w, int ch_in, int ch_out,
    data_t zp_in,
    int8_t weight_zp
) {
    // #ifndef __SYNTHESIS__
    // // Debug: Print Block 1 PW acc (before bias) in NCHW order
    // // Block 1 PW has h=120, ch_out=16
    // if (h == 120 && ch_out == 16) {
    //     printf("=== Block 1 Pointwise ACC (No Bias) NCHW ===\n");
    //     for (int co = 0; co < ch_out; co++) {
    //         for (int row = 0; row < h; row++) {
    //             for (int col = 0; col < w; col++) {
    //                 acc_t acc = 0;
    //                 for (int ci = 0; ci < ch_in; ci++) {
    //                     int idx_in = (ci * h * w) + (row * w) + col;
    //                     int idx_w  = (co * ch_in) + ci; 
    //                     acc_t in_val = (acc_t)(ap_uint<8>)input[idx_in] - (acc_t)(ap_uint<8>)zp_in;
    //                     acc_t w_val  = (acc_t)(weight[idx_w]) - (acc_t)(weight_zp);
    //                     acc += in_val * w_val;
    //                 }
    //                 printf("%d ", (int)acc);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
    // #endif

    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            for (int co = 0; co < ch_out; co++) {
                
                acc_t acc = 0;
                
                for (int ci = 0; ci < ch_in; ci++) {
                    int idx_in = (ci * h * w) + (row * w) + col;
                    // Pointwise Weight: [Cout][Cin]
                    int idx_w  = (co * ch_in) + ci; 
                    
                    acc_t in_val = (acc_t)(ap_uint<8>)input[idx_in] - (acc_t)(ap_uint<8>)zp_in;
                    acc_t w_val  = (acc_t)(weight[idx_w]) - (acc_t)(weight_zp);
                    acc += in_val * w_val;
                }
                
                // Pointwise 有 bias，这里 bias[co] 不为 0
                int idx_out = (co * h * w) + (row * w) + col;
                output[idx_out] = requantize(acc, bias[co], mult[co], shift[co], zp_out);
            }
        }
    }
}

// Max Pooling
void max_pool(
    data_t* input, 
    data_t* output, 
    int in_h, int in_w, int c, int k, int s
) {
    int out_h = in_h / s;
    int out_w = in_w / s;

    for (int ch = 0; ch < c; ch++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                
                data_t max_val = -128; // int8 最小值
                
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        int r = h * s + kh;
                        int c_idx = w * s + kw;
                        // 边界检查 (Pooling 有时不需要 padding)
                        if (r < in_h && c_idx < in_w) {
                            int idx_in = (ch * in_h * in_w) + (r * in_w) + c_idx;
                            if (input[idx_in] > max_val) max_val = input[idx_in];
                        }
                    }
                }
                
                int idx_out = (ch * out_h * out_w) + (h * out_w + w);
                output[idx_out] = max_val;
            }
        }
    }
}

// Global Average Pooling + Fully Connected
void classifier_layer(
    data_t* input,
    data_t* output,
    const int8_t* weight,
    const int32_t* bias,
    const int32_t* mult,
    const int32_t* shift,
    data_t zp_out,
    int ch_in, int h_in, int w_in, int classes,
    data_t zp_in,
    int8_t weight_zp
) {
    // 1. Global Average Pooling (手动计算)
    // 临时存储 GAP 结果，使用 int32 防止溢出，最后转回 int8
    // 注意：GAP 操作本身通常不重量化，或者视为 scale=1, zp=0 的变换
    // 为了简单，我们这里求和后除以像素数，并保持 int8 范围
    
    // 这里的 buffer 可以小一点
    data_t gap_out[128]; 
    int pixels = h_in * w_in;

    for(int c=0; c<ch_in; c++) {
        int sum = 0;
        for(int i=0; i<pixels; i++) {
             sum += input[c*pixels + i];
        }
        // 简单平均
        gap_out[c] = (data_t)(sum / pixels); 
    }

    // 2. Fully Connected
    for (int i = 0; i < classes; i++) {
        acc_t acc = 0;
        for (int j = 0; j < ch_in; j++) {
            int idx_w = i * ch_in + j;
            acc_t in_val = (acc_t)(ap_uint<8>)gap_out[j] - (acc_t)(ap_uint<8>)zp_in;
            acc_t w_val  = (acc_t)(weight[idx_w]) - (acc_t)(weight_zp);
            acc += in_val * w_val;
        }
        output[i] = requantize(acc, bias[i], mult[i], shift[i], zp_out);
    }
}

// 顶层设计
void litenet(ap_int<32> input_packed[3*128*128/4], ap_int<32> output_packed[12/4]) {
    // #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
    // #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE bram port=output_packed
	#pragma HLS INTERFACE bram port=input_packed
	#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS


    // 1. 定义内部 Buffer，用于存放解包后的 8-bit 数据
    static data_t input_local[3*128*128];
    // 优化：数组分区以提高并行读取速度（可选）
    #pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=4 dim=1
    // 2. 数据搬运与解包 (32-bit -> 4x 8-bit)
    // Zynq 是小端序 (Little Endian)，低位字节在低地址
    unpack_loop: for (int i = 0; i < (3*128*128)/4; i++) {
        #pragma HLS PIPELINE II=1
        ap_int<32> temp = input_packed[i];
        input_local[i*4 + 0] = temp.range(7, 0);   // Byte 0
        input_local[i*4 + 1] = temp.range(15, 8);  // Byte 1
        input_local[i*4 + 2] = temp.range(23, 16); // Byte 2
        input_local[i*4 + 3] = temp.range(31, 24); // Byte 3
    }

    // === Buffer 定义 (放在 BRAM 中) ===
    static data_t buf1[3 * 120 * 120];   // B1 DW out
    static data_t buf2[16 * 120 * 120];  // B1 PW out
    static data_t buf3[16 * 30 * 30];    // B1 Pool out (input to B2)
    
    static data_t buf4[16 * 26 * 26];    // B2 DW out
    static data_t buf5[32 * 26 * 26];    // B2 PW out
    static data_t buf6[32 * 8 * 8];      // B2 Pool out (input to B3)

    static data_t buf7[32 * 3 * 3];      // B3 DW out
    static data_t buf8[128 * 3 * 3];     // B3 PW out (input to Classifier)

    // 定义内部 Output Buffer
    data_t output_local[12];

    // ================= Block 1 =================
    // Depthwise: 3x128x128 -> 3x120x120 (k=9)
    // 传入 block1_depthwise_bias_int32 (全0)
    depthwise_conv(input_local, buf1, block1_depthwise_weights, block1_depthwise_bias_int32, 
                   block1_depthwise_mult, block1_depthwise_shift, block1_depthwise_zp_out, 
                   128, 128, 3, 9, INPUT_ZP, 0);

    #ifndef __SYNTHESIS__
    printf("=== Block 1 Depthwise Output (3x120x120) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf1[i]);
    }
    printf("\n");
    #endif
                   
    // Pointwise: 3x120x120 -> 16x120x120
    pointwise_conv(buf1, buf2, block1_pointwise_weights, block1_pointwise_bias_int32,
                   block1_pointwise_mult, block1_pointwise_shift, block1_pointwise_zp_out,
                   120, 120, 3, 16, block1_depthwise_zp_out, 0);

    #ifndef __SYNTHESIS__
    printf("=== Block 1 Pointwise Output (16x120x120) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf2[i]);
    }
    printf("\n");
    #endif

    // Pool: 16x120x120 -> 16x30x30 (k=4, s=4)
    max_pool(buf2, buf3, 120, 120, 16, 4, 4);

    #ifndef __SYNTHESIS__
    printf("=== Block 1 MaxPool Output (16x30x30) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf3[i]);
    }
    printf("\n");
    #endif

    // ================= Block 2 =================
    // Depthwise: 16x30x30 -> 16x26x26 (k=5)
    // Input ZP 来自上一层 Block1 Pointwise 的输出 ZP
    depthwise_conv(buf3, buf4, block2_depthwise_weights, block2_depthwise_bias_int32,
                   block2_depthwise_mult, block2_depthwise_shift, block2_depthwise_zp_out,
                   30, 30, 16, 5, block1_pointwise_zp_out, 0);

    #ifndef __SYNTHESIS__
    printf("=== Block 2 Depthwise Output (16x26x26) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf4[i]);
    }
    printf("\n");
    #endif
                   
    // Pointwise: 16x26x26 -> 32x26x26
    pointwise_conv(buf4, buf5, block2_pointwise_weights, block2_pointwise_bias_int32,
                   block2_pointwise_mult, block2_pointwise_shift, block2_pointwise_zp_out,
                   26, 26, 16, 32, block2_depthwise_zp_out, 0);

    #ifndef __SYNTHESIS__
    printf("=== Block 2 Pointwise Output (32x26x26) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf5[i]);
    }
    printf("\n");
    #endif

    // Pool: 32x26x26 -> 32x8x8 (k=3, s=3)
    // 26/3 = 8
    max_pool(buf5, buf6, 26, 26, 32, 3, 3);

    #ifndef __SYNTHESIS__
    printf("=== Block 2 MaxPool Output (32x8x8) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf6[i]);
    }
    printf("\n");
    #endif

    // ================= Block 3 =================
    // Depthwise: 32x8x8 -> 32x3x3 (k=6)
    depthwise_conv(buf6, buf7, block3_depthwise_weights, block3_depthwise_bias_int32,
                   block3_depthwise_mult, block3_depthwise_shift, block3_depthwise_zp_out,
                   8, 8, 32, 6, block2_pointwise_zp_out, 0);

    #ifndef __SYNTHESIS__
    printf("=== Block 3 Depthwise Output (32x3x3) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf7[i]);
    }
    printf("\n");
    #endif

    // Pointwise: 32x3x3 -> 128x3x3
    pointwise_conv(buf7, buf8, block3_pointwise_weights, block3_pointwise_bias_int32,
                   block3_pointwise_mult, block3_pointwise_shift, block3_pointwise_zp_out,
                   3, 3, 32, 128, block3_depthwise_zp_out, 0);

    #ifndef __SYNTHESIS__
    printf("=== Block 3 Pointwise Output (128x3x3) ===\n");
    printf("First 20 values: ");
    for(int i=0; i<20; i++) {
        printf("%d ", (int)buf8[i]);
    }
    printf("\n");
    #endif

    // ================= Classifier =================
    // GAP (128x3x3 -> 128) + Linear (128 -> 12)
    classifier_layer(buf8, output_local, classifier_weights, classifier_bias_int32,
                    classifier_mult, classifier_shift, classifier_zp_out,
                    128, 3, 3, 12, block3_pointwise_zp_out, 0);

    // === 3. 数据打包 (Packing): 12x 8-bit -> 3x 32-bit ===
    pack_loop: for (int i = 0; i < 3; i++) {
        #pragma HLS PIPELINE II=1
        ap_int<32> packed_val;
        // 将 4 个 int8 拼成 1 个 int32
        packed_val.range(7, 0)   = output_local[i*4 + 0];
        packed_val.range(15, 8)  = output_local[i*4 + 1];
        packed_val.range(23, 16) = output_local[i*4 + 2];
        packed_val.range(31, 24) = output_local[i*4 + 3];
        
        output_packed[i] = packed_val;
    }


    #ifndef __SYNTHESIS__
    printf("=== Classifier Output_local (12 classes) ===\n");
    printf("All values: ");
    for(int i=0; i<12; i++) {
        printf("%d ", (int)output_local[i]);
    }
    printf("\n");
    #endif
}