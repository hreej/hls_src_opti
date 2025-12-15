#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

// ================= Block 1 =================
#include "weights/block1_depthwise/weights.h"
#include "weights/block1_depthwise/mult.h"
#include "weights/block1_depthwise/shift.h"
#include "weights/block1_depthwise/zp.h"
// Block1 DW 通道数 = 3, 定义全0 bias
static const int32_t block1_depthwise_bias_int32[3] = {0, 0, 0};

#include "weights/block1_pointwise/weights.h"
#include "weights/block1_pointwise/bias_int32.h"
#include "weights/block1_pointwise/mult.h"
#include "weights/block1_pointwise/shift.h"
#include "weights/block1_pointwise/zp.h"

// ================= Block 2 =================
#include "weights/block2_depthwise/weights.h"
#include "weights/block2_depthwise/mult.h"
#include "weights/block2_depthwise/shift.h"
#include "weights/block2_depthwise/zp.h"
// Block2 DW 通道数 = 16, 定义全0 bias
static const int32_t block2_depthwise_bias_int32[16] = {0};

#include "weights/block2_pointwise/weights.h"
#include "weights/block2_pointwise/bias_int32.h"
#include "weights/block2_pointwise/mult.h"
#include "weights/block2_pointwise/shift.h"
#include "weights/block2_pointwise/zp.h"

// ================= Block 3 =================
#include "weights/block3_depthwise/weights.h"
#include "weights/block3_depthwise/mult.h"
#include "weights/block3_depthwise/shift.h"
#include "weights/block3_depthwise/zp.h"
// Block3 DW 通道数 = 32, 定义全0 bias
static const int32_t block3_depthwise_bias_int32[32] = {0};

#include "weights/block3_pointwise/weights.h"
#include "weights/block3_pointwise/bias_int32.h"
#include "weights/block3_pointwise/mult.h"
#include "weights/block3_pointwise/shift.h"
#include "weights/block3_pointwise/zp.h"

// ================= Classifier =================
#include "weights/classifier/weights.h"
#include "weights/classifier/bias_int32.h"
#include "weights/classifier/mult.h"
#include "weights/classifier/shift.h"
#include "weights/classifier/zp.h"

#endif