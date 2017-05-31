N N
C Ci
K Co

H iH
W iW

P oH
Q oW
pad_h
pad_w

        if N == 1:
            shlN = 0
        elif N < 32:
            shlN = len(bin(N-1))-2
        else:
            shlN = 5

        shlY, shlX, maskY, shrY, maskX, shrX, maskN, supY, supX = {
            0 : (4, 5, 0x18, 3, 0x07, 0, 0x00, 0x203, 0x300), # 4x8  yyxxx
            1 : (4, 4, 0x18, 3, 0x06, 1, 0x01, 0x203, 0x201), # 4x4  yyxxn
            2 : (3, 4, 0x10, 4, 0x0c, 2, 0x03, 0x104, 0x202), # 2x4  yxxnn
            3 : (2, 4, 0x00, 0, 0x18, 3, 0x07, 0x000, 0x203), # 1x4  xxnnn
            4 : (2, 3, 0x00, 0, 0x10, 4, 0x0f, 0x000, 0x104), # 1x2  xnnnn
            5 : (2, 2, 0x00, 0, 0x00, 0, 0x1f, 0x000, 0x000), # 1x1  nnnnn
        }.get(shlN)

        R, S = 3, 3
        GYS  = _ceil_div(P, 1 << shlY)  # ~= oH
        GXS  = _ceil_div(Q, 1 << shlX)  # ~= oW
        GN   = _ceil_div(N, 1 << shlN)  # ~= N // 32
        GK   = _ceil_div(K, 32)         # ~= Co // 32
        GYS2 = GYS // 2   # ~= oH//2
        GXS2 = GXS  * 2   # ~= oW * 2
        k    = _closest_divisor(GK, 4)
        Xk   = GXS*k
        YXk  = GYS*Xk

__device__ __forceinline__ int fp32_to_int32(float val)
{
    int ret;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}

__device__ __forceinline__ int div64(int value, int div_mul, int div_shift)
{
    int result;
    // if the divisor is a power of two the magic will be 1 and it's just a simple right shift
    if (magic == 1)
        result = value >> shift;
    // Otherwise multiply by magic and right shift just the high bits
    else
        asm(".reg .u64 res64;\n\t"
            ".reg .u32 lo32, hi32;\n\t"
            "mul.wide.u32 res64, %%1, %%2;\n\t"
            "mov.b64 {lo32, hi32}, res64;\n\t"
            "shr.b32 %%0, hi32, %%3;\n\t"
            : "=r"(result) : "r"(value), "r"(div_mul), "r"(div_shift));
    return result;
}

// grid = ( GN, GYS*GXS, C )    # (N // 32, oH * oW, Ci)
// block = (32,1,1)
// Y = iH
// X = iW
__global__ void xprop_image_trans_4x4(
    float* Out, const float* In,
    int Y, int X, int N, int pad_y, int pad_x,
    int GXS, int GYS2, int GXS2, int div_mul_GXS2, int div_shift_GXS2,
    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,
    int YXN, int XN, int GYS_GXS_C_1152, int GXS_C_1152, int C_1152)
{
    int tid   = threadIdx.x;
    int blkN  = gridDim.x - blockIdx.x - 1;
    int blkYX = gridDim.y - blockIdx.y - 1;
    int c     = gridDim.z - blockIdx.z - 1;

    // unpack y,x from blockIdx.x
    int gy2 = div64(blkYX, div_mul_GXS2, div_shift_GXS2);
    int gx2 = blkYX - gy2*GXS2;

    // Implement a square wave block id remapping
    // (for all but last row (if odd number of rows))
    int gy = gy2 << 1;
    int gx = gx2;
    if (gy2 != GYS2)
    {
        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);
        gx  = gx2 >> 1;
    }
    // Scan backwards on odd rows
    if (gy2 & 1)
        gx = GXS - gx - 1;

    // Super block YXN coordinates
    int y0 = (gy << shlY) + (((tid & maskY) >> shrY) << 2) - pad_y;
    int x0 = (gx << shlX) + (((tid & maskX) >> shrX) << 2) - pad_x;
    int n  = (blkN << shlN) + (tid & maskN);

    int out_offset = blkN*GYS_GXS_C_1152 + gy*GXS_C_1152 + gx*C_1152 + c*1152 + tid;

    bool valid = n < N;

    bool xin[6], yin[6];
    float I[6][6];

    for (int i = 0; i < 6; i++)
    {
        xin[i] = x0 + i >= 0 && x0 + i < X && valid;
        yin[i] = y0 + i >= 0 && y0 + i < Y;
    }

    int offset = c*YXN + y0*XN + x0*N + n;

    for (int y = 0; y < 6; y++)
    {
        if (y) offset += XN;

        for (int x = 0; x < 6; x++)
        {
            float val = 0;
            if (yin[y] && xin[x])
                val = __ldg(In + offset + x*N);
            I[y][x] = (val);
        }
    }

    // calculates B^T I[:][i]
    // each I[:][i] gives a [6][1] vector
    // concatenating these, this gives a [6][6] matrix
    // overall we calculate B^T I  , where I is a 6x6 input tile
    float T[6][6];
    for (int i = 0; i < 6; i++)
    {
        float t0 = __fmaf_rn(I[2][i], -4.0f, I[4][i]);
        float t1 = __fmaf_rn(I[1][i], -4.0f, I[3][i]);
        float t2 = I[4][i] - I[2][i];
        float t3 = I[3][i] - I[1][i];
        float t4 = __fmaf_rn(I[2][i], -5.0f, I[4][i]);
        float t5 = __fmaf_rn(I[3][i], -5.0f, I[5][i]);
        T[0][i] = __fmaf_rn(I[0][i], 4.0f, t4);
        T[1][i] = t0 + t1;
        T[2][i] = t0 - t1;
        T[3][i] = __fmaf_rn(t3,  2.0f, t2);
        T[4][i] = __fmaf_rn(t3, -2.0f, t2);
        T[5][i] = __fmaf_rn(I[1][i], 4.0f, t5);
        
        // T[0][i] = 4.0f * I[0][i] - 5.0f * I[2][i] + I[4][i];
        // tmp1 = fma(I[1][i], -4.0f, I[3][i])
        // T[1][i] = fma(I[2[i], -4.0f, tmp1)
        // T[i][i] 
    }
    // calculates (B^T I) B, giving a 6x6 transformed input tile
    for (int i = 0; i < 6; i++)
    {
        float t0 = __fmaf_rn(T[i][2], -4.0f, T[i][4]);
        float t1 = __fmaf_rn(T[i][1], -4.0f, T[i][3]);
        float t2 = T[i][4] - T[i][2];
        float t3 = T[i][3] - T[i][1];
        float t4 = __fmaf_rn(T[i][2], -5.0f, T[i][4]); // t4 = T[i][4] - 5.0f * T[i2][2]
        float t5 = __fmaf_rn(T[i][3], -5.0f, T[i][5]);
        Out[out_offset + 32*(i*6 + 0)] = (__fmaf_rn(T[i][0], 4.0f, t4));
        Out[out_offset + 32*(i*6 + 1)] = (t0 + t1);
        Out[out_offset + 32*(i*6 + 2)] = (t0 - t1);
        Out[out_offset + 32*(i*6 + 3)] = (__fmaf_rn(t3,  2.0f, t2));
        Out[out_offset + 32*(i*6 + 4)] = (__fmaf_rn(t3, -2.0f, t2));
        Out[out_offset + 32*(i*6 + 5)] = (__fmaf_rn(T[i][1], 4.0f, t5));
    }
}

