#include "algorithms/rw4b_grad.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"

namespace rw4b {

static constexpr int kGradBlock = 256;
static constexpr float kTauFo = 0.10f;
static constexpr float kStepClip = 0.35f;

// ---------------- Device smooth math helpers ----------------

__device__ inline float softplusF(float x, float tau) {
    float z = x / tau;
    if (z > 20.0f) return x;
    if (z < -20.0f) return 0.0f;
    return tau * log1pf(expf(z));
}

__device__ inline float sigmoidF(float x, float tau) {
    float z = x / tau;
    if (z > 20.0f) return 1.0f;
    if (z < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}

__device__ inline float lse2W(float a, float b, float gamma, float & w0, float & w1) {
    float m = a >= b ? a : b;
    float e0 = expf((a - m) / gamma);
    float e1 = expf((b - m) / gamma);
    float s = e0 + e1;
    w0 = e0 / s;
    w1 = e1 / s;
    return m + gamma * logf(s);
}

__device__ inline void readLitSmooth(int lit, int nPIs, float piAlpha, float tauFo, bool piPhaseAware,
                                     const float * d_expFo, const float * d_arr, const float * d_slew,
                                     float & aOut, float & sOut) {
    int u = lit >> 1;
    if (u == 0) {
        aOut = 0.0f;
        sOut = 0.0f;
        return;
    }
    if (u <= nPIs) {
        if (piPhaseAware && (lit & 1)) {
            float ld = softplusF(d_expFo[u] - 1.0f, tauFo);
            aOut = piAlpha * ld;
            sOut = ld;
        } else {
            aOut = 0.0f;
            sOut = 0.0f;
        }
        return;
    }
    aOut = d_arr[u];
    sOut = d_slew[u];
}

__device__ inline void readLitHard(int lit, int nPIs, float piAlpha, bool piPhaseAware,
                                   const float * d_expFo, const float * d_arr, const float * d_slew,
                                   float & aOut, float & sOut) {
    int u = lit >> 1;
    if (u == 0) {
        aOut = 0.0f;
        sOut = 0.0f;
        return;
    }
    if (u <= nPIs) {
        if (piPhaseAware && (lit & 1)) {
            float ld = d_expFo[u] > 1.0f ? (d_expFo[u] - 1.0f) : 0.0f;
            aOut = piAlpha * ld;
            sOut = ld;
        } else {
            aOut = 0.0f;
            sOut = 0.0f;
        }
        return;
    }
    aOut = d_arr[u];
    sOut = d_slew[u];
}

__device__ inline void addLitGrad(int lit, int nPIs, bool piPhaseAware, float ga, float gs,
                                  float * d_gArr, float * d_gSlew,
                                  float * d_gPiNegArr, float * d_gPiNegSlew) {
    int u = lit >> 1;
    if (u == 0) return;
    if (u <= nPIs) {
        if (piPhaseAware && (lit & 1)) {
            if (ga != 0.0f) atomicAdd(&d_gPiNegArr[u], ga);
            if (gs != 0.0f) atomicAdd(&d_gPiNegSlew[u], gs);
        }
        return;
    }
    if (ga != 0.0f) atomicAdd(&d_gArr[u], ga);
    if (gs != 0.0f) atomicAdd(&d_gSlew[u], gs);
}

// ---------------- Kernel 1: Softmax & Expected Fanout CSR Aggregation ----------------

__global__ void k_init_exp_fo(int nObjs, const int * d_foBase, float * d_expFo) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nObjs) return;
    d_expFo[u] = static_cast<float>(d_foBase[u]);
}

__global__ void k_softmax_and_exp_fo(
    int M, float tauP,
    const int * d_candCount, const float * d_logits,
    const int * d_foDeltaOffset, const int * d_foDeltaNode, const int * d_foDeltaVal,
    float * d_prob, float * d_expFo) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int K = d_candCount[m];
    int base = m * kGradMaxK;

    float maxZ = d_logits[base];
    for (int k = 1; k < K; ++k) {
        float z = d_logits[base + k];
        if (z > maxZ) maxZ = z;
    }
    float sumE = 0.0f;
    float ex[kGradMaxK];
    for (int k = 0; k < K; ++k) {
        ex[k] = expf((d_logits[base + k] - maxZ) / tauP);
        sumE += ex[k];
    }
    for (int k = 0; k < K; ++k) {
        float pk = ex[k] / sumE;
        d_prob[base + k] = pk;
        if (k > 0 && pk > 0.0f) {
            int st = d_foDeltaOffset[base + k];
            int ed = d_foDeltaOffset[base + k + 1];
            for (int e = st; e < ed; ++e) {
                int u = d_foDeltaNode[e];
                float dv = static_cast<float>(d_foDeltaVal[e]) * pk;
                atomicAdd(&d_expFo[u], dv);
            }
        }
    }
    for (int k = K; k < kGradMaxK; ++k) d_prob[base + k] = 0.0f;
}

// Discrete version for multi-threshold line search (sets p_{m,k*} = 1 if prob >= theta, else p_{m,0} = 1)
__global__ void k_discretize_and_exp_fo(
    int M, float theta,
    const int * d_candCount, const float * d_prob, const int * d_candDeltaAnd,
    const int * d_foDeltaOffset, const int * d_foDeltaNode, const int * d_foDeltaVal,
    int * d_discChoice, float * d_expFo, int * d_totalDeltaAnd) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int K = d_candCount[m];
    int base = m * kGradMaxK;

    int bestK = 0;
    float bestP = d_prob[base];
    for (int k = 1; k < K; ++k) {
        if (d_prob[base + k] > bestP) {
            bestP = d_prob[base + k];
            bestK = k;
        }
    }
    if (bestK > 0 && bestP < theta) bestK = 0;
    d_discChoice[m] = bestK;

    if (bestK > 0) {
        atomicAdd(d_totalDeltaAnd, d_candDeltaAnd[base + bestK]);
        int st = d_foDeltaOffset[base + bestK];
        int ed = d_foDeltaOffset[base + bestK + 1];
        for (int e = st; e < ed; ++e) {
            atomicAdd(&d_expFo[d_foDeltaNode[e]], static_cast<float>(d_foDeltaVal[e]));
        }
    }
}

// ---------------- Kernel 2: Levelized Smooth Micro-Subgraph Forward ----------------

__global__ void k_fwd_level(
    int startId, int count, int nPIs,
    float alpha, float piAlpha, float beta, float gamma, float gammaS, float tauFo, bool piPhaseAware,
    const int * d_f0, const int * d_f1, const int * d_isRegionRoot,
    const int * d_candCount, const int * d_candWireLit, const int * d_candNumGates,
    const SubGate * d_candSubGates, const float * d_prob, const float * d_expFo,
    float * d_arr, float * d_slew, float * d_slewIn, float * d_fwdW,
    float * d_candArr, float * d_candOutSlew, float * d_candRootInSlew, float * d_candFwdW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int v = startId + idx;
    int m = d_isRegionRoot[v];
    float ldV = softplusF(d_expFo[v] - 1.0f, tauFo);

    if (m < 0) {
        float a0, s0, a1, s1;
        readLitSmooth(d_f0[v], nPIs, piAlpha, tauFo, piPhaseAware, d_expFo, d_arr, d_slew, a0, s0);
        readLitSmooth(d_f1[v], nPIs, piAlpha, tauFo, piPhaseAware, d_expFo, d_arr, d_slew, a1, s1);
        float wa0, wa1, ws0, ws1;
        float aIn = lse2W(a0, a1, gamma, wa0, wa1);
        float sIn = lse2W(s0, s1, gammaS, ws0, ws1);
        d_fwdW[v * 2 + 0] = wa0;
        d_fwdW[v * 2 + 1] = ws0;
        d_slewIn[v] = sIn;
        d_arr[v] = aIn + 1.0f + alpha * ldV * (1.0f + beta * sIn);
        d_slew[v] = ldV;
        return;
    }

    int K = d_candCount[m];
    int base = m * kGradMaxK;
    float aMix = 0.0f, sMix = 0.0f;

    for (int k = 0; k < K; ++k) {
        int slot = base + k;
        float pk = d_prob[slot];
        int wLit = d_candWireLit[slot];
        float aK, sOutK, sInRoot = 0.0f;

        if (wLit >= 0) {
            readLitSmooth(wLit, nPIs, piAlpha, tauFo, piPhaseAware, d_expFo, d_arr, d_slew, aK, sOutK);
        } else {
            int nG = d_candNumGates[slot];
            float gaArr[kGradMaxSubGates];
            float gaSlew[kGradMaxSubGates];
            int gBase = slot * kGradMaxSubGates;
            for (int g = 0; g < nG; ++g) {
                SubGate sg = d_candSubGates[gBase + g];
                float a0, s0, a1, s1;
                if (sg.src0 >= 0) {
                    readLitSmooth(sg.src0, nPIs, piAlpha, tauFo, piPhaseAware, d_expFo, d_arr, d_slew, a0, s0);
                } else {
                    int p = -1 - sg.src0;
                    a0 = gaArr[p];
                    s0 = gaSlew[p];
                }
                if (sg.src1 >= 0) {
                    readLitSmooth(sg.src1, nPIs, piAlpha, tauFo, piPhaseAware, d_expFo, d_arr, d_slew, a1, s1);
                } else {
                    int p = -1 - sg.src1;
                    a1 = gaArr[p];
                    s1 = gaSlew[p];
                }
                float wa0, wa1, ws0, ws1;
                float aIn = lse2W(a0, a1, gamma, wa0, wa1);
                float sIn = lse2W(s0, s1, gammaS, ws0, ws1);
                d_candFwdW[(gBase + g) * 2 + 0] = wa0;
                d_candFwdW[(gBase + g) * 2 + 1] = ws0;
                float ldG = (g == nG - 1) ? ldV : sg.load;
                gaArr[g] = aIn + 1.0f + alpha * ldG * (1.0f + beta * sIn);
                gaSlew[g] = ldG;
                if (g == nG - 1) sInRoot = sIn;
            }
            aK = gaArr[nG - 1];
            sOutK = ldV;
        }
        d_candArr[slot] = aK;
        d_candOutSlew[slot] = sOutK;
        d_candRootInSlew[slot] = sInRoot;
        aMix += pk * aK;
        sMix += pk * sOutK;
    }
    d_arr[v] = aMix;
    d_slew[v] = sMix;
    d_slewIn[v] = 0.0f;
}

// Hard-max discrete forward kernel for multi-threshold discrete line search
__global__ void k_fwd_level_hard(
    int startId, int count, int nPIs,
    float alpha, float piAlpha, float beta, bool piPhaseAware,
    const int * d_f0, const int * d_f1, const int * d_isRegionRoot,
    const int * d_discChoice, const int * d_candWireLit, const int * d_candNumGates,
    const SubGate * d_candSubGates, const float * d_expFo,
    float * d_arr, float * d_slew) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int v = startId + idx;
    int m = d_isRegionRoot[v];
    float ldV = d_expFo[v] > 1.0f ? (d_expFo[v] - 1.0f) : 0.0f;

    if (m < 0) {
        float a0, s0, a1, s1;
        readLitHard(d_f0[v], nPIs, piAlpha, piPhaseAware, d_expFo, d_arr, d_slew, a0, s0);
        readLitHard(d_f1[v], nPIs, piAlpha, piPhaseAware, d_expFo, d_arr, d_slew, a1, s1);
        float aIn = a0 >= a1 ? a0 : a1;
        float sIn = s0 >= s1 ? s0 : s1;
        d_arr[v] = aIn + 1.0f + alpha * ldV * (1.0f + beta * sIn);
        d_slew[v] = ldV;
        return;
    }

    int k = d_discChoice[m];
    int slot = m * kGradMaxK + k;
    int wLit = d_candWireLit[slot];
    if (wLit >= 0) {
        float aK, sK;
        readLitHard(wLit, nPIs, piAlpha, piPhaseAware, d_expFo, d_arr, d_slew, aK, sK);
        d_arr[v] = aK;
        d_slew[v] = sK;
        return;
    }

    int nG = d_candNumGates[slot];
    float gaArr[kGradMaxSubGates];
    float gaSlew[kGradMaxSubGates];
    int gBase = slot * kGradMaxSubGates;
    for (int g = 0; g < nG; ++g) {
        SubGate sg = d_candSubGates[gBase + g];
        float a0, s0, a1, s1;
        if (sg.src0 >= 0) {
            readLitHard(sg.src0, nPIs, piAlpha, piPhaseAware, d_expFo, d_arr, d_slew, a0, s0);
        } else {
            int p = -1 - sg.src0;
            a0 = gaArr[p];
            s0 = gaSlew[p];
        }
        if (sg.src1 >= 0) {
            readLitHard(sg.src1, nPIs, piAlpha, piPhaseAware, d_expFo, d_arr, d_slew, a1, s1);
        } else {
            int p = -1 - sg.src1;
            a1 = gaArr[p];
            s1 = gaSlew[p];
        }
        float aIn = a0 >= a1 ? a0 : a1;
        float sIn = s0 >= s1 ? s0 : s1;
        float ldG = (g == nG - 1) ? ldV : sg.load;
        gaArr[g] = aIn + 1.0f + alpha * ldG * (1.0f + beta * sIn);
        gaSlew[g] = ldG;
    }
    d_arr[v] = gaArr[nG - 1];
    d_slew[v] = ldV;
}

// ---------------- Kernel 3: Loss Reduction (double) & PO Adjoints ----------------

__global__ void k_extract_po_arr(
    int nPOs, int nPIs, float piAlpha, float tauFo, bool piPhaseAware, bool hardMode,
    const int * d_outs, const float * d_expFo, const float * d_arr, const float * d_slew,
    float * d_poArr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPOs) return;
    float a, s;
    if (hardMode) {
        readLitHard(d_outs[i], nPIs, piAlpha, piPhaseAware, d_expFo, d_arr, d_slew, a, s);
    } else {
        readLitSmooth(d_outs[i], nPIs, piAlpha, tauFo, piPhaseAware, d_expFo, d_arr, d_slew, a, s);
    }
    d_poArr[i] = a;
}

__global__ void k_po_loss_stats(
    int nPOs, float poMax, float gammaPo, float tTarget,
    const float * d_poArr, double * d_wnsSumExp, double * d_tnsSum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPOs) return;
    float a = d_poArr[i];
    double e = exp(static_cast<double>(a - poMax) / static_cast<double>(gammaPo));
    atomicAdd(d_wnsSumExp, e);
    double sp = static_cast<double>(softplusF(a - tTarget, gammaPo));
    atomicAdd(d_tnsSum, sp);
}

__global__ void k_po_grad_init(
    int nPOs, int nPIs, bool piPhaseAware,
    float poMax, float gammaPo, float tTarget, float t1, float t2, float depth0, float poNorm,
    double wnsSumExp, const int * d_outs, const float * d_poArr,
    float * d_gArr, float * d_gSlew, float * d_gPiNegArr, float * d_gPiNegSlew) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPOs) return;
    float a = d_poArr[i];
    float wWns = static_cast<float>(exp(static_cast<double>(a - poMax) / static_cast<double>(gammaPo)) / wnsSumExp);
    float dWns = (t2 / depth0) * wWns;
    float dTns = (t1 / (depth0 * poNorm)) * sigmoidF(a - tTarget, gammaPo);
    addLitGrad(d_outs[i], nPIs, piPhaseAware, dWns + dTns, 0.0f, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
}

__global__ void k_aux_loss_terms(
    int M, int nObjs, int nNodes, float lamArea, float lamFo, float fTh, float tauFo, float discLam,
    const int * d_candCount, const int * d_candDeltaAnd, const int * d_foBase,
    const float * d_prob, const float * d_expFo,
    double * d_areaLoss, double * d_foLoss, double * d_discLoss) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double invV0 = 1.0 / static_cast<double>(nNodes > 0 ? nNodes : 1);
    if (tid < M) {
        int K = d_candCount[tid];
        int base = tid * kGradMaxK;
        double ea = 0.0, sumP2 = 0.0;
        for (int k = 0; k < K; ++k) {
            double pk = static_cast<double>(d_prob[base + k]);
            ea += pk * static_cast<double>(d_candDeltaAnd[base + k]);
            sumP2 += pk * pk;
        }
        atomicAdd(d_areaLoss, static_cast<double>(lamArea) * ea * invV0);
        atomicAdd(d_discLoss, static_cast<double>(discLam) * (1.0 - sumP2) * invV0);
    }
    if (lamFo > 0.0f && tid > 0 && tid < nObjs) {
        float bU = fmaxf(static_cast<float>(d_foBase[tid]), fTh);
        float sp = softplusF(d_expFo[tid] - bU, tauFo);
        if (sp > 0.0f) {
            atomicAdd(d_foLoss, static_cast<double>(lamFo) * static_cast<double>(sp) * invV0);
        }
    }
}

// ---------------- Kernel 4: Reverse-Level Arrival & Slew Backprop (Path 1) ----------------

__global__ void k_bwd_level(
    int startId, int count, int nPIs,
    float alpha, float beta, float tauFo, bool piPhaseAware,
    const int * d_f0, const int * d_f1, const int * d_isRegionRoot,
    const int * d_candCount, const int * d_candWireLit, const int * d_candNumGates,
    const SubGate * d_candSubGates, const float * d_prob, const float * d_expFo,
    const float * d_fwdW, const float * d_candFwdW,
    float * d_gArr, float * d_gSlew, float * d_gPiNegArr, float * d_gPiNegSlew) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int v = startId + idx;
    float gA = d_gArr[v];
    float gS = d_gSlew[v];
    if (gA == 0.0f && gS == 0.0f) return;

    int m = d_isRegionRoot[v];
    float ldV = softplusF(d_expFo[v] - 1.0f, tauFo);

    if (m < 0) {
        if (gA == 0.0f) return; // ordinary gate output slew depends only on expFo[v], not on fanin slews
        float wa0 = d_fwdW[v * 2 + 0];
        float ws0 = d_fwdW[v * 2 + 1];
        float wa1 = 1.0f - wa0;
        float ws1 = 1.0f - ws0;
        float cSlew = gA * alpha * beta * ldV;
        addLitGrad(d_f0[v], nPIs, piPhaseAware, gA * wa0, cSlew * ws0, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
        addLitGrad(d_f1[v], nPIs, piPhaseAware, gA * wa1, cSlew * ws1, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
        return;
    }

    int K = d_candCount[m];
    int base = m * kGradMaxK;
    for (int k = 0; k < K; ++k) {
        int slot = base + k;
        float pk = d_prob[slot];
        if (pk == 0.0f) continue;
        int wLit = d_candWireLit[slot];
        if (wLit >= 0) {
            addLitGrad(wLit, nPIs, piPhaseAware, gA * pk, gS * pk, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
            continue;
        }
        if (gA == 0.0f) continue;
        int nG = d_candNumGates[slot];
        float da[kGradMaxSubGates];
        for (int g = 0; g < nG; ++g) da[g] = 0.0f;
        da[nG - 1] = gA * pk;

        int gBase = slot * kGradMaxSubGates;
        for (int g = nG - 1; g >= 0; --g) {
            float dAg = da[g];
            if (dAg == 0.0f) continue;
            SubGate sg = d_candSubGates[gBase + g];
            float wa0 = d_candFwdW[(gBase + g) * 2 + 0];
            float ws0 = d_candFwdW[(gBase + g) * 2 + 1];
            float wa1 = 1.0f - wa0;
            float ws1 = 1.0f - ws0;
            float ldG = (g == nG - 1) ? ldV : sg.load;
            float dS_in = dAg * alpha * beta * ldG;

            if (sg.src0 >= 0) {
                addLitGrad(sg.src0, nPIs, piPhaseAware, dAg * wa0, dS_in * ws0, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
            } else {
                da[-1 - sg.src0] += dAg * wa0;
            }
            if (sg.src1 >= 0) {
                addLitGrad(sg.src1, nPIs, piPhaseAware, dAg * wa1, dS_in * ws1, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
            } else {
                da[-1 - sg.src1] += dAg * wa1;
            }
        }
    }
}

// ---------------- Kernel 5: Exact Fanout Gradient Across All 3 Node Types (Path 2) ----------------

__global__ void k_fanout_grad(
    int nPIs, int nObjs, int nNodes,
    float alpha, float piAlpha, float beta, float lamFo, float fTh, float tauFo,
    const int * d_isRegionRoot, const int * d_candCount, const int * d_candWireLit, const int * d_foBase,
    const float * d_prob, const float * d_expFo, const float * d_slewIn, const float * d_candRootInSlew,
    const float * d_gArr, const float * d_gSlew, const float * d_gPiNegArr, const float * d_gPiNegSlew,
    float * d_gFo) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u <= 0 || u >= nObjs) {
        if (u == 0) d_gFo[0] = 0.0f;
        return;
    }
    float ef = d_expFo[u];
    float sig = sigmoidF(ef - 1.0f, tauFo);
    float rEx = 0.0f;
    if (lamFo > 0.0f) {
        float bU = fmaxf(static_cast<float>(d_foBase[u]), fTh);
        rEx = (lamFo / static_cast<float>(nNodes > 0 ? nNodes : 1)) * sigmoidF(ef - bU, tauFo);
    }

    if (u <= nPIs) {
        d_gFo[u] = sig * (d_gPiNegArr[u] * piAlpha + d_gPiNegSlew[u]) + rEx;
        return;
    }

    int m = d_isRegionRoot[u];
    if (m < 0) {
        d_gFo[u] = sig * (d_gArr[u] * alpha * (1.0f + beta * d_slewIn[u]) + d_gSlew[u]) + rEx;
        return;
    }

    int K = d_candCount[m];
    int base = m * kGradMaxK;
    float gA = d_gArr[u], gS = d_gSlew[u];
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int slot = base + k;
        if (d_candWireLit[slot] >= 0) continue; // wire-fold does not use load(r_m)
        float pk = d_prob[slot];
        acc += pk * (gA * alpha * (1.0f + beta * d_candRootInSlew[slot]) + gS);
    }
    d_gFo[u] = sig * acc + rEx;
}

// ---------------- Kernel 6: Region Gradient Assembly, Softmax Jacobian & Clipped Adam ----------------

__global__ void k_region_grad_and_adam(
    int M, int nNodes, float tauP, float lamArea, float discLam,
    float lr, float stepClip, int stepIdx, bool doUpdate,
    const int * d_regionRoot, const int * d_candCount, const int * d_candDeltaAnd,
    const int * d_foDeltaOffset, const int * d_foDeltaNode, const int * d_foDeltaVal,
    const float * d_prob, const float * d_candArr, const float * d_candOutSlew,
    const float * d_gArr, const float * d_gSlew, const float * d_gFo,
    float * d_logits, float * d_adamM, float * d_adamV, float * d_gLogitsOut) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    int K = d_candCount[m];
    int base = m * kGradMaxK;
    int r = d_regionRoot[m];
    float gAr = d_gArr[r];
    float gSr = d_gSlew[r];
    double invV0 = 1.0 / static_cast<double>(nNodes > 0 ? nNodes : 1);

    double gp[kGradMaxK];
    double dotPG = 0.0;
    for (int k = 0; k < K; ++k) {
        int slot = base + k;
        double pk = static_cast<double>(d_prob[slot]);
        double g1 = static_cast<double>(gAr) * static_cast<double>(d_candArr[slot]) +
                    static_cast<double>(gSr) * static_cast<double>(d_candOutSlew[slot]);
        double g2 = 0.0;
        if (k > 0) {
            int st = d_foDeltaOffset[slot];
            int ed = d_foDeltaOffset[slot + 1];
            for (int e = st; e < ed; ++e) {
                g2 += static_cast<double>(d_gFo[d_foDeltaNode[e]]) * static_cast<double>(d_foDeltaVal[e]);
            }
        }
        double g3 = static_cast<double>(lamArea) * static_cast<double>(d_candDeltaAnd[slot]) * invV0;
        double g4 = -2.0 * static_cast<double>(discLam) * pk * invV0;
        gp[k] = g1 + g2 + g3 + g4;
        dotPG += pk * gp[k];
    }

    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float b1Corr = 1.0f - powf(b1, static_cast<float>(stepIdx));
    float b2Corr = 1.0f - powf(b2, static_cast<float>(stepIdx));

    for (int k = 0; k < K; ++k) {
        int slot = base + k;
        float gz = static_cast<float>((static_cast<double>(d_prob[slot]) / static_cast<double>(tauP)) * (gp[k] - dotPG));
        if (d_gLogitsOut) d_gLogitsOut[slot] = gz;
        if (doUpdate) {
            float mNew = b1 * d_adamM[slot] + (1.0f - b1) * gz;
            float vNew = b2 * d_adamV[slot] + (1.0f - b2) * gz * gz;
            d_adamM[slot] = mNew;
            d_adamV[slot] = vNew;
            float mHat = mNew / b1Corr;
            float vHat = vNew / b2Corr;
            float dz = lr * mHat / (sqrtf(vHat) + eps);
            if (dz > stepClip) dz = stepClip;
            if (dz < -stepClip) dz = -stepClip;
            d_logits[slot] -= dz;
        }
    }
    for (int k = K; k < kGradMaxK; ++k) {
        if (d_gLogitsOut) d_gLogitsOut[base + k] = 0.0f;
    }
}

// ---------------- Host Orchestrator ----------------

GradSolveStats solveJointGradGPU(const GradProblem & prob, const Rw4BParams & params,
                                 std::vector<int> & chosenCand,
                                 std::vector<float> * chosenProb) {
    GradSolveStats stats;
    chosenCand.assign(prob.M, 0);
    if (chosenProb) chosenProb->assign(prob.M, 1.0f);
    if (prob.M <= 0 || prob.nNodes <= 0 || prob.nPOs <= 0) return stats;

    auto tLoop0 = std::chrono::steady_clock::now();

    const int nPIs = prob.nPIs;
    const int nPOs = prob.nPOs;
    const int nObjs = prob.nObjs;
    const int nNodes = prob.nNodes;
    const int maxLv = prob.maxLv;
    const int M = prob.M;
    const int MK = M * kGradMaxK;
    const int nnz = static_cast<int>(prob.foDeltaNode.size());
    const auto & lvOff = *prob.lvOff;

    // Allocate device arrays once per outer round
    int *d_f0 = nullptr, *d_f1 = nullptr, *d_outs = nullptr, *d_foBase = nullptr;
    int *d_regionRoot = nullptr, *d_isRegionRoot = nullptr, *d_candCount = nullptr;
    int *d_candDeltaAnd = nullptr, *d_candWireLit = nullptr, *d_candNumGates = nullptr;
    int *d_foDeltaOffset = nullptr, *d_foDeltaNode = nullptr, *d_foDeltaVal = nullptr;
    int *d_discChoice = nullptr, *d_totalDeltaAnd = nullptr;
    SubGate * d_candSubGates = nullptr;
    float *d_logits = nullptr, *d_prob = nullptr, *d_adamM = nullptr, *d_adamV = nullptr, *d_gLogits = nullptr;
    float *d_expFo = nullptr, *d_arr = nullptr, *d_slew = nullptr, *d_slewIn = nullptr, *d_fwdW = nullptr;
    float *d_candArr = nullptr, *d_candOutSlew = nullptr, *d_candRootInSlew = nullptr, *d_candFwdW = nullptr;
    float *d_gArr = nullptr, *d_gSlew = nullptr, *d_gPiNegArr = nullptr, *d_gPiNegSlew = nullptr, *d_gFo = nullptr;
    float *d_poArr = nullptr;
    double *d_lossBuf = nullptr; // [5]: wnsSumExp, tnsSum, areaLoss, foLoss, discLoss

    CHECK_CUDA(cudaMalloc(&d_f0, nObjs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_f1, nObjs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_outs, nPOs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_foBase, nObjs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_regionRoot, M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_isRegionRoot, nObjs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_candCount, M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_candDeltaAnd, MK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_candWireLit, MK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_candNumGates, MK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_candSubGates, static_cast<size_t>(MK) * kGradMaxSubGates * sizeof(SubGate)));
    CHECK_CUDA(cudaMalloc(&d_foDeltaOffset, (MK + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_foDeltaNode, std::max(1, nnz) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_foDeltaVal, std::max(1, nnz) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_discChoice, M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_totalDeltaAnd, sizeof(int)));

    CHECK_CUDA(cudaMalloc(&d_logits, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_prob, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_adamM, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_adamV, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gLogits, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expFo, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_arr, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_slew, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_slewIn, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fwdW, static_cast<size_t>(nObjs) * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_candArr, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_candOutSlew, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_candRootInSlew, MK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_candFwdW, static_cast<size_t>(MK) * kGradMaxSubGates * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gArr, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gSlew, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gPiNegArr, (nPIs + 1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gPiNegSlew, (nPIs + 1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gFo, nObjs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_poArr, nPOs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lossBuf, 5 * sizeof(double)));

    // Upload static problem buffers
    CHECK_CUDA(cudaMemcpy(d_f0, prob.f0->data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_f1, prob.f1->data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_outs, prob.outs->data(), nPOs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_foBase, prob.foBase.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_regionRoot, prob.regionRoot.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_isRegionRoot, prob.isRegionRoot.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_candCount, prob.candCount.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_candDeltaAnd, prob.candDeltaAnd.data(), MK * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_candWireLit, prob.candWireLit.data(), MK * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_candNumGates, prob.candNumGates.data(), MK * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_candSubGates, prob.candSubGates.data(),
                          static_cast<size_t>(MK) * kGradMaxSubGates * sizeof(SubGate), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_foDeltaOffset, prob.foDeltaOffset.data(), (MK + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz > 0) {
        CHECK_CUDA(cudaMemcpy(d_foDeltaNode, prob.foDeltaNode.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_foDeltaVal, prob.foDeltaVal.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Parameterized warm-start initializer (z_{m,0} = z0Init, others 0).
    auto initWarmLogits = [&](std::vector<float> & zVec, float z0Init) {
        std::fill(zVec.begin(), zVec.end(), 0.0f);
        for (int m = 0; m < M; ++m) {
            zVec[m * kGradMaxK + 0] = z0Init;
        }
    };
    std::vector<float> hLogits(MK, 0.0f);
    initWarmLogits(hLogits, 1.2f);
    CHECK_CUDA(cudaMemcpy(d_logits, hLogits.data(), MK * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_adamM, 0, MK * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_adamV, 0, MK * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_arr, 0, (nPIs + 1) * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_slew, 0, (nPIs + 1) * sizeof(float)));

    const float alpha = params.fanoutAlpha;
    const float piAlpha = params.piFanoutAlpha;
    const float beta = params.slewBeta;
    const bool piPhaseAware = params.piPhaseAware;
    const float depth0 = std::max(prob.depth0, 1.0f);
    const float tau0 = std::max(prob.tau0, 1.0f);
    const float tTarget = params.tnsRho * depth0;
    const float lamArea = 1.0f;
    // Bounded linear smooth net-fanout-increase penalty:
    // L_fo_inc = (lamFo / |V0|) * sum_u SoftPlus_0.1(E[fo(u)] - max(fo_base(u), fTh))
    const float lamFo = params.foIncLam;
    const float fTh = params.foIncTh;

    std::vector<float> hPoArr(nPOs, 0.0f);
    float poNorm = 1.0f;
    float lastPoMax = depth0;

    // Helper: run forward + loss (and optionally backward + Adam update)
    auto runStep = [&](float tauP, float gamma, float gammaS, float gammaPo,
                       float t1, float t2, float discLam, bool doBackward, bool doUpdate,
                       int stepIdx, bool computePoNorm) -> double {
        k_init_exp_fo<<<(nObjs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(nObjs, d_foBase, d_expFo);
        k_softmax_and_exp_fo<<<(M + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
            M, tauP, d_candCount, d_logits, d_foDeltaOffset, d_foDeltaNode, d_foDeltaVal, d_prob, d_expFo);

        for (int l = 1; l <= maxLv; ++l) {
            int cnt = lvOff[l + 1] - lvOff[l];
            if (cnt <= 0) continue;
            k_fwd_level<<<(cnt + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
                lvOff[l], cnt, nPIs, alpha, piAlpha, beta, gamma, gammaS, kTauFo, piPhaseAware,
                d_f0, d_f1, d_isRegionRoot, d_candCount, d_candWireLit, d_candNumGates,
                d_candSubGates, d_prob, d_expFo, d_arr, d_slew, d_slewIn, d_fwdW,
                d_candArr, d_candOutSlew, d_candRootInSlew, d_candFwdW);
        }

        k_extract_po_arr<<<(nPOs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
            nPOs, nPIs, piAlpha, kTauFo, piPhaseAware, false, d_outs, d_expFo, d_arr, d_slew, d_poArr);
        CHECK_CUDA(cudaMemcpy(hPoArr.data(), d_poArr, nPOs * sizeof(float), cudaMemcpyDeviceToHost));
        float poMax = hPoArr[0];
        for (int i = 1; i < nPOs; ++i)
            if (hPoArr[i] > poMax) poMax = hPoArr[i];
        lastPoMax = poMax;

        if (computePoNorm) {
            int cntNear = 0;
            for (int i = 0; i < nPOs; ++i)
                if (hPoArr[i] >= tTarget - gammaPo) cntNear++;
            poNorm = static_cast<float>(std::max(1, cntNear));
        }

        CHECK_CUDA(cudaMemset(d_lossBuf, 0, 5 * sizeof(double)));
        k_po_loss_stats<<<(nPOs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
            nPOs, poMax, gammaPo, tTarget, d_poArr, d_lossBuf + 0, d_lossBuf + 1);
        int maxAux = std::max(M, nObjs);
        k_aux_loss_terms<<<(maxAux + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
            M, nObjs, nNodes, lamArea, lamFo, fTh, kTauFo, discLam,
            d_candCount, d_candDeltaAnd, d_foBase, d_prob, d_expFo,
            d_lossBuf + 2, d_lossBuf + 3, d_lossBuf + 4);

        double hLoss[5];
        CHECK_CUDA(cudaMemcpy(hLoss, d_lossBuf, 5 * sizeof(double), cudaMemcpyDeviceToHost));
        double wnsSumExp = hLoss[0];
        double wns = static_cast<double>(poMax) + static_cast<double>(gammaPo) * std::log(std::max(wnsSumExp, 1e-30));
        double tns = hLoss[1] / static_cast<double>(poNorm);
        double totalLoss = static_cast<double>(t1) * (tns / static_cast<double>(depth0)) +
                           static_cast<double>(t2) * (wns / static_cast<double>(depth0)) +
                           hLoss[2] + hLoss[3] + hLoss[4];

        if (doBackward) {
            CHECK_CUDA(cudaMemset(d_gArr, 0, nObjs * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_gSlew, 0, nObjs * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_gPiNegArr, 0, (nPIs + 1) * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_gPiNegSlew, 0, (nPIs + 1) * sizeof(float)));

            k_po_grad_init<<<(nPOs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
                nPOs, nPIs, piPhaseAware, poMax, gammaPo, tTarget, t1, t2, depth0, poNorm,
                wnsSumExp, d_outs, d_poArr, d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);

            for (int l = maxLv; l >= 1; --l) {
                int cnt = lvOff[l + 1] - lvOff[l];
                if (cnt <= 0) continue;
                k_bwd_level<<<(cnt + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
                    lvOff[l], cnt, nPIs, alpha, beta, kTauFo, piPhaseAware,
                    d_f0, d_f1, d_isRegionRoot, d_candCount, d_candWireLit, d_candNumGates,
                    d_candSubGates, d_prob, d_expFo, d_fwdW, d_candFwdW,
                    d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew);
            }

            k_fanout_grad<<<(nObjs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
                nPIs, nObjs, nNodes, alpha, piAlpha, beta, lamFo, fTh, kTauFo,
                d_isRegionRoot, d_candCount, d_candWireLit, d_foBase, d_prob, d_expFo, d_slewIn, d_candRootInSlew,
                d_gArr, d_gSlew, d_gPiNegArr, d_gPiNegSlew, d_gFo);

            k_region_grad_and_adam<<<(M + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
                M, nNodes, tauP, lamArea, discLam, params.gradLr, kStepClip, stepIdx, doUpdate,
                d_regionRoot, d_candCount, d_candDeltaAnd, d_foDeltaOffset, d_foDeltaNode, d_foDeltaVal,
                d_prob, d_candArr, d_candOutSlew, d_gArr, d_gSlew, d_gFo,
                d_logits, d_adamM, d_adamV, d_gLogits);
        }
        return totalLoss;
    };

    // Initialize poNorm using initial state
    float wTotal = params.adpW;

    // Optional Stage B-1 Hard Gate: Exact Double-Precision Central Finite-Difference Check
    // Compares GPU float32 analytical backward gradients (hAnaGrad) against exact fp64 central FD (eps=1e-6).
    if (params.gradCheck) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < prob.candCount[m]; ++k) {
                hLogits[m * kGradMaxK + k] = 0.15f * static_cast<float>((m * 7 + k * 3) % 5 - 2);
            }
        }
        CHECK_CUDA(cudaMemcpy(d_logits, hLogits.data(), MK * sizeof(float), cudaMemcpyHostToDevice));
        float chkTauP = 0.8f, chkGamma = 1.0f * tau0, chkGammaS = 1.0f, chkDisc = 0.35f;
        float chkT1 = 0.6f * wTotal, chkT2 = 0.4f * wTotal;
        runStep(chkTauP, chkGamma, chkGammaS, chkGamma, chkT1, chkT2, chkDisc, true, false, 1, true);
        std::vector<float> hAnaGrad(MK, 0.0f);
        CHECK_CUDA(cudaMemcpy(hAnaGrad.data(), d_gLogits, MK * sizeof(float), cudaMemcpyDeviceToHost));

        // Double-precision host mirror of exact smooth forward + loss (used ONLY for --grad-check)
        auto softplusD = [](double x, double tau) -> double {
            double z = x / tau;
            if (z > 30.0) return x;
            if (z < -30.0) return 0.0;
            return tau * std::log1p(std::exp(z));
        };
        auto lse2D = [](double a, double b, double g) -> double {
            double mx = a >= b ? a : b;
            return mx + g * std::log(std::exp((a - mx) / g) + std::exp((b - mx) / g));
        };
        std::vector<double> zD(MK);
        for (int i = 0; i < MK; ++i) zD[i] = static_cast<double>(hLogits[i]);

        auto evalLossDouble = [&]() -> double {
            std::vector<double> pD(MK, 0.0), expFoD(nObjs, 0.0), arrD(nObjs, 0.0), slewD(nObjs, 0.0);
            for (int u = 0; u < nObjs; ++u) expFoD[u] = static_cast<double>(prob.foBase[u]);
            for (int m = 0; m < M; ++m) {
                int K = prob.candCount[m], base = m * kGradMaxK;
                double mx = zD[base];
                for (int k = 1; k < K; ++k) if (zD[base + k] > mx) mx = zD[base + k];
                double sE = 0.0;
                for (int k = 0; k < K; ++k) { pD[base + k] = std::exp((zD[base + k] - mx) / chkTauP); sE += pD[base + k]; }
                for (int k = 0; k < K; ++k) {
                    pD[base + k] /= sE;
                    if (k > 0) {
                        for (int e = prob.foDeltaOffset[base + k]; e < prob.foDeltaOffset[base + k + 1]; ++e) {
                            expFoD[prob.foDeltaNode[e]] += static_cast<double>(prob.foDeltaVal[e]) * pD[base + k];
                        }
                    }
                }
            }
            auto readLitD = [&](int lit, double & aO, double & sO) {
                int u = lit >> 1;
                if (u == 0) { aO = 0.0; sO = 0.0; return; }
                if (u <= nPIs) {
                    if (piPhaseAware && (lit & 1)) {
                        double ld = softplusD(expFoD[u] - 1.0, kTauFo);
                        aO = static_cast<double>(piAlpha) * ld;
                        sO = ld;
                    } else { aO = 0.0; sO = 0.0; }
                    return;
                }
                aO = arrD[u]; sO = slewD[u];
            };
            const auto & hf0 = *prob.f0;
            const auto & hf1 = *prob.f1;
            const auto & houts = *prob.outs;
            for (int l = 1; l <= maxLv; ++l) {
                for (int v = lvOff[l]; v < lvOff[l + 1]; ++v) {
                    double ldV = softplusD(expFoD[v] - 1.0, kTauFo);
                    int m = prob.isRegionRoot[v];
                    if (m < 0) {
                        double a0, s0, a1, s1;
                        readLitD(hf0[v], a0, s0); readLitD(hf1[v], a1, s1);
                        double aIn = lse2D(a0, a1, chkGamma), sIn = lse2D(s0, s1, chkGammaS);
                        arrD[v] = aIn + 1.0 + static_cast<double>(alpha) * ldV * (1.0 + static_cast<double>(beta) * sIn);
                        slewD[v] = ldV;
                    } else {
                        int K = prob.candCount[m], base = m * kGradMaxK;
                        double aMix = 0.0, sMix = 0.0;
                        for (int k = 0; k < K; ++k) {
                            int slot = base + k, wLit = prob.candWireLit[slot];
                            double aK, sOutK;
                            if (wLit >= 0) {
                                readLitD(wLit, aK, sOutK);
                            } else {
                                int nG = prob.candNumGates[slot], gBase = slot * kGradMaxSubGates;
                                double gaA[kGradMaxSubGates], gaS[kGradMaxSubGates];
                                for (int g = 0; g < nG; ++g) {
                                    const SubGate & sg = prob.candSubGates[gBase + g];
                                    double a0, s0, a1, s1;
                                    if (sg.src0 >= 0) readLitD(sg.src0, a0, s0);
                                    else { a0 = gaA[-1 - sg.src0]; s0 = gaS[-1 - sg.src0]; }
                                    if (sg.src1 >= 0) readLitD(sg.src1, a1, s1);
                                    else { a1 = gaA[-1 - sg.src1]; s1 = gaS[-1 - sg.src1]; }
                                    double aIn = lse2D(a0, a1, chkGamma), sIn = lse2D(s0, s1, chkGammaS);
                                    double ldG = (g == nG - 1) ? ldV : static_cast<double>(sg.load);
                                    gaA[g] = aIn + 1.0 + static_cast<double>(alpha) * ldG * (1.0 + static_cast<double>(beta) * sIn);
                                    gaS[g] = ldG;
                                }
                                aK = gaA[nG - 1]; sOutK = ldV;
                            }
                            aMix += pD[slot] * aK;
                            sMix += pD[slot] * sOutK;
                        }
                        arrD[v] = aMix; slewD[v] = sMix;
                    }
                }
            }
            std::vector<double> poD(nPOs);
            double poMx = -1e30;
            for (int i = 0; i < nPOs; ++i) {
                double a, s; readLitD(houts[i], a, s); poD[i] = a;
                if (a > poMx) poMx = a;
            }
            double wnsSE = 0.0, tnsS = 0.0;
            for (int i = 0; i < nPOs; ++i) {
                wnsSE += std::exp((poD[i] - poMx) / chkGamma);
                tnsS += softplusD(poD[i] - tTarget, chkGamma);
            }
            double wns = poMx + chkGamma * std::log(std::max(wnsSE, 1e-30));
            double tns = tnsS / static_cast<double>(poNorm);
            double invV0 = 1.0 / static_cast<double>(nNodes);
            double areaL = 0.0, discL = 0.0, foL = 0.0;
            for (int m = 0; m < M; ++m) {
                int K = prob.candCount[m], base = m * kGradMaxK;
                double ea = 0.0, sp2 = 0.0;
                for (int k = 0; k < K; ++k) {
                    ea += pD[base + k] * static_cast<double>(prob.candDeltaAnd[base + k]);
                    sp2 += pD[base + k] * pD[base + k];
                }
                areaL += static_cast<double>(lamArea) * ea * invV0;
                discL += static_cast<double>(chkDisc) * (1.0 - sp2) * invV0;
            }
            if (lamFo > 0.0f) {
                for (int u = 1; u < nObjs; ++u) {
                    double bU = std::max(static_cast<double>(prob.foBase[u]), static_cast<double>(fTh));
                    double sp = softplusD(expFoD[u] - bU, kTauFo);
                    foL += static_cast<double>(lamFo) * sp * invV0;
                }
            }
            return static_cast<double>(chkT1) * (tns / depth0) +
                   static_cast<double>(chkT2) * (wns / depth0) + areaL + foL + discL;
        };

        std::vector<int> sampleM;
        for (int m = 0; m < M && static_cast<int>(sampleM.size()) < 8; ++m) {
            bool hasWire = false;
            for (int k = 1; k < prob.candCount[m]; ++k)
                if (prob.candWireLit[m * kGradMaxK + k] >= 0) hasWire = true;
            if (hasWire) sampleM.push_back(m);
        }
        int stride = std::max(1, M / 32);
        for (int m = 0; m < M && static_cast<int>(sampleM.size()) < 32; m += stride) {
            if (std::find(sampleM.begin(), sampleM.end(), m) == sampleM.end()) sampleM.push_back(m);
        }

        const double epsFD = 1e-6;
        double maxRel = 0.0, maxAbs = 0.0;
        int worstM = -1, worstK = -1;
        double worstAna = 0.0, worstNum = 0.0;
        int tested = 0;
        for (int m : sampleM) {
            for (int k = 0; k < prob.candCount[m]; ++k) {
                int slot = m * kGradMaxK + k;
                double origZ = zD[slot];
                zD[slot] = origZ + epsFD;
                double lp = evalLossDouble();
                zD[slot] = origZ - epsFD;
                double lm = evalLossDouble();
                zD[slot] = origZ;

                double numG = (lp - lm) / (2.0 * epsFD);
                double anaG = static_cast<double>(hAnaGrad[slot]);
                double absErr = std::fabs(anaG - numG);
                double denom = std::max({std::fabs(anaG), std::fabs(numG), 1e-5});
                double relErr = (absErr <= 1e-7) ? 0.0 : (absErr / denom);
                if (relErr > maxRel) {
                    maxRel = relErr;
                    worstM = m; worstK = k; worstAna = anaG; worstNum = numG;
                }
                if (absErr > maxAbs) maxAbs = absErr;
                tested++;
            }
        }
        stats.gradCheckTested = tested;
        stats.gradCheckMaxRelErr = maxRel;
        stats.gradCheckMaxAbsErr = maxAbs;
        stats.gradCheckPassed = (maxRel < 1e-4);
        printf("rw4b-grad CHECK: M=%d tested=%d max_rel_err=%.3e max_abs_err=%.3e (worst m=%d k=%d wire=%d ana=%.6e num=%.6e) [%s]\n",
               M, tested, maxRel, maxAbs, worstM, worstK,
               (worstM >= 0 ? (prob.candWireLit[worstM * kGradMaxK + worstK] >= 0 ? 1 : 0) : 0),
               worstAna, worstNum, (maxRel < 1e-4 ? "PASS<1e-4" : "FAIL"));

        // Reset logits before backtracking warm-start
        initWarmLogits(hLogits, 1.2f);
        CHECK_CUDA(cudaMemcpy(d_logits, hLogits.data(), MK * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Backtracking Warm-Start: probe z_0 in {1.2, 2.0, 2.8, 3.8} with forward-only t=1 pass
    // and pick the smallest z_0 that avoids a t=1 fractional fanout/delay explosion.
    {
        static constexpr float kZ0Ladder[4] = {1.2f, 2.0f, 2.8f, 3.8f};
        float selectedZ0 = kZ0Ladder[3];
        for (float candZ0 : kZ0Ladder) {
            initWarmLogits(hLogits, candZ0);
            CHECK_CUDA(cudaMemcpy(d_logits, hLogits.data(), MK * sizeof(float), cudaMemcpyHostToDevice));
            double probeLoss = runStep(1.0f, 1.5f * tau0, 1.0f, 1.5f * tau0,
                                       0.85f * wTotal, 0.15f * wTotal, 0.0f,
                                       false, false, 0, true);
            selectedZ0 = candZ0;
            if (probeLoss <= 1.15 && lastPoMax <= 1.15f * depth0) {
                break;
            }
        }
        stats.chosenZ0 = selectedZ0;
    }

    // Main Inner Joint Gradient Loop (3-Stage Annealing Schedule)
    // Gini magnet stays strictly OFF for the first 60% of steps so pure timing+area gradients
    // decide winners without no-op pull-back, then ramps linearly over the final 40%.
    const int T = std::max(1, params.gradSteps);
    for (int t = 1; t <= T; ++t) {
        float frac = static_cast<float>(t - 1) / static_cast<float>(std::max(1, T - 1));
        float t1, t2, gamma, gammaS, tauP, discLam;
        if (frac <= 0.40f) {
            // Stage 1: Global negotiation (TNS-heavy, wide LSE, magnet OFF)
            t1 = 0.85f * wTotal;
            t2 = 0.15f * wTotal;
            gamma = 1.5f * tau0;
            gammaS = 1.0f;
            tauP = 1.0f;
            discLam = 0.0f;
        } else if (frac <= 0.60f) {
            // Stage 2a: Bottleneck focus (transition TNS -> WNS, anneal tau, magnet STILL OFF)
            float u = (frac - 0.40f) / 0.20f;
            t1 = (0.85f - 0.45f * u) * wTotal;
            t2 = wTotal - t1;
            gamma = (1.5f - 0.5f * u) * tau0;
            gammaS = 1.0f - 0.3f * u;
            tauP = 1.0f - 0.25f * u;
            discLam = 0.0f;
        } else {
            // Stage 3: Discretization pull-in (WNS-heavy, sharp tau, linear Gini magnet ramp)
            float u = (frac - 0.60f) / 0.40f;
            t1 = (0.40f - 0.15f * u) * wTotal;
            t2 = wTotal - t1;
            gamma = (1.0f - 0.5f * u) * tau0;
            gammaS = 0.7f - 0.3f * u;
            tauP = 0.75f - 0.30f * u;
            discLam = params.discLam * u;
        }
        double loss = runStep(tauP, gamma, gammaS, gamma, t1, t2, discLam, true, true, t, t == 1);
        if (t == 1) stats.initLoss = loss;
        if (t == T) stats.finalLoss = loss;
    }

    // Final probabilities at final tauP=0.45
    k_init_exp_fo<<<(nObjs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(nObjs, d_foBase, d_expFo);
    k_softmax_and_exp_fo<<<(M + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
        M, 0.45f, d_candCount, d_logits, d_foDeltaOffset, d_foDeltaNode, d_foDeltaVal, d_prob, d_expFo);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto tLoop1 = std::chrono::steady_clock::now();
    stats.msGradLoop = std::chrono::duration<double, std::milli>(tLoop1 - tLoop0).count();

    // Compute average max probability for diagnostics
    std::vector<float> hProb(MK, 0.0f);
    CHECK_CUDA(cudaMemcpy(hProb.data(), d_prob, MK * sizeof(float), cudaMemcpyDeviceToHost));
    double sumMaxP = 0.0;
    for (int m = 0; m < M; ++m) {
        float mp = 0.0f;
        for (int k = 0; k < prob.candCount[m]; ++k)
            if (hProb[m * kGradMaxK + k] > mp) mp = hProb[m * kGradMaxK + k];
        sumMaxP += mp;
    }
    stats.avgMaxProb = static_cast<float>(sumMaxP / static_cast<double>(M));

    // Module 5.4: GPU Multi-Threshold Discrete Line Search (8 confidence levels)
    const float thetas[8] = {0.35f, 0.50f, 0.65f, 0.75f, 0.85f, 0.92f, 0.96f, 0.985f};
    float bestAdpDelta = 0.0f; // 0.0 = all no-op baseline (strict monotonic non-degradation)
    float bestTheta = -1.0f;
    float bestDelay = depth0;
    int bestDa = 0;
    std::vector<int> bestChoices(M, 0), tmpChoices(M, 0);

    for (float th : thetas) {
        k_init_exp_fo<<<(nObjs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(nObjs, d_foBase, d_expFo);
        CHECK_CUDA(cudaMemset(d_totalDeltaAnd, 0, sizeof(int)));
        k_discretize_and_exp_fo<<<(M + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
            M, th, d_candCount, d_prob, d_candDeltaAnd, d_foDeltaOffset, d_foDeltaNode, d_foDeltaVal,
            d_discChoice, d_expFo, d_totalDeltaAnd);

        for (int l = 1; l <= maxLv; ++l) {
            int cnt = lvOff[l + 1] - lvOff[l];
            if (cnt <= 0) continue;
            k_fwd_level_hard<<<(cnt + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
                lvOff[l], cnt, nPIs, alpha, piAlpha, beta, piPhaseAware,
                d_f0, d_f1, d_isRegionRoot, d_discChoice, d_candWireLit, d_candNumGates,
                d_candSubGates, d_expFo, d_arr, d_slew);
        }
        k_extract_po_arr<<<(nPOs + kGradBlock - 1) / kGradBlock, kGradBlock>>>(
            nPOs, nPIs, piAlpha, kTauFo, piPhaseAware, true, d_outs, d_expFo, d_arr, d_slew, d_poArr);
        CHECK_CUDA(cudaMemcpy(hPoArr.data(), d_poArr, nPOs * sizeof(float), cudaMemcpyDeviceToHost));
        float dHard = hPoArr[0];
        for (int i = 1; i < nPOs; ++i)
            if (hPoArr[i] > dHard) dHard = hPoArr[i];
        int hDa = 0;
        CHECK_CUDA(cudaMemcpy(&hDa, d_totalDeltaAnd, sizeof(int), cudaMemcpyDeviceToHost));

        float adpDelta = wTotal * ((dHard - depth0) / depth0) +
                         static_cast<float>(hDa) / static_cast<float>(nNodes);
        if (adpDelta < bestAdpDelta - 1e-7f) {
            bestAdpDelta = adpDelta;
            bestTheta = th;
            bestDelay = dHard;
            bestDa = hDa;
            CHECK_CUDA(cudaMemcpy(tmpChoices.data(), d_discChoice, M * sizeof(int), cudaMemcpyDeviceToHost));
            bestChoices.swap(tmpChoices);
        }
    }

    chosenCand.swap(bestChoices);
    if (chosenProb) {
        chosenProb->resize(M);
        for (int m = 0; m < M; ++m) (*chosenProb)[m] = hProb[m * kGradMaxK + chosenCand[m]];
    }
    int nonZero = 0;
    for (int c : chosenCand)
        if (c > 0) nonZero++;

    auto tLoop2 = std::chrono::steady_clock::now();
    stats.msDiscSearch = std::chrono::duration<double, std::milli>(tLoop2 - tLoop1).count();
    stats.bestTheta = bestTheta;
    stats.bestDiscDelay = bestDelay;
    stats.bestDiscDeltaAnd = bestDa;
    stats.bestDiscAdpDelta = bestAdpDelta;
    stats.selectedNonZero = nonZero;

    // Free device buffers
    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_outs); cudaFree(d_foBase);
    cudaFree(d_regionRoot); cudaFree(d_isRegionRoot); cudaFree(d_candCount);
    cudaFree(d_candDeltaAnd); cudaFree(d_candWireLit); cudaFree(d_candNumGates);
    cudaFree(d_candSubGates); cudaFree(d_foDeltaOffset); cudaFree(d_foDeltaNode); cudaFree(d_foDeltaVal);
    cudaFree(d_discChoice); cudaFree(d_totalDeltaAnd);
    cudaFree(d_logits); cudaFree(d_prob); cudaFree(d_adamM); cudaFree(d_adamV); cudaFree(d_gLogits);
    cudaFree(d_expFo); cudaFree(d_arr); cudaFree(d_slew); cudaFree(d_slewIn); cudaFree(d_fwdW);
    cudaFree(d_candArr); cudaFree(d_candOutSlew); cudaFree(d_candRootInSlew); cudaFree(d_candFwdW);
    cudaFree(d_gArr); cudaFree(d_gSlew); cudaFree(d_gPiNegArr); cudaFree(d_gPiNegSlew); cudaFree(d_gFo);
    cudaFree(d_poArr); cudaFree(d_lossBuf);
    return stats;
}

} // namespace rw4b
