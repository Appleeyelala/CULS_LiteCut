#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "aig_manager.h"
#include "algorithms/rewrite.h"
#include "algorithms/rw4b.h"
#include "algorithms/rw4b_grad.h"
#include "common.h"
#include "robin_hood.h"

namespace {
#include "algorithms/rewrite_library.inc"
} // namespace

namespace rw4b {

static constexpr int kCutSetSize = 12;   // slot 0: trivial 1-leaf; slots 1..5: timing-priority; slots 6..11: area-priority
static constexpr int kTimingSlotStart = 1;
static constexpr int kTimingSlotEnd = 6; // [1, 6) -> 5 timing-priority slots
static constexpr int kAreaSlotStart = 6;
static constexpr int kAreaSlotEnd = 12;  // [6, 12) -> 6 native area/sharing slots
static constexpr int kBlockSize = 256;
static constexpr int kMaxMffc = 24;
static constexpr int kMaxCone = 32;
static constexpr int kTopK = 2;
static constexpr float kInf = 1e30f;

struct Cut4B {
    int sign;
    int leaves[4];
    float maxArr;
    float sumArr;
    unsigned short truthtable;
    unsigned short value;
    unsigned char nLeaves;
    unsigned char used;
};

struct CandResult {
    float score;
    float maxArr;
    float req;
    int da;
    int cutSlot;
    int subgr;
};

struct Top3Bounds {
    float s[3];
    int c[3]; // consumer AND node id, -2 for PO, -1 for empty
};

struct TableNode4B {
    int val;
    int next;
};

// ---------------- GPU helpers ----------------

__device__ inline float loadF(int fo) { return fo > 1 ? static_cast<float>(fo - 1) : 0.0f; }
__device__ inline int loadI(int fo) { return fo > 1 ? (fo - 1) : 0; }
__device__ inline int hqI(int x, int pivot) {
    int h = x - pivot;
    return h > 0 ? h * h : 0;
}

__device__ inline void atomicMinFloat(float * addr, float val) {
    int * addr_i = reinterpret_cast<int *>(addr);
    int old = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ inline float edgeArr(int lit, const float * d_arr, const float * d_negExtra, bool piPhaseAware) {
    int n = lit >> 1;
    float a = d_arr[n];
    if (piPhaseAware && (lit & 1) && d_negExtra != nullptr) a += d_negExtra[n];
    return a;
}

__device__ inline float edgeSlew(int lit, int nPIs, const float * d_outSlew, const float * d_negSlew, bool piPhaseAware) {
    int n = lit >> 1;
    if (n <= nPIs) {
        return (piPhaseAware && (lit & 1) && d_negSlew != nullptr) ? d_negSlew[n] : 0.0f;
    }
    return d_outSlew[n];
}

__device__ inline float lse2(float a, float b, float tau) {
    float m = a >= b ? a : b;
    return m + tau * logf(expf((a - m) / tau) + expf((b - m) / tau));
}

// ---------------- GPU timing & sensitivity kernels ----------------

__global__ void countRefsKernel(const int * d_f0, const int * d_f1, const int * d_outs,
                                int nPIs, int nObjs, int nPOs, int * d_fo, int * d_negFo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx < nNodes) {
        int id = nPIs + 1 + idx;
        int l0 = d_f0[id], l1 = d_f1[id];
        atomicAdd(&d_fo[l0 >> 1], 1);
        atomicAdd(&d_fo[l1 >> 1], 1);
        if (l0 & 1) atomicAdd(&d_negFo[l0 >> 1], 1);
        if (l1 & 1) atomicAdd(&d_negFo[l1 >> 1], 1);
    }
    if (idx < nPOs) {
        int l = d_outs[idx];
        atomicAdd(&d_fo[l >> 1], 1);
        if (l & 1) atomicAdd(&d_negFo[l >> 1], 1);
    }
}

__global__ void initPIsKernel(int nPIs, const int * d_fo, const int * d_negFo,
                              float piAlpha, bool piPhaseAware,
                              float * d_arr, float * d_gateDelay, float * d_negExtra,
                              float * d_outSlew, float * d_negSlew) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id == 0) {
        d_arr[0] = 0.0f;
        d_gateDelay[0] = 0.0f;
        d_outSlew[0] = 0.0f;
        if (d_negExtra) d_negExtra[0] = 0.0f;
        if (d_negSlew) d_negSlew[0] = 0.0f;
        return;
    }
    if (id <= nPIs) {
        d_outSlew[id] = 0.0f;
        if (piPhaseAware) {
            float ln = loadF(d_negFo[id]);
            d_gateDelay[id] = 0.0f;
            d_arr[id] = 0.0f;
            d_negExtra[id] = piAlpha * ln;
            d_negSlew[id] = ln;
        } else {
            float l = loadF(d_fo[id]);
            d_gateDelay[id] = piAlpha * l;
            d_arr[id] = piAlpha * l;
            if (d_negExtra) d_negExtra[id] = 0.0f;
            if (d_negSlew) d_negSlew[id] = 0.0f;
        }
    }
}

__global__ void propagateForwardLevelKernel(int startId, int count, int nPIs,
                                            const int * d_f0, const int * d_f1, const int * d_fo,
                                            float alpha, float beta, bool piPhaseAware,
                                            const float * d_negExtra, const float * d_negSlew,
                                            float * d_arr, float * d_gateDelay, float * d_outSlew,
                                            float * d_slewIn, int * d_slewArg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int id = startId + idx;
    int l0 = d_f0[id], l1 = d_f1[id];
    float ld = loadF(d_fo[id]);
    float d;
    if (beta > 0.0f) {
        float s0 = edgeSlew(l0, nPIs, d_outSlew, d_negSlew, piPhaseAware);
        float s1 = edgeSlew(l1, nPIs, d_outSlew, d_negSlew, piPhaseAware);
        float sIn = s0 >= s1 ? s0 : s1;
        d_slewIn[id] = sIn;
        d_slewArg[id] = s0 >= s1 ? 0 : 1;
        d = 1.0f + alpha * ld * (1.0f + beta * sIn);
    } else {
        d_slewIn[id] = 0.0f;
        d_slewArg[id] = -1;
        d = 1.0f + alpha * ld;
    }
    d_outSlew[id] = ld;
    d_gateDelay[id] = d;
    float a0 = edgeArr(l0, d_arr, d_negExtra, piPhaseAware);
    float a1 = edgeArr(l1, d_arr, d_negExtra, piPhaseAware);
    d_arr[id] = d + (a0 >= a1 ? a0 : a1);
}

__global__ void extractPOArrivalsKernel(const int * d_outs, int nPOs,
                                        const float * d_arr, const float * d_negExtra,
                                        bool piPhaseAware, float * d_poArr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPOs) d_poArr[idx] = edgeArr(d_outs[idx], d_arr, d_negExtra, piPhaseAware);
}

__global__ void initReqPOKernel(const int * d_outs, int nPOs, int nPIs, float depth,
                                const float * d_negExtra, bool piPhaseAware,
                                float * d_req, float * d_reqNeg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPOs) return;
    int lit = d_outs[idx];
    int drv = lit >> 1;
    if (piPhaseAware && (lit & 1) && drv <= nPIs && d_reqNeg) {
        atomicMinFloat(&d_reqNeg[drv], depth);
    }
    float extra = (piPhaseAware && (lit & 1) && d_negExtra) ? d_negExtra[drv] : 0.0f;
    atomicMinFloat(&d_req[drv], depth - extra);
}

__global__ void propagateReqLevelKernel(int startId, int count, int nPIs,
                                        const int * d_f0, const int * d_f1,
                                        const float * d_gateDelay, const float * d_negExtra,
                                        bool piPhaseAware, float * d_req, float * d_reqNeg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int id = startId + idx;
    float r = d_req[id];
    if (r >= kInf * 0.5f) return;
    float b = r - d_gateDelay[id];
    int lits[2] = {d_f0[id], d_f1[id]};
    for (int k = 0; k < 2; ++k) {
        int lit = lits[k];
        int pred = lit >> 1;
        if (piPhaseAware && (lit & 1) && pred <= nPIs && d_reqNeg) {
            atomicMinFloat(&d_reqNeg[pred], b);
        }
        float extra = (piPhaseAware && (lit & 1) && d_negExtra) ? d_negExtra[pred] : 0.0f;
        atomicMinFloat(&d_req[pred], b - extra);
    }
}

__global__ void initSensPOKernel(const int * d_outs, int nPOs, float depth, float tau,
                                 const float * d_arr, const float * d_negExtra,
                                 bool piPhaseAware, float * d_sens, float * d_phi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPOs) return;
    int lit = d_outs[idx];
    float a = edgeArr(lit, d_arr, d_negExtra, piPhaseAware);
    float w = expf((a - depth) / tau);
    atomicAdd(&d_sens[lit >> 1], w);
    if (d_phi) atomicAdd(d_phi, w);
}

__global__ void propagateSensLevelKernel(int startId, int count, float tau,
                                         const int * d_f0, const int * d_f1,
                                         const float * d_arr, const float * d_negExtra,
                                         bool piPhaseAware, float * d_sens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int id = startId + idx;
    float s = d_sens[id];
    if (s == 0.0f) return;
    int l0 = d_f0[id], l1 = d_f1[id];
    float a0 = edgeArr(l0, d_arr, d_negExtra, piPhaseAware);
    float a1 = edgeArr(l1, d_arr, d_negExtra, piPhaseAware);
    float m = a0 >= a1 ? a0 : a1;
    float w0 = expf((a0 - m) / tau);
    float w1 = expf((a1 - m) / tau);
    float denom = w0 + w1;
    atomicAdd(&d_sens[l0 >> 1], s * (w0 / denom));
    atomicAdd(&d_sens[l1 >> 1], s * (w1 / denom));
}

__global__ void slewCouplingKernel(int nPIs, int nObjs, float alpha, float beta, bool piPhaseAware,
                                   const int * d_f0, const int * d_f1, const int * d_fo,
                                   const int * d_slewArg, const float * d_sens,
                                   float * d_slewCoef, float * d_slewCoefNeg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx >= nNodes || beta <= 0.0f) return;
    int g = nPIs + 1 + idx;
    float s = d_sens[g];
    if (s == 0.0f) return;
    float ld = loadF(d_fo[g]);
    if (ld == 0.0f) return;
    int arg = d_slewArg[g];
    if (arg < 0) return;
    int lit = arg == 0 ? d_f0[g] : d_f1[g];
    int src = lit >> 1;
    float c = s * alpha * beta * ld;
    if (src <= nPIs) {
        if (piPhaseAware && (lit & 1)) atomicAdd(&d_slewCoefNeg[src], c);
    } else {
        atomicAdd(&d_slewCoef[src], c);
    }
}

__global__ void scatterFanoutsKernel(const int * d_f0, const int * d_f1, const int * d_outs,
                                     int nPIs, int nObjs, int nPOs,
                                     int * d_foCursor, int * d_foEdge) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx < nNodes) {
        int v = nPIs + 1 + idx;
        int l0 = d_f0[v], l1 = d_f1[v];
        int p0 = atomicAdd(&d_foCursor[l0 >> 1], 1);
        d_foEdge[p0] = (v << 1) | (l0 & 1);
        int p1 = atomicAdd(&d_foCursor[l1 >> 1], 1);
        d_foEdge[p1] = (v << 1) | (l1 & 1);
    }
    if (idx < nPOs) {
        int l = d_outs[idx];
        int p = atomicAdd(&d_foCursor[l >> 1], 1);
        d_foEdge[p] = ((nObjs + idx) << 1) | (l & 1);
    }
}

__device__ inline void insertTop3(Top3Bounds & b, float sl, int cid) {
    if (sl < b.s[0]) {
        b.s[2] = b.s[1]; b.c[2] = b.c[1];
        b.s[1] = b.s[0]; b.c[1] = b.c[0];
        b.s[0] = sl;     b.c[0] = cid;
    } else if (sl < b.s[1]) {
        b.s[2] = b.s[1]; b.c[2] = b.c[1];
        b.s[1] = sl;     b.c[1] = cid;
    } else if (sl < b.s[2]) {
        b.s[2] = sl;     b.c[2] = cid;
    }
}

__global__ void computeConsumerBoundsKernel(
    int nPIs, int nObjs, float depth, bool piPhaseAware,
    const int * d_foOff, const int * d_foEdge,
    const float * d_arr, const float * d_req, const float * d_gd, const float * d_negExtra,
    Top3Bounds * d_top3, Top3Bounds * d_negTop3) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u <= 0 || u >= nObjs) return;
    Top3Bounds b = {{kInf, kInf, kInf}, {-1, -1, -1}};
    Top3Bounds nb = {{kInf, kInf, kInf}, {-1, -1, -1}};
    int st = d_foOff[u], ed = d_foOff[u + 1];
    bool isPI = (u <= nPIs);
    for (int e = st; e < ed; ++e) {
        int code = d_foEdge[e];
        int cons = code >> 1;
        int neg = code & 1;
        float ea = edgeArr((u << 1) | neg, d_arr, d_negExtra, piPhaseAware);
        float reqE = (cons >= nObjs) ? depth : (d_req[cons] - d_gd[cons]);
        float sl = reqE - ea;
        int cid = (cons >= nObjs) ? -2 : cons;
        if (isPI && piPhaseAware) {
            if (neg) insertTop3(nb, sl, cid);
        } else {
            insertTop3(b, sl, cid);
        }
    }
    d_top3[u] = b;
    if (isPI) d_negTop3[u] = nb;
}

// ---------------- GPU 4-cut enumeration (adapted from CULS rewrite.cu) ----------------

__device__ inline int cutVal(const Cut4B & c, const int * fo) {
    if (c.nLeaves < 2) return 1001;
    int val = 0, ones = 0;
    for (int i = 0; i < c.nLeaves; ++i) {
        int r = fo[c.leaves[i]];
        val += r;
        ones += (r == 1);
    }
    if (val > 1000) val = 1000;
    if (ones > 3) val = 5 - ones;
    return val;
}

__device__ inline int popcnt32(unsigned x) { return __popc(x); }

__device__ inline void computeCutTiming(Cut4B & c, const float * d_arr) {
    float mx = -1.0f, sm = 0.0f;
    for (int i = 0; i < c.nLeaves; ++i) {
        float a = d_arr[c.leaves[i]];
        if (a > mx) mx = a;
        sm += a;
    }
    c.maxArr = mx;
    c.sumArr = sm;
}

__device__ inline bool timingBetter(const Cut4B & a, const Cut4B & b) {
    if (a.maxArr < b.maxArr - 1e-5f) return true;
    if (a.maxArr > b.maxArr + 1e-5f) return false;
    return a.sumArr < b.sumArr - 1e-5f;
}

__device__ inline int mergeCutOrd(const Cut4B & a, const Cut4B & b, Cut4B * out) {
    int c = 0, i = 0, k = 0;
    for (c = 0; c < 4; ++c) {
        if (k == b.nLeaves) {
            if (i == a.nLeaves) { out->nLeaves = c; return 1; }
            out->leaves[c] = a.leaves[i++];
        } else if (i == a.nLeaves) {
            out->leaves[c] = b.leaves[k++];
        } else if (a.leaves[i] < b.leaves[k]) {
            out->leaves[c] = a.leaves[i++];
        } else if (a.leaves[i] > b.leaves[k]) {
            out->leaves[c] = b.leaves[k++];
        } else {
            out->leaves[c] = a.leaves[i++];
            k++;
        }
    }
    if (i < a.nLeaves || k < b.nLeaves) return 0;
    out->nLeaves = c;
    return 1;
}

__device__ inline int mergeCut(const Cut4B & a, const Cut4B & b, Cut4B * out) {
    int ok = (a.nLeaves >= b.nLeaves) ? mergeCutOrd(a, b, out) : mergeCutOrd(b, a, out);
    if (!ok) return 0;
    out->sign = a.sign | b.sign;
    out->used = 1;
    return 1;
}

__device__ inline int cutDom(const Cut4B & a, const Cut4B & b) {
    for (int i = 0; i < a.nLeaves; ++i) {
        int found = 0;
        for (int j = 0; j < b.nLeaves; ++j)
            if (b.leaves[j] == a.leaves[i]) { found = 1; break; }
        if (!found) return 0;
    }
    return 1;
}

__device__ inline int cutFilterCand(Cut4B * cuts, const Cut4B & cand) {
    for (int i = 0; i < kCutSetSize; ++i) {
        if (!cuts[i].used) continue;
        if (cuts[i].nLeaves > cand.nLeaves) {
            if ((cuts[i].sign & cand.sign) == cand.sign && cutDom(cand, cuts[i]))
                cuts[i].used = 0;
        } else {
            if ((cuts[i].sign & cand.sign) == cuts[i].sign && cutDom(cuts[i], cand))
                return 1; // cand is dominated by existing cut i
        }
    }
    return 0;
}

__device__ inline void insertHybridCut(Cut4B * cuts, Cut4B cand) {
    // Partition 1: Timing-priority slots [kTimingSlotStart, kTimingSlotEnd)
    int freeT = -1, worstT = -1;
    for (int i = kTimingSlotStart; i < kTimingSlotEnd; ++i) {
        if (!cuts[i].used) { freeT = i; break; }
        if (worstT == -1 || timingBetter(cuts[worstT], cuts[i])) worstT = i;
    }
    if (freeT >= 0) {
        cuts[freeT] = cand;
        return;
    }
    if (worstT >= 0 && timingBetter(cand, cuts[worstT])) {
        Cut4B evicted = cuts[worstT];
        cuts[worstT] = cand;
        cand = evicted; // cascade evicted cut to area partition
    }

    // Partition 2: Native area/sharing-priority slots [kAreaSlotStart, kAreaSlotEnd)
    int freeA = -1;
    for (int i = kAreaSlotStart; i < kAreaSlotEnd; ++i) {
        if (!cuts[i].used) { freeA = i; break; }
    }
    if (freeA >= 0) {
        cuts[freeA] = cand;
        return;
    }
    int worstA = -1;
    for (int pass = 0; pass < 3 && worstA == -1; ++pass) {
        for (int i = kAreaSlotStart; i < kAreaSlotEnd; ++i) {
            bool ok = (pass == 0 && cuts[i].nLeaves < 2) ||
                      (pass == 1 && cuts[i].nLeaves == 2) ||
                      (pass == 2 && cuts[i].nLeaves > 2);
            if (ok && (worstA == -1 || cuts[i].value < cuts[worstA].value)) worstA = i;
        }
    }
    if (worstA >= 0 && cand.value > cuts[worstA].value) {
        cuts[worstA] = cand;
    }
}

__device__ inline int ttSwap(int u, int v) {
    if (v == 0) return (u & 0x99999999) | ((u & 0x22222222) << 1) | ((u & 0x44444444) >> 1);
    if (v == 1) return (u & 0xC3C3C3C3) | ((u & 0x0C0C0C0C) << 2) | ((u & 0x30303030) >> 2);
    if (v == 2) return (u & 0xF00FF00F) | ((u & 0x00F000F0) << 4) | ((u & 0x0F000F00) >> 4);
    return u;
}

__device__ inline int cutPhase(const Cut4B & x, const Cut4B & c) {
    int ph = 0;
    for (int i = 0; i < c.nLeaves; ++i)
        for (int j = 0; j < x.nLeaves; ++j)
            if (x.leaves[j] == c.leaves[i]) ph |= (1 << i);
    return ph;
}

__device__ inline int ttStretch(int tt, int nVar, int ph) {
    int v = nVar - 1;
    for (int i = 3; i >= 0; --i) {
        if ((ph >> i) & 1) {
            for (int k = v; k < i; ++k) tt = ttSwap(tt, k);
            v--;
        }
    }
    return tt;
}

__device__ inline int ttShrink(int tt, int ph) {
    int v = 0;
    for (int i = 0; i < 4; ++i) {
        if ((ph >> i) & 1) {
            for (int k = i - 1; k >= v; --k) tt = ttSwap(tt, k);
            v++;
        }
    }
    return tt;
}

__device__ inline int minCutSupport(Cut4B * c) {
    const int m0[4] = {0x5555, 0x3333, 0x0F0F, 0x00FF};
    const int m1[4] = {0xAAAA, 0xCCCC, 0xF0F0, 0xFF00};
    int ph = 0, tt = c->truthtable & 0xFFFF, nl = c->nLeaves;
    for (int i = 0; i < c->nLeaves; ++i) {
        if ((tt & m0[i]) == ((tt & m1[i]) >> (1 << i))) nl--;
        else ph |= (1 << i);
    }
    if (nl == c->nLeaves) return 0;
    c->truthtable = static_cast<unsigned short>(ttShrink(tt, ph) & 0xFFFF);
    c->sign = 0;
    for (int i = 0, k = 0; i < c->nLeaves; ++i) {
        if ((ph >> i) & 1) {
            c->leaves[k++] = c->leaves[i];
            c->sign |= (1 << (c->leaves[i] & 31));
        }
    }
    c->nLeaves = static_cast<unsigned char>(nl);
    return 1;
}

__global__ void initCutInputsKernel(int nPIs, const int * d_fo, const float * d_arr, Cut4B * d_cuts) {
    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (id > nPIs) return;
    Cut4B * c = d_cuts + static_cast<size_t>(id) * kCutSetSize;
    c->used = 1;
    c->sign = 1 << (id & 31);
    c->truthtable = 0xAAAA;
    c->nLeaves = 1;
    c->leaves[0] = id;
    c->value = static_cast<unsigned short>(cutVal(*c, d_fo));
    computeCutTiming(*c, d_arr);
}

__global__ void cutEnumLevelKernel(int startId, int count, const int * d_f0, const int * d_f1,
                                   const int * d_fo, const float * d_arr, Cut4B * d_cuts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int id = startId + idx;
    Cut4B * myCuts = d_cuts + static_cast<size_t>(id) * kCutSetSize;
    myCuts[0].used = 1;
    myCuts[0].sign = 1 << (id & 31);
    myCuts[0].truthtable = 0xAAAA;
    myCuts[0].nLeaves = 1;
    myCuts[0].leaves[0] = id;
    myCuts[0].value = static_cast<unsigned short>(cutVal(myCuts[0], d_fo));
    computeCutTiming(myCuts[0], d_arr);

    int l0 = d_f0[id], l1 = d_f1[id];
    int in0 = l0 >> 1, in1 = l1 >> 1;
    int c0 = l0 & 1, c1 = l1 & 1;
    const Cut4B * cuts0 = d_cuts + static_cast<size_t>(in0) * kCutSetSize;
    const Cut4B * cuts1 = d_cuts + static_cast<size_t>(in1) * kCutSetSize;

    for (int i = 0; i < kCutSetSize; ++i) {
        if (!cuts0[i].used) continue;
        Cut4B a = cuts0[i];
        for (int j = 0; j < kCutSetSize; ++j) {
            if (!cuts1[j].used) continue;
            Cut4B b = cuts1[j];
            if (popcnt32(static_cast<unsigned>(a.sign | b.sign)) > 4) continue;
            Cut4B cand;
            if (!mergeCut(a, b, &cand)) continue;
            int tt0 = c0 ? ~static_cast<int>(a.truthtable) : static_cast<int>(a.truthtable);
            int tt1 = c1 ? ~static_cast<int>(b.truthtable) : static_cast<int>(b.truthtable);
            tt0 = ttStretch(tt0, a.nLeaves, cutPhase(a, cand));
            tt1 = ttStretch(tt1, b.nLeaves, cutPhase(b, cand));
            cand.truthtable = static_cast<unsigned short>((tt0 & tt1) & 0xFFFF);
            minCutSupport(&cand);
            if (cand.nLeaves < 1) continue;
            if (cutFilterCand(myCuts, cand)) continue;
            cand.value = static_cast<unsigned short>(cutVal(cand, d_fo));
            computeCutTiming(cand, d_arr);
            insertHybridCut(myCuts, cand);
            if (cand.nLeaves < 2) return;
        }
    }
}

// ---------------- GPU strash hash table ----------------

__device__ inline int hashBucket(int l0, int l1, int P) {
    if (l0 > l1) { int t = l0; l0 = l1; l1 = t; }
    unsigned long long k = (static_cast<unsigned long long>(l0) * 7937ULL) ^
                           (static_cast<unsigned long long>(l1) * 2971ULL);
    return static_cast<int>(k % static_cast<unsigned long long>(P));
}

__global__ void buildHashKernel(int nPIs, int nObjs, int P,
                                const int * d_f0, const int * d_f1, TableNode4B * d_ht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx >= nNodes) return;
    int id = nPIs + 1 + idx;
    int key = hashBucket(d_f0[id], d_f1[id], P);
    int slot = P + idx;
    d_ht[slot].val = id;
    while (true) {
        int prev = atomicCAS(&d_ht[key].next, -1, slot);
        if (prev == -1) break;
        key = prev;
    }
}

__device__ inline int lookupHash(int l0, int l1, int P, const TableNode4B * ht,
                                 const int * f0, const int * f1) {
    if (l0 > l1) { int t = l0; l0 = l1; l1 = t; }
    int key = hashBucket(l0, l1, P);
    for (int cur = ht[key].next; cur != -1; cur = ht[cur].next) {
        int val = ht[cur].val;
        int a = f0[val], b = f1[val];
        if (a > b) { int t = a; a = b; b = t; }
        if (a == l0 && b == l1) return val;
    }
    return -1;
}

// ---------------- GPU EvaluateNode4B kernel (our 4B_v4_slew core) ----------------

__device__ inline int isLeafIdx(int n, const int leaves[4]) {
    for (int i = 0; i < 4; ++i)
        if (leaves[i] == n) return i;
    return -1;
}

__device__ inline bool inList(int n, const int * arr, int cnt) {
    for (int i = 0; i < cnt; ++i)
        if (arr[i] == n) return true;
    return false;
}

__device__ inline float querySlackOtherDev(
    int u, bool negOnly, int nObjs, float depth, bool piPhaseAware,
    const int * remIds, int remCnt,
    const int * d_foOff, const int * d_foEdge,
    const float * d_arr, const float * d_req, const float * d_gd, const float * d_negExtra,
    const Top3Bounds * d_top3, const Top3Bounds * d_negTop3) {
    const Top3Bounds & b = negOnly ? d_negTop3[u] : d_top3[u];
    for (int k = 0; k < 3; ++k) {
        int cid = b.c[k];
        if (cid == -1) return kInf;
        if (cid == -2 || !inList(cid, remIds, remCnt)) return b.s[k];
    }
    // Exact fallback when all 3 tightest consumers lie inside remIds
    float best = kInf;
    int st = d_foOff[u], ed = d_foOff[u + 1];
    for (int e = st; e < ed; ++e) {
        int code = d_foEdge[e];
        int cons = code >> 1;
        int neg = code & 1;
        if (negOnly && !neg) continue;
        if (cons < nObjs && inList(cons, remIds, remCnt)) continue;
        float ea = edgeArr((u << 1) | neg, d_arr, d_negExtra, piPhaseAware);
        float reqE = (cons >= nObjs) ? depth : (d_req[cons] - d_gd[cons]);
        float sl = reqE - ea;
        if (sl < best) best = sl;
    }
    return best;
}

__global__ void evaluateNode4BKernel(
    int nPIs, int nObjs, int P, float depth, float phi,
    const int * d_f0, const int * d_f1, const int * d_fo, const int * d_negFo,
    const int * d_foOff, const int * d_foEdge,
    const float * d_arr, const float * d_req, const float * d_gateDelay,
    const float * d_negExtra, const float * d_outSlew, const float * d_negSlew,
    const float * d_slewIn, const float * d_sens, const float * d_slewCoef, const float * d_slewCoefNeg,
    const Top3Bounds * d_top3, const Top3Bounds * d_negTop3,
    const Cut4B * d_cuts, const rw::Library * lib, const TableNode4B * d_ht,
    float alpha, float piAlpha, float beta, float lamArea, float regW, int pivot, float tau,
    float budgetTol, bool piPhaseAware, bool strashAware, bool fanoutBudget,
    bool adpScore, float adpW, float slackKappa, int foCap,
    CandResult * d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx >= nNodes) return;
    int root = nPIs + 1 + idx;
    d_out[root * kTopK + 0] = {0.0f, 0.0f, 0.0f, 0, -1, -1};
    d_out[root * kTopK + 1] = {0.0f, 0.0f, 0.0f, 0, -1, -1};

    float req = d_req[root];
    if (req >= kInf * 0.5f) return;
    float sRoot = d_sens[root];
    float nodeLv = d_arr[root];
    int rootFo = d_fo[root];
    float slackRoot = req - nodeLv;
    bool isNearCrit = (slackKappa <= 0.0f) ? true : (slackRoot <= slackKappa * tau + 1e-5f);
    bool allowDupHit = (foCap > 0 && isNearCrit);

    const Cut4B * myCuts = d_cuts + static_cast<size_t>(root) * kCutSetSize;
    CandResult topCand[kTopK] = {
        {0.0f, 0.0f, req, 0, -1, -1},
        {0.0f, 0.0f, req, 0, -1, -1}
    };

    int remIds[kMaxMffc];
    int trkIds[kMaxMffc], trkRem[kMaxMffc];
    int coneIds[kMaxCone];
    int match[54];
    float cvMax[54], cvLse[54], cvSlew[54];

    for (int slot = 0; slot < kCutSetSize; ++slot) {
        Cut4B cut = myCuts[slot];
        if (!cut.used || cut.nLeaves < 3) continue;
        if (cut.nLeaves == 3) cut.leaves[3] = 0;

        int uPhase = lib->pPhases[cut.truthtable];
        int cls = lib->pMap[cut.truthtable];
        const int * perm = lib->pPerms4[lib->pPerms[cut.truthtable]];
        if (lib->nSubgr[cls] == 0) continue;

        int leaves[4];
        int leavesCompl[4];
        for (int j = 0; j < 4; ++j) {
            leaves[j] = cut.leaves[perm[j]];
            leavesCompl[j] = (uPhase >> j) & 1;
        }

        // Bounded MFFC discovery (separate trkIds/trkRem so remIds never overwrites tracked entries)
        int remCnt = 0;
        int tracked = 0;
        remIds[remCnt++] = root;
        int qHead = 0;
        while (qHead < remCnt) {
            int u = remIds[qHead++];
            int lits[2] = {d_f0[u], d_f1[u]};
            for (int k = 0; k < 2; ++k) {
                int src = lits[k] >> 1;
                if (src <= nPIs || isLeafIdx(src, leaves) >= 0) continue;
                int pos = -1;
                for (int t = 0; t < tracked; ++t) {
                    if (trkIds[t] == src) { pos = t; break; }
                }
                int r;
                if (pos >= 0) {
                    r = --trkRem[pos];
                } else {
                    if (tracked >= kMaxMffc) continue;
                    trkIds[tracked] = src;
                    r = d_fo[src] - 1;
                    trkRem[tracked] = r;
                    pos = tracked++;
                }
                if (r == 0 && remCnt < kMaxMffc && !inList(src, remIds, remCnt)) {
                    trkIds[pos] = trkIds[tracked - 1];
                    trkRem[pos] = trkRem[tracked - 1];
                    tracked--;
                    remIds[remCnt++] = src;
                }
            }
        }
        int saved = remCnt;

        // Sort remIds descending for exact topo backward distance & local_sens
        for (int a = 1; a < remCnt; ++a) {
            int key = remIds[a], b = a - 1;
            while (b >= 0 && remIds[b] < key) { remIds[b + 1] = remIds[b]; --b; }
            remIds[b + 1] = key;
        }

        // 1. Compute orig_refs and orig_refs_neg over MFFC (remIds)
        int origRefs[4] = {0, 0, 0, 0};
        int origRefsNeg[4] = {0, 0, 0, 0};
        for (int a = 0; a < remCnt; ++a) {
            int u = remIds[a];
            int lits[2] = {d_f0[u], d_f1[u]};
            for (int k = 0; k < 2; ++k) {
                int lit = lits[k];
                int li = isLeafIdx(lit >> 1, leaves);
                if (li >= 0) {
                    origRefs[li]++;
                    if (lit & 1) origRefsNeg[li]++;
                }
            }
        }

        // 2. Discover FULL 4-cut interior cone (including shared multi-fanout nodes!) for exact jOrig & local_sens
        int coneCnt = 0;
        coneIds[coneCnt++] = root;
        int cHead = 0;
        while (cHead < coneCnt) {
            int u = coneIds[cHead++];
            int lits[2] = {d_f0[u], d_f1[u]};
            for (int k = 0; k < 2; ++k) {
                int pred = lits[k] >> 1;
                if (pred <= nPIs || isLeafIdx(pred, leaves) >= 0) continue;
                if (coneCnt < kMaxCone && !inList(pred, coneIds, coneCnt)) {
                    coneIds[coneCnt++] = pred;
                }
            }
        }
        // Sort coneIds descending for exact topological backward distance & local_sens
        for (int a = 1; a < coneCnt; ++a) {
            int key = coneIds[a], b = a - 1;
            while (b >= 0 && coneIds[b] < key) { coneIds[b + 1] = coneIds[b]; --b; }
            coneIds[b + 1] = key;
        }

        float coneDist[kMaxCone], coneSens[kMaxCone];
        for (int a = 0; a < coneCnt; ++a) { coneDist[a] = -1.0f; coneSens[a] = 0.0f; }
        coneDist[0] = d_gateDelay[root];
        coneSens[0] = sRoot;
        float leafDelays[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        float leafLocalSens[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int a = 0; a < coneCnt; ++a) {
            int u = coneIds[a];
            float du = coneDist[a];
            float su = coneSens[a];
            int l0 = d_f0[u], l1 = d_f1[u];
            float a0 = edgeArr(l0, d_arr, d_negExtra, piPhaseAware);
            float a1 = edgeArr(l1, d_arr, d_negExtra, piPhaseAware);
            float m = a0 >= a1 ? a0 : a1;
            float w0 = expf((a0 - m) / tau), w1 = expf((a1 - m) / tau);
            float denom = w0 + w1;
            int lits[2] = {l0, l1};
            float ws[2] = {w0 / denom, w1 / denom};
            for (int k = 0; k < 2; ++k) {
                int lit = lits[k];
                int pred = lit >> 1;
                float extra = (piPhaseAware && (lit & 1) && d_negExtra) ? d_negExtra[pred] : 0.0f;
                int li = isLeafIdx(pred, leaves);
                if (li >= 0) {
                    if (du >= 0.0f) {
                        float dp = du + extra;
                        if (dp > leafDelays[li]) leafDelays[li] = dp;
                    }
                    leafLocalSens[li] += su * ws[k];
                } else {
                    for (int b = a + 1; b < coneCnt; ++b) {
                        if (coneIds[b] == pred) {
                            if (du >= 0.0f) {
                                float dp = du + d_gateDelay[pred] + extra;
                                if (dp > coneDist[b]) coneDist[b] = dp;
                            }
                            coneSens[b] += su * ws[k];
                            break;
                        }
                    }
                }
            }
        }

        // Compute j_orig (LSE over used leaves)
        float maxOrigRel = -kInf;
        int usedOrig = 0;
        float origRel[4];
        for (int i = 0; i < 4; ++i) {
            if (leafDelays[i] >= 0.0f) {
                origRel[usedOrig] = d_arr[leaves[i]] + leafDelays[i] - nodeLv;
                if (origRel[usedOrig] > maxOrigRel) maxOrigRel = origRel[usedOrig];
                usedOrig++;
            }
        }
        float jOrig = -nodeLv;
        if (usedOrig > 0) {
            float sumE = 0.0f;
            for (int i = 0; i < usedOrig; ++i) sumE += expf((origRel[i] - maxOrigRel) / tau);
            jOrig = maxOrigRel + tau * logf(sumE);
        }
        float sensOther[4];
        for (int i = 0; i < 4; ++i) {
            float s = d_sens[leaves[i]] - leafLocalSens[i];
            sensOther[i] = s > 0.0f ? s : 0.0f;
        }

        // Strash-aware match across NPN library class
        uint64_t isC = 0;
        for (int j = 0; j < 54; ++j) match[j] = -1;
        for (int j = 0; j < 4; ++j) {
            match[j] = leaves[j];
            if (leavesCompl[j]) isC |= (1ULL << j);
        }
        int nLibNodes = lib->nNodes[cls];
        for (int j = 0; j < nLibNodes; ++j) {
            int num = j + 4;
            int in0 = lib->fanin0[cls][j], in1 = lib->fanin1[cls][j];
            if (!strashAware) continue;
            if (match[in0] == -1 || match[in1] == -1 || match[in0] == root || match[in1] == root) continue;
            int c0 = ((isC >> in0) & 1) ^ lib->isC0[cls][j];
            int c1 = ((isC >> in1) & 1) ^ lib->isC1[cls][j];
            int lit0 = (match[in0] << 1) | c0;
            int lit1 = (match[in1] << 1) | c1;
            int hit = lookupHash(lit0, lit1, P, d_ht, d_f0, d_f1);
            if (hit != -1 && hit < root && !inList(hit, remIds, remCnt)) {
                // P0-2: if near-critical and hit already at/above foCap, treat as new gate to align with rebuild
                if (!(allowDupHit && d_fo[hit] >= foCap)) {
                    match[num] = hit;
                }
            }
        }

        // Evaluate all subgraphs in this NPN class (keep best per cut, then Top-K across cuts)
        float cutBestScore = 0.0f;
        CandResult cutBest = {0.0f, 0.0f, req, 0, -1, -1};
        int nSubgr = lib->nSubgr[cls];
        for (int out = 0; out < nSubgr; ++out) {
            int rt = lib->pSubgr[cls][out];
            if (match[rt] == root) continue; // self

            uint64_t usedMask = 1ULL << rt;
            for (int j = rt; j >= 4; --j) {
                if ((usedMask >> j) & 1ULL) {
                    if (match[j] != -1) continue; // reused existing node stops expansion
                    usedMask |= (1ULL << lib->fanin0[cls][j - 4]);
                    usedMask |= (1ULL << lib->fanin1[cls][j - 4]);
                }
            }

            int nodesAdded = 0;
            int candRefs[4] = {0, 0, 0, 0};
            int candRefsNeg[4] = {0, 0, 0, 0};
            int eFo[54];
            for (int j = 4; j <= rt; ++j) eFo[j] = 0;

            int hitIds[8], hitDelta[8];
            int hitCnt = 0;
            auto addHit = [&](int u, int d) {
                if (u <= 0) return;
                for (int t = 0; t < hitCnt; ++t) {
                    if (hitIds[t] == u) { hitDelta[t] += d; return; }
                }
                if (hitCnt < 8) { hitIds[hitCnt] = u; hitDelta[hitCnt] = d; hitCnt++; }
            };

            if (rt < 4) {
                // Wire directly to leaf rt (conservative negative-phase accounting for PI)
                candRefs[rt] += rootFo;
                if (piPhaseAware && leaves[rt] > 0 && leaves[rt] <= nPIs) {
                    int rNeg = d_negFo[root];
                    int rPos = rootFo - rNeg;
                    candRefsNeg[rt] += (rNeg > rPos ? rNeg : rPos);
                }
            } else if (match[rt] != -1) {
                addHit(match[rt], rootFo);
            } else {
                for (int j = 4; j <= rt; ++j) {
                    if (!((usedMask >> j) & 1ULL) || match[j] != -1) continue;
                    nodesAdded++;
                    int in0 = lib->fanin0[cls][j - 4], in1 = lib->fanin1[cls][j - 4];
                    int c0 = lib->isC0[cls][j - 4], c1 = lib->isC1[cls][j - 4];
                    int ins[2] = {in0, in1};
                    int cs[2] = {c0, c1};
                    for (int k = 0; k < 2; ++k) {
                        int en = ins[k];
                        if (en < 4) {
                            candRefs[en]++;
                            if (leavesCompl[en] ^ cs[k]) candRefsNeg[en]++;
                        } else if (match[en] != -1) {
                            addHit(match[en], 1);
                        } else {
                            eFo[en]++;
                        }
                    }
                }
            }

            // Compute leaf fanout delta costs + exact slack_other hard budget guard
            float leafDd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int leafDl[4] = {0, 0, 0, 0};
            float side = 0.0f, reg = 0.0f, slewCost = 0.0f;
            bool budgetOk = true;

            for (int i = 0; i < 4; ++i) {
                int lf = leaves[i];
                if (lf <= 0) continue;
                bool isPI = (lf <= nPIs);
                bool negOnly = (isPI && piPhaseAware);
                int dfo = negOnly ? (candRefsNeg[i] - origRefsNeg[i]) : (candRefs[i] - origRefs[i]);
                if (dfo == 0) continue;
                float coef = isPI ? piAlpha : (beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[lf]) : alpha);
                if (coef <= 0.0f) continue;
                int baseCnt = negOnly ? d_negFo[lf] : d_fo[lf];
                int newCnt = baseCnt + dfo;
                if (newCnt < 0) newCnt = 0;
                int dl = loadI(newCnt) - loadI(baseCnt);
                float dd = coef * static_cast<float>(dl);
                if (fanoutBudget && dd > 1e-6f && !(newCnt <= pivot && dd <= 0.5f)) {
                    float so = querySlackOtherDev(
                        lf, negOnly, nObjs, depth, piPhaseAware, remIds, remCnt,
                        d_foOff, d_foEdge, d_arr, d_req, d_gateDelay, d_negExtra, d_top3, d_negTop3);
                    if (so < kInf * 0.5f && dd > (so > 0.0f ? so : 0.0f) + budgetTol) {
                        budgetOk = false;
                        break;
                    }
                }
                leafDd[i] = dd;
                leafDl[i] = dl;
                side += sensOther[i] * dd;
                if (regW > 0.0f) reg += regW * static_cast<float>(hqI(newCnt, pivot) - hqI(baseCnt, pivot));
                if (beta > 0.0f && dl != 0) {
                    float sc = negOnly ? d_slewCoefNeg[lf] : d_slewCoef[lf];
                    if (sc > 0.0f) slewCost += sc * static_cast<float>(dl);
                }
            }
            if (!budgetOk) continue;
            for (int t = 0; t < hitCnt; ++t) {
                int u = hitIds[t], dfo = hitDelta[t];
                if (dfo == 0 || u <= nPIs) continue;
                float coef = beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[u]) : alpha;
                int baseCnt = d_fo[u];
                int newCnt = baseCnt + dfo;
                if (newCnt < 0) newCnt = 0;
                int dl = loadI(newCnt) - loadI(baseCnt);
                float dd = coef * static_cast<float>(dl);
                if (fanoutBudget && dd > 1e-6f && !(newCnt <= pivot && dd <= 0.5f)) {
                    float so = querySlackOtherDev(
                        u, false, nObjs, depth, piPhaseAware, remIds, remCnt,
                        d_foOff, d_foEdge, d_arr, d_req, d_gateDelay, d_negExtra, d_top3, d_negTop3);
                    if (so < kInf * 0.5f && dd > (so > 0.0f ? so : 0.0f) + budgetTol) {
                        budgetOk = false;
                        break;
                    }
                }
                side += d_sens[u] * dd;
                if (regW > 0.0f) reg += regW * static_cast<float>(hqI(newCnt, pivot) - hqI(baseCnt, pivot));
                if (beta > 0.0f && dl != 0 && d_slewCoef[u] > 0.0f) slewCost += d_slewCoef[u] * static_cast<float>(dl);
            }
            if (!budgetOk) continue;

            // Propagate candidate timing
            for (int i = 0; i < 4; ++i) {
                int lf = leaves[i];
                if (lf == 0) {
                    cvMax[i] = 0.0f; cvLse[i] = -nodeLv; cvSlew[i] = 0.0f;
                } else {
                    float a = d_arr[lf] + ((lf <= nPIs && piPhaseAware) ? 0.0f : leafDd[i]);
                    cvMax[i] = a;
                    cvLse[i] = a - nodeLv;
                    cvSlew[i] = (lf <= nPIs) ? 0.0f : (d_outSlew[lf] + static_cast<float>(leafDl[i]));
                }
            }

            auto getIn = [&](int en, int ec, float & aOut, float & rOut, float & sOut) {
                if (en < 4) {
                    int lf = leaves[en];
                    if (lf == 0) { aOut = 0.0f; rOut = -nodeLv; sOut = 0.0f; return; }
                    if (lf <= nPIs && piPhaseAware) {
                        int ph = leavesCompl[en] ^ ec;
                        aOut = d_arr[lf] + (ph ? (d_negExtra[lf] + leafDd[en]) : 0.0f);
                        rOut = aOut - nodeLv;
                        sOut = ph ? (d_negSlew[lf] + static_cast<float>(leafDl[en])) : 0.0f;
                        return;
                    }
                    aOut = cvMax[en]; rOut = cvLse[en]; sOut = cvSlew[en];
                    return;
                }
                if (match[en] != -1) {
                    int u = match[en];
                    int dlU = 0;
                    for (int t = 0; t < hitCnt; ++t) if (hitIds[t] == u) { dlU = loadI(d_fo[u] + hitDelta[t]) - loadI(d_fo[u]); break; }
                    float coef = beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[u]) : alpha;
                    aOut = d_arr[u] + coef * static_cast<float>(dlU);
                    rOut = aOut - nodeLv;
                    sOut = d_outSlew[u] + static_cast<float>(dlU);
                    return;
                }
                aOut = cvMax[en]; rOut = cvLse[en]; sOut = cvSlew[en];
            };

            float maxArrAbs, jCand;
            if (rt < 4) {
                float sTmp;
                getIn(rt, 0, maxArrAbs, jCand, sTmp);
            } else if (match[rt] != -1) {
                float sTmp;
                getIn(rt, 0, maxArrAbs, jCand, sTmp);
            } else {
                for (int j = 4; j <= rt; ++j) {
                    if (!((usedMask >> j) & 1ULL) || match[j] != -1) continue;
                    float a0, r0, s0, a1, r1, s1;
                    getIn(lib->fanin0[cls][j - 4], lib->isC0[cls][j - 4], a0, r0, s0);
                    getIn(lib->fanin1[cls][j - 4], lib->isC1[cls][j - 4], a1, r1, s1);
                    float sIn = s0 >= s1 ? s0 : s1;
                    float ld = (j == rt) ? loadF(rootFo) : loadF(eFo[j]);
                    float gd = (beta > 0.0f) ? (1.0f + alpha * ld * (1.0f + beta * sIn))
                                             : ((j == rt) ? d_gateDelay[root] : (1.0f + alpha * ld));
                    cvMax[j] = gd + (a0 >= a1 ? a0 : a1);
                    cvLse[j] = gd + lse2(r0, r1, tau);
                    cvSlew[j] = ld;
                }
                maxArrAbs = cvMax[rt];
                jCand = cvLse[rt];
            }

            if (maxArrAbs > req + 1e-5f) continue;
            int da = nodesAdded - saved;
            float deltaJ = jOrig - jCand;
            float rawTiming = sRoot * deltaJ - side - slewCost;
            float effDelta = rawTiming - reg;
            float score;
            float minScore = adpScore ? 1e-9f : 1e-6f;
            if (adpScore) {
                // P0-3: hard near-critical slack gate for area-increasing rewrites
                if (da > 0 && !isNearCrit) continue;
                float dD = rawTiming / phi;
                float invDepth = 1.0f / fmaxf(depth, 1.0f);
                float invArea = 1.0f / fmaxf(static_cast<float>(nNodes), 1.0f);
                if (da <= 0) {
                    if (deltaJ < -1e-5f || effDelta < -1e-5f) continue;
                    if (da == 0 && effDelta <= 1e-5f) continue; // avoid float32 zero-gain churn
                    score = adpW * (dD * invDepth) - (reg * invDepth) + static_cast<float>(-da) * invArea;
                } else {
                    score = adpW * (dD * invDepth) - (reg * invDepth) - static_cast<float>(da) * invArea;
                }
            } else {
                if (da <= 0) {
                    if (deltaJ < -1e-5f || effDelta < -1e-5f) continue;
                    if (da == 0 && effDelta <= 1e-5f) continue; // avoid float32 zero-gain churn
                    score = effDelta + 1e-4f * static_cast<float>(-da);
                } else {
                    score = effDelta - lamArea * static_cast<float>(da);
                }
            }
            if (score > cutBestScore && score > minScore) {
                cutBestScore = score;
                cutBest = {score, maxArrAbs, req, da, slot, out};
            }
        }
        if (cutBest.cutSlot >= 0) {
            if (cutBest.score > topCand[0].score) {
                topCand[1] = topCand[0];
                topCand[0] = cutBest;
            } else if (cutBest.score > topCand[1].score) {
                topCand[1] = cutBest;
            }
        }
    }
    d_out[root * kTopK + 0] = topCand[0];
    d_out[root * kTopK + 1] = topCand[1];
}

// ---------------- Stage B: GPU Stratified Pareto Menu Extraction Kernel ----------------

struct MenuSlot {
    float score;
    float deltaJ;
    float posFoInc;
    int da;
    int cutSlot;
    int subgr;
};

__global__ void evaluateNode4BMenuKernel(
    int nPIs, int nObjs, int P, float depth, float phi,
    const int * d_f0, const int * d_f1, const int * d_fo, const int * d_negFo,
    const int * d_foOff, const int * d_foEdge, const int * d_nodeLv,
    const float * d_arr, const float * d_req, const float * d_gateDelay,
    const float * d_negExtra, const float * d_outSlew, const float * d_negSlew,
    const float * d_slewIn, const float * d_sens, const float * d_slewCoef, const float * d_slewCoefNeg,
    const Top3Bounds * d_top3, const Top3Bounds * d_negTop3,
    const Cut4B * d_cuts, const rw::Library * lib, const TableNode4B * d_ht,
    float alpha, float piAlpha, float beta, float regW, int pivot, float tau,
    float budgetTol, bool piPhaseAware, bool strashAware, bool fanoutBudget,
    float adpW, float tnsRho,
    MenuSlot * d_menuOut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx >= nNodes) return;
    int root = nPIs + 1 + idx;
    for (int s = 0; s < 4; ++s) {
        d_menuOut[root * 4 + s] = {-kInf, -kInf, kInf, 999999, -1, -1};
    }

    float req = d_req[root];
    if (req >= kInf * 0.5f) return;
    float sRoot = d_sens[root];
    float nodeLv = d_arr[root];
    int rootFo = d_fo[root];
    int rootUnitLv = d_nodeLv[root];
    float slackRoot = req - nodeLv;
    float poolSlackLimit = fmaxf(3.0f * tau, (1.0f - tnsRho) * depth);
    bool inVision = (slackRoot <= poolSlackLimit + 1e-5f);

    const Cut4B * myCuts = d_cuts + static_cast<size_t>(root) * kCutSetSize;
    MenuSlot bestSpeed = {-kInf, -kInf, kInf, 999999, -1, -1};
    MenuSlot bestAdp   = {-kInf, -kInf, kInf, 999999, -1, -1};
    MenuSlot bestArea  = {-kInf, -kInf, kInf, 999999, -1, -1};
    MenuSlot bestLowFo = {-kInf, -kInf, kInf, 999999, -1, -1};

    int remIds[kMaxMffc], trkIds[kMaxMffc], trkRem[kMaxMffc], coneIds[kMaxCone], match[54];
    float cvMax[54], cvLse[54], cvSlew[54];
    float invDepth = 1.0f / fmaxf(depth, 1.0f);
    float invArea = 1.0f / fmaxf(static_cast<float>(nNodes), 1.0f);

    for (int slot = 0; slot < kCutSetSize; ++slot) {
        Cut4B cut = myCuts[slot];
        if (!cut.used || cut.nLeaves < 3) continue;
        if (cut.nLeaves == 3) cut.leaves[3] = 0;

        int uPhase = lib->pPhases[cut.truthtable];
        int cls = lib->pMap[cut.truthtable];
        const int * perm = lib->pPerms4[lib->pPerms[cut.truthtable]];
        if (lib->nSubgr[cls] == 0) continue;

        int leaves[4], leavesCompl[4];
        for (int j = 0; j < 4; ++j) {
            leaves[j] = cut.leaves[perm[j]];
            leavesCompl[j] = (uPhase >> j) & 1;
        }

        // Bounded MFFC discovery
        int remCnt = 0, tracked = 0;
        remIds[remCnt++] = root;
        int qHead = 0;
        while (qHead < remCnt) {
            int u = remIds[qHead++];
            int lits[2] = {d_f0[u], d_f1[u]};
            for (int k = 0; k < 2; ++k) {
                int src = lits[k] >> 1;
                if (src <= nPIs || isLeafIdx(src, leaves) >= 0) continue;
                int pos = -1;
                for (int t = 0; t < tracked; ++t) {
                    if (trkIds[t] == src) { pos = t; break; }
                }
                int r;
                if (pos >= 0) {
                    r = --trkRem[pos];
                } else {
                    if (tracked >= kMaxMffc) continue;
                    trkIds[tracked] = src;
                    r = d_fo[src] - 1;
                    trkRem[tracked] = r;
                    pos = tracked++;
                }
                if (r == 0 && remCnt < kMaxMffc && !inList(src, remIds, remCnt)) {
                    trkIds[pos] = trkIds[tracked - 1];
                    trkRem[pos] = trkRem[tracked - 1];
                    tracked--;
                    remIds[remCnt++] = src;
                }
            }
        }
        int saved = remCnt;

        int origRefs[4] = {0, 0, 0, 0}, origRefsNeg[4] = {0, 0, 0, 0};
        for (int a = 0; a < remCnt; ++a) {
            int u = remIds[a];
            int lits[2] = {d_f0[u], d_f1[u]};
            for (int k = 0; k < 2; ++k) {
                int li = isLeafIdx(lits[k] >> 1, leaves);
                if (li >= 0) {
                    origRefs[li]++;
                    if (lits[k] & 1) origRefsNeg[li]++;
                }
            }
        }

        // Full 4-cut interior cone (coneIds) for leafDelays, jOrig, leafLocalSens
        int coneCnt = 0;
        coneIds[coneCnt++] = root;
        int cHead = 0;
        while (cHead < coneCnt) {
            int u = coneIds[cHead++];
            int lits[2] = {d_f0[u], d_f1[u]};
            for (int k = 0; k < 2; ++k) {
                int pred = lits[k] >> 1;
                if (pred <= nPIs || isLeafIdx(pred, leaves) >= 0) continue;
                if (coneCnt < kMaxCone && !inList(pred, coneIds, coneCnt)) coneIds[coneCnt++] = pred;
            }
        }
        for (int a = 1; a < coneCnt; ++a) {
            int key = coneIds[a], b = a - 1;
            while (b >= 0 && coneIds[b] < key) { coneIds[b + 1] = coneIds[b]; --b; }
            coneIds[b + 1] = key;
        }
        float coneDist[kMaxCone], coneSens[kMaxCone];
        for (int a = 0; a < coneCnt; ++a) { coneDist[a] = -1.0f; coneSens[a] = 0.0f; }
        coneDist[0] = d_gateDelay[root];
        coneSens[0] = sRoot;
        float leafDelays[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        float leafLocalSens[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int a = 0; a < coneCnt; ++a) {
            int u = coneIds[a];
            float du = coneDist[a], su = coneSens[a];
            int l0 = d_f0[u], l1 = d_f1[u];
            float a0 = edgeArr(l0, d_arr, d_negExtra, piPhaseAware);
            float a1 = edgeArr(l1, d_arr, d_negExtra, piPhaseAware);
            float m = a0 >= a1 ? a0 : a1;
            float w0 = expf((a0 - m) / tau), w1 = expf((a1 - m) / tau);
            float denom = w0 + w1;
            int lits[2] = {l0, l1};
            float ws[2] = {w0 / denom, w1 / denom};
            for (int k = 0; k < 2; ++k) {
                int lit = lits[k], pred = lit >> 1;
                float extra = (piPhaseAware && (lit & 1) && d_negExtra) ? d_negExtra[pred] : 0.0f;
                int li = isLeafIdx(pred, leaves);
                if (li >= 0) {
                    if (du >= 0.0f && du + extra > leafDelays[li]) leafDelays[li] = du + extra;
                    leafLocalSens[li] += su * ws[k];
                } else {
                    for (int b = a + 1; b < coneCnt; ++b) {
                        if (coneIds[b] == pred) {
                            if (du >= 0.0f && du + d_gateDelay[pred] + extra > coneDist[b])
                                coneDist[b] = du + d_gateDelay[pred] + extra;
                            coneSens[b] += su * ws[k];
                            break;
                        }
                    }
                }
            }
        }
        float maxOrigRel = -kInf, origRel[4];
        int usedOrig = 0;
        for (int i = 0; i < 4; ++i) {
            if (leafDelays[i] >= 0.0f) {
                origRel[usedOrig] = d_arr[leaves[i]] + leafDelays[i] - nodeLv;
                if (origRel[usedOrig] > maxOrigRel) maxOrigRel = origRel[usedOrig];
                usedOrig++;
            }
        }
        float jOrig = -nodeLv;
        if (usedOrig > 0) {
            float sumE = 0.0f;
            for (int i = 0; i < usedOrig; ++i) sumE += expf((origRel[i] - maxOrigRel) / tau);
            jOrig = maxOrigRel + tau * logf(sumE);
        }
        float sensOther[4];
        for (int i = 0; i < 4; ++i) {
            float s = d_sens[leaves[i]] - leafLocalSens[i];
            sensOther[i] = s > 0.0f ? s : 0.0f;
        }

        // Strash-aware match enforcing strict level(hit) < level(root) (Stage B Invariant 3)
        uint64_t isC = 0;
        for (int j = 0; j < 54; ++j) match[j] = -1;
        for (int j = 0; j < 4; ++j) {
            match[j] = leaves[j];
            if (leavesCompl[j]) isC |= (1ULL << j);
        }
        int nLibNodes = lib->nNodes[cls];
        for (int j = 0; j < nLibNodes; ++j) {
            int num = j + 4;
            if (!strashAware) continue;
            int in0 = lib->fanin0[cls][j], in1 = lib->fanin1[cls][j];
            if (match[in0] == -1 || match[in1] == -1 || match[in0] == root || match[in1] == root) continue;
            int c0 = ((isC >> in0) & 1) ^ lib->isC0[cls][j];
            int c1 = ((isC >> in1) & 1) ^ lib->isC1[cls][j];
            int hit = lookupHash((match[in0] << 1) | c0, (match[in1] << 1) | c1, P, d_ht, d_f0, d_f1);
            if (hit != -1 && d_nodeLv[hit] < rootUnitLv && !inList(hit, remIds, remCnt)) {
                match[num] = hit;
            }
        }

        int nSubgr = lib->nSubgr[cls];
        for (int out = 0; out < nSubgr; ++out) {
            int rt = lib->pSubgr[cls][out];
            if (match[rt] == root) continue;

            uint64_t usedMask = 1ULL << rt;
            for (int j = rt; j >= 4; --j) {
                if ((usedMask >> j) & 1ULL) {
                    if (match[j] != -1) continue;
                    usedMask |= (1ULL << lib->fanin0[cls][j - 4]);
                    usedMask |= (1ULL << lib->fanin1[cls][j - 4]);
                }
            }

            int nodesAdded = 0;
            int candRefs[4] = {0, 0, 0, 0}, candRefsNeg[4] = {0, 0, 0, 0};
            int eFo[54];
            for (int j = 4; j <= rt; ++j) eFo[j] = 0;
            int hitIds[8], hitDelta[8], hitCnt = 0;
            auto addHit = [&](int u, int d) {
                if (u <= 0) return;
                for (int t = 0; t < hitCnt; ++t) if (hitIds[t] == u) { hitDelta[t] += d; return; }
                if (hitCnt < 8) { hitIds[hitCnt] = u; hitDelta[hitCnt] = d; hitCnt++; }
            };

            if (rt < 4) {
                candRefs[rt] += rootFo;
                if (piPhaseAware && leaves[rt] > 0 && leaves[rt] <= nPIs) {
                    int rNeg = d_negFo[root], rPos = rootFo - rNeg;
                    candRefsNeg[rt] += (rNeg > rPos ? rNeg : rPos);
                }
            } else if (match[rt] != -1) {
                addHit(match[rt], rootFo);
            } else {
                for (int j = 4; j <= rt; ++j) {
                    if (!((usedMask >> j) & 1ULL) || match[j] != -1) continue;
                    nodesAdded++;
                    int ins[2] = {lib->fanin0[cls][j - 4], lib->fanin1[cls][j - 4]};
                    int cs[2] = {lib->isC0[cls][j - 4], lib->isC1[cls][j - 4]};
                    for (int k = 0; k < 2; ++k) {
                        int en = ins[k];
                        if (en < 4) {
                            candRefs[en]++;
                            if (leavesCompl[en] ^ cs[k]) candRefsNeg[en]++;
                        } else if (match[en] != -1) {
                            addHit(match[en], 1);
                        } else {
                            eFo[en]++;
                        }
                    }
                }
            }
            if (nodesAdded > kGradMaxSubGates) continue;

            int da = nodesAdded - saved;
            if (!inVision && da > 0) continue; // Module 1.1 stratified filter

            float leafDd[4] = {0, 0, 0, 0};
            int leafDl[4] = {0, 0, 0, 0};
            float side = 0.0f, reg = 0.0f, slewCost = 0.0f, posFoInc = 0.0f;
            float effTol = (da > 0 && inVision) ? (budgetTol + 0.25f * tau) : budgetTol;
            bool budgetOk = true;
            for (int i = 0; i < 4; ++i) {
                int lf = leaves[i];
                if (lf <= 0) continue;
                bool isPI = (lf <= nPIs), negOnly = (isPI && piPhaseAware);
                int dfo = negOnly ? (candRefsNeg[i] - origRefsNeg[i]) : (candRefs[i] - origRefs[i]);
                if (dfo > 0) posFoInc += static_cast<float>(dfo);
                if (dfo == 0) continue;
                float coef = isPI ? piAlpha : (beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[lf]) : alpha);
                if (coef <= 0.0f) continue;
                int baseCnt = negOnly ? d_negFo[lf] : d_fo[lf];
                int newCnt = baseCnt + dfo;
                if (newCnt < 0) newCnt = 0;
                int dl = loadI(newCnt) - loadI(baseCnt);
                float dd = coef * static_cast<float>(dl);
                if (fanoutBudget && dd > 1e-6f && !(newCnt <= pivot && dd <= 0.5f)) {
                    float so = querySlackOtherDev(
                        lf, negOnly, nObjs, depth, piPhaseAware, remIds, remCnt,
                        d_foOff, d_foEdge, d_arr, d_req, d_gateDelay, d_negExtra, d_top3, d_negTop3);
                    if (so < kInf * 0.5f && dd > (so > 0.0f ? so : 0.0f) + effTol) {
                        budgetOk = false;
                        break;
                    }
                }
                leafDd[i] = dd;
                leafDl[i] = dl;
                side += sensOther[i] * dd;
                if (regW > 0.0f) reg += regW * static_cast<float>(hqI(newCnt, pivot) - hqI(baseCnt, pivot));
                if (beta > 0.0f && dl != 0) {
                    float sc = negOnly ? d_slewCoefNeg[lf] : d_slewCoef[lf];
                    if (sc > 0.0f) slewCost += sc * static_cast<float>(dl);
                }
            }
            if (!budgetOk) continue;
            for (int t = 0; t < hitCnt; ++t) {
                int u = hitIds[t], dfo = hitDelta[t];
                if (dfo > 0) posFoInc += static_cast<float>(dfo);
                if (dfo == 0 || u <= nPIs) continue;
                float coef = beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[u]) : alpha;
                int baseCnt = d_fo[u], newCnt = baseCnt + dfo;
                if (newCnt < 0) newCnt = 0;
                int dl = loadI(newCnt) - loadI(baseCnt);
                float dd = coef * static_cast<float>(dl);
                if (fanoutBudget && dd > 1e-6f && !(newCnt <= pivot && dd <= 0.5f)) {
                    float so = querySlackOtherDev(
                        u, false, nObjs, depth, piPhaseAware, remIds, remCnt,
                        d_foOff, d_foEdge, d_arr, d_req, d_gateDelay, d_negExtra, d_top3, d_negTop3);
                    if (so < kInf * 0.5f && dd > (so > 0.0f ? so : 0.0f) + effTol) {
                        budgetOk = false;
                        break;
                    }
                }
                side += d_sens[u] * dd;
                if (regW > 0.0f) reg += regW * static_cast<float>(hqI(newCnt, pivot) - hqI(baseCnt, pivot));
                if (beta > 0.0f && dl != 0 && d_slewCoef[u] > 0.0f) slewCost += d_slewCoef[u] * static_cast<float>(dl);
            }
            if (!budgetOk) continue;

            for (int i = 0; i < 4; ++i) {
                int lf = leaves[i];
                if (lf == 0) { cvMax[i] = 0.0f; cvLse[i] = -nodeLv; cvSlew[i] = 0.0f; }
                else {
                    float a = d_arr[lf] + ((lf <= nPIs && piPhaseAware) ? 0.0f : leafDd[i]);
                    cvMax[i] = a; cvLse[i] = a - nodeLv;
                    cvSlew[i] = (lf <= nPIs) ? 0.0f : (d_outSlew[lf] + static_cast<float>(leafDl[i]));
                }
            }
            auto getIn = [&](int en, int ec, float & aOut, float & rOut, float & sOut) {
                if (en < 4) {
                    int lf = leaves[en];
                    if (lf == 0) { aOut = 0.0f; rOut = -nodeLv; sOut = 0.0f; return; }
                    if (lf <= nPIs && piPhaseAware) {
                        int ph = leavesCompl[en] ^ ec;
                        aOut = d_arr[lf] + (ph ? (d_negExtra[lf] + leafDd[en]) : 0.0f);
                        rOut = aOut - nodeLv;
                        sOut = ph ? (d_negSlew[lf] + static_cast<float>(leafDl[en])) : 0.0f;
                        return;
                    }
                    aOut = cvMax[en]; rOut = cvLse[en]; sOut = cvSlew[en];
                    return;
                }
                if (match[en] != -1) {
                    int u = match[en], dlU = 0;
                    for (int t = 0; t < hitCnt; ++t) if (hitIds[t] == u) { dlU = loadI(d_fo[u] + hitDelta[t]) - loadI(d_fo[u]); break; }
                    float coef = beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[u]) : alpha;
                    aOut = d_arr[u] + coef * static_cast<float>(dlU);
                    rOut = aOut - nodeLv;
                    sOut = d_outSlew[u] + static_cast<float>(dlU);
                    return;
                }
                aOut = cvMax[en]; rOut = cvLse[en]; sOut = cvSlew[en];
            };
            float maxArrAbs, jCand;
            if (rt < 4 || match[rt] != -1) {
                float sTmp;
                getIn(rt, 0, maxArrAbs, jCand, sTmp);
            } else {
                for (int j = 4; j <= rt; ++j) {
                    if (!((usedMask >> j) & 1ULL) || match[j] != -1) continue;
                    float a0, r0, s0, a1, r1, s1;
                    getIn(lib->fanin0[cls][j - 4], lib->isC0[cls][j - 4], a0, r0, s0);
                    getIn(lib->fanin1[cls][j - 4], lib->isC1[cls][j - 4], a1, r1, s1);
                    float sIn = s0 >= s1 ? s0 : s1;
                    float ld = (j == rt) ? loadF(rootFo) : loadF(eFo[j]);
                    float gd = (beta > 0.0f) ? (1.0f + alpha * ld * (1.0f + beta * sIn))
                                             : ((j == rt) ? d_gateDelay[root] : (1.0f + alpha * ld));
                    cvMax[j] = gd + (a0 >= a1 ? a0 : a1);
                    cvLse[j] = gd + lse2(r0, r1, tau);
                    cvSlew[j] = ld;
                }
                maxArrAbs = cvMax[rt];
                jCand = cvLse[rt];
            }
            if (maxArrAbs > req + (inVision ? 0.25f * tau : 1e-5f)) continue;

            float deltaJ = jOrig - jCand;
            if (da <= 0 && deltaJ < -1e-5f) continue; // area-saving rewrites must not degrade local cone timing
            if (!inVision && !(da < 0 || (da == 0 && deltaJ > 1e-4f))) continue;
            if (inVision && !(deltaJ > 1e-4f || da < 0)) continue;

            float rawTiming = sRoot * deltaJ - side - slewCost;
            float adpScore = adpW * ((rawTiming / phi) * invDepth) - (reg * invDepth) - static_cast<float>(da) * invArea;
            // Prune speculative zero/negative-ADP noise unless it offers meaningful local speedup (> 0.5)
            if (adpScore <= 1e-9f && !(inVision && deltaJ > 0.5f)) continue;
            MenuSlot cur = {adpScore, deltaJ, posFoInc, da, slot, out};

            if (deltaJ > bestSpeed.deltaJ || (deltaJ == bestSpeed.deltaJ && adpScore > bestSpeed.score)) bestSpeed = cur;
            if (adpScore > bestAdp.score) bestAdp = cur;
            if (da < bestArea.da || (da == bestArea.da && deltaJ > bestArea.deltaJ)) bestArea = cur;
            if (deltaJ > 1e-4f && (posFoInc < bestLowFo.posFoInc || (posFoInc == bestLowFo.posFoInc && deltaJ > bestLowFo.deltaJ)))
                bestLowFo = cur;
        }
    }
    d_menuOut[root * 4 + 0] = bestSpeed;
    d_menuOut[root * 4 + 1] = bestAdp;
    d_menuOut[root * 4 + 2] = bestArea;
    d_menuOut[root * 4 + 3] = bestLowFo;
}

// ---------------- Host helpers ----------------

static bool isPrime(int x) {
    if (x < 2) return false;
    for (int i = 2; i * i <= x; ++i)
        if (x % i == 0) return false;
    return true;
}

static int nextPrime(int x) {
    while (!isPrime(x)) ++x;
    return x;
}

static void loadLibrary(rw::Library & lib) {
    int cur = 0;
    for (int i = 0; i < (1 << 16); ++i) lib.pPhases[i] = LIBRARY[cur++];
    for (int i = 0; i < (1 << 16); ++i) lib.pPerms[i] = LIBRARY[cur++];
    for (int i = 0; i < 24; ++i)
        for (int j = 0; j < 4; ++j) lib.pPerms4[i][j] = LIBRARY[cur++];
    for (int i = 0; i < (1 << 16); ++i) lib.pMap[i] = LIBRARY[cur++];
    for (int i = 0; i < 222; ++i) lib.nNodes[i] = LIBRARY[cur++];
    for (int i = 0; i < 222; ++i) lib.nSubgr[i] = LIBRARY[cur++];
    for (int c = 0; c < 222; ++c) {
        for (int i = 0; i < lib.nNodes[c]; ++i) {
            lib.fanin0[c][i] = LIBRARY[cur++];
            lib.fanin1[c][i] = LIBRARY[cur++];
            lib.isC0[c][i] = LIBRARY[cur++];
            lib.isC1[c][i] = LIBRARY[cur++];
        }
        for (int j = 0; j < lib.nSubgr[c]; ++j) lib.pSubgr[c][j] = LIBRARY[cur++];
    }
}

struct AigStats {
    float depth;
    float depthPhSlew;
    int maxFo;
    long long excessFo2;
};

static bool hasMultiInputAndCover(int nPIs, int nPOs, int nObjs, const int * f0, const int * f1, const int * outs) {
    std::vector<int> fo(nObjs, 0);
    for (int i = nPIs + 1; i < nObjs; ++i) {
        fo[f0[i] >> 1]++;
        fo[f1[i] >> 1]++;
    }
    for (int i = 0; i < nPOs; ++i) fo[outs[i] >> 1]++;
    for (int i = nPIs + 1; i < nObjs; ++i) {
        int l0 = f0[i], l1 = f1[i];
        int c0 = l0 >> 1, c1 = l1 >> 1;
        if (((l0 & 1) == 0 && c0 > nPIs && fo[c0] == 1) ||
            ((l1 & 1) == 0 && c1 > nPIs && fo[c1] == 1)) {
            return true;
        }
    }
    return false;
}

// Reorder AIG into contiguous unit-level buckets: 0=const, 1..nPIs=PIs, nPIs+1..nObjs-1 sorted by unit level
static void reorderByLevel(int nPIs, int nPOs, int nObjs,
                           std::vector<int> & f0, std::vector<int> & f1, std::vector<int> & outs,
                           std::vector<int> & levelOffsets) {
    std::vector<int> lv(nObjs, 0);
    int maxLv = 0;
    for (int i = nPIs + 1; i < nObjs; ++i) {
        lv[i] = 1 + std::max(lv[f0[i] >> 1], lv[f1[i] >> 1]);
        if (lv[i] > maxLv) maxLv = lv[i];
    }
    std::vector<int> cnt(maxLv + 1, 0);
    for (int i = nPIs + 1; i < nObjs; ++i) cnt[lv[i]]++;
    levelOffsets.assign(maxLv + 2, nPIs + 1);
    for (int l = 1; l <= maxLv; ++l) levelOffsets[l + 1] = levelOffsets[l] + cnt[l];

    std::vector<int> cur = levelOffsets;
    std::vector<int> old2new(nObjs, 0);
    for (int i = 0; i <= nPIs; ++i) old2new[i] = i;
    for (int i = nPIs + 1; i < nObjs; ++i) old2new[i] = cur[lv[i]]++;

    std::vector<int> nf0(nObjs, 0), nf1(nObjs, 0);
    for (int i = nPIs + 1; i < nObjs; ++i) {
        int nid = old2new[i];
        int l0 = (old2new[f0[i] >> 1] << 1) | (f0[i] & 1);
        int l1 = (old2new[f1[i] >> 1] << 1) | (f1[i] & 1);
        if (l0 > l1) std::swap(l0, l1);
        nf0[nid] = l0;
        nf1[nid] = l1;
    }
    for (int i = 0; i < nPOs; ++i) {
        outs[i] = (old2new[outs[i] >> 1] << 1) | (outs[i] & 1);
    }
    f0.swap(nf0);
    f1.swap(nf1);
}

// Reorder reachable AND nodes in DFS post-order (preserves ABC map cut locality & guarantees l0 <= l1)
static void reorderDfsPostOrder(int nPIs, int nPOs,
                                std::vector<int> & f0, std::vector<int> & f1, std::vector<int> & outs) {
    int nObjs = static_cast<int>(f0.size());
    if (nObjs <= nPIs + 1) return;
    std::vector<uint8_t> state(nObjs, 0); // 0=unvisited, 1=visiting, 2=done
    std::vector<int> old2new(nObjs, 0);
    for (int i = 0; i <= nPIs; ++i) { state[i] = 2; old2new[i] = i; }

    std::vector<int> nf0(nPIs + 1, 0), nf1(nPIs + 1, 0);
    nf0.reserve(nObjs);
    nf1.reserve(nObjs);
    std::vector<int> stk;
    stk.reserve(nObjs);

    for (int i = 0; i < nPOs; ++i) {
        int root = outs[i] >> 1;
        if (root <= nPIs || state[root] == 2) continue;
        stk.push_back(root);
        while (!stk.empty()) {
            int u = stk.back();
            if (state[u] == 2) { stk.pop_back(); continue; }
            if (state[u] == 0) {
                state[u] = 1;
                int s1 = f1[u] >> 1, s0 = f0[u] >> 1;
                if (s1 > nPIs && state[s1] == 0) stk.push_back(s1);
                if (s0 > nPIs && state[s0] == 0) stk.push_back(s0);
            } else {
                stk.pop_back();
                state[u] = 2;
                int nid = static_cast<int>(nf0.size());
                old2new[u] = nid;
                int l0 = (old2new[f0[u] >> 1] << 1) | (f0[u] & 1);
                int l1 = (old2new[f1[u] >> 1] << 1) | (f1[u] & 1);
                if (l0 > l1) std::swap(l0, l1);
                nf0.push_back(l0);
                nf1.push_back(l1);
            }
        }
    }
    for (int i = 0; i < nPOs; ++i) {
        outs[i] = (old2new[outs[i] >> 1] << 1) | (outs[i] & 1);
    }
    f0.swap(nf0);
    f1.swap(nf1);
}

static AigStats computeV3StatsHost(int nPIs, int nPOs, int nObjs,
                                   const int * f0, const int * f1, const int * outs,
                                   float alpha = 0.20f, float piAlpha = 0.10f, float slewBeta = 0.005f) {
    std::vector<int> fo(nObjs, 0), nfo(nObjs, 0);
    for (int i = nPIs + 1; i < nObjs; ++i) {
        fo[f0[i] >> 1]++; fo[f1[i] >> 1]++;
        if (f0[i] & 1) nfo[f0[i] >> 1]++;
        if (f1[i] & 1) nfo[f1[i] >> 1]++;
    }
    for (int i = 0; i < nPOs; ++i) {
        fo[outs[i] >> 1]++;
        if (outs[i] & 1) nfo[outs[i] >> 1]++;
    }
    // 1. Non-phase fanout depth (v3)
    std::vector<float> arr(nObjs, 0.0f);
    for (int i = 1; i <= nPIs; ++i) arr[i] = piAlpha * (fo[i] > 1 ? float(fo[i] - 1) : 0.0f);
    for (int i = nPIs + 1; i < nObjs; ++i) {
        float gd = 1.0f + alpha * (fo[i] > 1 ? float(fo[i] - 1) : 0.0f);
        arr[i] = gd + std::max(arr[f0[i] >> 1], arr[f1[i] >> 1]);
    }
    float depth = 0.0f;
    for (int i = 0; i < nPOs; ++i) depth = std::max(depth, arr[outs[i] >> 1]);

    // 2. Phase-aware + slew depth (v4_slew)
    std::vector<float> arrP(nObjs, 0.0f), negEx(nPIs + 1, 0.0f), outSl(nObjs, 0.0f), negSl(nPIs + 1, 0.0f);
    for (int i = 1; i <= nPIs; ++i) {
        float ln = nfo[i] > 1 ? float(nfo[i] - 1) : 0.0f;
        negEx[i] = piAlpha * ln;
        negSl[i] = ln;
    }
    auto eArrP = [&](int l) {
        int u = l >> 1;
        return arrP[u] + ((u <= nPIs && (l & 1)) ? negEx[u] : 0.0f);
    };
    auto eSlP = [&](int l) {
        int u = l >> 1;
        if (u <= nPIs) return (l & 1) ? negSl[u] : 0.0f;
        return outSl[u];
    };
    for (int i = nPIs + 1; i < nObjs; ++i) {
        float ld = fo[i] > 1 ? float(fo[i] - 1) : 0.0f;
        float sIn = std::max(eSlP(f0[i]), eSlP(f1[i]));
        float gd = 1.0f + alpha * ld * (1.0f + slewBeta * sIn);
        outSl[i] = ld;
        arrP[i] = gd + std::max(eArrP(f0[i]), eArrP(f1[i]));
    }
    float depthPhSlew = 0.0f;
    for (int i = 0; i < nPOs; ++i) depthPhSlew = std::max(depthPhSlew, eArrP(outs[i]));

    int maxFo = 0;
    long long ex2 = 0;
    for (int i = 1; i < nObjs; ++i) {
        if (fo[i] > maxFo) maxFo = fo[i];
        if (fo[i] > 6) {
            long long h = fo[i] - 6;
            ex2 += h * h;
        }
    }
    return {depth, depthPhSlew, maxFo, ex2};
}

static uint16_t computeConeTruth(int root, const int leaves[4], int nPIs,
                                 const std::vector<int> & f0, const std::vector<int> & f1,
                                 bool & ok) {
    static const uint16_t kPats[4] = {0xAAAA, 0xCCCC, 0xF0F0, 0xFF00};
    std::unordered_map<int, uint16_t> vals;
    vals[0] = 0;
    for (int i = 0; i < 4; ++i)
        if (leaves[i] > 0) vals[leaves[i]] = kPats[i];
    std::vector<int> st = {root};
    std::vector<int> interior;
    std::unordered_set<int> seen;
    while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        if (vals.count(u) || seen.count(u)) continue;
        if (u <= nPIs) { ok = false; return 0; }
        seen.insert(u);
        interior.push_back(u);
        st.push_back(f0[u] >> 1);
        st.push_back(f1[u] >> 1);
    }
    std::sort(interior.begin(), interior.end());
    for (int u : interior) {
        uint16_t v0 = vals[f0[u] >> 1] ^ ((f0[u] & 1) ? 0xFFFF : 0);
        uint16_t v1 = vals[f1[u] >> 1] ^ ((f1[u] & 1) ? 0xFFFF : 0);
        vals[u] = v0 & v1;
    }
    ok = true;
    return vals[root];
}

static bool verifyCandFlip(uint16_t origTT, int cls, int rt, int uPhase,
                           const rw::Library & lib, int & flipOut) {
    static const uint16_t kPats[4] = {0xAAAA, 0xCCCC, 0xF0F0, 0xFF00};
    uint16_t cv[54];
    for (int j = 0; j < 4; ++j) cv[j] = kPats[j] ^ (((uPhase >> j) & 1) ? 0xFFFF : 0);
    for (int j = 0; j < lib.nNodes[cls] && j + 4 <= rt; ++j) {
        uint16_t v0 = cv[lib.fanin0[cls][j]] ^ (lib.isC0[cls][j] ? 0xFFFF : 0);
        uint16_t v1 = cv[lib.fanin1[cls][j]] ^ (lib.isC1[cls][j] ? 0xFFFF : 0);
        cv[j + 4] = v0 & v1;
    }
    uint16_t outTT = cv[rt];
    if (outTT == origTT) { flipOut = 0; return true; }
    if (outTT == static_cast<uint16_t>(origTT ^ 0xFFFF)) { flipOut = 1; return true; }
    return false;
}

static void collectBoundedMffcHost(int root, const int leaves[4], int nPIs,
                                   const std::vector<int> & f0, const std::vector<int> & f1,
                                   const std::vector<int> & fo,
                                   std::vector<int> & remAll, std::vector<int> & remStrict) {
    remAll.clear();
    remStrict.clear();
    auto isLf = [&](int n) {
        return n == leaves[0] || n == leaves[1] || n == leaves[2] || n == leaves[3];
    };
    std::unordered_map<int, int> rem;
    std::vector<int> pending = {root};
    std::unordered_set<int> added;
    while (!pending.empty()) {
        int u = pending.back();
        pending.pop_back();
        if (u <= nPIs || isLf(u) || added.count(u)) continue;
        added.insert(u);
        remAll.push_back(u);
        if (u != root) remStrict.push_back(u);
        int s0 = f0[u] >> 1, s1 = f1[u] >> 1;
        for (int s : {s0, s1}) {
            if (s <= nPIs || isLf(s)) continue;
            auto it = rem.find(s);
            int val = (it == rem.end() ? fo[s] : it->second) - 1;
            rem[s] = val;
            if (val == 0) pending.push_back(s);
        }
    }
}

struct StrashRebuild {
    int nPIs;
    int foCap;
    std::vector<int> f0, f1, foCnt;
    robin_hood::unordered_flat_map<uint64_t, int> ht;

    explicit StrashRebuild(int nPIs_, int reserveNodes, int foCap_ = 0) : nPIs(nPIs_), foCap(foCap_) {
        f0.assign(nPIs + 1, 0);
        f1.assign(nPIs + 1, 0);
        foCnt.assign(nPIs + 1, 0);
        f0.reserve(nPIs + reserveNodes + 64);
        f1.reserve(nPIs + reserveNodes + 64);
        foCnt.reserve(nPIs + reserveNodes + 64);
        ht.reserve(reserveNodes * 2 + 64);
    }

    int andLit(int l0, int l1, bool allowDup = false) {
        if (l0 == 0 || l1 == 0) return 0;
        if (l0 == 1) return l1;
        if (l1 == 1) return l0;
        if (l0 == l1) return l0;
        if (l0 == (l1 ^ 1)) return 0;
        if (l0 > l1) std::swap(l0, l1);
        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(l0)) << 32) |
                       static_cast<uint32_t>(l1);
        auto it = ht.find(key);
        if (it != ht.end()) {
            int hit = it->second;
            if (!(allowDup && foCap > 0 && foCnt[hit] >= foCap)) {
                return hit << 1;
            }
        }
        int nid = static_cast<int>(f0.size());
        f0.push_back(l0);
        f1.push_back(l1);
        foCnt.push_back(0);
        foCnt[l0 >> 1]++;
        foCnt[l1 >> 1]++;
        ht[key] = nid;
        return nid << 1;
    }
};

static void copyHostToAigMan(AIGMan & aigman, int nPIs, int nPOs,
                             const std::vector<int> & f0, const std::vector<int> & f1,
                             const std::vector<int> & outs) {
    if (aigman.deviceAllocated) aigman.clearDevice();
    aigman.clearHost();
    aigman.nPIs = nPIs;
    aigman.nPOs = nPOs;
    aigman.nObjs = static_cast<int>(f0.size());
    aigman.nNodes = aigman.nObjs - nPIs - 1;
    aigman.allocHost();
    std::memcpy(aigman.pFanin0, f0.data(), aigman.nObjs * sizeof(int));
    std::memcpy(aigman.pFanin1, f1.data(), aigman.nObjs * sizeof(int));
    for (int i = nPIs + 1; i < aigman.nObjs; ++i) {
        if (aigman.pFanin0[i] > aigman.pFanin1[i]) {
            std::swap(aigman.pFanin0[i], aigman.pFanin1[i]);
        }
    }
    std::memcpy(aigman.pOuts, outs.data(), nPOs * sizeof(int));
    std::memset(aigman.pNumFanouts, 0, aigman.nObjs * sizeof(int));
    for (int i = nPIs + 1; i < aigman.nObjs; ++i) {
        aigman.pNumFanouts[f0[i] >> 1]++;
        aigman.pNumFanouts[f1[i] >> 1]++;
    }
    for (int i = 0; i < nPOs; ++i) aigman.pNumFanouts[outs[i] >> 1]++;
    std::vector<int> lv(aigman.nObjs, 0);
    int maxLv = 0;
    for (int i = nPIs + 1; i < aigman.nObjs; ++i) {
        lv[i] = 1 + std::max(lv[f0[i] >> 1], lv[f1[i] >> 1]);
        if (lv[i] > maxLv) maxLv = lv[i];
    }
    aigman.nLevels = maxLv;
    aigman.setPrevCmdRewrite(0);
}

static bool conditionalBalanceV3(AIGMan & aigman, const Rw4BParams & params, const char * tag) {
    if (aigman.deviceAllocated) {
        aigman.toHost();
        aigman.clearDevice();
    }
    if (!hasMultiInputAndCover(aigman.nPIs, aigman.nPOs, aigman.nObjs, aigman.pFanin0, aigman.pFanin1, aigman.pOuts)) {
        return false;
    }
    AigStats s0 = computeV3StatsHost(aigman.nPIs, aigman.nPOs, aigman.nObjs,
                                     aigman.pFanin0, aigman.pFanin1, aigman.pOuts,
                                     params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta);
    std::vector<int> bakF0(aigman.pFanin0, aigman.pFanin0 + aigman.nObjs);
    std::vector<int> bakF1(aigman.pFanin1, aigman.pFanin1 + aigman.nObjs);
    std::vector<int> bakOuts(aigman.pOuts, aigman.pOuts + aigman.nPOs);

    int savedVerb = aigman.verbose;
    aigman.verbose = 0;
    aigman.balance(1, true);
    aigman.verbose = savedVerb;
    if (aigman.deviceAllocated) {
        aigman.toHost();
        aigman.clearDevice();
    }

    AigStats s1 = computeV3StatsHost(aigman.nPIs, aigman.nPOs, aigman.nObjs,
                                     aigman.pFanin0, aigman.pFanin1, aigman.pOuts,
                                     params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta);
    bool accept = (s1.maxFo <= std::max(64, s0.maxFo)) &&
                  (s1.excessFo2 <= std::max(1000LL, s0.excessFo2)) &&
                  (s1.depth <= s0.depth - 1.0f);
    if (!accept) {
        copyHostToAigMan(aigman, aigman.nPIs, aigman.nPOs, bakF0, bakF1, bakOuts);
    }
    if (params.verbose >= 1) {
        printf("rw4b [%s]: D=%.1f->%.1f Dp=%.1f->%.1f maxFo=%d->%d ex2=%lld->%lld -> %s\n",
               tag, s0.depth, s1.depth, s0.depthPhSlew, s1.depthPhSlew, s0.maxFo, s1.maxFo, s0.excessFo2, s1.excessFo2,
               accept ? "ACCEPT" : "REJECT");
    }
    return accept;
}

static void selectBestPreMacroGPU(AIGMan & aigman, const Rw4BParams & params) {
    if (aigman.deviceAllocated) {
        aigman.toHost();
        aigman.clearDevice();
    }
    int nPIs = aigman.nPIs, nPOs = aigman.nPOs;
    AigStats origS = computeV3StatsHost(nPIs, nPOs, aigman.nObjs,
                                        aigman.pFanin0, aigman.pFanin1, aigman.pOuts,
                                        params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta);
    std::vector<int> origF0(aigman.pFanin0, aigman.pFanin0 + aigman.nObjs);
    std::vector<int> origF1(aigman.pFanin1, aigman.pFanin1 + aigman.nObjs);
    std::vector<int> origOuts(aigman.pOuts, aigman.pOuts + nPOs);

    std::string bestTag = "orig";
    AigStats bestS = origS;
    std::vector<int> bestF0 = origF0, bestF1 = origF1, bestOuts = origOuts;

    auto passesV3 = [&](const AigStats & s) {
        return (s.maxFo <= std::max(64, origS.maxFo)) &&
               (s.excessFo2 <= std::max(1000LL, origS.excessFo2)) &&
               (s.depth <= origS.depth - 1.0f);
    };
    auto betterThanBest = [&](const AigStats & s) {
        if (s.depth < bestS.depth - 1e-4f) return true;
        if (std::fabs(s.depth - bestS.depth) <= 1e-4f && s.excessFo2 < bestS.excessFo2) return true;
        return false;
    };
    auto snapshotCurrent = [&](std::vector<int> & f0, std::vector<int> & f1, std::vector<int> & outs) -> AigStats {
        if (aigman.deviceAllocated) {
            aigman.toHost();
            aigman.clearDevice();
        }
        f0.assign(aigman.pFanin0, aigman.pFanin0 + aigman.nObjs);
        f1.assign(aigman.pFanin1, aigman.pFanin1 + aigman.nObjs);
        outs.assign(aigman.pOuts, aigman.pOuts + nPOs);
        return computeV3StatsHost(nPIs, nPOs, aigman.nObjs,
                                  aigman.pFanin0, aigman.pFanin1, aigman.pOuts,
                                  params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta);
    };

    int savedVerb = aigman.verbose;
    aigman.verbose = 0;

    AigStats balS = origS, brS = origS, rfS = origS;
    bool canBal = hasMultiInputAndCover(nPIs, nPOs, aigman.nObjs, aigman.pFanin0, aigman.pFanin1, aigman.pOuts);

    // 1. bal
    if (canBal) {
        aigman.balance(1, true);
        std::vector<int> balF0, balF1, balOuts;
        balS = snapshotCurrent(balF0, balF1, balOuts);
        if (passesV3(balS) && betterThanBest(balS)) {
            bestTag = "bal"; bestS = balS; bestF0 = balF0; bestF1 = balF1; bestOuts = balOuts;
        }

        // 2. bal_rfz (continue from bal before restoring orig)
        aigman.refactor(true, true, 10);
        aigman.strash(false, true);
        std::vector<int> brF0, brF1, brOuts;
        brS = snapshotCurrent(brF0, brF1, brOuts);
        if (passesV3(brS) && betterThanBest(brS)) {
            bestTag = "bal_rfz"; bestS = brS; bestF0 = std::move(brF0); bestF1 = std::move(brF1); bestOuts = std::move(brOuts);
        }
        copyHostToAigMan(aigman, nPIs, nPOs, origF0, origF1, origOuts);
    }

    // 3. rfz (from orig)
    aigman.refactor(true, true, 10);
    aigman.strash(false, true);
    std::vector<int> rfF0, rfF1, rfOuts;
    rfS = snapshotCurrent(rfF0, rfF1, rfOuts);
    if (passesV3(rfS) && betterThanBest(rfS)) {
        bestTag = "rfz"; bestS = rfS; bestF0 = std::move(rfF0); bestF1 = std::move(rfF1); bestOuts = std::move(rfOuts);
    }

    aigman.verbose = savedVerb;
    copyHostToAigMan(aigman, nPIs, nPOs, bestF0, bestF1, bestOuts);
    if (params.verbose >= 1) {
        printf("rw4b [pre_struct]: orig(D=%.1f) bal(D=%.1f) rfz(D=%.1f) bal_rfz(D=%.1f) -> %s (D=%.1f)\n",
               origS.depth, balS.depth, rfS.depth, brS.depth, bestTag.c_str(), bestS.depth);
    }
}

static float querySlackOtherHost(
    int u, bool negOnly, int nObjs, float depth, bool piPhaseAware,
    const std::vector<int> & remAll,
    const std::vector<int> & hFoOff, const std::vector<int> & hFoEdge,
    const std::vector<float> & hArr, const std::vector<float> & hReq,
    const std::vector<float> & hGd, const std::vector<float> & hNegExtra,
    const std::vector<Top3Bounds> & hTop3, const std::vector<Top3Bounds> & hNegTop3) {
    auto inRem = [&](int x) {
        for (int r : remAll) if (r == x) return true;
        return false;
    };
    const Top3Bounds & b = negOnly ? hNegTop3[u] : hTop3[u];
    for (int k = 0; k < 3; ++k) {
        int cid = b.c[k];
        if (cid == -1) return kInf;
        if (cid == -2 || !inRem(cid)) return b.s[k];
    }
    float best = kInf;
    for (int e = hFoOff[u]; e < hFoOff[u + 1]; ++e) {
        int code = hFoEdge[e];
        int cons = code >> 1;
        int neg = code & 1;
        if (negOnly && !neg) continue;
        if (cons < nObjs && inRem(cons)) continue;
        float ea = hArr[u] + ((piPhaseAware && neg) ? hNegExtra[u] : 0.0f);
        float reqE = (cons >= nObjs) ? depth : (hReq[cons] - hGd[cons]);
        float sl = reqE - ea;
        if (sl < best) best = sl;
    }
    return best;
}

// ---------------- Main GPU rw4b entry ----------------

void runRw4B(AIGMan & aigman, const Rw4BParams & params) {
    if (aigman.nNodes == 0) return;
    auto tTotal0 = std::chrono::steady_clock::now();

    if (aigman.deviceAllocated) {
        aigman.toHost();
        aigman.clearDevice();
    }

    if (params.macroMode == 1) {
        conditionalBalanceV3(aigman, params, "pre_bal");
    } else if (params.macroMode == 2) {
        selectBestPreMacroGPU(aigman, params);
    }

    // Load NPN library once onto GPU
    static bool libLoaded = false;
    static rw::Library hLib;
    if (!libLoaded) {
        loadLibrary(hLib);
        libLoaded = true;
    }
    rw::Library * d_lib = nullptr;
    CHECK_CUDA(cudaMalloc(&d_lib, sizeof(rw::Library)));
    CHECK_CUDA(cudaMemcpy(d_lib, &hLib, sizeof(rw::Library), cudaMemcpyHostToDevice));

    const float scales[3] = {1.5f, 1.0f, 0.5f};
    double msTiming = 0.0, msEnum = 0.0, msEval = 0.0, msSelectRebuild = 0.0;
    int totalApplied = 0;

    std::vector<int> f0(aigman.pFanin0, aigman.pFanin0 + aigman.nObjs);
    std::vector<int> f1(aigman.pFanin1, aigman.pFanin1 + aigman.nObjs);
    std::vector<int> outs(aigman.pOuts, aigman.pOuts + aigman.nPOs);
    int nPIs = aigman.nPIs;
    int nPOs = aigman.nPOs;

    for (int rnd = 0; rnd < params.rounds; ++rnd) {
        int nObjs = static_cast<int>(f0.size());
        int nNodes = nObjs - nPIs - 1;
        if (nNodes <= 0) break;

        std::vector<int> lvOff;
        reorderByLevel(nPIs, nPOs, nObjs, f0, f1, outs, lvOff);
        int maxLv = static_cast<int>(lvOff.size()) - 2;

        // Allocate GPU buffers for this round
        int *d_f0 = nullptr, *d_f1 = nullptr, *d_outs = nullptr, *d_fo = nullptr, *d_negFo = nullptr, *d_slewArg = nullptr;
        int *d_foOff = nullptr, *d_foCursor = nullptr, *d_foEdge = nullptr;
        float *d_arr = nullptr, *d_req = nullptr, *d_reqNeg = nullptr, *d_gd = nullptr;
        float *d_negExtra = nullptr, *d_outSlew = nullptr, *d_negSlew = nullptr, *d_slewIn = nullptr;
        float *d_sens = nullptr, *d_slewCoef = nullptr, *d_slewCoefNeg = nullptr, *d_poArr = nullptr, *d_phi = nullptr;
        Top3Bounds *d_top3 = nullptr, *d_negTop3 = nullptr;
        Cut4B * d_cuts = nullptr;
        TableNode4B * d_ht = nullptr;
        CandResult * d_candOut = nullptr;

        int totalEdges = std::max(1, 2 * nNodes + nPOs);
        int P = nextPrime(std::max(101, nNodes * 2 + 1));
        CHECK_CUDA(cudaMalloc(&d_f0, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_f1, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_outs, std::max(1, nPOs) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_fo, (nObjs + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_negFo, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_foOff, (nObjs + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_foCursor, (nObjs + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_foEdge, totalEdges * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_slewArg, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_arr, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_req, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_reqNeg, (nPIs + 1) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gd, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_negExtra, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_outSlew, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_negSlew, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_slewIn, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sens, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_slewCoef, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_slewCoefNeg, (nPIs + 1) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_poArr, std::max(1, nPOs) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_phi, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_top3, nObjs * sizeof(Top3Bounds)));
        CHECK_CUDA(cudaMalloc(&d_negTop3, (nPIs + 1) * sizeof(Top3Bounds)));
        CHECK_CUDA(cudaMalloc(&d_cuts, static_cast<size_t>(nObjs) * kCutSetSize * sizeof(Cut4B)));
        CHECK_CUDA(cudaMalloc(&d_ht, static_cast<size_t>(P + nNodes) * sizeof(TableNode4B)));
        CHECK_CUDA(cudaMalloc(&d_candOut, static_cast<size_t>(nObjs) * kTopK * sizeof(CandResult)));

        CHECK_CUDA(cudaMemcpy(d_f0, f0.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_f1, f1.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
        if (nPOs > 0) CHECK_CUDA(cudaMemcpy(d_outs, outs.data(), nPOs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_fo, 0, (nObjs + 1) * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_negFo, 0, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_sens, 0, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_phi, 0, sizeof(float)));
        CHECK_CUDA(cudaMemset(d_slewCoef, 0, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_slewCoefNeg, 0, (nPIs + 1) * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_negExtra, 0, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_negSlew, 0, nObjs * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_cuts, 0, static_cast<size_t>(nObjs) * kCutSetSize * sizeof(Cut4B)));
        CHECK_CUDA(cudaMemset(d_ht, -1, static_cast<size_t>(P + nNodes) * sizeof(TableNode4B)));

        std::vector<float> infArr(nObjs, kInf);
        CHECK_CUDA(cudaMemcpy(d_req, infArr.data(), nObjs * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_reqNeg, infArr.data(), (nPIs + 1) * sizeof(float), cudaMemcpyHostToDevice));

        auto t0 = std::chrono::steady_clock::now();
        // 1. Refs + CSR fanout list + Forward delay propagation
        int maxWork = std::max(nNodes, nPOs);
        countRefsKernel<<<(maxWork + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            d_f0, d_f1, d_outs, nPIs, nObjs, nPOs, d_fo, d_negFo);
        {
            thrust::device_ptr<int> foPtr(d_fo);
            thrust::device_ptr<int> offPtr(d_foOff);
            thrust::exclusive_scan(foPtr, foPtr + nObjs + 1, offPtr);
            CHECK_CUDA(cudaMemcpy(d_foCursor, d_foOff, (nObjs + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
            scatterFanoutsKernel<<<(maxWork + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                d_f0, d_f1, d_outs, nPIs, nObjs, nPOs, d_foCursor, d_foEdge);
        }
        initPIsKernel<<<(nPIs + 1 + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            nPIs, d_fo, d_negFo, params.piFanoutAlpha, params.piPhaseAware,
            d_arr, d_gd, d_negExtra, d_outSlew, d_negSlew);
        for (int l = 1; l <= maxLv; ++l) {
            int cnt = lvOff[l + 1] - lvOff[l];
            if (cnt <= 0) continue;
            propagateForwardLevelKernel<<<(cnt + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                lvOff[l], cnt, nPIs, d_f0, d_f1, d_fo, params.fanoutAlpha, params.slewBeta,
                params.piPhaseAware, d_negExtra, d_negSlew, d_arr, d_gd, d_outSlew, d_slewIn, d_slewArg);
        }

        // Depth & adaptive tau via GPU sort over PO arrivals
        float depth = 0.0f;
        float tau = params.minTau;
        if (nPOs > 0) {
            extractPOArrivalsKernel<<<(nPOs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                d_outs, nPOs, d_arr, d_negExtra, params.piPhaseAware, d_poArr);
            thrust::device_ptr<float> poPtr(d_poArr);
            thrust::sort(poPtr, poPtr + nPOs);
            CHECK_CUDA(cudaMemcpy(&depth, d_poArr + (nPOs - 1), sizeof(float), cudaMemcpyDeviceToHost));
            float q75 = depth;
            int q75Idx = (3 * (nPOs - 1)) / 4;
            CHECK_CUDA(cudaMemcpy(&q75, d_poArr + q75Idx, sizeof(float), cudaMemcpyDeviceToHost));
            float spread = std::max(0.0f, depth - q75);
            float scale = scales[rnd < 3 ? rnd : 2];
            float raw = std::max(params.minTau, std::min(params.maxTau, params.spreadCoeff * spread)) * scale;
            tau = std::max(params.minTau, std::min(params.maxTau, raw));
        }

        // Backward required + sensitivity + slew coupling + consumer slack bounds
        float phi = 1.0f;
        if (nPOs > 0) {
            initReqPOKernel<<<(nPOs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                d_outs, nPOs, nPIs, depth, d_negExtra, params.piPhaseAware, d_req, d_reqNeg);
            initSensPOKernel<<<(nPOs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                d_outs, nPOs, depth, tau, d_arr, d_negExtra, params.piPhaseAware, d_sens, d_phi);
            CHECK_CUDA(cudaMemcpy(&phi, d_phi, sizeof(float), cudaMemcpyDeviceToHost));
            if (phi < 1e-6f) phi = 1.0f;
        }
        for (int l = maxLv; l >= 1; --l) {
            int cnt = lvOff[l + 1] - lvOff[l];
            if (cnt <= 0) continue;
            propagateReqLevelKernel<<<(cnt + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                lvOff[l], cnt, nPIs, d_f0, d_f1, d_gd, d_negExtra, params.piPhaseAware, d_req, d_reqNeg);
            propagateSensLevelKernel<<<(cnt + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                lvOff[l], cnt, tau, d_f0, d_f1, d_arr, d_negExtra, params.piPhaseAware, d_sens);
        }
        if (params.slewBeta > 0.0f) {
            slewCouplingKernel<<<(nNodes + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                nPIs, nObjs, params.fanoutAlpha, params.slewBeta, params.piPhaseAware,
                d_f0, d_f1, d_fo, d_slewArg, d_sens, d_slewCoef, d_slewCoefNeg);
        }
        computeConsumerBoundsKernel<<<(nObjs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            nPIs, nObjs, depth, params.piPhaseAware, d_foOff, d_foEdge,
            d_arr, d_req, d_gd, d_negExtra, d_top3, d_negTop3);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        msTiming += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. GPU hybrid 12-slot 4-cut enumeration (1 trivial + 5 timing-priority + 6 area-priority)
        if (nPIs > 0) {
            initCutInputsKernel<<<(nPIs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(nPIs, d_fo, d_arr, d_cuts);
        }
        for (int l = 1; l <= maxLv; ++l) {
            int cnt = lvOff[l + 1] - lvOff[l];
            if (cnt <= 0) continue;
            cutEnumLevelKernel<<<(cnt + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                lvOff[l], cnt, d_f0, d_f1, d_fo, d_arr, d_cuts);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        msEnum += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // 3. GPU strash table + evaluation (Stage B joint-grad path OR Stage A greedy path)
        buildHashKernel<<<(nNodes + kBlockSize - 1) / kBlockSize, kBlockSize>>>(nPIs, nObjs, P, d_f0, d_f1, d_ht);

        if (params.jointGrad) {
            std::vector<int> hNodeLv(nObjs, 0);
            for (int l = 1; l <= maxLv; ++l) {
                for (int id = lvOff[l]; id < lvOff[l + 1]; ++id) hNodeLv[id] = l;
            }
            int * d_nodeLv = nullptr;
            MenuSlot * d_menuOut = nullptr;
            CHECK_CUDA(cudaMalloc(&d_nodeLv, nObjs * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_menuOut, static_cast<size_t>(nObjs) * 4 * sizeof(MenuSlot)));
            CHECK_CUDA(cudaMemcpy(d_nodeLv, hNodeLv.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));

            evaluateNode4BMenuKernel<<<(nNodes + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                nPIs, nObjs, P, depth, phi, d_f0, d_f1, d_fo, d_negFo, d_foOff, d_foEdge, d_nodeLv,
                d_arr, d_req, d_gd, d_negExtra, d_outSlew, d_negSlew, d_slewIn, d_sens, d_slewCoef, d_slewCoefNeg,
                d_top3, d_negTop3, d_cuts, d_lib, d_ht, params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta,
                params.fanoutReg, params.fanoutPivot, tau, params.budgetTol, params.piPhaseAware, params.strashAware,
                params.fanoutBudget, params.adpW, params.tnsRho, d_menuOut);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto t3 = std::chrono::steady_clock::now();
            msEval += std::chrono::duration<double, std::milli>(t3 - t2).count();

            std::vector<MenuSlot> hMenu(static_cast<size_t>(nObjs) * 4);
            std::vector<int> hFo(nObjs), hNegFo(nObjs), hFoOff(nObjs + 1), hFoEdge(totalEdges);
            std::vector<float> hArr(nObjs), hReq(nObjs), hGd(nObjs), hNegExtra(nObjs), hSlewIn(nObjs), hSens(nObjs);
            std::vector<Top3Bounds> hTop3(nObjs), hNegTop3(nPIs + 1);
            std::vector<Cut4B> hCuts(static_cast<size_t>(nObjs) * kCutSetSize);
            CHECK_CUDA(cudaMemcpy(hMenu.data(), d_menuOut, static_cast<size_t>(nObjs) * 4 * sizeof(MenuSlot), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hFo.data(), d_fo, nObjs * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hNegFo.data(), d_negFo, nObjs * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hFoOff.data(), d_foOff, (nObjs + 1) * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hFoEdge.data(), d_foEdge, totalEdges * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hArr.data(), d_arr, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hReq.data(), d_req, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hGd.data(), d_gd, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hNegExtra.data(), d_negExtra, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hSlewIn.data(), d_slewIn, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hSens.data(), d_sens, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hTop3.data(), d_top3, nObjs * sizeof(Top3Bounds), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hNegTop3.data(), d_negTop3, (nPIs + 1) * sizeof(Top3Bounds), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(hCuts.data(), d_cuts, static_cast<size_t>(nObjs) * kCutSetSize * sizeof(Cut4B), cudaMemcpyDeviceToHost));

            cudaFree(d_nodeLv); cudaFree(d_menuOut);
            cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_outs); cudaFree(d_fo); cudaFree(d_negFo); cudaFree(d_slewArg);
            cudaFree(d_foOff); cudaFree(d_foCursor); cudaFree(d_foEdge);
            cudaFree(d_arr); cudaFree(d_req); cudaFree(d_reqNeg); cudaFree(d_gd);
            cudaFree(d_negExtra); cudaFree(d_outSlew); cudaFree(d_negSlew); cudaFree(d_slewIn);
            cudaFree(d_sens); cudaFree(d_slewCoef); cudaFree(d_slewCoefNeg); cudaFree(d_poArr); cudaFree(d_phi);
            cudaFree(d_top3); cudaFree(d_negTop3);
            cudaFree(d_cuts); cudaFree(d_ht); cudaFree(d_candOut);

            robin_hood::unordered_flat_map<uint64_t, int> origStrash;
            if (params.strashAware) {
                origStrash.reserve(nNodes * 2 + 64);
                for (int id = nPIs + 1; id < nObjs; ++id) {
                    int l0 = f0[id], l1 = f1[id];
                    if (l0 > l1) std::swap(l0, l1);
                    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(l0)) << 32) | static_cast<uint32_t>(l1);
                    origStrash.emplace(key, id);
                }
            }

            struct GradCandHost {
                int cls, rt, uPhase, flip, deltaAnd, wireLit;
                float score = -kInf, deltaJ = 0.0f;
                int leaves[4];
                int match[54];
                int hitCnt, hitIds[8];
                std::vector<int> remAll, remStrict;
                std::vector<SubGate> subGates;
                std::vector<std::pair<int, int>> foDeltas; // external (node, delta)
            };
            struct GradRegionHost {
                int root;
                float weight;
                int warmBestK; // 0 if bestAdp <= 0, else 1-based index into cands (+1)
                std::vector<GradCandHost> cands; // non-noop candidates (1..4)
                std::vector<int> remStrictUnion;
                std::vector<int> refsUnion;
            };

            std::vector<GradRegionHost> candRegions;
            candRegions.reserve(nNodes / 4 + 16);

            auto addUnique = [](std::vector<int> & v, int x) {
                if (x <= 0) return;
                for (int y : v) if (y == x) return;
                v.push_back(x);
            };

            for (int root = nPIs + 1; root < nObjs; ++root) {
                MenuSlot slots[4];
                int nRaw = 0;
                for (int s = 0; s < 4; ++s) {
                    const MenuSlot & ms = hMenu[static_cast<size_t>(root) * 4 + s];
                    if (ms.cutSlot < 0 || ms.subgr < 0) continue;
                    bool dup = false;
                    for (int prev = 0; prev < nRaw; ++prev) {
                        if (slots[prev].cutSlot == ms.cutSlot && slots[prev].subgr == ms.subgr) { dup = true; break; }
                    }
                    if (!dup) slots[nRaw++] = ms;
                }
                if (nRaw == 0) continue;

                GradRegionHost reg;
                reg.root = root;
                reg.weight = 0.0f;
                reg.warmBestK = 0;
                float bestAdp = -kInf, bestDj = 0.0f;
                for (int s = 0; s < nRaw; ++s) {
                    const MenuSlot & ms = slots[s];
                    Cut4B cut = hCuts[static_cast<size_t>(root) * kCutSetSize + ms.cutSlot];
                    if (!cut.used || cut.nLeaves < 3) continue;
                    if (cut.nLeaves == 3) cut.leaves[3] = 0;

                    int uPhase = hLib.pPhases[cut.truthtable];
                    int cls = hLib.pMap[cut.truthtable];
                    const int * perm = hLib.pPerms4[hLib.pPerms[cut.truthtable]];
                    int leaves[4];
                    for (int j = 0; j < 4; ++j) leaves[j] = cut.leaves[perm[j]];
                    int rt = hLib.pSubgr[cls][ms.subgr];

                    bool ttOk = false;
                    uint16_t origTT = computeConeTruth(root, leaves, nPIs, f0, f1, ttOk);
                    int flip = 0;
                    if (!ttOk || !verifyCandFlip(origTT, cls, rt, uPhase, hLib, flip)) continue;

                    std::vector<int> remAll, remStrict;
                    collectBoundedMffcHost(root, leaves, nPIs, f0, f1, hFo, remAll, remStrict);

                    int match[54];
                    uint64_t isC = 0;
                    for (int j = 0; j < 54; ++j) match[j] = -1;
                    for (int j = 0; j < 4; ++j) {
                        match[j] = leaves[j];
                        if ((uPhase >> j) & 1) isC |= (1ULL << j);
                    }
                    auto inRemAll = [&](int u) {
                        for (int r : remAll) if (r == u) return true;
                        return false;
                    };
                    if (params.strashAware) {
                        int nLibN = hLib.nNodes[cls];
                        for (int j = 0; j < nLibN; ++j) {
                            int num = j + 4;
                            int i0 = hLib.fanin0[cls][j], i1 = hLib.fanin1[cls][j];
                            if (match[i0] == -1 || match[i1] == -1 || match[i0] == root || match[i1] == root) continue;
                            int c0 = ((isC >> i0) & 1) ^ hLib.isC0[cls][j];
                            int c1 = ((isC >> i1) & 1) ^ hLib.isC1[cls][j];
                            int l0 = (match[i0] << 1) | c0, l1 = (match[i1] << 1) | c1;
                            if (l0 > l1) std::swap(l0, l1);
                            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(l0)) << 32) | static_cast<uint32_t>(l1);
                            auto it = origStrash.find(key);
                            if (it != origStrash.end() && hNodeLv[it->second] < hNodeLv[root] && !inRemAll(it->second)) {
                                match[num] = it->second;
                            }
                        }
                    }

                    // Build compact micro-subgraph & external fanout deltas
                    int origRefs[4] = {0, 0, 0, 0}, origRefsNeg[4] = {0, 0, 0, 0};
                    for (int u : remAll) {
                        int lits[2] = {f0[u], f1[u]};
                        for (int l : lits) {
                            int src = l >> 1;
                            for (int j = 0; j < 4; ++j) {
                                if (leaves[j] == src) {
                                    origRefs[j]++;
                                    if (l & 1) origRefsNeg[j]++;
                                }
                            }
                        }
                    }
                    int candRefs[4] = {0, 0, 0, 0}, candRefsNeg[4] = {0, 0, 0, 0};
                    int hitIds[8], hitDelta[8], hitCnt = 0;
                    auto addHit = [&](int u, int d) {
                        if (u <= 0) return;
                        for (int t = 0; t < hitCnt; ++t) if (hitIds[t] == u) { hitDelta[t] += d; return; }
                        if (hitCnt < 8) { hitIds[hitCnt] = u; hitDelta[hitCnt] = d; hitCnt++; }
                    };

                    GradCandHost gc;
                    gc.cls = cls; gc.rt = rt; gc.uPhase = uPhase; gc.flip = flip;
                    gc.score = ms.score; gc.deltaJ = ms.deltaJ;
                    gc.wireLit = -1;
                    for (int j = 0; j < 4; ++j) gc.leaves[j] = leaves[j];
                    for (int j = 0; j < 54; ++j) gc.match[j] = match[j];

                    int nodesAdded = 0;
                    if (rt < 4) {
                        int p = (((uPhase >> rt) & 1) ^ flip);
                        gc.wireLit = (leaves[rt] << 1) | p;
                        candRefs[rt] += hFo[root];
                        if (leaves[rt] > 0 && leaves[rt] <= nPIs && params.piPhaseAware) {
                            int rNeg = hNegFo[root];
                            candRefsNeg[rt] += (p == 0) ? rNeg : (hFo[root] - rNeg);
                        }
                    } else if (match[rt] != -1) {
                        gc.wireLit = (match[rt] << 1) | flip;
                        addHit(match[rt], hFo[root]);
                    } else {
                        uint64_t um = 1ULL << rt;
                        for (int j = rt; j >= 4; --j) {
                            if ((um >> j) & 1ULL) {
                                if (match[j] != -1) continue;
                                um |= (1ULL << hLib.fanin0[cls][j - 4]);
                                um |= (1ULL << hLib.fanin1[cls][j - 4]);
                            }
                        }
                        int localIdx[54];
                        int eFo[54];
                        for (int j = 0; j < 54; ++j) { localIdx[j] = -1; eFo[j] = 0; }
                        for (int j = 4; j <= rt; ++j) {
                            if (!((um >> j) & 1ULL) || match[j] != -1) continue;
                            localIdx[j] = nodesAdded++;
                        }
                        if (nodesAdded == 0 || nodesAdded > kGradMaxSubGates) continue;
                        for (int j = 4; j <= rt; ++j) {
                            if (localIdx[j] < 0) continue;
                            int ins[2] = {hLib.fanin0[cls][j - 4], hLib.fanin1[cls][j - 4]};
                            int cs[2] = {hLib.isC0[cls][j - 4], hLib.isC1[cls][j - 4]};
                            for (int b = 0; b < 2; ++b) {
                                int en = ins[b];
                                if (en < 4) {
                                    candRefs[en]++;
                                    if ((((uPhase >> en) & 1) ^ cs[b]) != 0) candRefsNeg[en]++;
                                } else if (match[en] != -1) {
                                    addHit(match[en], 1);
                                } else {
                                    eFo[en]++;
                                }
                            }
                        }
                        gc.subGates.resize(nodesAdded);
                        for (int j = 4; j <= rt; ++j) {
                            if (localIdx[j] < 0) continue;
                            int g = localIdx[j];
                            int ins[2] = {hLib.fanin0[cls][j - 4], hLib.fanin1[cls][j - 4]};
                            int cs[2] = {hLib.isC0[cls][j - 4], hLib.isC1[cls][j - 4]};
                            int enc[2];
                            for (int b = 0; b < 2; ++b) {
                                int en = ins[b];
                                if (en < 4) {
                                    int ph = (((uPhase >> en) & 1) ^ cs[b]);
                                    enc[b] = (leaves[en] << 1) | ph;
                                } else if (match[en] != -1) {
                                    enc[b] = (match[en] << 1) | cs[b];
                                } else {
                                    enc[b] = -1 - localIdx[en];
                                }
                            }
                            float ld = (j == rt) ? 0.0f : (eFo[j] > 1 ? static_cast<float>(eFo[j] - 1) : 0.0f);
                            gc.subGates[g] = {enc[0], enc[1], ld};
                        }
                    }

                    gc.deltaAnd = nodesAdded - static_cast<int>(remAll.size());
                    gc.hitCnt = hitCnt;
                    for (int t = 0; t < hitCnt; ++t) gc.hitIds[t] = hitIds[t];
                    gc.remAll = std::move(remAll);
                    gc.remStrict = std::move(remStrict);

                    // Sparse external fanout deltas (only external leaves and hitIds, never remStrict)
                    for (int j = 0; j < 4; ++j) {
                        int lf = leaves[j];
                        if (lf <= 0) continue;
                        bool negOnly = (lf <= nPIs && params.piPhaseAware);
                        int d = negOnly ? (candRefsNeg[j] - origRefsNeg[j]) : (candRefs[j] - origRefs[j]);
                        if (d != 0) gc.foDeltas.emplace_back(lf, d);
                    }
                    for (int t = 0; t < hitCnt; ++t) {
                        if (hitDelta[t] != 0 && hitIds[t] > nPIs) gc.foDeltas.emplace_back(hitIds[t], hitDelta[t]);
                    }

                    for (int u : gc.remStrict) addUnique(reg.remStrictUnion, u);
                    for (int j = 0; j < 4; ++j) addUnique(reg.refsUnion, leaves[j]);
                    for (int t = 0; t < hitCnt; ++t) addUnique(reg.refsUnion, hitIds[t]);

                    if (ms.score > bestAdp) {
                        bestAdp = ms.score;
                        if (ms.score > 1e-9f) reg.warmBestK = static_cast<int>(reg.cands.size()) + 1;
                    }
                    if (ms.deltaJ > bestDj) bestDj = ms.deltaJ;
                    reg.cands.push_back(std::move(gc));
                }
                if (!reg.cands.empty()) {
                    // Strictly prioritize positive global ADP regions over speculative off-critical regions
                    reg.weight = (bestAdp > 1e-9f)
                        ? bestAdp
                        : (1e-11f * std::max(0.0f, bestDj) / std::max(depth, 1.0f));
                    candRegions.push_back(std::move(reg));
                }
            }

            // Critical-Cone Hybrid Partitioning:
            // Split candRegions into Critical Cone (slack <= 3*tau or sens > 0, capped at params.critMax)
            // for GPU joint-grad optimization, and Non-Critical pool for safe Stage A discrete merge.
            std::vector<int> critPool, nonCritPool;
            critPool.reserve(candRegions.size());
            if (params.critMax <= 0) {
                for (size_t i = 0; i < candRegions.size(); ++i) critPool.push_back(static_cast<int>(i));
            } else {
                nonCritPool.reserve(candRegions.size());
                for (size_t i = 0; i < candRegions.size(); ++i) {
                    int r = candRegions[i].root;
                    float sl = hReq[r] - hArr[r];
                    if (sl <= 3.0f * tau + 1e-5f || hSens[r] > 1e-6f) {
                        critPool.push_back(static_cast<int>(i));
                    } else {
                        nonCritPool.push_back(static_cast<int>(i));
                    }
                }
                if (static_cast<int>(critPool.size()) > params.critMax) {
                    std::sort(critPool.begin(), critPool.end(), [&](int a, int b) {
                        float sa = hSens[candRegions[a].root], sb = hSens[candRegions[b].root];
                        if (sa != sb) return sa > sb;
                        if (candRegions[a].weight != candRegions[b].weight) return candRegions[a].weight > candRegions[b].weight;
                        return candRegions[a].root < candRegions[b].root;
                    });
                    for (size_t i = static_cast<size_t>(params.critMax); i < critPool.size(); ++i) {
                        nonCritPool.push_back(critPool[i]);
                    }
                    critPool.resize(static_cast<size_t>(params.critMax));
                }
            }

            // Linear-time GWMIN conflict-free region selection on Critical Cone (4 strict invariants)
            std::vector<int> cntRemStrict(nObjs, 0), cntRefs(nObjs, 0), cntRoot(nObjs, 0);
            for (int idxR : critPool) {
                const auto & reg = candRegions[idxR];
                cntRoot[reg.root] = 1;
                for (int u : reg.remStrictUnion) cntRemStrict[u]++;
                for (int u : reg.refsUnion) cntRefs[u]++;
            }
            std::vector<float> gwScore(critPool.size(), 0.0f);
            std::vector<int> critOrder(critPool.size());
            for (size_t i = 0; i < critPool.size(); ++i) {
                critOrder[i] = static_cast<int>(i);
                const auto & reg = candRegions[critPool[i]];
                int deg = cntRemStrict[reg.root];
                for (int u : reg.remStrictUnion) deg += (cntRemStrict[u] - 1) + cntRefs[u] + cntRoot[u];
                for (int u : reg.refsUnion) deg += cntRemStrict[u];
                gwScore[i] = reg.weight / (1.0f + static_cast<float>(std::max(0, deg)));
            }
            std::sort(critOrder.begin(), critOrder.end(), [&](int a, int b) {
                if (gwScore[a] != gwScore[b]) return gwScore[a] > gwScore[b];
                return candRegions[critPool[a]].root < candRegions[critPool[b]].root;
            });

            std::vector<uint8_t> usedRoot(nObjs, 0), delStrict(nObjs, 0), prot(nObjs, 0), refByDownstream(nObjs, 0);
            std::vector<int> activeIdx;
            activeIdx.reserve(critPool.size());
            for (int ordPos : critOrder) {
                int idxR = critPool[ordPos];
                const auto & reg = candRegions[idxR];
                if (usedRoot[reg.root] || delStrict[reg.root]) continue;
                bool conflict = false;
                for (int u : reg.remStrictUnion) {
                    if (delStrict[u] || prot[u] || usedRoot[u]) { conflict = true; break; }
                }
                if (!conflict) {
                    for (int u : reg.refsUnion) {
                        if (delStrict[u]) { conflict = true; break; }
                    }
                }
                if (conflict) continue;

                usedRoot[reg.root] = 1;
                prot[reg.root] = 1;
                for (int u : reg.remStrictUnion) delStrict[u] = 1;
                for (int u : reg.refsUnion) { prot[u] = 1; refByDownstream[u] = 1; }
                activeIdx.push_back(idxR);
            }

            // Invariant 4: strip wire-fold candidates from any active root referenced by a downstream active region
            std::vector<int> finalActiveIdx;
            finalActiveIdx.reserve(activeIdx.size());
            for (int idxR : activeIdx) {
                auto & reg = candRegions[idxR];
                if (refByDownstream[reg.root]) {
                    std::vector<GradCandHost> kept;
                    int newWarmK = 0;
                    for (size_t ci = 0; ci < reg.cands.size(); ++ci) {
                        if (reg.cands[ci].wireLit < 0) {
                            kept.push_back(std::move(reg.cands[ci]));
                            if (reg.warmBestK == static_cast<int>(ci) + 1) newWarmK = static_cast<int>(kept.size());
                        }
                    }
                    reg.cands.swap(kept);
                    reg.warmBestK = newWarmK;
                }
                if (!reg.cands.empty()) finalActiveIdx.push_back(idxR);
            }

            int M = static_cast<int>(finalActiveIdx.size());
            if (M == 0 && nonCritPool.empty()) {
                auto t4 = std::chrono::steady_clock::now();
                msSelectRebuild += std::chrono::duration<double, std::milli>(t4 - t3).count();
                break;
            }

            std::vector<int> foBaseHost(nObjs, 0);
            for (int u = 1; u <= nPIs; ++u) foBaseHost[u] = params.piPhaseAware ? hNegFo[u] : hFo[u];
            for (int u = nPIs + 1; u < nObjs; ++u) foBaseHost[u] = hFo[u];

            auto hLoad = [](int fo) { return fo > 1 ? (fo - 1) : 0; };
            std::vector<int> accFo(nObjs, 0);
            std::vector<uint8_t> finalDelStrict(nObjs, 0), lockedRoot(nObjs, 0), lockedProt(nObjs, 0);
            std::vector<const GradCandHost *> chosenOp(nObjs, nullptr);

            auto checkFanoutBudgetHost = [&](const GradCandHost & gc) -> bool {
                if (!params.fanoutBudget) return true;
                for (const auto & kv : gc.foDeltas) {
                    int u = kv.first, d = kv.second;
                    if (d <= 0) continue;
                    bool negOnly = (u <= nPIs && params.piPhaseAware);
                    float coef = (u <= nPIs)
                        ? params.piFanoutAlpha
                        : (params.slewBeta > 0.0f ? params.fanoutAlpha * (1.0f + params.slewBeta * hSlewIn[u]) : params.fanoutAlpha);
                    if (coef <= 0.0f) continue;
                    int base = foBaseHost[u];
                    int nCnt = std::max(0, base + accFo[u] + d);
                    float inc = coef * static_cast<float>(hLoad(nCnt) - hLoad(base));
                    if (inc > 1e-6f && !(nCnt <= params.fanoutPivot && inc <= 0.5f)) {
                        float so = querySlackOtherHost(u, negOnly, nObjs, depth, params.piPhaseAware, gc.remAll,
                                                       hFoOff, hFoEdge, hArr, hReq, hGd, hNegExtra, hTop3, hNegTop3);
                        if (so < kInf * 0.5f && inc > std::max(0.0f, so) + params.budgetTol) {
                            return false;
                        }
                    }
                }
                return true;
            };

            auto commitChosenOp = [&](int root, const GradCandHost & gc) {
                for (const auto & kv : gc.foDeltas) accFo[kv.first] += kv.second;
                chosenOp[root] = &gc;
                lockedRoot[root] = 1;
                lockedProt[root] = 1;
                for (int u : gc.remStrict) finalDelStrict[u] = 1;
                for (int j = 0; j < 4; ++j) if (gc.leaves[j] > 0) lockedProt[gc.leaves[j]] = 1;
                for (int t = 0; t < gc.hitCnt; ++t) if (gc.hitIds[t] > 0) lockedProt[gc.hitIds[t]] = 1;
            };

            // Step 1: Critical-Cone GPU Joint Gradient Optimization + Live Fanout-Spike Guard
            GradSolveStats gStats;
            int critApplied = 0;
            if (M > 0) {
                GradProblem prob;
                prob.nPIs = nPIs; prob.nPOs = nPOs; prob.nObjs = nObjs; prob.nNodes = nNodes;
                prob.maxLv = maxLv; prob.depth0 = depth; prob.tau0 = tau;
                prob.f0 = &f0; prob.f1 = &f1; prob.outs = &outs; prob.lvOff = &lvOff;
                prob.foBase = foBaseHost;

                prob.M = M;
                prob.regionRoot.resize(M);
                prob.isRegionRoot.assign(nObjs, -1);
                prob.candCount.resize(M);
                prob.candWarmBestK.assign(M, 0);
                prob.candDeltaAnd.assign(M * kGradMaxK, 0);
                prob.candWireLit.assign(M * kGradMaxK, -1);
                prob.candNumGates.assign(M * kGradMaxK, 0);
                prob.candSubGates.assign(static_cast<size_t>(M) * kGradMaxK * kGradMaxSubGates, {0, 0, 0.0f});
                prob.foDeltaOffset.assign(M * kGradMaxK + 1, 0);

                for (int m = 0; m < M; ++m) {
                    const auto & reg = candRegions[finalActiveIdx[m]];
                    int r = reg.root;
                    prob.regionRoot[m] = r;
                    prob.isRegionRoot[r] = m;
                    int K = std::min(kGradMaxK, 1 + static_cast<int>(reg.cands.size()));
                    prob.candCount[m] = K;
                    prob.candWarmBestK[m] = (reg.warmBestK < K) ? reg.warmBestK : 0;

                    int slot0 = m * kGradMaxK + 0;
                    prob.candDeltaAnd[slot0] = 0;
                    prob.candWireLit[slot0] = -1;
                    prob.candNumGates[slot0] = 1;
                    prob.candSubGates[slot0 * kGradMaxSubGates + 0] = {f0[r], f1[r], 0.0f};
                    prob.foDeltaOffset[slot0 + 1] = static_cast<int>(prob.foDeltaNode.size());

                    for (int k = 1; k < K; ++k) {
                        const auto & gc = reg.cands[k - 1];
                        int slot = m * kGradMaxK + k;
                        prob.candDeltaAnd[slot] = gc.deltaAnd;
                        prob.candWireLit[slot] = gc.wireLit;
                        if (gc.wireLit < 0) {
                            int nG = static_cast<int>(gc.subGates.size());
                            prob.candNumGates[slot] = nG;
                            for (int g = 0; g < nG; ++g) {
                                prob.candSubGates[slot * kGradMaxSubGates + g] = gc.subGates[g];
                            }
                        }
                        for (const auto & kv : gc.foDeltas) {
                            prob.foDeltaNode.push_back(kv.first);
                            prob.foDeltaVal.push_back(kv.second);
                        }
                        prob.foDeltaOffset[slot + 1] = static_cast<int>(prob.foDeltaNode.size());
                    }
                    for (int k = K; k < kGradMaxK; ++k) {
                        prob.foDeltaOffset[m * kGradMaxK + k + 1] = static_cast<int>(prob.foDeltaNode.size());
                    }
                }

                std::vector<int> chosenCand;
                std::vector<float> chosenProb;
                gStats = solveJointGradGPU(prob, params, chosenCand, &chosenProb);

                // Gradient-Stall Guarded Fallback with Global Discrete Monotonicity Gate:
                // Only when joint-grad truly stalls (discD >= D0 && raw < 10% M), build a trial
                // live fanout-budgeted greedy ADP selection and accept it IFF it strictly reduces
                // exact global discrete delay (fbDiscD < D0 - 1e-4).
                bool gradStalled = (gStats.bestDiscDelay >= depth - 1e-4f &&
                                    gStats.selectedNonZero < std::max(1, M / 10));
                bool usedFallback = false;
                if (gradStalled) {
                    std::vector<int> fbOrder;
                    fbOrder.reserve(candRegions.size());
                    for (size_t i = 0; i < candRegions.size(); ++i) {
                        if (candRegions[i].warmBestK > 0) fbOrder.push_back(static_cast<int>(i));
                    }
                    std::sort(fbOrder.begin(), fbOrder.end(), [&](int a, int b) {
                        if (candRegions[a].weight != candRegions[b].weight)
                            return candRegions[a].weight > candRegions[b].weight;
                        return candRegions[a].root < candRegions[b].root;
                    });
                    std::vector<int> fbAccFo(nObjs, 0);
                    std::vector<uint8_t> fbDelStrict(nObjs, 0), fbLockedRoot(nObjs, 0), fbLockedProt(nObjs, 0);
                    std::vector<const GradCandHost *> fbOp(nObjs, nullptr);
                    int fbCount = 0;
                    for (int idxR : fbOrder) {
                        const auto & reg = candRegions[idxR];
                        const auto & gc = reg.cands[reg.warmBestK - 1];
                        if (fbLockedRoot[reg.root] || fbDelStrict[reg.root]) continue;
                        bool conflict = false;
                        for (int u : gc.remStrict) {
                            if (fbDelStrict[u] || fbLockedProt[u] || fbLockedRoot[u]) { conflict = true; break; }
                        }
                        if (!conflict) {
                            for (int j = 0; j < 4; ++j) {
                                if (gc.leaves[j] > 0 && fbDelStrict[gc.leaves[j]]) { conflict = true; break; }
                            }
                        }
                        if (!conflict) {
                            for (int t = 0; t < gc.hitCnt; ++t) {
                                if (gc.hitIds[t] > 0 && fbDelStrict[gc.hitIds[t]]) { conflict = true; break; }
                            }
                        }
                        if (conflict) continue;
                        bool budgetOk = true;
                        if (params.fanoutBudget) {
                            for (const auto & kv : gc.foDeltas) {
                                int u = kv.first, d = kv.second;
                                if (d <= 0) continue;
                                bool negOnly = (u <= nPIs && params.piPhaseAware);
                                float coef = (u <= nPIs)
                                    ? params.piFanoutAlpha
                                    : (params.slewBeta > 0.0f ? params.fanoutAlpha * (1.0f + params.slewBeta * hSlewIn[u]) : params.fanoutAlpha);
                                if (coef <= 0.0f) continue;
                                int base = foBaseHost[u];
                                int nCnt = std::max(0, base + fbAccFo[u] + d);
                                float inc = coef * static_cast<float>(hLoad(nCnt) - hLoad(base));
                                if (inc > 1e-6f && !(nCnt <= params.fanoutPivot && inc <= 0.5f)) {
                                    float so = querySlackOtherHost(u, negOnly, nObjs, depth, params.piPhaseAware, gc.remAll,
                                                                   hFoOff, hFoEdge, hArr, hReq, hGd, hNegExtra, hTop3, hNegTop3);
                                    if (so < kInf * 0.5f && inc > std::max(0.0f, so) + params.budgetTol) {
                                        budgetOk = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if (!budgetOk) continue;
                        for (const auto & kv : gc.foDeltas) fbAccFo[kv.first] += kv.second;
                        fbOp[reg.root] = &gc;
                        fbLockedRoot[reg.root] = 1;
                        fbLockedProt[reg.root] = 1;
                        for (int u : gc.remStrict) fbDelStrict[u] = 1;
                        for (int j = 0; j < 4; ++j) if (gc.leaves[j] > 0) fbLockedProt[gc.leaves[j]] = 1;
                        for (int t = 0; t < gc.hitCnt; ++t) if (gc.hitIds[t] > 0) fbLockedProt[gc.hitIds[t]] = 1;
                        fbCount++;
                    }

                    if (fbCount > 0) {
                        std::vector<float> fbArr(nObjs, 0.0f), fbSlew(nObjs, 0.0f);
                        auto readFbLit = [&](int lit, float & aOut, float & sOut) {
                            int u = lit >> 1, c = lit & 1;
                            if (u <= 0) { aOut = 0.0f; sOut = 0.0f; return; }
                            if (u <= nPIs) {
                                if (!params.piPhaseAware || c != 0) {
                                    float ld = static_cast<float>(hLoad(std::max(0, foBaseHost[u] + fbAccFo[u])));
                                    aOut = params.piFanoutAlpha * ld;
                                    sOut = ld;
                                } else {
                                    aOut = 0.0f; sOut = 0.0f;
                                }
                                return;
                            }
                            aOut = fbArr[u]; sOut = fbSlew[u];
                        };
                        for (int id = nPIs + 1; id < nObjs; ++id) {
                            if (fbDelStrict[id]) continue;
                            const GradCandHost * op = fbOp[id];
                            float rootLd = static_cast<float>(hLoad(std::max(0, foBaseHost[id] + fbAccFo[id])));
                            if (!op) {
                                float a0, s0, a1, s1;
                                readFbLit(f0[id], a0, s0);
                                readFbLit(f1[id], a1, s1);
                                float gd = 1.0f + params.fanoutAlpha * rootLd * (1.0f + params.slewBeta * std::max(s0, s1));
                                fbArr[id] = gd + std::max(a0, a1);
                                fbSlew[id] = rootLd;
                            } else if (op->wireLit >= 0) {
                                readFbLit(op->wireLit, fbArr[id], fbSlew[id]);
                            } else {
                                int nG = static_cast<int>(op->subGates.size());
                                float locA[kGradMaxSubGates], locS[kGradMaxSubGates];
                                for (int g = 0; g < nG; ++g) {
                                    const SubGate & sg = op->subGates[g];
                                    float a0, s0, a1, s1;
                                    if (sg.in0 < 0) { a0 = locA[-1 - sg.in0]; s0 = locS[-1 - sg.in0]; }
                                    else readFbLit(sg.in0, a0, s0);
                                    if (sg.in1 < 0) { a1 = locA[-1 - sg.in1]; s1 = locS[-1 - sg.in1]; }
                                    else readFbLit(sg.in1, a1, s1);
                                    float ld = (g == nG - 1) ? rootLd : sg.internalLoad;
                                    float gd = 1.0f + params.fanoutAlpha * ld * (1.0f + params.slewBeta * std::max(s0, s1));
                                    locA[g] = gd + std::max(a0, a1);
                                    locS[g] = ld;
                                }
                                fbArr[id] = locA[nG - 1];
                                fbSlew[id] = locS[nG - 1];
                            }
                        }
                        float fbDiscD = 0.0f;
                        for (int l : outs) {
                            float a, s;
                            readFbLit(l, a, s);
                            if (a > fbDiscD) fbDiscD = a;
                        }
                        if (fbDiscD < depth - 1e-4f) {
                            accFo.swap(fbAccFo);
                            finalDelStrict.swap(fbDelStrict);
                            lockedRoot.swap(fbLockedRoot);
                            lockedProt.swap(fbLockedProt);
                            chosenOp.swap(fbOp);
                            critApplied = fbCount;
                            gStats.bestTheta = 0.0f;
                            gStats.bestDiscDelay = fbDiscD;
                            usedFallback = true;
                        }
                    }
                }
                if (!usedFallback && gStats.selectedNonZero > 0) {
                    std::vector<int> selRegions;
                    selRegions.reserve(gStats.selectedNonZero);
                    for (int m = 0; m < M; ++m) {
                        if (chosenCand[m] > 0) selRegions.push_back(m);
                    }
                    std::sort(selRegions.begin(), selRegions.end(), [&](int a, int b) {
                        if (chosenProb[a] != chosenProb[b]) return chosenProb[a] > chosenProb[b];
                        return candRegions[finalActiveIdx[a]].weight > candRegions[finalActiveIdx[b]].weight;
                    });
                    for (int m : selRegions) {
                        const auto & reg = candRegions[finalActiveIdx[m]];
                        const auto & gc = reg.cands[chosenCand[m] - 1];
                        if (!checkFanoutBudgetHost(gc)) continue;
                        commitChosenOp(reg.root, gc);
                        critApplied++;
                    }
                }
            }

            // Step 2: Non-Critical Safe Discrete Merge (Stage A rules on remaining conflict-free regions)
            // Only safe area-saving / zero-area non-degrading rewrites (deltaAnd <= 0, deltaJ >= 0, score > 0)
            // obeying the live accFo and slack_other redline updated by Step 1.
            int nonCritApplied = 0;
            if (!nonCritPool.empty()) {
                struct SafeNonCritItem {
                    int regIdx;
                    int candIdx;
                    float score;
                };
                std::vector<SafeNonCritItem> safeItems;
                safeItems.reserve(nonCritPool.size());
                for (int idxR : nonCritPool) {
                    const auto & reg = candRegions[idxR];
                    int bestCi = -1;
                    float bestSc = 1e-9f;
                    for (size_t ci = 0; ci < reg.cands.size(); ++ci) {
                        const auto & gc = reg.cands[ci];
                        if (gc.deltaAnd <= 0 && gc.deltaJ >= -1e-5f && gc.score > bestSc) {
                            bestSc = gc.score;
                            bestCi = static_cast<int>(ci);
                        }
                    }
                    if (bestCi >= 0) safeItems.push_back({idxR, bestCi, bestSc});
                }
                std::sort(safeItems.begin(), safeItems.end(), [&](const SafeNonCritItem & a, const SafeNonCritItem & b) {
                    if (a.score != b.score) return a.score > b.score;
                    return candRegions[a.regIdx].root < candRegions[b.regIdx].root;
                });
                for (const auto & item : safeItems) {
                    const auto & reg = candRegions[item.regIdx];
                    const auto & gc = reg.cands[item.candIdx];
                    if (lockedRoot[reg.root] || finalDelStrict[reg.root]) continue;
                    bool conflict = false;
                    for (int u : gc.remStrict) {
                        if (finalDelStrict[u] || lockedProt[u] || lockedRoot[u]) { conflict = true; break; }
                    }
                    if (!conflict) {
                        for (int j = 0; j < 4; ++j) {
                            if (gc.leaves[j] > 0 && finalDelStrict[gc.leaves[j]]) { conflict = true; break; }
                        }
                    }
                    if (!conflict) {
                        for (int t = 0; t < gc.hitCnt; ++t) {
                            if (gc.hitIds[t] > 0 && finalDelStrict[gc.hitIds[t]]) { conflict = true; break; }
                        }
                    }
                    if (conflict) continue;
                    if (!checkFanoutBudgetHost(gc)) continue;
                    commitChosenOp(reg.root, gc);
                    nonCritApplied++;
                }
            }

            int guardedApplied = critApplied + nonCritApplied;
            if (params.verbose >= 1) {
                printf("rw4b-grad round %d: M=%d applied=%d(raw=%d) z0=%.1f theta=%.2f avgMaxP=%.3f loss=%.5f->%.5f discD=%.2f(D0=%.2f) dAND=%d grad=%.1fms disc=%.1fms crit=%d nc=%d\n",
                       rnd + 1, M, guardedApplied, gStats.selectedNonZero, gStats.chosenZ0, gStats.bestTheta, gStats.avgMaxProb,
                       gStats.initLoss, gStats.finalLoss, gStats.bestDiscDelay, depth,
                       gStats.bestDiscDeltaAnd, gStats.msGradLoop, gStats.msDiscSearch, critApplied, nonCritApplied);
            }

            if (guardedApplied == 0) {
                auto t4 = std::chrono::steady_clock::now();
                msSelectRebuild += std::chrono::duration<double, std::milli>(t4 - t3).count();
                break;
            }
            totalApplied += guardedApplied;

            StrashRebuild builder(nPIs, nNodes, 0);
            std::vector<int> litMap(nObjs, 0);
            for (int i = 0; i <= nPIs; ++i) litMap[i] = i << 1;

            for (int id = nPIs + 1; id < nObjs; ++id) {
                if (finalDelStrict[id]) continue;
                const GradCandHost * op = chosenOp[id];
                if (!op) {
                    int nl0 = litMap[f0[id] >> 1] ^ (f0[id] & 1);
                    int nl1 = litMap[f1[id] >> 1] ^ (f1[id] & 1);
                    litMap[id] = builder.andLit(nl0, nl1, false);
                } else {
                    int localLit[54];
                    for (int j = 0; j < 4; ++j) {
                        int base = op->leaves[j] > 0 ? litMap[op->leaves[j]] : 0;
                        localLit[j] = base ^ (((op->uPhase >> j) & 1) ? 1 : 0);
                    }
                    uint64_t um = 1ULL << op->rt;
                    for (int j = op->rt; j >= 4; --j) {
                        if ((um >> j) & 1ULL) {
                            if (op->match[j] != -1) continue;
                            um |= (1ULL << hLib.fanin0[op->cls][j - 4]);
                            um |= (1ULL << hLib.fanin1[op->cls][j - 4]);
                        }
                    }
                    for (int j = 4; j <= op->rt; ++j) {
                        if (!((um >> j) & 1ULL)) continue;
                        if (op->match[j] != -1) {
                            localLit[j] = litMap[op->match[j]];
                            continue;
                        }
                        int i0 = hLib.fanin0[op->cls][j - 4], i1 = hLib.fanin1[op->cls][j - 4];
                        int nl0 = localLit[i0] ^ (hLib.isC0[op->cls][j - 4] ? 1 : 0);
                        int nl1 = localLit[i1] ^ (hLib.isC1[op->cls][j - 4] ? 1 : 0);
                        localLit[j] = builder.andLit(nl0, nl1, false);
                    }
                    litMap[id] = localLit[op->rt] ^ op->flip;
                }
            }

            std::vector<int> newOuts(nPOs);
            for (int i = 0; i < nPOs; ++i) newOuts[i] = litMap[outs[i] >> 1] ^ (outs[i] & 1);

            int bObjs = static_cast<int>(builder.f0.size());
            std::vector<uint8_t> keep(bObjs, 0);
            std::vector<int> stk;
            stk.reserve(bObjs);
            for (int l : newOuts) {
                int n = l >> 1;
                if (n > nPIs && !keep[n]) { keep[n] = 1; stk.push_back(n); }
            }
            while (!stk.empty()) {
                int u = stk.back(); stk.pop_back();
                for (int s : {builder.f0[u] >> 1, builder.f1[u] >> 1}) {
                    if (s > nPIs && !keep[s]) { keep[s] = 1; stk.push_back(s); }
                }
            }
            std::vector<int> remap(bObjs, 0);
            for (int i = 0; i <= nPIs; ++i) remap[i] = i;
            std::vector<int> nextF0(nPIs + 1, 0), nextF1(nPIs + 1, 0);
            for (int i = nPIs + 1; i < bObjs; ++i) {
                if (!keep[i]) continue;
                int nid = static_cast<int>(nextF0.size());
                remap[i] = nid;
                int l0 = (remap[builder.f0[i] >> 1] << 1) | (builder.f0[i] & 1);
                int l1 = (remap[builder.f1[i] >> 1] << 1) | (builder.f1[i] & 1);
                if (l0 > l1) std::swap(l0, l1);
                nextF0.push_back(l0);
                nextF1.push_back(l1);
            }
            for (int i = 0; i < nPOs; ++i) newOuts[i] = (remap[newOuts[i] >> 1] << 1) | (newOuts[i] & 1);
            f0.swap(nextF0); f1.swap(nextF1); outs.swap(newOuts);

            auto t4 = std::chrono::steady_clock::now();
            msSelectRebuild += std::chrono::duration<double, std::milli>(t4 - t3).count();
            continue;
        }

        evaluateNode4BKernel<<<(nNodes + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            nPIs, nObjs, P, depth, phi, d_f0, d_f1, d_fo, d_negFo, d_foOff, d_foEdge,
            d_arr, d_req, d_gd, d_negExtra, d_outSlew, d_negSlew, d_slewIn, d_sens, d_slewCoef, d_slewCoefNeg,
            d_top3, d_negTop3, d_cuts, d_lib, d_ht,
            params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta,
            params.lamArea, params.fanoutReg, params.fanoutPivot, tau,
            params.budgetTol, params.piPhaseAware, params.strashAware, params.fanoutBudget,
            params.adpScore, params.adpW, params.slackKappa, params.foCap, d_candOut);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t3 = std::chrono::steady_clock::now();
        msEval += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 4. Copy compact candidate winners + timing/bounds to host for lazy fanout-budget selection
        std::vector<CandResult> hCand(static_cast<size_t>(nObjs) * kTopK);
        std::vector<int> hFo(nObjs), hNegFo(nObjs), hFoOff(nObjs + 1), hFoEdge(totalEdges);
        std::vector<float> hArr(nObjs), hReq(nObjs), hGd(nObjs), hNegExtra(nObjs), hSlewIn(nObjs);
        std::vector<Top3Bounds> hTop3(nObjs), hNegTop3(nPIs + 1);
        std::vector<Cut4B> hCuts(static_cast<size_t>(nObjs) * kCutSetSize);
        CHECK_CUDA(cudaMemcpy(hCand.data(), d_candOut, static_cast<size_t>(nObjs) * kTopK * sizeof(CandResult), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hFo.data(), d_fo, nObjs * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hNegFo.data(), d_negFo, nObjs * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hFoOff.data(), d_foOff, (nObjs + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hFoEdge.data(), d_foEdge, totalEdges * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hArr.data(), d_arr, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hReq.data(), d_req, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hGd.data(), d_gd, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hNegExtra.data(), d_negExtra, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hSlewIn.data(), d_slewIn, nObjs * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hTop3.data(), d_top3, nObjs * sizeof(Top3Bounds), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hNegTop3.data(), d_negTop3, (nPIs + 1) * sizeof(Top3Bounds), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hCuts.data(), d_cuts, static_cast<size_t>(nObjs) * kCutSetSize * sizeof(Cut4B), cudaMemcpyDeviceToHost));

        // Free round GPU buffers early
        cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_outs); cudaFree(d_fo); cudaFree(d_negFo); cudaFree(d_slewArg);
        cudaFree(d_foOff); cudaFree(d_foCursor); cudaFree(d_foEdge);
        cudaFree(d_arr); cudaFree(d_req); cudaFree(d_reqNeg); cudaFree(d_gd);
        cudaFree(d_negExtra); cudaFree(d_outSlew); cudaFree(d_negSlew); cudaFree(d_slewIn);
        cudaFree(d_sens); cudaFree(d_slewCoef); cudaFree(d_slewCoefNeg); cudaFree(d_poArr); cudaFree(d_phi);
        cudaFree(d_top3); cudaFree(d_negTop3);
        cudaFree(d_cuts); cudaFree(d_ht); cudaFree(d_candOut);

        // Build host strash map of current AIG for exact touched/inputs budget accounting
        robin_hood::unordered_flat_map<uint64_t, int> origStrash;
        if (params.strashAware) {
            origStrash.reserve(nNodes * 2 + 64);
            for (int id = nPIs + 1; id < nObjs; ++id) {
                int l0 = f0[id], l1 = f1[id];
                if (l0 > l1) std::swap(l0, l1);
                uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(l0)) << 32) | static_cast<uint32_t>(l1);
                origStrash.emplace(key, id);
            }
        }

        struct TouchEntry {
            int node;
            bool negOnly;
            int d;
            float coef;
            int base;
            float slackOther;
        };
        struct InputEntry {
            int node;
            bool negOnly;
            float coef;
            int base;
        };
        struct SelectedItem {
            int root;
            float score;
            float maxArr;
            float req;
            int da;
            int subgr;
            int cls;
            int rt;
            int uPhase;
            int flip;
            bool allowDup;
            int leaves[4];
            int hitCnt;
            int hitIds[8];
            int match[54];
            std::vector<int> remAll;
            std::vector<int> remStrict;
            std::vector<TouchEntry> touched;
            std::vector<InputEntry> inputs;
        };

        auto hostLoad = [](int fo) { return fo > 1 ? (fo - 1) : 0; };
        auto hostHq = [&](int x) { int h = x - params.fanoutPivot; return h > 0 ? h * h : 0; };
        auto hostCoef = [&](int u, bool negOnly) {
            if (u <= nPIs) return params.piFanoutAlpha;
            return params.slewBeta > 0.0f ? params.fanoutAlpha * (1.0f + params.slewBeta * hSlewIn[u]) : params.fanoutAlpha;
        };

        std::vector<SelectedItem> entries;
        entries.reserve(nNodes / 2);

        float minScore = params.adpScore ? 1e-9f : 1e-6f;
        float regScale = params.adpScore ? (1.0f / std::max(depth, 1.0f)) : 1.0f;

        for (int root = nPIs + 1; root < nObjs; ++root) {
            int kLimit = std::min(kTopK, std::max(1, params.perRootK));
            for (int k = 0; k < kLimit; ++k) {
                const CandResult & cr = hCand[static_cast<size_t>(root) * kTopK + k];
                if (cr.score <= minScore || cr.cutSlot < 0) continue;

                Cut4B cut = hCuts[static_cast<size_t>(root) * kCutSetSize + cr.cutSlot];
                if (!cut.used || cut.nLeaves < 3) continue;
                if (cut.nLeaves == 3) cut.leaves[3] = 0;

                int uPhase = hLib.pPhases[cut.truthtable];
                int cls = hLib.pMap[cut.truthtable];
                const int * perm = hLib.pPerms4[hLib.pPerms[cut.truthtable]];
                int leaves[4];
                for (int j = 0; j < 4; ++j) leaves[j] = cut.leaves[perm[j]];

                int rt = hLib.pSubgr[cls][cr.subgr];
                bool ttOk = false;
                uint16_t origTT = computeConeTruth(root, leaves, nPIs, f0, f1, ttOk);
                int flip = 0;
                if (!ttOk || !verifyCandFlip(origTT, cls, rt, uPhase, hLib, flip)) continue;

                std::vector<int> remAll, remStrict;
                collectBoundedMffcHost(root, leaves, nPIs, f0, f1, hFo, remAll, remStrict);

                // Compute exact strash match on host
                int match[54];
                uint64_t isC = 0;
                for (int j = 0; j < 54; ++j) match[j] = -1;
                for (int j = 0; j < 4; ++j) {
                    match[j] = leaves[j];
                    if ((uPhase >> j) & 1) isC |= (1ULL << j);
                }
                auto inRemAll = [&](int u) {
                    for (int r : remAll) if (r == u) return true;
                    return false;
                };
                float slackRootH = cr.req - hArr[root];
                bool isNearCritH = (params.slackKappa <= 0.0f) ? true : (slackRootH <= params.slackKappa * tau + 1e-5f);
                bool allowDupOp = (params.foCap > 0 && isNearCritH && cr.da > 0);
                if (params.strashAware) {
                    int nLibN = hLib.nNodes[cls];
                    for (int j = 0; j < nLibN; ++j) {
                        int num = j + 4;
                        int i0 = hLib.fanin0[cls][j], i1 = hLib.fanin1[cls][j];
                        if (match[i0] == -1 || match[i1] == -1 || match[i0] == root || match[i1] == root) continue;
                        int c0 = ((isC >> i0) & 1) ^ hLib.isC0[cls][j];
                        int c1 = ((isC >> i1) & 1) ^ hLib.isC1[cls][j];
                        int l0 = (match[i0] << 1) | c0;
                        int l1 = (match[i1] << 1) | c1;
                        if (l0 > l1) std::swap(l0, l1);
                        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(l0)) << 32) | static_cast<uint32_t>(l1);
                        auto it = origStrash.find(key);
                        if (it != origStrash.end() && it->second < root && !inRemAll(it->second)) {
                            if (!(allowDupOp && hFo[it->second] >= params.foCap)) {
                                match[num] = it->second;
                            }
                        }
                    }
                }

                // Exact touched & inputs lists
                int origRefs[4] = {0, 0, 0, 0}, origRefsNeg[4] = {0, 0, 0, 0};
                for (int u : remAll) {
                    int lits[2] = {f0[u], f1[u]};
                    for (int l : lits) {
                        int s = l >> 1;
                        for (int j = 0; j < 4; ++j) {
                            if (leaves[j] == s) {
                                origRefs[j]++;
                                if (l & 1) origRefsNeg[j]++;
                            }
                        }
                    }
                }
                int candRefs[4] = {0, 0, 0, 0}, candRefsNeg[4] = {0, 0, 0, 0};
                int hitIds[8], hitDelta[8], hitCnt = 0;
                auto addHit = [&](int u, int d) {
                    if (u <= 0) return;
                    for (int t = 0; t < hitCnt; ++t) if (hitIds[t] == u) { hitDelta[t] += d; return; }
                    if (hitCnt < 8) { hitIds[hitCnt] = u; hitDelta[hitCnt] = d; hitCnt++; }
                };
                std::vector<InputEntry> inputs;
                auto addInput = [&](int u, bool negOnly) {
                    if (u <= 0) return;
                    for (const auto & e : inputs) if (e.node == u && e.negOnly == negOnly) return;
                    float coef = hostCoef(u, negOnly);
                    if (coef <= 0.0f) return;
                    int base = negOnly ? hNegFo[u] : hFo[u];
                    inputs.push_back({u, negOnly, coef, base});
                };

                if (rt < 4) {
                    candRefs[rt] += hFo[root];
                    if (leaves[rt] > 0 && leaves[rt] <= nPIs && params.piPhaseAware) {
                        int p = (((uPhase >> rt) & 1) ^ flip);
                        int rNeg = hNegFo[root];
                        int moved = (p == 0) ? rNeg : (hFo[root] - rNeg);
                        candRefsNeg[rt] += moved;
                    }
                    if (leaves[rt] > 0) addInput(leaves[rt], (leaves[rt] <= nPIs && params.piPhaseAware));
                } else if (match[rt] != -1) {
                    addHit(match[rt], hFo[root]);
                    addInput(match[rt], false);
                } else {
                    uint64_t um = 1ULL << rt;
                    for (int j = rt; j >= 4; --j) {
                        if ((um >> j) & 1ULL) {
                            if (match[j] != -1) continue;
                            um |= (1ULL << hLib.fanin0[cls][j - 4]);
                            um |= (1ULL << hLib.fanin1[cls][j - 4]);
                        }
                    }
                    for (int j = 4; j <= rt; ++j) {
                        if (!((um >> j) & 1ULL) || match[j] != -1) continue;
                        int ins[2] = {hLib.fanin0[cls][j - 4], hLib.fanin1[cls][j - 4]};
                        int cs[2] = {hLib.isC0[cls][j - 4], hLib.isC1[cls][j - 4]};
                        for (int m = 0; m < 2; ++m) {
                            int en = ins[m];
                            if (en < 4) {
                                candRefs[en]++;
                                bool isNeg = (((uPhase >> en) & 1) ^ cs[m]) != 0;
                                if (isNeg) candRefsNeg[en]++;
                                if (leaves[en] > 0) {
                                    bool piNeg = (leaves[en] <= nPIs && params.piPhaseAware);
                                    if (!piNeg || isNeg) addInput(leaves[en], piNeg);
                                }
                            } else if (match[en] != -1) {
                                addHit(match[en], 1);
                                addInput(match[en], false);
                            }
                        }
                    }
                }

                std::vector<TouchEntry> touched;
                for (int j = 0; j < 4; ++j) {
                    int lf = leaves[j];
                    if (lf <= 0) continue;
                    bool negOnly = (lf <= nPIs && params.piPhaseAware);
                    int d = negOnly ? (candRefsNeg[j] - origRefsNeg[j]) : (candRefs[j] - origRefs[j]);
                    if (d == 0) continue;
                    float coef = hostCoef(lf, negOnly);
                    if (coef <= 0.0f) continue;
                    int base = negOnly ? hNegFo[lf] : hFo[lf];
                    float so = (d > 0 && params.fanoutBudget)
                        ? querySlackOtherHost(lf, negOnly, nObjs, depth, params.piPhaseAware, remAll,
                                              hFoOff, hFoEdge, hArr, hReq, hGd, hNegExtra, hTop3, hNegTop3)
                        : kInf;
                    touched.push_back({lf, negOnly, d, coef, base, so});
                }
                for (int t = 0; t < hitCnt; ++t) {
                    int u = hitIds[t], d = hitDelta[t];
                    if (d == 0 || u <= nPIs) continue;
                    float coef = hostCoef(u, false);
                    int base = hFo[u];
                    float so = (d > 0 && params.fanoutBudget)
                        ? querySlackOtherHost(u, false, nObjs, depth, params.piPhaseAware, remAll,
                                              hFoOff, hFoEdge, hArr, hReq, hGd, hNegExtra, hTop3, hNegTop3)
                        : kInf;
                    touched.push_back({u, false, d, coef, base, so});
                }

                SelectedItem item;
                item.root = root;
                item.score = cr.score;
                item.maxArr = cr.maxArr;
                item.req = cr.req;
                item.da = cr.da;
                item.subgr = cr.subgr;
                item.cls = cls;
                item.rt = rt;
                item.uPhase = uPhase;
                item.flip = flip;
                item.allowDup = allowDupOp;
                for (int j = 0; j < 4; ++j) item.leaves[j] = leaves[j];
                item.hitCnt = hitCnt;
                for (int t = 0; t < hitCnt; ++t) item.hitIds[t] = hitIds[t];
                for (int j = 0; j < 54; ++j) item.match[j] = match[j];
                item.remAll = std::move(remAll);
                item.remStrict = std::move(remStrict);
                item.touched = std::move(touched);
                item.inputs = std::move(inputs);
                entries.push_back(std::move(item));
            }
        }

        // Lazy greedy non-conflicting selection with live fanout budget (exact port of select4b._select_with_budget)
        struct HeapNode {
            float score;
            int negDa;
            int negRoot;
            int negSubgr;
            int idx;
            bool operator<(const HeapNode & o) const {
                if (score != o.score) return score < o.score;
                if (negDa != o.negDa) return negDa < o.negDa;
                if (negRoot != o.negRoot) return negRoot < o.negRoot;
                return negSubgr < o.negSubgr;
            }
        };
        std::priority_queue<HeapNode> pq;
        for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
            const auto & e = entries[i];
            pq.push({e.score, -e.da, -e.root, -e.subgr, i});
        }

        std::vector<uint8_t> usedRoot(nObjs, 0), delStrict(nObjs, 0), prot(nObjs, 0);
        std::vector<int> accFo(nObjs, 0), accNegFo(nPIs + 1, 0);
        std::vector<uint8_t> requeueCnt(entries.size(), 0);
        std::vector<int> chosenIdx(nObjs, -1);
        int chosenCount = 0;

        while (!pq.empty()) {
            HeapNode top = pq.top();
            pq.pop();
            const SelectedItem & op = entries[top.idx];
            if (usedRoot[op.root] || delStrict[op.root]) continue;

            bool conflict = false;
            for (int j = 0; j < 4; ++j)
                if (op.leaves[j] > 0 && delStrict[op.leaves[j]]) { conflict = true; break; }
            if (!conflict) {
                for (int t = 0; t < op.hitCnt; ++t) if (delStrict[op.hitIds[t]]) { conflict = true; break; }
            }
            if (!conflict) {
                for (int u : op.remStrict) if (prot[u]) { conflict = true; break; }
            }
            if (!conflict) {
                for (int u : op.remAll) if (delStrict[u]) { conflict = true; break; }
            }
            if (conflict) continue;

            bool ok = true;
            float extraReg = 0.0f;
            if (params.fanoutBudget) {
                for (const auto & te : op.touched) {
                    int a_ = te.negOnly ? accNegFo[te.node] : accFo[te.node];
                    if (a_ != 0 && params.fanoutReg > 0.0f) {
                        extraReg += params.fanoutReg * float(
                            (hostHq(te.base + a_ + te.d) - hostHq(te.base + a_)) -
                            (hostHq(te.base + te.d) - hostHq(te.base)));
                    }
                    if (te.d > 0 && te.slackOther < kInf * 0.5f) {
                        int nCnt = std::max(0, te.base + a_ + te.d);
                        float inc = te.coef * float(hostLoad(nCnt) - hostLoad(te.base));
                        if (inc > 1e-6f && !(nCnt <= params.fanoutPivot && inc <= 0.5f) &&
                            inc > std::max(0.0f, te.slackOther) + params.budgetTol) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok) {
                    float arrInc = 0.0f;
                    for (const auto & ie : op.inputs) {
                        int a_ = ie.negOnly ? accNegFo[ie.node] : accFo[ie.node];
                        if (a_ > 0) {
                            float inc = ie.coef * float(hostLoad(ie.base + a_) - hostLoad(ie.base));
                            if (inc > arrInc) arrInc = inc;
                        }
                    }
                    if (op.maxArr + arrInc > op.req + 1e-5f) ok = false;
                }
            }
            if (!ok) continue;

            float liveScore = op.score - extraReg * regScale;
            if (liveScore <= minScore) continue;
            if (params.fanoutBudget && requeueCnt[top.idx] < 4 && !pq.empty() && liveScore < pq.top().score - minScore) {
                requeueCnt[top.idx]++;
                pq.push({liveScore, top.negDa, top.negRoot, top.negSubgr, top.idx});
                continue;
            }

            usedRoot[op.root] = 1;
            for (int u : op.remStrict) delStrict[u] = 1;
            for (int u : op.remAll) prot[u] = 1;
            for (int j = 0; j < 4; ++j) if (op.leaves[j] > 0) prot[op.leaves[j]] = 1;
            for (int t = 0; t < op.hitCnt; ++t) prot[op.hitIds[t]] = 1;
            for (const auto & te : op.touched) {
                if (te.negOnly) accNegFo[te.node] += te.d;
                else accFo[te.node] += te.d;
            }
            chosenIdx[op.root] = top.idx;
            chosenCount++;
        }

        if (chosenCount == 0) {
            auto t4 = std::chrono::steady_clock::now();
            msSelectRebuild += std::chrono::duration<double, std::milli>(t4 - t3).count();
            break;
        }
        totalApplied += chosenCount;

        // Exact topological strashed rebuild (skip deleted remStrict, stop at matched nodes, P0-2 allowDup gate)
        StrashRebuild builder(nPIs, nNodes, params.foCap);
        std::vector<int> litMap(nObjs, 0);
        for (int i = 0; i <= nPIs; ++i) litMap[i] = i << 1;

        for (int id = nPIs + 1; id < nObjs; ++id) {
            if (delStrict[id]) continue; // P0-1: skip deleted MFFC interior nodes so they never inflate foCnt
            int cIdx = chosenIdx[id];
            if (cIdx < 0) {
                int nl0 = litMap[f0[id] >> 1] ^ (f0[id] & 1);
                int nl1 = litMap[f1[id] >> 1] ^ (f1[id] & 1);
                litMap[id] = builder.andLit(nl0, nl1, false);
            } else {
                const SelectedItem & op = entries[cIdx];
                int localLit[54];
                for (int j = 0; j < 4; ++j) {
                    int base = op.leaves[j] > 0 ? litMap[op.leaves[j]] : 0;
                    localLit[j] = base ^ (((op.uPhase >> j) & 1) ? 1 : 0);
                }
                uint64_t um = 1ULL << op.rt;
                for (int j = op.rt; j >= 4; --j) {
                    if ((um >> j) & 1ULL) {
                        if (op.match[j] != -1) continue; // P0-1: stop expansion at strash-matched node
                        um |= (1ULL << hLib.fanin0[op.cls][j - 4]);
                        um |= (1ULL << hLib.fanin1[op.cls][j - 4]);
                    }
                }
                for (int j = 4; j <= op.rt; ++j) {
                    if (!((um >> j) & 1ULL)) continue;
                    if (op.match[j] != -1) {
                        localLit[j] = litMap[op.match[j]];
                        continue;
                    }
                    int i0 = hLib.fanin0[op.cls][j - 4], i1 = hLib.fanin1[op.cls][j - 4];
                    int nl0 = localLit[i0] ^ (hLib.isC0[op.cls][j - 4] ? 1 : 0);
                    int nl1 = localLit[i1] ^ (hLib.isC1[op.cls][j - 4] ? 1 : 0);
                    localLit[j] = builder.andLit(nl0, nl1, op.allowDup);
                }
                litMap[id] = localLit[op.rt] ^ op.flip;
            }
        }

        std::vector<int> newOuts(nPOs);
        for (int i = 0; i < nPOs; ++i) newOuts[i] = litMap[outs[i] >> 1] ^ (outs[i] & 1);

        // Prune dead nodes reachable from POs
        int bObjs = static_cast<int>(builder.f0.size());
        std::vector<uint8_t> keep(bObjs, 0);
        std::vector<int> stk;
        stk.reserve(bObjs);
        for (int l : newOuts) {
            int n = l >> 1;
            if (n > nPIs && !keep[n]) { keep[n] = 1; stk.push_back(n); }
        }
        while (!stk.empty()) {
            int u = stk.back();
            stk.pop_back();
            for (int s : {builder.f0[u] >> 1, builder.f1[u] >> 1}) {
                if (s > nPIs && !keep[s]) { keep[s] = 1; stk.push_back(s); }
            }
        }
        std::vector<int> remap(bObjs, 0);
        for (int i = 0; i <= nPIs; ++i) remap[i] = i;
        std::vector<int> nextF0(nPIs + 1, 0), nextF1(nPIs + 1, 0);
        for (int i = nPIs + 1; i < bObjs; ++i) {
            if (!keep[i]) continue;
            int nid = static_cast<int>(nextF0.size());
            remap[i] = nid;
            int l0 = (remap[builder.f0[i] >> 1] << 1) | (builder.f0[i] & 1);
            int l1 = (remap[builder.f1[i] >> 1] << 1) | (builder.f1[i] & 1);
            if (l0 > l1) std::swap(l0, l1);
            nextF0.push_back(l0);
            nextF1.push_back(l1);
        }
        for (int i = 0; i < nPOs; ++i) {
            newOuts[i] = (remap[newOuts[i] >> 1] << 1) | (newOuts[i] & 1);
        }
        f0.swap(nextF0);
        f1.swap(nextF1);
        outs.swap(newOuts);

        auto t4 = std::chrono::steady_clock::now();
        msSelectRebuild += std::chrono::duration<double, std::milli>(t4 - t3).count();

        if (params.verbose >= 1) {
            printf("rw4b round %d: tau=%.2f phi=%.2f applied=%d ands=%zu\n",
                   rnd + 1, tau, phi, chosenCount, f0.size() - nPIs - 1);
        }
    }

    cudaFree(d_lib);
    reorderDfsPostOrder(nPIs, nPOs, f0, f1, outs);
    copyHostToAigMan(aigman, nPIs, nPOs, f0, f1, outs);

    if (params.macroMode >= 1) {
        conditionalBalanceV3(aigman, params, "post_bal");
    }

    auto tTotal1 = std::chrono::steady_clock::now();
    double msTotal = std::chrono::duration<double, std::milli>(tTotal1 - tTotal0).count();
    printf("rw4b SUMMARY: ands=%d level=%d applied=%d | total=%.1fms (gpu_timing=%.1fms gpu_enum=%.1fms gpu_eval=%.1fms cpu_sel_rebuild=%.1fms)\n",
           aigman.nNodes, aigman.nLevels, totalApplied, msTotal, msTiming, msEnum, msEval, msSelectRebuild);
}

} // namespace rw4b
