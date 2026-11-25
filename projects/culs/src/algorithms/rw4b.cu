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
#include <thrust/sort.h>

#include "aig_manager.h"
#include "algorithms/rewrite.h"
#include "algorithms/rw4b.h"
#include "common.h"
#include "robin_hood.h"

namespace {
#include "algorithms/rewrite_library.inc"
} // namespace

namespace rw4b {

static constexpr int kCutSetSize = 8;
static constexpr int kBlockSize = 256;
static constexpr int kMaxMffc = 24;
static constexpr float kInf = 1e30f;

struct Cut4B {
    int sign;
    int leaves[4];
    unsigned short truthtable;
    unsigned short value;
    unsigned char nLeaves;
    unsigned char used;
};

struct CandResult {
    float score;
    float maxArr;
    int da;
    int cutSlot;
    int subgr;
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
                                 bool piPhaseAware, float * d_sens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPOs) return;
    int lit = d_outs[idx];
    float a = edgeArr(lit, d_arr, d_negExtra, piPhaseAware);
    float w = expf((a - depth) / tau);
    atomicAdd(&d_sens[lit >> 1], w);
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

__device__ inline int findCutSlot(Cut4B * cuts) {
    for (int i = 0; i < kCutSetSize; ++i)
        if (!cuts[i].used) return i;
    int ans = -1;
    for (int pass = 0; pass < 3 && ans == -1; ++pass) {
        for (int i = 0; i < kCutSetSize; ++i) {
            bool ok = (pass == 0 && cuts[i].nLeaves > 2) ||
                      (pass == 1 && cuts[i].nLeaves == 2) ||
                      (pass == 2 && cuts[i].nLeaves < 2);
            if (ok && (ans == -1 || cuts[i].value < cuts[ans].value)) ans = i;
        }
    }
    cuts[ans].used = 0;
    return ans;
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

__device__ inline int cutFilter(Cut4B * cuts, int slot) {
    for (int i = 0; i < kCutSetSize; ++i) {
        if (i == slot || !cuts[i].used) continue;
        if (cuts[i].nLeaves > cuts[slot].nLeaves) {
            if ((cuts[i].sign & cuts[slot].sign) == cuts[slot].sign && cutDom(cuts[slot], cuts[i]))
                cuts[i].used = 0;
        } else {
            if ((cuts[i].sign & cuts[slot].sign) == cuts[i].sign && cutDom(cuts[i], cuts[slot])) {
                cuts[slot].used = 0;
                return 1;
            }
        }
    }
    return 0;
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

__global__ void initCutInputsKernel(int nPIs, const int * d_fo, Cut4B * d_cuts) {
    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (id > nPIs) return;
    Cut4B * c = d_cuts + static_cast<size_t>(id) * kCutSetSize;
    c->used = 1;
    c->sign = 1 << (id & 31);
    c->truthtable = 0xAAAA;
    c->nLeaves = 1;
    c->leaves[0] = id;
    c->value = static_cast<unsigned short>(cutVal(*c, d_fo));
}

__global__ void cutEnumLevelKernel(int startId, int count, const int * d_f0, const int * d_f1,
                                   const int * d_fo, Cut4B * d_cuts) {
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
            int slot = findCutSlot(myCuts);
            if (!mergeCut(a, b, &myCuts[slot])) continue;
            if (cutFilter(myCuts, slot)) continue;
            int tt0 = c0 ? ~static_cast<int>(a.truthtable) : static_cast<int>(a.truthtable);
            int tt1 = c1 ? ~static_cast<int>(b.truthtable) : static_cast<int>(b.truthtable);
            tt0 = ttStretch(tt0, a.nLeaves, cutPhase(a, myCuts[slot]));
            tt1 = ttStretch(tt1, b.nLeaves, cutPhase(b, myCuts[slot]));
            myCuts[slot].truthtable = static_cast<unsigned short>((tt0 & tt1) & 0xFFFF);
            if (minCutSupport(&myCuts[slot])) cutFilter(myCuts, slot);
            myCuts[slot].value = static_cast<unsigned short>(cutVal(myCuts[slot], d_fo));
            if (myCuts[slot].nLeaves < 2) return;
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

__global__ void evaluateNode4BKernel(
    int nPIs, int nObjs, int P,
    const int * d_f0, const int * d_f1, const int * d_fo, const int * d_negFo,
    const float * d_arr, const float * d_req, const float * d_gateDelay,
    const float * d_negExtra, const float * d_outSlew, const float * d_negSlew,
    const float * d_slewIn, const float * d_sens, const float * d_slewCoef, const float * d_slewCoefNeg,
    const Cut4B * d_cuts, const rw::Library * lib, const TableNode4B * d_ht,
    float alpha, float piAlpha, float beta, float lamArea, float regW, int pivot, float tau,
    bool piPhaseAware, bool strashAware,
    CandResult * d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nNodes = nObjs - nPIs - 1;
    if (idx >= nNodes) return;
    int root = nPIs + 1 + idx;
    d_out[root] = {0.0f, 0.0f, 0, -1, -1};

    float req = d_req[root];
    if (req >= kInf * 0.5f) return;
    float sRoot = d_sens[root];
    float nodeLv = d_arr[root];
    int rootFo = d_fo[root];

    const Cut4B * myCuts = d_cuts + static_cast<size_t>(root) * kCutSetSize;
    float bestScore = 0.0f;
    CandResult bestRes = {0.0f, 0.0f, 0, -1, -1};

    int remIds[kMaxMffc], remRem[kMaxMffc];
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

        // Bounded MFFC discovery (iterative BFS/stack up to kMaxMffc)
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
                    if (remIds[remCnt + t] == src) { pos = t; break; }
                }
                int r;
                if (pos >= 0) {
                    r = --remRem[pos];
                } else {
                    if (remCnt + tracked >= kMaxMffc) continue;
                    remIds[remCnt + tracked] = src;
                    r = d_fo[src] - 1;
                    remRem[tracked] = r;
                    tracked++;
                }
                if (r == 0 && remCnt < kMaxMffc && !inList(src, remIds, remCnt)) {
                    // move src into remIds[remCnt]
                    // compact tracked entry
                    if (pos >= 0) {
                        remIds[remCnt + pos] = remIds[remCnt + tracked - 1];
                        remRem[pos] = remRem[tracked - 1];
                        tracked--;
                    } else {
                        tracked--;
                    }
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

        // Compute orig_refs, orig_refs_neg, leaf_delays, local_sens over remIds
        int origRefs[4] = {0, 0, 0, 0};
        int origRefsNeg[4] = {0, 0, 0, 0};
        float remDist[kMaxMffc], remSens[kMaxMffc];
        for (int a = 0; a < remCnt; ++a) { remDist[a] = -1.0f; remSens[a] = 0.0f; }
        remDist[0] = d_gateDelay[root];
        remSens[0] = sRoot;
        float leafDelays[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        float leafLocalSens[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int a = 0; a < remCnt; ++a) {
            int u = remIds[a];
            float du = remDist[a];
            float su = remSens[a];
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
                    origRefs[li]++;
                    if (lit & 1) origRefsNeg[li]++;
                    float dp = du + extra;
                    if (dp > leafDelays[li]) leafDelays[li] = dp;
                    leafLocalSens[li] += su * ws[k];
                } else {
                    for (int b = a + 1; b < remCnt; ++b) {
                        if (remIds[b] == pred) {
                            float dp = du + d_gateDelay[pred];
                            if (dp > remDist[b]) remDist[b] = dp;
                            remSens[b] += su * ws[k];
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
            if (hit != -1 && hit != root && !inList(hit, remIds, remCnt)) {
                match[num] = hit;
            }
        }

        // Evaluate all subgraphs in this NPN class
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
                // Wire directly to leaf rt
                candRefs[rt] += rootFo;
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

            // Compute leaf fanout delta costs
            float leafDd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int leafDl[4] = {0, 0, 0, 0};
            float side = 0.0f, reg = 0.0f, slewCost = 0.0f;

            for (int i = 0; i < 4; ++i) {
                int lf = leaves[i];
                if (lf <= 0) continue;
                bool isPI = (lf <= nPIs);
                int dfo = (isPI && piPhaseAware) ? (candRefsNeg[i] - origRefsNeg[i]) : (candRefs[i] - origRefs[i]);
                if (dfo == 0) continue;
                float coef = isPI ? piAlpha : (beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[lf]) : alpha);
                if (coef <= 0.0f) continue;
                int baseCnt = (isPI && piPhaseAware) ? d_negFo[lf] : d_fo[lf];
                int newCnt = baseCnt + dfo;
                if (newCnt < 0) newCnt = 0;
                int dl = loadI(newCnt) - loadI(baseCnt);
                float dd = coef * static_cast<float>(dl);
                leafDd[i] = dd;
                leafDl[i] = dl;
                side += sensOther[i] * dd;
                if (regW > 0.0f) reg += regW * static_cast<float>(hqI(newCnt, pivot) - hqI(baseCnt, pivot));
                if (beta > 0.0f && dl != 0) {
                    float sc = (isPI && piPhaseAware) ? d_slewCoefNeg[lf] : d_slewCoef[lf];
                    if (sc > 0.0f) slewCost += sc * static_cast<float>(dl);
                }
            }
            for (int t = 0; t < hitCnt; ++t) {
                int u = hitIds[t], dfo = hitDelta[t];
                if (dfo == 0 || u <= nPIs) continue;
                float coef = beta > 0.0f ? alpha * (1.0f + beta * d_slewIn[u]) : alpha;
                int baseCnt = d_fo[u];
                int newCnt = baseCnt + dfo;
                if (newCnt < 0) newCnt = 0;
                int dl = loadI(newCnt) - loadI(baseCnt);
                float dd = coef * static_cast<float>(dl);
                side += d_sens[u] * dd;
                if (regW > 0.0f) reg += regW * static_cast<float>(hqI(newCnt, pivot) - hqI(baseCnt, pivot));
                if (beta > 0.0f && dl != 0 && d_slewCoef[u] > 0.0f) slewCost += d_slewCoef[u] * static_cast<float>(dl);
            }

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
            float effDelta = sRoot * deltaJ - side - reg - slewCost;
            float score;
            if (da <= 0) {
                if (deltaJ < -1e-5f || effDelta < -1e-5f) continue;
                if (da == 0 && effDelta <= 1e-5f) continue; // avoid float32 zero-gain churn
                score = effDelta + 1e-4f * static_cast<float>(-da);
            } else {
                score = effDelta - lamArea * static_cast<float>(da);
            }
            if (score > bestScore && score > 1e-6f) {
                bestScore = score;
                bestRes = {score, maxArrAbs, da, slot, out};
            }
        }
    }
    d_out[root] = bestRes;
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
    int maxFo;
    long long excessFo2;
};

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
        nf0[nid] = l0;
        nf1[nid] = l1;
    }
    for (int i = 0; i < nPOs; ++i) {
        outs[i] = (old2new[outs[i] >> 1] << 1) | (outs[i] & 1);
    }
    f0.swap(nf0);
    f1.swap(nf1);
}

static AigStats computeV3StatsHost(int nPIs, int nPOs, int nObjs,
                                   const int * f0, const int * f1, const int * outs,
                                   float alpha = 0.20f, float piAlpha = 0.10f) {
    std::vector<int> fo(nObjs, 0);
    for (int i = nPIs + 1; i < nObjs; ++i) {
        fo[f0[i] >> 1]++;
        fo[f1[i] >> 1]++;
    }
    for (int i = 0; i < nPOs; ++i) fo[outs[i] >> 1]++;
    std::vector<float> arr(nObjs, 0.0f);
    for (int i = 1; i <= nPIs; ++i) arr[i] = piAlpha * (fo[i] > 1 ? float(fo[i] - 1) : 0.0f);
    for (int i = nPIs + 1; i < nObjs; ++i) {
        float gd = 1.0f + alpha * (fo[i] > 1 ? float(fo[i] - 1) : 0.0f);
        arr[i] = gd + std::max(arr[f0[i] >> 1], arr[f1[i] >> 1]);
    }
    float depth = 0.0f;
    for (int i = 0; i < nPOs; ++i) depth = std::max(depth, arr[outs[i] >> 1]);
    int maxFo = 0;
    long long ex2 = 0;
    for (int i = 1; i < nObjs; ++i) {
        if (fo[i] > maxFo) maxFo = fo[i];
        if (fo[i] > 6) {
            long long h = fo[i] - 6;
            ex2 += h * h;
        }
    }
    return {depth, maxFo, ex2};
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
    std::vector<int> f0, f1;
    robin_hood::unordered_flat_map<uint64_t, int> ht;

    explicit StrashRebuild(int nPIs_, int reserveNodes) : nPIs(nPIs_) {
        f0.assign(nPIs + 1, 0);
        f1.assign(nPIs + 1, 0);
        f0.reserve(nPIs + reserveNodes + 64);
        f1.reserve(nPIs + reserveNodes + 64);
        ht.reserve(reserveNodes * 2 + 64);
    }

    int andLit(int l0, int l1) {
        if (l0 == 0 || l1 == 0) return 0;
        if (l0 == 1) return l1;
        if (l1 == 1) return l0;
        if (l0 == l1) return l0;
        if (l0 == (l1 ^ 1)) return 0;
        if (l0 > l1) std::swap(l0, l1);
        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(l0)) << 32) |
                       static_cast<uint32_t>(l1);
        auto it = ht.find(key);
        if (it != ht.end()) return it->second << 1;
        int nid = static_cast<int>(f0.size());
        f0.push_back(l0);
        f1.push_back(l1);
        ht.emplace(key, nid);
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
    AigStats s0 = computeV3StatsHost(aigman.nPIs, aigman.nPOs, aigman.nObjs,
                                     aigman.pFanin0, aigman.pFanin1, aigman.pOuts,
                                     params.fanoutAlpha, params.piFanoutAlpha);
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
                                     params.fanoutAlpha, params.piFanoutAlpha);
    bool accept = (s1.maxFo <= std::max(64, s0.maxFo)) &&
                  (s1.excessFo2 <= std::max(1000LL, static_cast<long long>(s0.excessFo2 * 1.02))) &&
                  (s1.depth <= s0.depth - 1.0f);
    if (!accept) {
        copyHostToAigMan(aigman, aigman.nPIs, aigman.nPOs, bakF0, bakF1, bakOuts);
    }
    if (params.verbose >= 1) {
        printf("rw4b [%s]: D=%.1f->%.1f maxFo=%d->%d ex2=%lld->%lld -> %s\n",
               tag, s0.depth, s1.depth, s0.maxFo, s1.maxFo, s0.excessFo2, s1.excessFo2,
               accept ? "ACCEPT" : "REJECT");
    }
    return accept;
}

// ---------------- Main GPU rw4b entry ----------------

void runRw4B(AIGMan & aigman, const Rw4BParams & params) {
    if (aigman.nNodes == 0) return;
    auto tTotal0 = std::chrono::steady_clock::now();

    if (aigman.deviceAllocated) {
        aigman.toHost();
        aigman.clearDevice();
    }

    if (params.condBalance) {
        conditionalBalanceV3(aigman, params, "pre_bal");
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
        float *d_arr = nullptr, *d_req = nullptr, *d_reqNeg = nullptr, *d_gd = nullptr;
        float *d_negExtra = nullptr, *d_outSlew = nullptr, *d_negSlew = nullptr, *d_slewIn = nullptr;
        float *d_sens = nullptr, *d_slewCoef = nullptr, *d_slewCoefNeg = nullptr, *d_poArr = nullptr;
        Cut4B * d_cuts = nullptr;
        TableNode4B * d_ht = nullptr;
        CandResult * d_candOut = nullptr;

        int P = nextPrime(std::max(101, nNodes * 2 + 1));
        CHECK_CUDA(cudaMalloc(&d_f0, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_f1, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_outs, nPOs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_fo, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_negFo, nObjs * sizeof(int)));
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
        CHECK_CUDA(cudaMalloc(&d_cuts, static_cast<size_t>(nObjs) * kCutSetSize * sizeof(Cut4B)));
        CHECK_CUDA(cudaMalloc(&d_ht, static_cast<size_t>(P + nNodes) * sizeof(TableNode4B)));
        CHECK_CUDA(cudaMalloc(&d_candOut, nObjs * sizeof(CandResult)));

        CHECK_CUDA(cudaMemcpy(d_f0, f0.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_f1, f1.data(), nObjs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_outs, outs.data(), nPOs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_fo, 0, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_negFo, 0, nObjs * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_sens, 0, nObjs * sizeof(float)));
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
        // 1. Refs + Forward delay propagation
        int maxWork = std::max(nNodes, nPOs);
        countRefsKernel<<<(maxWork + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            d_f0, d_f1, d_outs, nPIs, nObjs, nPOs, d_fo, d_negFo);
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

        // Backward required + sensitivity + slew coupling
        if (nPOs > 0) {
            initReqPOKernel<<<(nPOs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                d_outs, nPOs, nPIs, depth, d_negExtra, params.piPhaseAware, d_req, d_reqNeg);
            initSensPOKernel<<<(nPOs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                d_outs, nPOs, depth, tau, d_arr, d_negExtra, params.piPhaseAware, d_sens);
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
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        msTiming += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. GPU 4-cut enumeration
        if (nPIs > 0) {
            initCutInputsKernel<<<(nPIs + kBlockSize - 1) / kBlockSize, kBlockSize>>>(nPIs, d_fo, d_cuts);
        }
        for (int l = 1; l <= maxLv; ++l) {
            int cnt = lvOff[l + 1] - lvOff[l];
            if (cnt <= 0) continue;
            cutEnumLevelKernel<<<(cnt + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
                lvOff[l], cnt, d_f0, d_f1, d_fo, d_cuts);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        msEnum += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // 3. GPU strash table + 4B evaluation
        buildHashKernel<<<(nNodes + kBlockSize - 1) / kBlockSize, kBlockSize>>>(nPIs, nObjs, P, d_f0, d_f1, d_ht);
        evaluateNode4BKernel<<<(nNodes + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            nPIs, nObjs, P, d_f0, d_f1, d_fo, d_negFo, d_arr, d_req, d_gd,
            d_negExtra, d_outSlew, d_negSlew, d_slewIn, d_sens, d_slewCoef, d_slewCoefNeg,
            d_cuts, d_lib, d_ht, params.fanoutAlpha, params.piFanoutAlpha, params.slewBeta,
            params.lamArea, params.fanoutReg, params.fanoutPivot, tau,
            params.piPhaseAware, params.strashAware, d_candOut);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t3 = std::chrono::steady_clock::now();
        msEval += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 4. Copy compact candidate winners + fanouts to host for conflict-free + live-budget selection
        std::vector<CandResult> hCand(nObjs);
        std::vector<int> hFo(nObjs);
        std::vector<Cut4B> hCuts(static_cast<size_t>(nObjs) * kCutSetSize);
        CHECK_CUDA(cudaMemcpy(hCand.data(), d_candOut, nObjs * sizeof(CandResult), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hFo.data(), d_fo, nObjs * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hCuts.data(), d_cuts, static_cast<size_t>(nObjs) * kCutSetSize * sizeof(Cut4B), cudaMemcpyDeviceToHost));

        // Free round GPU buffers early
        cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_outs); cudaFree(d_fo); cudaFree(d_negFo); cudaFree(d_slewArg);
        cudaFree(d_arr); cudaFree(d_req); cudaFree(d_reqNeg); cudaFree(d_gd);
        cudaFree(d_negExtra); cudaFree(d_outSlew); cudaFree(d_negSlew); cudaFree(d_slewIn);
        cudaFree(d_sens); cudaFree(d_slewCoef); cudaFree(d_slewCoefNeg); cudaFree(d_poArr);
        cudaFree(d_cuts); cudaFree(d_ht); cudaFree(d_candOut);

        struct SelectedItem {
            int root;
            float score;
            int da;
            int subgr;
            int cls;
            int rt;
            int uPhase;
            int flip;
            int leaves[4];
            std::vector<int> remAll;
            std::vector<int> remStrict;
        };

        std::vector<int> winRoots;
        winRoots.reserve(nNodes / 4);
        for (int id = nPIs + 1; id < nObjs; ++id) {
            if (hCand[id].score > 0.0f && hCand[id].cutSlot >= 0) winRoots.push_back(id);
        }
        std::sort(winRoots.begin(), winRoots.end(), [&](int a, int b) {
            if (hCand[a].score != hCand[b].score) return hCand[a].score > hCand[b].score;
            if (hCand[a].da != hCand[b].da) return hCand[a].da < hCand[b].da;
            return a < b;
        });

        std::vector<uint8_t> usedRoot(nObjs, 0), delStrict(nObjs, 0), prot(nObjs, 0);
        std::vector<int> accFo(nObjs, 0);
        std::unordered_map<int, SelectedItem> chosen;
        chosen.reserve(winRoots.size());

        for (int root : winRoots) {
            if (usedRoot[root] || delStrict[root]) continue;
            Cut4B cut = hCuts[static_cast<size_t>(root) * kCutSetSize + hCand[root].cutSlot];
            if (!cut.used || cut.nLeaves < 3) continue;
            if (cut.nLeaves == 3) cut.leaves[3] = 0;

            int uPhase = hLib.pPhases[cut.truthtable];
            int cls = hLib.pMap[cut.truthtable];
            const int * perm = hLib.pPerms4[hLib.pPerms[cut.truthtable]];
            int leaves[4];
            for (int j = 0; j < 4; ++j) leaves[j] = cut.leaves[perm[j]];

            bool leafConflict = false;
            for (int j = 0; j < 4; ++j)
                if (leaves[j] > 0 && delStrict[leaves[j]]) { leafConflict = true; break; }
            if (leafConflict) continue;

            std::vector<int> remAll, remStrict;
            collectBoundedMffcHost(root, leaves, nPIs, f0, f1, hFo, remAll, remStrict);
            bool conflict = false;
            for (int u : remStrict) if (prot[u]) { conflict = true; break; }
            if (!conflict) {
                for (int u : remAll) if (delStrict[u]) { conflict = true; break; }
            }
            if (conflict) continue;

            int rt = hLib.pSubgr[cls][hCand[root].subgr];
            bool ttOk = false;
            uint16_t origTT = computeConeTruth(root, leaves, nPIs, f0, f1, ttOk);
            int flip = 0;
            if (!ttOk || !verifyCandFlip(origTT, cls, rt, uPhase, hLib, flip)) continue;

            // Live fanout budget regularizer check on leaves
            int origRefs[4] = {0, 0, 0, 0}, candRefs[4] = {0, 0, 0, 0};
            for (int u : remAll) {
                for (int s : {f0[u] >> 1, f1[u] >> 1}) {
                    for (int j = 0; j < 4; ++j) if (leaves[j] == s) origRefs[j]++;
                }
            }
            if (rt < 4) {
                candRefs[rt] += hFo[root];
            } else {
                uint64_t um = 1ULL << rt;
                for (int j = rt; j >= 4; --j) {
                    if ((um >> j) & 1ULL) {
                        um |= (1ULL << hLib.fanin0[cls][j - 4]);
                        um |= (1ULL << hLib.fanin1[cls][j - 4]);
                    }
                }
                for (int j = 4; j <= rt; ++j) {
                    if (!((um >> j) & 1ULL)) continue;
                    int i0 = hLib.fanin0[cls][j - 4], i1 = hLib.fanin1[cls][j - 4];
                    if (i0 < 4) candRefs[i0]++;
                    if (i1 < 4) candRefs[i1]++;
                }
            }
            if (params.fanoutBudget && params.fanoutReg > 0.0f) {
                float extraReg = 0.0f;
                auto hq = [&](int x) { int h = x - params.fanoutPivot; return h > 0 ? h * h : 0; };
                for (int j = 0; j < 4; ++j) {
                    int lf = leaves[j];
                    if (lf <= 0) continue;
                    int d = candRefs[j] - origRefs[j];
                    int a_ = accFo[lf];
                    if (d != 0 && a_ != 0) {
                        int base = hFo[lf];
                        extraReg += params.fanoutReg * float((hq(base + a_ + d) - hq(base + a_)) - (hq(base + d) - hq(base)));
                    }
                }
                if (hCand[root].score - extraReg <= 0.0f) continue;
            }

            usedRoot[root] = 1;
            for (int u : remStrict) delStrict[u] = 1;
            for (int u : remAll) prot[u] = 1;
            for (int j = 0; j < 4; ++j) {
                if (leaves[j] > 0) {
                    prot[leaves[j]] = 1;
                    accFo[leaves[j]] += (candRefs[j] - origRefs[j]);
                }
            }

            SelectedItem item;
            item.root = root;
            item.score = hCand[root].score;
            item.da = hCand[root].da;
            item.subgr = hCand[root].subgr;
            item.cls = cls;
            item.rt = rt;
            item.uPhase = uPhase;
            item.flip = flip;
            for (int j = 0; j < 4; ++j) item.leaves[j] = leaves[j];
            item.remAll = std::move(remAll);
            item.remStrict = std::move(remStrict);
            chosen.emplace(root, std::move(item));
        }

        if (chosen.empty()) {
            auto t4 = std::chrono::steady_clock::now();
            msSelectRebuild += std::chrono::duration<double, std::milli>(t4 - t3).count();
            break;
        }
        totalApplied += static_cast<int>(chosen.size());

        // Exact topological strashed rebuild + PO reachability pruning
        StrashRebuild builder(nPIs, nNodes);
        std::vector<int> litMap(nObjs, 0);
        for (int i = 0; i <= nPIs; ++i) litMap[i] = i << 1;

        for (int id = nPIs + 1; id < nObjs; ++id) {
            auto it = chosen.find(id);
            if (it == chosen.end()) {
                int nl0 = litMap[f0[id] >> 1] ^ (f0[id] & 1);
                int nl1 = litMap[f1[id] >> 1] ^ (f1[id] & 1);
                litMap[id] = builder.andLit(nl0, nl1);
            } else {
                const SelectedItem & op = it->second;
                int localLit[54];
                for (int j = 0; j < 4; ++j) {
                    int base = op.leaves[j] > 0 ? litMap[op.leaves[j]] : 0;
                    localLit[j] = base ^ (((op.uPhase >> j) & 1) ? 1 : 0);
                }
                uint64_t um = 1ULL << op.rt;
                for (int j = op.rt; j >= 4; --j) {
                    if ((um >> j) & 1ULL) {
                        um |= (1ULL << hLib.fanin0[op.cls][j - 4]);
                        um |= (1ULL << hLib.fanin1[op.cls][j - 4]);
                    }
                }
                for (int j = 4; j <= op.rt; ++j) {
                    if (!((um >> j) & 1ULL)) continue;
                    int i0 = hLib.fanin0[op.cls][j - 4], i1 = hLib.fanin1[op.cls][j - 4];
                    int nl0 = localLit[i0] ^ (hLib.isC0[op.cls][j - 4] ? 1 : 0);
                    int nl1 = localLit[i1] ^ (hLib.isC1[op.cls][j - 4] ? 1 : 0);
                    localLit[j] = builder.andLit(nl0, nl1);
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
            printf("rw4b round %d: tau=%.2f applied=%zu ands=%zu\n",
                   rnd + 1, tau, chosen.size(), f0.size() - nPIs - 1);
        }
    }

    cudaFree(d_lib);
    copyHostToAigMan(aigman, nPIs, nPOs, f0, f1, outs);

    if (params.condBalance) {
        conditionalBalanceV3(aigman, params, "post_bal");
    }

    auto tTotal1 = std::chrono::steady_clock::now();
    double msTotal = std::chrono::duration<double, std::milli>(tTotal1 - tTotal0).count();
    printf("rw4b SUMMARY: ands=%d level=%d applied=%d | total=%.1fms (gpu_timing=%.1fms gpu_enum=%.1fms gpu_eval=%.1fms cpu_sel_rebuild=%.1fms)\n",
           aigman.nNodes, aigman.nLevels, totalApplied, msTotal, msTiming, msEnum, msEval, msSelectRebuild);
}

} // namespace rw4b
