#pragma once

#include <vector>

#include "algorithms/rw4b.h"

namespace rw4b {

static constexpr int kGradMaxK = 5;         // k=0 no-op + up to 4 Pareto candidates
static constexpr int kGradMaxSubGates = 12; // max internal new AND gates in a 4-input NPN subgraph

// Compact 2-input micro-gate inside a candidate subgraph:
//   src0, src1:
//     >= 0 : external AIG literal (nodeId << 1) | compl
//     <  0 : previous internal micro-gate index p = -1 - src (0 <= p < g)
//   load: fixed internal load max(0, eFo[g] - 1) for non-root gates (ignored for root gate g == nGates - 1)
struct SubGate {
    int src0;
    int src1;
    float load;
};

struct GradProblem {
    int nPIs = 0;
    int nPOs = 0;
    int nObjs = 0;
    int nNodes = 0;
    int maxLv = 0;
    float depth0 = 1.0f;
    float tau0 = 1.0f;

    // Levelized original AIG
    const std::vector<int> * f0 = nullptr;
    const std::vector<int> * f1 = nullptr;
    const std::vector<int> * outs = nullptr;
    const std::vector<int> * lvOff = nullptr;
    // Base fanouts: foBase[u] = fo[u] for u > nPIs; foBase[u] = negFo[u] for 1 <= u <= nPIs; foBase[0] = 0
    std::vector<int> foBase;

    // Active regions (M = regionRoot.size())
    int M = 0;
    std::vector<int> regionRoot;       // [M]
    std::vector<int> isRegionRoot;     // [nObjs], -1 if ordinary node, else region idx m
    std::vector<int> candCount;        // [M], 2 <= K_m <= kGradMaxK
    std::vector<int> candWarmBestK;    // [M], 0 if no positive Stage A cand, else 1..K_m-1 best Stage A ADP cand
    std::vector<int> candDeltaAnd;     // [M * kGradMaxK]
    std::vector<int> candWireLit;      // [M * kGradMaxK], >= 0 if wire-fold literal, -1 if micro-subgraph
    std::vector<int> candNumGates;     // [M * kGradMaxK], 0 if wire-fold, else 1..kGradMaxSubGates (k=0 has 1)
    std::vector<SubGate> candSubGates; // [M * kGradMaxK * kGradMaxSubGates]

    // Sparse external fanout-delta CSR (indexed by slot = m * kGradMaxK + k)
    // Only external nodes u not in remStrict(R_m) are included.
    // For 1 <= u <= nPIs, val is delta negFo(u); for u > nPIs, val is delta fo(u).
    std::vector<int> foDeltaOffset; // [M * kGradMaxK + 1]
    std::vector<int> foDeltaNode;   // [NNZ]
    std::vector<int> foDeltaVal;    // [NNZ]
};

struct GradSolveStats {
    bool gradCheckPassed = true;
    double gradCheckMaxRelErr = 0.0;
    double gradCheckMaxAbsErr = 0.0;
    int gradCheckTested = 0;
    double initLoss = 0.0;
    double finalLoss = 0.0;
    float avgMaxProb = 0.0f;
    float bestTheta = -1.0f;
    float bestDiscDelay = 0.0f;
    int bestDiscDeltaAnd = 0;
    float bestDiscAdpDelta = 0.0f;
    int selectedNonZero = 0;
    double msGradLoop = 0.0;
    double msDiscSearch = 0.0;
};

GradSolveStats solveJointGradGPU(const GradProblem & prob, const Rw4BParams & params,
                                 std::vector<int> & chosenCand);

} // namespace rw4b
