#pragma once

#include <string>

class AIGMan;

namespace rw4b {

struct Rw4BParams {
    int rounds = 3;
    float fanoutAlpha = 0.20f;
    float piFanoutAlpha = 0.10f;
    float slewBeta = 0.005f;
    float lamArea = 0.05f;
    float fanoutReg = 0.0005f;
    int fanoutPivot = 6;
    float minTau = 1.0f;
    float maxTau = 6.0f;
    float spreadCoeff = 0.2f;
    bool piPhaseAware = true;
    bool strashAware = true;
    bool fanoutBudget = true;
    int perRootK = 2;
    float budgetTol = 1e-4f;
    int foCap = 0;          // Fanout-capped strash limit (0 = unlimited / no duplicate gates)
    bool adpScore = true;   // P0-3: scale-invariant ADP first-order scoring w*(dD/D) - da/A
    float adpW = 2.0f;      // P0-3: delay weight w in ADP score
    float slackKappa = 2.0f;// P0-2/P0-3: near-critical gate slack(root) <= kappa * tau
    int macroMode = 0; // 0: pure rw4b, 1: v3-gated GPU balance (-b), 2: v3-gated GPU struct {orig,bal,rfz,bal_rfz} (-m)
    // Stage B: GPU multi-region end-to-end differentiable joint gradient optimization
    bool jointGrad = false;
    int gradSteps = 25;
    bool gradCheck = false;
    float tnsRho = 0.85f;
    float discLam = 0.30f;
    float gradLr = 0.20f;
    int verbose = 1;
};

void runRw4B(AIGMan & aigman, const Rw4BParams & params);

} // namespace rw4b
