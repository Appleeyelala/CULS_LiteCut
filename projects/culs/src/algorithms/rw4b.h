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
    int macroMode = 0; // 0: pure rw4b, 1: v3-gated GPU balance (-b), 2: v3-gated GPU struct {orig,bal,rfz,bal_rfz} (-m)
    int verbose = 1;
};

void runRw4B(AIGMan & aigman, const Rw4BParams & params);

} // namespace rw4b
