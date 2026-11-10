#pragma once

// Gate 3C: enumerate the decisions produced by the GPU rewrite evaluator on
// the supplied graph. The command is read-only and does not replace nodes.
int gate3cRewriteDecisionDump(const char * outputDir, const char * inputPath,
                              int nObjs, int nPIs, int nPOs, int nNodes,
                              const int * pFanin0, const int * pFanin1,
                              const int * pOuts, const int * pNumFanouts);
