#pragma once

// Lightweight declaration for the read-only Gate 3A command.  Keep this
// header independent of CUDA/Thrust so the CLI translation unit can include
// it with the host C++ compiler.
int gate3aMFFCDump(const char * outputDir, const char * inputPath,
                   int nObjs, int nPIs, int nPOs, int nNodes,
                   const int * pFanin0, const int * pFanin1, const int * pOuts,
                   const int * pNumFanouts, int cutSize, int verbose);
