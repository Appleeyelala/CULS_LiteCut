#pragma once

// Gate 3E: execute ordered, explicitly selected (root, cut, subgraph)
// choices through the existing GPU replacement builder and CPU install loop.
int gate3eExecutionReplay(const char * outputDir, const char * inputPath,
                          const char * regionMapPath, const char * configPath,
                          int nObjs, int nPIs, int nPOs, int nNodes,
                          const int * pFanin0, const int * pFanin1,
                          const int * pOuts, const int * pNumFanouts);
