#pragma once

// Gate 3D: provenance audit of the existing GPU rewrite preselection,
// replacement-DAG construction, and CPU sequential install loop.
int gate3dRewriteExecutionAudit(const char * outputDir, const char * inputPath,
                                const char * regionMapPath,
                                int nObjs, int nPIs, int nPOs, int nNodes,
                                const int * pFanin0, const int * pFanin1,
                                const int * pOuts, const int * pNumFanouts);
