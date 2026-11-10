#pragma once
#include <cub/cub.cuh>
#include "common.h"

namespace Aig {

__global__ void getElemTruthTable(unsigned * vTruthElem, int nVars);
__global__ void getElemTruthTable(uint64 * vTruthElem, int nVars);
__host__ void getElemTruthTableCPU(unsigned * vTruthElem, int nVars);
__host__ void getCutTruthTableSingleCPU(const int * pFanin0, const int * pFanin1, 
                                        unsigned * pTruth, const unsigned * vTruthElem, 
                                        const int * pCut, int nVars, int rootId, 
                                        int nMaxCutSize, int nPIs);

__host__ __device__ __forceinline__
int compareTruth(unsigned * truth1, unsigned* truth2, int nVars){
    int nWords = dUtils::TruthWordNum(nVars);
    for(int w=0; w<nWords; w++){
        if(truth1[w] ^ truth2[w])    return 1;
    }
    return 0;
}

__host__ __device__ __forceinline__
void cutTruthIter(int nodeId, const int * pFanin0, const int * pFanin1, 
                  unsigned * vTruthMem, int * visited, int * pVisitedSize, int nWords) {
    int fanin0Id, fanin1Id, fanin0TruthIdx, fanin1TruthIdx;
    int lit0, lit1;

    fanin0Id = dUtils::AigNodeID(pFanin0[nodeId]);
    fanin1Id = dUtils::AigNodeID(pFanin1[nodeId]);
    fanin0TruthIdx = fanin1TruthIdx = -1;
    for (int j = 0; j < *pVisitedSize; j++)
        if (visited[j] == fanin0Id) {
            fanin0TruthIdx = j;
            break;
        }
    for (int j = 0; j < *pVisitedSize; j++)
        if (visited[j] == fanin1Id) {
            fanin1TruthIdx = j;
            break;
        }
    if(!(fanin0TruthIdx != -1 && fanin1TruthIdx != -1)){
        printf("error: nodeId %d, fanin0: %d %d, fanin1 %d %d\n", nodeId, fanin0Id, fanin0TruthIdx, fanin1Id, fanin1TruthIdx);
    }
    assert(fanin0TruthIdx != -1 && fanin1TruthIdx != -1);

    lit0 = pFanin0[nodeId], lit1 = pFanin1[nodeId];
    if (!dUtils::AigNodeIsComplement(lit0) && !dUtils::AigNodeIsComplement(lit1))
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                vTruthMem[fanin0TruthIdx * nWords + i] & vTruthMem[fanin1TruthIdx * nWords + i];
    else if (!dUtils::AigNodeIsComplement(lit0) && dUtils::AigNodeIsComplement(lit1))
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                vTruthMem[fanin0TruthIdx * nWords + i] & ~vTruthMem[fanin1TruthIdx * nWords + i];
    else if (dUtils::AigNodeIsComplement(lit0) && !dUtils::AigNodeIsComplement(lit1))
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                ~vTruthMem[fanin0TruthIdx * nWords + i] & vTruthMem[fanin1TruthIdx * nWords + i];
    else
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                ~vTruthMem[fanin0TruthIdx * nWords + i] & ~vTruthMem[fanin1TruthIdx * nWords + i];

    visited[(*pVisitedSize)++] = nodeId;
}

__host__ __device__ __forceinline__
void printTruthTable(unsigned* vTruth, int nVars){
    // in momery:  76543210 21110198
    // display: 01234567 89101112
    int nWords = dUtils::TruthWordNum(nVars);
    printf("nVars: %d, nwords: %d | ", nVars, nWords);
    const unsigned MASK = (unsigned)1;
    for(int w=0; w<nWords; w++){
        unsigned value = *(vTruth+w);
        for(int i=0; i<8*sizeof(unsigned); i++){
            printf("%c", (value & MASK ? '1' : '0'));
            value >>= 1;
            if((i+1) % 8 == 0)   printf(" ");
        }
    }
    printf("\n");
}

/**
 * Device funtion of computing the truth table of one cut.
 * vTruthMem is the scratch memory used for intermediate node truth tables.
 * If it is not provided, it will be allocated dynamically.
 * vNumSaved is for debug uses in refactoring.
 */
template <int STACK_SIZE>
__device__ __forceinline__
void getCutTruthTableSingle(const int * pFanin0, const int * pFanin1, 
                            unsigned * pTruth, const unsigned * vTruthElem, 
                            const int * pCut, int nVars, int rootId, int nMaxCutSize, int nPIs,
                            unsigned * vTruthMem = NULL, const int * vNumSaved = NULL,
                            int* intNodesInd=NULL, int* intNodesSize=NULL) {
    int nodeId;
    int fVisited, fAllocated, nWords, nWordsElem, nIntNodes;
    int stack[STACK_SIZE], stackRes[STACK_SIZE], visited[STACK_SIZE];
    int stackTop, stackResTop, visitedSize;

    nWords = dUtils::TruthWordNum(nVars);
    nWordsElem = dUtils::TruthWordNum(nMaxCutSize);

    visitedSize = 0;
    for (int i = 0; i < nVars; i++) {
        visited[visitedSize++] = pCut[i];
    }

    // 1. traversal to collect intermediate nodes
    stackTop = stackResTop = -1;
    stack[++stackTop] = rootId;
    while (stackTop != -1) {
        // skip if already visited
        nodeId = stack[stackTop--];
        fVisited = 0;
        for (int i = 0; i < visitedSize; i++) {
            if (visited[i] == nodeId) {
                fVisited = 1;
                break;
            }
        }
        if (fVisited) continue;

        assert(dUtils::AigIsNode(nodeId, nPIs));
        assert(visitedSize < STACK_SIZE);
        visited[visitedSize++] = nodeId;

        // save result. make sure the nodes in stackRes are in reversed topo order (decreasing id)
        int j = stackResTop++;
        for (; j >= 0 && stackRes[j] < nodeId; j--) // insertion sort
            stackRes[j + 1] = stackRes[j];
        stackRes[j + 1] = nodeId;

        // push fanins into stack
        assert(stackTop < STACK_SIZE - 2);
        stack[++stackTop] = dUtils::AigNodeID(pFanin1[nodeId]);
        stack[++stackTop] = dUtils::AigNodeID(pFanin0[nodeId]);
    }
    nIntNodes = stackResTop + 1;
    assert(stackRes[0] == rootId);
    // debug
    if (vNumSaved != NULL)
        assert(nIntNodes >= vNumSaved[rootId] && nIntNodes + nVars < STACK_SIZE);
    else
        assert(nIntNodes + nVars < STACK_SIZE);
    
    // 2. compute truth table
    // allocate memory for intermediate truth tables, if it is not provided
    if (vTruthMem == NULL) {
        vTruthMem = (unsigned *) malloc((nVars + nIntNodes) * nWords * sizeof(unsigned));
        assert(vTruthMem != NULL); // if NULL, then not enough heap memory
        fAllocated = 1;
    } else
        fAllocated = 0;
    
    visitedSize = 0;
    // collect elementary truth tables for the cut nodes
    for (int i = 0; i < nVars; i++) {
        for (int j = 0; j < nWords; j++)
            vTruthMem[i * nWords + j] = vTruthElem[i * nWordsElem + j];
        visited[visitedSize++] = pCut[i];
    }
    for (int i = stackResTop; i >= 0; i--) {
        cutTruthIter(stackRes[i], pFanin0, pFanin1, vTruthMem, visited, &visitedSize, nWords);
    }
    assert(visited[visitedSize - 1] == rootId);
    if(intNodesInd!=NULL){
        memcpy(intNodesInd, visited, sizeof(int)*visitedSize);
        assert(visitedSize == nIntNodes + nVars);
        *intNodesSize = visitedSize;
    }

    // copy the truth table of rootId to pTruth
    for (int i = 0; i < nWords; i++)
        pTruth[i] = vTruthMem[(visitedSize - 1) * nWords + i];
    
    if (fAllocated)
        free(vTruthMem);
}

/**
 * Get the truth tables of cuts rooted at vIndices, saved in table format.
 * The resulting truth tables are saved in the consecutive array vTruth. 
 **/
template <int CUT_TABLE_NUM_COLS, int STACK_SIZE>
__global__ void getCutTruthTable(const int * pFanin0, const int * pFanin1, const int * vNumSaved,
                                 const int * vIndices, const int * vCutTable, const int * vCutSizes, 
                                 unsigned * vTruth, const int * vTruthRanges, const unsigned * vTruthElem,
                                 int nIndices, int nPIs, int nMaxCutSize,
                                 int* intNodesInd=NULL, unsigned* intNodesTruth=NULL, int* intNodesSize=NULL) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId, nVars;
    int startIdx;
    int nWords = dUtils::TruthWordNum(nMaxCutSize);

    for (; idx < nIndices; idx += nThreads) {
        rootId = vIndices[idx];
        nVars = vCutSizes[rootId];
        startIdx = (idx == 0 ? 0 : vTruthRanges[idx - 1]);
        getCutTruthTableSingle<STACK_SIZE>(pFanin0, pFanin1, vTruth + startIdx, vTruthElem,
                                           vCutTable + rootId * CUT_TABLE_NUM_COLS, nVars,
                                           rootId, nMaxCutSize, nPIs, 
                                           (intNodesTruth == NULL ? NULL : intNodesTruth+idx*STACK_SIZE*nWords),
                                           vNumSaved,
                                           intNodesInd+idx*STACK_SIZE, intNodesSize+idx);
    }
}


/**
 * Collect the internal nodes of a cut-bounded cone.
 *
 * This traversal is shared by the truth-table path and the Gate 3A audit
 * path.  Cut nodes are initialized as visited, so stackRes contains the
 * root and intermediate AND nodes, but not the cut boundary itself.
 */
template <int STACK_SIZE>
__device__ __forceinline__ int collectCutConeNodes(const int * pFanin0, const int * pFanin1,
                                                   const int * pCuts, int nVars, int rootId,
                                                   int nPIs, int * stack, int * stackRes,
                                                   int * visited) {
    int stackTop = -1;
    int stackResTop = -1;
    int visitedSize = 0;
    int nodeId;

    for (int i = 0; i < nVars; i++)
        visited[visitedSize++] = pCuts[i];

    stack[++stackTop] = rootId;
    while (stackTop != -1) {
        nodeId = stack[stackTop--];

        bool fVisited = false;
        for (int i = 0; i < visitedSize; i++) {
            if (visited[i] == nodeId) {
                fVisited = true;
                break;
            }
        }
        if (fVisited)
            continue;

        assert(dUtils::AigIsNode(nodeId, nPIs));
        assert(visitedSize < STACK_SIZE);
        visited[visitedSize++] = nodeId;

        // Preserve the original decreasing-id insertion order.
        int j = stackResTop++;
        for (; j >= 0 && stackRes[j] < nodeId; j--)
            stackRes[j + 1] = stackRes[j];
        stackRes[j + 1] = nodeId;

        assert(stackTop < STACK_SIZE - 2);
        stack[++stackTop] = dUtils::AigNodeID(pFanin1[nodeId]);
        stack[++stackTop] = dUtils::AigNodeID(pFanin0[nodeId]);
    }

    assert(stackResTop >= 0 && stackRes[0] == rootId);
    return stackResTop + 1;
}


/**
 * Same as getCutTruthTable, except that cuts are stored in a consecutive array.
 * Additionally, if vNode2ConeResynIdx is provided, mark the nodes in each cone using the 
 * corresponding cone index of vIndices.
 **/
template <int STACK_SIZE>
__global__ void getCutTruthTableConsecutive(const int * pFanin0, const int * pFanin1, const int * vNumSaved,
                                            const int * vIndices, const int * vCuts, const int * vCutRanges,
                                            unsigned * vTruth, const int * vTruthRanges, const unsigned * vTruthElem,
                                            int nIndices, int nPIs, int nMaxCutSize, int * vNode2ConeResynIdx = NULL) {
    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[THREAD_PER_BLOCK / 32];
    __shared__ unsigned * vTruthMemAlloc[THREAD_PER_BLOCK / 32];
    
    int nThreads = NUM_BLOCKS(nIndices, 32) * 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x / 32, laneIdx = threadIdx.x % 32;
    int stack[STACK_SIZE], stackRes[STACK_SIZE], visited[STACK_SIZE];
    int visitedSize;
    int rootId;
    int nWords, nWordsElem, nVars, nIntNodes;
    unsigned * vTruthMem;
    int startIdx, endIdx;
    int localMemLen, totalMemLen = -1, startMemIdx = -1;

    assert(nIndices <= nThreads && nThreads - nIndices < 32);

    if (idx < nThreads) {
        if (idx < nIndices) {
            rootId = vIndices[idx];
            startIdx = (idx == 0 ? 0 : vCutRanges[idx - 1]);
            endIdx = vCutRanges[idx];
            nVars = endIdx - startIdx;
            nWords = dUtils::TruthWordNum(nVars);
            nWordsElem = dUtils::TruthWordNum(nMaxCutSize);

            // 1. traversal to collect intermediate nodes
            nIntNodes = collectCutConeNodes<STACK_SIZE>(
                pFanin0, pFanin1, vCuts + startIdx, nVars, rootId, nPIs,
                stack, stackRes, visited
            );
            if (vNode2ConeResynIdx) {
                for (int i = 0; i < nIntNodes; i++) {
                    if (stackRes[i] != rootId)
                        vNode2ConeResynIdx[stackRes[i]] = idx;
                }
            }
            assert(nIntNodes >= vNumSaved[rootId] && nIntNodes + nVars < STACK_SIZE);
            assert(stackRes[0] == rootId);
        }

        // 2. compute truth table
        // allocate memory for intermediate truth tables
        // allocate once per warp to reduce overhead
        localMemLen = (idx < nIndices ? (nVars + nIntNodes) * nWords : 0);
        WarpScan(temp_storage[warpIdx]).ExclusiveSum(localMemLen, startMemIdx, totalMemLen);
        assert(startMemIdx != -1 && totalMemLen != -1);

        if (laneIdx == 0) {
            vTruthMemAlloc[warpIdx] = (unsigned *) malloc(totalMemLen * sizeof(unsigned));
            assert(vTruthMemAlloc[warpIdx] != NULL); // if NULL, then not enough heap memory
        }
        __syncwarp();
        vTruthMem = vTruthMemAlloc[warpIdx] + startMemIdx;
        

        if (idx < nIndices) {
            visitedSize = 0;
            // collect elementary truth tables for the cut nodes
            for (int i = 0; i < nVars; i++) {
                for (int j = 0; j < nWords; j++)
                    vTruthMem[i * nWords + j] = vTruthElem[i * nWordsElem + j];
                visited[visitedSize++] = vCuts[startIdx + i];
            }
            for (int i = nIntNodes - 1; i >= 0; i--) {
                cutTruthIter(stackRes[i], pFanin0, pFanin1, vTruthMem, visited, &visitedSize, nWords);
            }
            assert(visited[visitedSize - 1] == rootId);

            // copy the truth table of rootId to vTruth
            startIdx = (idx == 0 ? 0 : vTruthRanges[idx - 1]);
            endIdx = vTruthRanges[idx];
            assert(endIdx - startIdx == nWords);
            for (int i = 0; i < nWords; i++)
                vTruth[startIdx + i] = vTruthMem[(visitedSize - 1) * nWords + i];
        }

        __syncwarp();
        if (laneIdx == 0)
            free(vTruthMemAlloc[warpIdx]);
    }
}


/**
 * Audit-only variant of the shared cut traversal.  It returns the actual
 * internal node list for every supplied root and performs no truth-table or
 * resynthesis work.
 */
template <int STACK_SIZE>
__global__ void getCutConeNodesConsecutive(const int * pFanin0, const int * pFanin1,
                                           const int * vIndices, const int * vCuts,
                                           const int * vCutRanges, int nIndices, int nPIs,
                                           int * vConeNodes, int * vConeSizes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nIndices)
        return;

    int startIdx = (idx == 0 ? 0 : vCutRanges[idx - 1]);
    int endIdx = vCutRanges[idx];
    int rootId = vIndices[idx];
    int nVars = endIdx - startIdx;
    int stack[STACK_SIZE], stackRes[STACK_SIZE], visited[STACK_SIZE];

    int nIntNodes = collectCutConeNodes<STACK_SIZE>(
        pFanin0, pFanin1, vCuts + startIdx, nVars, rootId, nPIs,
        stack, stackRes, visited
    );
    vConeSizes[idx] = nIntNodes;
    for (int i = 0; i < nIntNodes; i++)
        vConeNodes[idx * STACK_SIZE + i] = stackRes[i];
}

} // namespace Aig
