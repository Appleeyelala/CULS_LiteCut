#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>
#include "common.h"
#include "rewrite.h"
#include "robin_hood.h"
#include "rewrite_library.inc"

using namespace std;
using namespace rw;

#define BLOCK_SIZE 64
#define RATIO  2.3
#define CUT_SET_SIZE 8
#define TABLE_SIZE 8
#define LARGE_BLOCK_SIZE 512

__managed__ int P;
__managed__ int N; // # of graph nodes
__managed__ int GPUexpected = 0;
__managed__ int replaceHasFullCorrsp = 0;

#define ID(i, j) ((i) * CUT_SET_SIZE + (j))
#define BLOCK_NUMBER(n, m) (((n) + (m) - 1) / (m))

double ENUM_TIME = 0, EVAL_TIME = 0, REPLACE_TIME = 0, REORDER_TIME = 0, REDUNDANCY_TIME = 0;
double COPYBACK_TIME = 0, REID_TIME = 0, CHOICE_TIME = 0;

time_t wall_time = clock();
string time(time_t t = 0) {
    if (t == 0) t = clock();
    stringstream t_ss;
    t_ss << "[" << setprecision(3) << setw(6) << fixed << (float) t / 1000000 << "] ";
    return t_ss.str();
}
ostream& print() {
    return cout << time();
}
#define prt print()



__device__ int CutFindValue(Cut *cut, int *nRef) {
    int value = 0, nOnes = 0;
    for(int i = 0; i < cut->nLeaves; i++) {
        value += nRef[cut->leaves[i]];
        nOnes += (nRef[cut->leaves[i]] == 1);
    }
    if(cut->nLeaves < 2) return 1001;
    if(value > 1000)
        value = 1000;
    if(nOnes > 3)
        value = 5 - nOnes;
    return value;
}

__global__ void Inputs(int *nRef, Cut *cuts, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(idx <= n) {
        Cut* cut = cuts + ID(idx, 0);
        cut->used = 1;
        cut->sign = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves = 1;
        cut->leaves[0] = idx;
        cut->value = CutFindValue(cut, nRef);
    }
}

__device__ int CountOnes(unsigned uWord) {
    uWord = (uWord & 0x55555555) + ((uWord>>1) & 0x55555555);
    uWord = (uWord & 0x33333333) + ((uWord>>2) & 0x33333333);
    uWord = (uWord & 0x0F0F0F0F) + ((uWord>>4) & 0x0F0F0F0F);
    uWord = (uWord & 0x00FF00FF) + ((uWord>>8) & 0x00FF00FF);
    return  (uWord & 0x0000FFFF) + (uWord>>16);
}

__device__ int FindCut(int idx, Cut *cuts) {
    for(int i = 0; i < CUT_SET_SIZE; i++)
        if(cuts[ID(idx, i)].used == 0) return i;
    int ans = -1;
    for(int i = 0; i < CUT_SET_SIZE; i++)
        if(cuts[ID(idx, i)].nLeaves > 2) {
            if(ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                ans = i;
        }
    if(ans == -1) {
        for(int i = 0; i < CUT_SET_SIZE; i++)
            if(cuts[ID(idx, i)].nLeaves == 2) {
                if(ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    }
    if(ans == -1) {
        for(int i = 0; i < CUT_SET_SIZE; i++)
            if(cuts[ID(idx, i)].nLeaves < 2) {
                if(ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    }
    cuts[ID(idx, ans)].used = 0;
    return ans;
}

// a.nLeaves >= b.nLeaves
__device__ int MergeCutOrdered(Cut a, Cut b, Cut* cut) {
    int c = 0, i = 0, k = 0;
    for (c = 0; c < 4; c++)
    {
        if (k == b.nLeaves)
        {
            if (i == a.nLeaves)
            {
                cut->nLeaves = c;
                return 1;
            }
            cut->leaves[c] = a.leaves[i++];
            continue;
        }
        if (i == a.nLeaves)
        {
            if (k == b.nLeaves)
            {
                cut->nLeaves = c;
                return 1;
            }
            cut->leaves[c] = b.leaves[k++];
            continue;
        }
        if (a.leaves[i] < b.leaves[k])
        {
            cut->leaves[c] = a.leaves[i++];
            continue;
        }
        if (a.leaves[i] > b.leaves[k])
        {
            cut->leaves[c] = b.leaves[k++];
            continue;
        }
        cut->leaves[c] = a.leaves[i++];
        k++;
    }
    if (i < a.nLeaves || k < b.nLeaves)
        return 0;
    cut->nLeaves = c;
    return 1;
}

__device__ int MergeCut(Cut a, Cut b, Cut *cut) {
    if(a.nLeaves >= b.nLeaves) {
        if(MergeCutOrdered(a, b, cut) == 0) return 0;
    } else {
        if(MergeCutOrdered(b, a, cut) == 0) return 0;
    }
    cut->sign = a.sign | b.sign;
    cut->used = 1;
    return 1;
}

//check if cut a is a subset of cut b
__device__ int CutDominance(Cut a, Cut b) {
    for(int i = 0; i < a.nLeaves; i++) {
        int ok = 0;
        for(int j = 0; j < b.nLeaves; j++)
            if(b.leaves[j] == a.leaves[i]) ok = 1;
        if(!ok) return 0;
    }
    return 1;
}

//check if cut[id] is redundant, and filter out redundant cuts
__device__ int CutFilter(Cut *cut, int id) {
    for(int i = 0; i < CUT_SET_SIZE; i++) if(i != id && cut[i].used) {
        if(cut[i].nLeaves > cut[id].nLeaves) {
            if((cut[i].sign & cut[id].sign) != cut[id].sign) continue;
            if(CutDominance(cut[id], cut[i]))
                cut[i].used = 0;
        } else {
            if((cut[i].sign & cut[id].sign) != cut[i].sign) continue;
            if(CutDominance(cut[i], cut[id])) {
                cut[id].used = 0;
                return 1;
            }
        }
    }
    return 0;
}

__device__ int CutTruthPhase(Cut x, Cut cut) {
    int phase = 0;
    for(int i = 0; i < cut.nLeaves; i++)
        for(int j = 0; j < x.nLeaves; j++)
            if(x.leaves[j] == cut.leaves[i]) phase |= 1 << i;
    return phase;
}

__device__ int CutTruthSwapAdjacentVars(int uTruth, int iVar) {
    if ( iVar == 0 )
        return (uTruth & 0x99999999) | ((uTruth & 0x22222222) << 1) | ((uTruth & 0x44444444) >> 1);
    if ( iVar == 1 )
        return (uTruth & 0xC3C3C3C3) | ((uTruth & 0x0C0C0C0C) << 2) | ((uTruth & 0x30303030) >> 2);
    if ( iVar == 2 )
        return (uTruth & 0xF00FF00F) | ((uTruth & 0x00F000F0) << 4) | ((uTruth & 0x0F000F00) >> 4);
    return 0;
}

__device__ int CutTruthStretch(int truthtable, int nVar, int phase) {
    int Var = nVar - 1;
    for(int i = 3; i >= 0; i--) if(phase >> i & 1) {
        for(int k = Var; k < i; k++)
            truthtable = CutTruthSwapAdjacentVars(truthtable, k);
        Var--;
    }
    return truthtable;
}

__device__ int CutTruthtable(Cut cut, Cut a, Cut b, int aComplement, int bComplement) {
    int tt0 = aComplement ? ~a.truthtable : a.truthtable;
    int tt1 = bComplement ? ~b.truthtable : b.truthtable;
    tt0 = CutTruthStretch(tt0, a.nLeaves, CutTruthPhase(a, cut));
    tt1 = CutTruthStretch(tt1, b.nLeaves, CutTruthPhase(b, cut));
    return tt0 & tt1;
}

__device__ int CutTruthShrink(int uTruth, int nVars, int Phase) {
    int i, k, Var = 0;
    for ( i = 0; i < 4; i++ )
        if ( Phase & (1 << i) )
        {
            for ( k = i-1; k >= Var; k-- )
                uTruth = CutTruthSwapAdjacentVars( uTruth, k );
            Var++;
        }
    return uTruth;
}

__device__ int MinimizeCutSupport(Cut* cut) {
    int masks[4][2] = {
        { 0x5555, 0xAAAA },
        { 0x3333, 0xCCCC },
        { 0x0F0F, 0xF0F0 },
        { 0x00FF, 0xFF00 }
    };
    int phase = 0, truth = cut->truthtable & 0xFFFF, nLeaves = cut->nLeaves;
    for(int i = 0; i < cut->nLeaves; i++)
        if((truth & masks[i][0]) == ((truth & masks[i][1]) >> (1 << i)))
            nLeaves--;
        else
            phase |= 1 << i;
    if(nLeaves == cut->nLeaves) return 0;
    truth = CutTruthShrink(truth, cut->nLeaves, phase);
    cut->truthtable = truth & 0xFFFF;
    cut->sign = 0;
    for(int i = 0, k = 0; i < cut->nLeaves; i++) if(phase >> i & 1) {
        cut->leaves[k++] = cut->leaves[i];
        cut->sign |= 1 << (31 & cut->leaves[i]);
    }
    cut->nLeaves = nLeaves;
    return 1;
}

__global__ void CutEnumerate(int *fanin0, int *fanin1, int* isC0, int *isC1, int *nRef, Cut *cuts, int delta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        idx += delta + 1;
        Cut* cut = cuts + ID(idx, 0);
        cut->used = 1;
        cut->sign = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves = 1;
        cut->leaves[0] = idx;
        cut->value = CutFindValue(cut, nRef);
        int in0 = fanin0[idx], in1 = fanin1[idx];
        for(int i = 0; i < CUT_SET_SIZE; i++) if(cuts[ID(in0, i)].used == 1)
            for(int j = 0; j < CUT_SET_SIZE; j++) if(cuts[ID(in1, j)].used == 1) {
                Cut a = cuts[ID(in0, i)], b = cuts[ID(in1, j)];
                if(CountOnes(a.sign | b.sign) > 4) continue;
                int cutId = FindCut(idx, cuts);
                if(!MergeCut(a, b, cut + cutId)) continue;
                if(CutFilter(cut, cutId)) continue;
                cut[cutId].truthtable = 0xFFFF & CutTruthtable(cut[cutId], a, b, isC0[idx], isC1[idx]);
                if(MinimizeCutSupport(cut + cutId))
                    CutFilter(cut, cutId);
                cut[cutId].value = CutFindValue(cut + cutId, nRef);
                if(cut[cutId].nLeaves < 2) return;
            }
    }
}

__device__ int Decrease(int id, int *tableSize, int *tableId, int *tableNum, int *nRef, Cut *cut) {
    for(int i = 0; i < cut->nLeaves; i++)
        if(id == cut->leaves[i]) return 1;
    for(int i = 0; i < tableSize[0]; i++)
        if(tableId[i] == id) return --tableNum[i];
    if(tableSize[0] == TABLE_SIZE) {
        //printf("Warning in Decrease\n");
        return 1;
    }
    tableId[tableSize[0]] = id;
    tableNum[tableSize[0]] = nRef[id] - 1;
    tableSize[0]++;
    return nRef[id] - 1;
}


__device__ int IsDeleted(int id, int *tableSize, int *tableId, int *tableNum) {
    for(int i = 0; i < tableSize[0]; i++)
        if(tableId[i] == id) return tableNum[i] == 0;
    return 0;
}


__device__ int CalcMFFC(int cur, Cut* cut, int *fanin0, int *fanin1, int *tableSize, int *tableId, int *tableNum, int *nRef, int root) {
    int ans = 1;
    if(Decrease(fanin0[cur], tableSize, tableId, tableNum, nRef, cut) == 0)
        ans += CalcMFFC(fanin0[cur], cut, fanin0, fanin1, tableSize, tableId, tableNum, nRef, root);
    if(Decrease(fanin1[cur], tableSize, tableId, tableNum, nRef, cut) == 0)
        ans += CalcMFFC(fanin1[cur], cut, fanin0, fanin1, tableSize, tableId, tableNum, nRef, root);
    return ans;
}

__device__ void TableInsert(int in0, int in1, int C0, int C1, TableNode* hashTable, int idx, int offset=0) {
    int index = idx - offset - 1;
    unsigned long key = 0;
    key ^= in0 * 7937;
    key ^= in1 * 2971;
    key ^= C0 * 911;
    key ^= C1 * 353;
    key %= P;
    hashTable[P + index].val = idx;
    while(true) {
        int res = atomicCAS(&hashTable[key].next, -1, P + index);
        if(res == -1)
            break;
        else
            key = res;
    }
}

__device__ int TableLookup(int in0, int in1, int C0, int C1, TableNode* hashTable, int *fanin0, int *fanin1, int *isC0, int *isC1) {
    unsigned long key = 0;
    if(in0 > in1) {
        int temp = in0;
        in0 = in1;
        in1 = temp;
        temp = C0;
        C0 = C1;
        C1 = temp;
    }
    key ^= in0 * 7937;
    key ^= in1 * 2971;
    key ^= C0 * 911;
    key ^= C1 * 353;
    key %= P;
    for(int cur = hashTable[key].next; cur != -1; cur = hashTable[cur].next) {
        int val = hashTable[cur].val;
        if(fanin0[val] == in0 && fanin1[val] == in1 && isC0[val] == C0 && isC1[val] == C1)
            return val;
    }
    return -1;
}

__device__ int Eval(int cur, int *match, int Class, Library *lib, int curId) {
    if(match[cur] > -1) return 0;
    if(match[cur] == curId) return 0;
    match[cur] = curId;
    return 1 + Eval(lib->fanin0[Class][cur - 4], match, Class, lib, curId) + Eval(lib->fanin1[Class][cur - 4], match, Class, lib, curId);
}



__global__ void EvaluateNode(int sz, int *bestout, int *selectedCutSlots, int *fanin0, int *fanin1, int *isC0, int *isC1, int *nodeLevels, Cut *cuts, Cut* selectedCuts, int *nRef,
                             Library *lib, TableNode* hashTable, int fUseZeros,
                             unsigned char *candidateMasks) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x, reduction = -1, bestLevel = 99999999, bestCut = -1, bestOut;
    int match[54], tableSize, tableId[TABLE_SIZE], tableNum[TABLE_SIZE];
    int matchLevel[54];
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        if(cut->used == 0 || cut->nLeaves < 3) continue;
        int nleaves = cut->nLeaves;
        if(nleaves == 3)
            cut->leaves[cut->nLeaves++] = 0;
        tableSize = 0;
        int saved = CalcMFFC(id, cut, fanin0, fanin1, &tableSize, tableId, tableNum, nRef, id);
        int uPhase = lib->pPhases[cut->truthtable];
        int Class = lib->pMap[cut->truthtable];
        int *pPerm = lib->pPerms4[lib->pPerms[cut->truthtable]];
        for(int j = 0; j < 54; j++)
            match[j] = matchLevel[j] = -1;
	      uint64_t isC = 0;
        for(int j = 0; j < 4; j++) {
            match[j] = cut->leaves[pPerm[j]];
            matchLevel[j] = nodeLevels[match[j]];
            if(uPhase >> j & 1)
		            isC |= 1LL << j;
        }
        for(int j = 0; j < lib->nNodes[Class]; j++) {
            int num = j + 4;
            int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
            assert(matchLevel[in0] != -1 && matchLevel[in1] != -1);
            matchLevel[num] = 1 + (matchLevel[in0] > matchLevel[in1] ? matchLevel[in0] : matchLevel[in1]);

            if(match[in0] == -1 || match[in1] == -1 || match[in0] == id || match[in1] == id) continue;
            int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
            if(nodeId != -1 && !IsDeleted(nodeId, &tableSize, tableId, tableNum)) {
                match[num] = nodeId;
                matchLevel[num] = nodeLevels[nodeId];
            }
        }
        for(int out = 0; out < lib->nSubgr[Class]; out++) {
            int rt = lib->pSubgr[Class][out];
            if(match[rt] == id) continue;
            int nodesAdded = Eval(rt, match, Class, lib, -out - 2);
            int rtLevel = matchLevel[rt];
            assert(rtLevel != -1);
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded && bestCut == -1)) {
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded)) {
            //     reduction = saved - nodesAdded;
            //     bestCut = i;
            //     bestOut = out;
            // }
            if (saved - nodesAdded < 0 || (saved - nodesAdded == 0 && !fUseZeros))
                continue;
            candidateMasks[(ID(id, i) * MAX_REWRITE_SUBGR) + out] = 1;
            if (saved - nodesAdded < reduction || (saved - nodesAdded == reduction && rtLevel >= bestLevel))
                continue;
            reduction = saved - nodesAdded;
            bestLevel = rtLevel;
            bestCut = i;
            bestOut = out;
        }
        cut->nLeaves = nleaves;
    }
    if(bestCut != -1) {
	// atomicAdd(&GPUexpected, 1);
        selectedCuts[id] = cuts[ID(id, bestCut)];
        selectedCuts[id].used = 1;
        bestout[id] = bestOut;
        selectedCutSlots[id] = bestCut;
    } else
    {
        selectedCuts[id].used = 0;
        selectedCutSlots[id] = -1;
    }
}


__global__ void BuildHashTable(TableNode *hashTable, int sz, int *fanin0, int *fanin1, int *isC0, int *isC1) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < sz) {
        int id = index + 1;
        if(fanin0[id] == 0 && fanin1[id] == 0) return;
        unsigned long key = 0;
        key ^= fanin0[id] * 7937;
        key ^= fanin1[id] * 2971;
        key ^= isC0[id] * 911;
        key ^= isC1[id] * 353;
        key %= P;
        hashTable[P + index].val = id;
        while(true) {
            int res = atomicCAS(&hashTable[key].next, -1, P + index);
            if(res == -1)
                break;
            else
                key = res;
        }
    }

}

__device__ void BuildSubgr(int cur, int Class, Library *lib, int *match, uint64_t isC, int *fanin0, int *fanin1, int *isC0, int *isC1, int *phase, TableNode* newTable, int sz) {
    // only create the node that has no correspondence in original table, or not created by this thread (in this subgraph)
    if (match[cur] != -1) return;
    int in0 = lib->fanin0[Class][cur - 4];
    int in1 = lib->fanin1[Class][cur - 4];
    BuildSubgr(in0, Class, lib, match, isC, fanin0, fanin1, isC0, isC1, phase, newTable, sz);
    BuildSubgr(in1, Class, lib, match, isC, fanin0, fanin1, isC0, isC1, phase, newTable, sz);

    // Build the node
    int node = atomicAdd(&N, 1);
    // assert(node < (int) (sz * RATIO));
    node += 1;
    fanin0[node] = match[in0];
    fanin1[node] = match[in1];
    isC0[node] = lib->isC0[Class][cur - 4] ^ (isC >> in0 & 1);
    isC1[node] = lib->isC1[Class][cur - 4] ^ (isC >> in1 & 1);
    phase[node] = (phase[fanin0[node]] ^ isC0[node]) & (phase[fanin1[node]] ^ isC1[node]);

    if (fanin0[node] > fanin1[node]) {
        int temp = fanin0[node];
        fanin0[node] = fanin1[node];
        fanin1[node] = temp;
        temp = isC0[node];
        isC0[node] = isC1[node];
        isC1[node] = temp;
    }

    // newTable is used for ensuring unique insertion
    TableInsert(fanin0[node], fanin1[node], isC0[node], isC1[node], newTable, node, sz);
    int actual_node = TableLookup(fanin0[node], fanin1[node], isC0[node], isC1[node], newTable, fanin0, fanin1, isC0, isC1);
    if (actual_node != node) { // The node is already created (by other threads, in other subgraphs)
        fanin0[node] = fanin1[node] = -1;
        match[cur] = actual_node;
    } else {
        match[cur] = node;
    }
}

__global__ void ReplaceSubgr(int sz, int *bestout, int *fanin0, int *fanin1, int *isC0, int *isC1, Cut *selectedCuts, Library *lib, TableNode* hashTable, TableNode* newTable, int *phase, int *replace) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    //if (N >= (int) sz * RATIO) printf("Warning: too many new nodes to handle...\n");
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    Cut cut = selectedCuts[id];
    if (cut.used == 0 || cut.nLeaves < 3) return;
	//atomicAdd(&CNT, 1);
    if(cut.nLeaves == 3)
        cut.leaves[cut.nLeaves++] = 0;

    int uPhase = lib->pPhases[cut.truthtable];
    int Class = lib->pMap[cut.truthtable];
    int *pPerm = lib->pPerms4[lib->pPerms[cut.truthtable]];
    // match maps the new node id into the id in the original hashtable;
    // if there is no corresponding node in the original table, then match = -1
    int match[54];
	  uint64_t isC = 0;
    for(int j = 0; j < 54; j++) match[j] = -1;
    for(int j = 0; j < 4; j++) {
        match[j] = cut.leaves[pPerm[j]];
        if(uPhase >> j & 1)
        	  isC |= 1LL << j;
    }
    int rt = lib->pSubgr[Class][bestout[id]];
	  uint64_t used = 1LL << rt;
    for(int j = rt; j >= 4; j--) if(used >> j & 1) {
	      used |= 1LL << lib->fanin0[Class][j - 4];
	      used |= 1LL << lib->fanin1[Class][j - 4];
    }
    for(int j = 0; j < lib->nNodes[Class]; j++) if(used >> (j + 4) & 1) {
      	int num = j + 4;
      	int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
      	if(match[in0] == -1 || match[in1] == -1) continue;
      	int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
      	if(nodeId != -1) match[num] = nodeId;
    }

    // the root of the new graph has correspondence, so the whole new graph exists in the original table
    // if (match[rt] == id) {atomicAdd(&replaceHasFullCorrsp, 1);  return;}
    if (match[rt] == id) return;
    BuildSubgr(rt, Class, lib, match, isC, fanin0, fanin1, isC0, isC1, phase, newTable, sz);
    replace[id] = (match[rt] * 2) + (phase[match[rt]] ^ phase[id]); // the corresponding new node
    // if (id > 100300 && id < 100400)
    //     printf("replace[%d] = %d\n", id, replace[id]);
}

__global__ void DetachAndAttach(int sz, int *fanin0, int *fanin1, int *replace) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (replace[id] == -1) return;
    fanin0[id] = 1;
    fanin1[id] = replace[id];
}

int IsPrime(int n) {
    for(int i = 2; i * i <= n; i++)
        if(n % i == 0) return 0;
    return 1;
}

int GetPrime(int n) {
    while(!IsPrime(n))
        n++;
    return n;
}

void ShowMemory() {
    size_t free_byte ;

    size_t total_byte ;

    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){

        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

        exit(1);

    }

    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", 
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

__global__ void printCuts(int id, Cut *cuts) {
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        printf("cut#%d details: truthtable %d, used%d, nLeaves=%d, leaves=%d %d %d %d\n", i, cut->truthtable, cut->used, cut->nLeaves, cut->leaves[0], cut->leaves[1], cut->leaves[2], cut->leaves[3]);
    }
}


__global__ void Convert(int *fanin, int *isC, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n) {
        isC[id] = fanin[id] & 1;
        fanin[id] >>= 1;
    }
}

__global__ void Revert(int *fanin, int *isC, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id <= n) {
        fanin[id] <<= 1;
        fanin[id] += isC[id];
    }
}

__global__ void print(int n, Cut *selected, int *bestOut) {
    for(int i = 1; i <= n; i++) if(selected[i].used == 1)
        printf("Selected %d: %d %d\n", i, selected[i].truthtable, selected[i].sign);
}




void GPUSolver::Init(int n) {
    P = GetPrime(n + 1);
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaStackSize = limit;
    printf("GPUSolver: setting cudaLimitStackSize = %lu KB\n", limit * 12 / 1024);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 12);
    cudaMalloc(&fanin0, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&fanin1, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&nRef, (n + 1) * sizeof(int));
    cudaMalloc(&isComplement0, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&isComplement1, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&bestSubgraph, (n + 1) * sizeof(int));
    cudaMalloc(&selectedCutSlots, (n + 1) * sizeof(int));
    cudaMalloc(&phase, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&replace, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMalloc(&lib, sizeof(Library));
    // ShowMemory();
    cudaMalloc(&newTable, (2 * n + 1 + P) * sizeof(TableNode));
    cudaMalloc(&hashTable, (2 * n + 1 + P) * sizeof(TableNode));
    // ShowMemory();
    cudaMalloc(&selectedCuts, (n + 1) * sizeof(Cut));
    cudaMalloc(&candidateMasks,
               static_cast<size_t>(n + 1) * CUT_SET_SIZE * MAX_REWRITE_SUBGR * sizeof(unsigned char));
    ShowMemory();
    printf("memory needed for Cut: %.2f MB\n", 1.0*sizeof(Cut) * (n + 1) * CUT_SET_SIZE / 1024/1024);
    CHECK_CUDA( cudaMalloc(&cuts, sizeof(Cut) * (n + 1) * CUT_SET_SIZE ) );
    ShowMemory();
}

void GPUSolver::Free() {
    cudaFree(fanin0);
    cudaFree(fanin1);
    cudaFree(nRef);
    cudaFree(isComplement0);
    cudaFree(isComplement1);
    cudaFree(bestSubgraph);
    cudaFree(selectedCutSlots);
    cudaFree(phase);
    cudaFree(replace);
    cudaFree(lib);
    cudaFree(newTable);
    cudaFree(hashTable);
    cudaFree(selectedCuts);
    cudaFree(candidateMasks);
    cudaFree(cuts);

    printf("GPUSolver: setting cudaLimitStackSize = %lu\n", cudaStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, cudaStackSize);
}

void GPUSolver::GetResults(int n, int *CPUbestSubgraph, Cut *CPUcuts) {
    cudaMemcpy(CPUbestSubgraph, bestSubgraph, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(CPUcuts, selectedCuts, (n + 1) * sizeof(Cut), cudaMemcpyDeviceToHost);
}

void GPUSolver::GetSelectedCutSlots(int n, int *CPUselectedCutSlots) {
    cudaMemcpy(CPUselectedCutSlots, selectedCutSlots,
               (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
}

void GPUSolver::GetEnumeratedCuts(int n, Cut *CPUcuts) {
    cudaMemcpy(CPUcuts, cuts,
               static_cast<size_t>(n + 1) * CUT_SET_SIZE * sizeof(Cut),
               cudaMemcpyDeviceToHost);
}

void GPUSolver::GetCandidateMasks(int n, unsigned char *CPUcandidateMasks) {
    cudaMemcpy(CPUcandidateMasks, candidateMasks,
               static_cast<size_t>(n + 1) * CUT_SET_SIZE * MAX_REWRITE_SUBGR * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
}

void GPUSolver::CopySelectedChoices(int n, const int *CPUbestSubgraph, const Cut *CPUcuts) {
    cudaMemcpy(bestSubgraph, CPUbestSubgraph, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(selectedCuts, CPUcuts, (n + 1) * sizeof(Cut), cudaMemcpyHostToDevice);
}

void GPUSolver::CopyLib(Library CPUlib) {
    cudaMemcpy(lib, &CPUlib, sizeof(Library), cudaMemcpyHostToDevice);
}

void GPUSolver::EnumerateAndPreEvaluate(int *level, const vector<int> &levelCount, int n, int *CPUfanin0, int *CPUfanin1, int *CPUref, bool fUseZeros) {
    int * nodeLevels = phase; // note, phase has not been used yet, so use its memory for now
    cudaMemcpy(nodeLevels, level, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(fanin0, CPUfanin0, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fanin1, CPUfanin1, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(nRef, CPUref, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(hashTable, -1, sizeof(TableNode) * (2 * n + 1 + P));
    cudaMemset(cuts, 0, CUT_SET_SIZE * (n + 1) * sizeof(Cut));
    cudaMemset(candidateMasks, 0,
               static_cast<size_t>(n + 1) * CUT_SET_SIZE * MAX_REWRITE_SUBGR * sizeof(unsigned char));
    cudaMemset(selectedCutSlots, -1, (n + 1) * sizeof(int));
    auto startTime = clock();
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin0, isComplement0, (n + 1));
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin1, isComplement1, (n + 1));
    Inputs<<<BLOCK_NUMBER(levelCount[0], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (nRef, cuts, levelCount[0]);        
    for(int i = 1; i < levelCount.size(); i++)
        CutEnumerate<<<BLOCK_NUMBER(levelCount[i] - levelCount[i - 1], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin0, fanin1, isComplement0, isComplement1, nRef, cuts, levelCount[i - 1], levelCount[i] - levelCount[i - 1]);
    cudaDeviceSynchronize();
    ENUM_TIME += clock() - startTime;
    startTime = clock();
    BuildHashTable<<<BLOCK_NUMBER(n, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (hashTable, n, fanin0, fanin1, isComplement0, isComplement1);           
    EvaluateNode<<<BLOCK_NUMBER(n, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (
        n, bestSubgraph, selectedCutSlots, fanin0, fanin1, isComplement0, isComplement1,
        nodeLevels, cuts, selectedCuts, nRef, lib, hashTable, fUseZeros == true,
        candidateMasks
    );
    gpuErrchk( cudaDeviceSynchronize() );
    std::cerr << cudaGetLastError() << " in EvaluateNode " << std::endl;

    EVAL_TIME += clock() - startTime;
} 

int GPUSolver::ReplaceSubgraphs(int n, int *CPUfanin0, int *CPUfanin1, int *CPUphase, int *CPUreplace) {
    prt << "Replacing sub-graphs" << endl;
    cudaMemcpy(phase, CPUphase, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(replace, -1, static_cast<int> (RATIO * (n + 1) * sizeof(int)));
    cudaMemset(newTable, -1, (2 * n + 1 + P) * sizeof(TableNode));
    N = n;
    ReplaceSubgr<<<BLOCK_NUMBER(n, BLOCK_SIZE), BLOCK_SIZE>>>(n, bestSubgraph, fanin0, fanin1, isComplement0, isComplement1, selectedCuts, lib, hashTable, newTable, phase, replace);
    gpuErrchk( cudaDeviceSynchronize() ); // wait until N
    std::cerr << cudaGetLastError() << " after replace " << std::endl;
printf("N = %d   n = %d   n * RATIO = %d\n", N, n, static_cast<int> (RATIO * n));
    Revert<<<BLOCK_NUMBER(N + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>>(fanin0, isComplement0, N + 1);
    Revert<<<BLOCK_NUMBER(N + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>>(fanin1, isComplement1, N + 1);
    cudaDeviceSynchronize();
    // Detach and attach sequentially
    auto startTime = clock();
    int *tempIn0 = (int*)malloc(5 * (n + 1) * sizeof(int));
    int *tempIn1 = (int*)malloc(5 * (n + 1) * sizeof(int));
    cudaMemcpy(tempIn0, fanin0, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tempIn1, fanin1, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(CPUreplace, replace, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    COPYBACK_TIME += clock() - startTime;

    startTime = clock();
    int nn = n;
    int *mapping = new int[N+1];
    memset(mapping, -1, (N+1)*sizeof(int));
    function<int(int)> rebuild = [&](int rId) {
        if (rId <= nn) return rId;
        if (mapping[rId] != -1) return mapping[rId];
        int in0 = (rebuild(tempIn0[rId] / 2) * 2 + (tempIn0[rId] & 1));
        int in1 = (rebuild(tempIn1[rId] / 2) * 2 + (tempIn1[rId] & 1));
        n++;
        CPUfanin0[n] = in0;
        CPUfanin1[n] = in1;
        mapping[rId] = n;
        return n;
    };
    printf("before rebuild, nn = %d\n", nn);

    // replace maps the old node id to new node id, or, indicate the "choice" node id of the old one
    // rebuild makes the new node ids consecutive
    for (int id = 1; id <= nn; id++) {
            //prt << "id " << id << " replace " << (CPUreplace[id] == -1 ? -1 : CPUreplace[id]/2)<< endl;
        if (CPUreplace[id] == -1) continue;
        else CPUreplace[id] = rebuild(CPUreplace[id] / 2) * 2 + (CPUreplace[id] & 1);
    }

    free(mapping);
    free(tempIn0);
    free(tempIn1);

    REID_TIME += clock() - startTime;
    return n;
}


void Message(string s) {
    cerr << fixed << setprecision(5) << "[" << 1.0 * clock() / CLOCKS_PER_SEC << "] " << s << endl;
}


namespace {

struct RewriteAuditLeafStatus {
    int leafInternal = -1;
    int leafOriginal = -1;
    bool deleted = false;
    int firstDeletedByInternalRoot = -1;
    int firstDeletedByOriginalRoot = -1;
    int firstDeletedByExecutionOrder = -1;
};

struct RewriteAuditRecord {
    int internalRoot = -1;
    int root = -1;
    int regionId = -1;
    int executionOrder = -1;
    int cutSlot = -1;
    bool preselected = false;
    std::string status;
    std::string statusReason;
    int selectedSubgraph = -1;
    std::vector<int> cutLeaves;
    std::vector<int> cutLeavesInternal;
    std::vector<int> removableNodesBefore;
    std::vector<int> removableNodesBeforeInternal;
    std::vector<int> removableNodesBeforeGeneratedInternal;
    std::vector<RewriteAuditLeafStatus> cutLeafStatusBefore;
    std::vector<int> damagedCutLeaves;
    std::vector<int> damagedByPriorRoot;
    std::vector<int> damagedByPriorExecutionOrder;
    int replacementLiteral = -1;
    int replacementNodeInternal = -1;
    int replacementNodeOriginal = -1;
    int saved = -1;
    int added = -1;
    bool savedAddedEvaluated = false;
    std::vector<int> deletedNodesInternal;
    std::vector<int> deletedNodes;
};

struct RewriteAuditMFFCResult {
    int saved = 0;
    std::vector<int> nodes;
};

std::string rewriteAuditJsonEscape(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (char c : value) {
        switch (c) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += c; break;
        }
    }
    return result;
}

void rewriteAuditWriteIntArray(std::ostream & out, const std::vector<int> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out << ',';
        out << values[i];
    }
    out << ']';
}

int rewriteAuditOriginalId(const RewriteAuditOptions & options, int internalId) {
    if (internalId < 0 || internalId >= static_cast<int>(options.internalToOriginal.size()))
        return -1;
    return options.internalToOriginal[internalId];
}

int rewriteAuditRegionId(const RewriteAuditOptions & options, int originalId) {
    if (originalId < 0 || originalId >= static_cast<int>(options.originalToRegion.size()))
        return -1;
    return options.originalToRegion[originalId];
}

RewriteAuditMFFCResult rewriteAuditCollectMFFC(
    int root, int numInputs, int graphSize,
    const int *fanin0, const int *fanin1, const int *refs,
    const std::vector<int> & cutLeaves) {
    RewriteAuditMFFCResult result;
    if (root <= numInputs || root > graphSize)
        return result;

    std::vector<int> temporaryRefs(refs, refs + graphSize + 1);
    std::vector<unsigned char> cutLeaf(graphSize + 1, 0);
    for (int leaf : cutLeaves)
        if (leaf >= 0 && leaf <= graphSize)
            cutLeaf[leaf] = 1;
    std::vector<unsigned char> touched(graphSize + 1, 0);
    std::vector<unsigned char> visited(graphSize + 1, 0);

    auto decrease = [&](int literal) {
        int node = id(literal);
        if (node < 0 || node > graphSize)
            return 1;
        if (cutLeaf[node])
            return 1;
        if (!touched[node]) {
            touched[node] = 1;
            temporaryRefs[node]--;
        } else {
            temporaryRefs[node]--;
        }
        return temporaryRefs[node];
    };

    std::function<void(int)> collect = [&](int cur) {
        if (cur <= numInputs || cur > graphSize || visited[cur])
            return;
        visited[cur] = 1;
        result.nodes.push_back(cur);
        if (!(fanin0[cur] == fanin1[cur] || fanin0[cur] == 1))
            ++result.saved;
        if (decrease(fanin0[cur]) == 0)
            collect(id(fanin0[cur]));
        if (decrease(fanin1[cur]) == 0)
            collect(id(fanin1[cur]));
    };

    collect(root);
    std::sort(result.nodes.begin(), result.nodes.end());
    return result;
}

void rewriteAuditWriteRecord(std::ostream & out, const RewriteAuditRecord & record) {
    out << "{\"root\":" << record.root
        << ",\"internal_root\":" << record.internalRoot
        << ",\"region_id\":" << record.regionId
        << ",\"execution_order\":" << record.executionOrder
        << ",\"cut_slot\":" << record.cutSlot
        << ",\"preselected\":" << (record.preselected ? "true" : "false")
        << ",\"final_status\":\"" << rewriteAuditJsonEscape(record.status) << "\""
        << ",\"status_reason\":\"" << rewriteAuditJsonEscape(record.statusReason) << "\""
        << ",\"selected_subgraph\":" << record.selectedSubgraph
        << ",\"cut_leaves\":";
    rewriteAuditWriteIntArray(out, record.cutLeaves);
    out << ",\"cut_leaves_internal\":";
    rewriteAuditWriteIntArray(out, record.cutLeavesInternal);
    out << ",\"removable_nodes_before_execution\":";
    rewriteAuditWriteIntArray(out, record.removableNodesBefore);
    out << ",\"removable_nodes_before_execution_internal\":";
    rewriteAuditWriteIntArray(out, record.removableNodesBeforeInternal);
    out << ",\"removable_nodes_before_execution_generated_internal\":";
    rewriteAuditWriteIntArray(out, record.removableNodesBeforeGeneratedInternal);
    out << ",\"cut_leaf_status_before_execution\":[";
    for (size_t i = 0; i < record.cutLeafStatusBefore.size(); ++i) {
        if (i != 0)
            out << ',';
        const auto & leaf = record.cutLeafStatusBefore[i];
        out << "{\"leaf\":" << leaf.leafOriginal
            << ",\"leaf_internal\":" << leaf.leafInternal
            << ",\"deleted\":" << (leaf.deleted ? "true" : "false")
            << ",\"first_deleted_by_prior_root\":" << leaf.firstDeletedByOriginalRoot
            << ",\"first_deleted_by_prior_root_internal\":" << leaf.firstDeletedByInternalRoot
            << ",\"first_deleted_by_prior_execution_order\":" << leaf.firstDeletedByExecutionOrder
            << '}';
    }
    out << ']';
    out << ",\"damaged_cut_leaves\":";
    rewriteAuditWriteIntArray(out, record.damagedCutLeaves);
    out << ",\"damaged_by_prior_root\":";
    rewriteAuditWriteIntArray(out, record.damagedByPriorRoot);
    out << ",\"damaged_by_prior_execution_order\":";
    rewriteAuditWriteIntArray(out, record.damagedByPriorExecutionOrder);
    out << ",\"replacement_identity\":{\"selected_subgraph\":" << record.selectedSubgraph
        << ",\"replace_literal\":" << record.replacementLiteral
        << ",\"replace_node_internal\":" << record.replacementNodeInternal
        << ",\"replace_node_original\":" << record.replacementNodeOriginal << '}'
        << ",\"saved\":" << record.saved
        << ",\"added\":" << record.added
        << ",\"saved_added_evaluated\":" << (record.savedAddedEvaluated ? "true" : "false")
        << ",\"deleted_nodes_internal\":";
    rewriteAuditWriteIntArray(out, record.deletedNodesInternal);
    out << ",\"deleted_nodes\":";
    rewriteAuditWriteIntArray(out, record.deletedNodes);
    out << "}\n";
}

void rewriteAuditWriteTrace(const RewriteAuditOptions & options,
                            const std::vector<RewriteAuditRecord> & records) {
    if (options.tracePath.empty())
        return;
    std::ofstream out(options.tracePath);
    if (!out)
        throw std::runtime_error("cannot open rewrite audit trace: " + options.tracePath);
    for (const auto & record : records)
        rewriteAuditWriteRecord(out, record);
}

} // namespace




void CPUSolver::Reset(int nInputs, int nOutputs, int nTotal, 
                      const int * pFanin0, const int * pFanin1, const int * pOuts) {
    // this should be called when the previous command is not rewrite
    FreeMem();
    AllocMem();
    
    int nAnds = nTotal - nInputs;
    numInputs = nInputs;
    numOutputs = nOutputs;
    n = nTotal;

    // CPUSolver uses AIGER literals (0=false, 1=true), unlike AIGMan.
    // Reorder/phase propagation must not rely on freshly allocated pages being
    // zero, particularly for constant outputs and repeated replay instances.
    std::fill_n(fanin0, numInputs + 1, 0);
    std::fill_n(fanin1, numInputs + 1, 0);
    std::fill_n(phase, n + 1, 0);
    std::fill_n(level, n + 1, 0);
    std::fill_n(isDeleted, n + 1, 0);
    std::fill_n(visMark, n + 1, 0);
    order[0] = 0;

    for (int i = 0; i < nAnds; i++) {
        size_t thisIdx = (size_t)(i + 1 + nInputs);
        fanin0[thisIdx] = invertConstTrueFalse(pFanin0[thisIdx]);
        fanin1[thisIdx] = invertConstTrueFalse(pFanin1[thisIdx]);
    }
    for(int i = 0; i < nOutputs; i++) {
        output[i] = invertConstTrueFalse(pOuts[i]);
    }

    // reset data
    memset(created, 0, sizeof(int) * 1000);
    visCnt = 0;
    expected = 0;

    for(int i = numInputs + 1; i <= n; i++)
        if(id(fanin0[i]) == 0 || id(fanin1[i]) == 0)
            printf("CONSTANT INPUTS FOUND\n");
    Reorder();
}

void CPUSolver::Rewrite(bool fUseZeros, bool GPUReplace) {
    expected = 0;
    gpuSolver = new GPUSolver(n);

    prt << "Rewrite Iteration" << endl;
    ReadLibrary();
    gpuSolver->CopyLib(lib);
    LevelCount();
    gpuSolver->EnumerateAndPreEvaluate(level, levelCount, n, fanin0, fanin1, ref, fUseZeros);
    prt << "Finished GPU enumeration and pre-evaluation" << endl;
    auto startTime = clock();
    if (GPUReplace) {
        gpuSolver->GetResults(n, bestOut, bestCut);
        gpuSolver->GetSelectedCutSlots(n, bestCutSlot);
        if (rewriteAuditOptions.replayMode || !rewriteAuditOptions.replayChoices.empty()) {
            std::vector<Cut> allCuts(static_cast<size_t>(n + 1) * CUT_SET_SIZE);
            std::vector<unsigned char> candidateMasks(
                static_cast<size_t>(n + 1) * CUT_SET_SIZE * MAX_REWRITE_SUBGR
            );
            gpuSolver->GetEnumeratedCuts(n, allCuts.data());
            gpuSolver->GetCandidateMasks(n, candidateMasks.data());

            std::fill(bestOut, bestOut + n + 1, -1);
            std::fill(bestCutSlot, bestCutSlot + n + 1, -1);
            std::fill(bestCut, bestCut + n + 1, Cut{});
            for (const RewriteReplayChoice & choice : rewriteAuditOptions.replayChoices) {
                if (choice.internalRoot <= numInputs || choice.internalRoot > n ||
                    choice.cutSlot < 0 || choice.cutSlot >= CUT_SET_SIZE ||
                    choice.subgraphIndex < 0 || choice.subgraphIndex >= MAX_REWRITE_SUBGR) {
                    throw std::runtime_error("Gate 3E replay choice is outside the reordered graph");
                }
                const Cut & candidateCut = allCuts[ID(choice.internalRoot, choice.cutSlot)];
                const size_t maskIndex =
                    (static_cast<size_t>(ID(choice.internalRoot, choice.cutSlot)) * MAX_REWRITE_SUBGR)
                    + choice.subgraphIndex;
                if (!candidateCut.used || candidateCut.nLeaves < 3 || !candidateMasks[maskIndex]) {
                    throw std::runtime_error("Gate 3E replay choice is not an executable GPU candidate");
                }
                if (bestCut[choice.internalRoot].used) {
                    throw std::runtime_error("Gate 3E replay contains duplicate roots");
                }
                bestCut[choice.internalRoot] = candidateCut;
                bestCut[choice.internalRoot].used = 1;
                bestOut[choice.internalRoot] = choice.subgraphIndex;
                bestCutSlot[choice.internalRoot] = choice.cutSlot;
            }
            gpuSolver->CopySelectedChoices(n, bestOut, bestCut);
        }
        int nn = n;
        int *replace = (int*)malloc((n + 1) * sizeof(int));
        n = gpuSolver->ReplaceSubgraphs(n, fanin0, fanin1, phase, replace);
        printf("after rebuild, n = %d\n", n);
        // printf(" ** pre-eval good cones: %d, replace has full corrspondence: %d\n", GPUexpected, replaceHasFullCorrsp);
        auto choiceStartTime = clock();
        for (int i = nn + 1; i <= n; i++) { // initially, new nodes are marked as deleted
            isDeleted[i] = 1;
            ref[i] = 0;
        }

        function<int(int)> install = [&](int idx) {
            if (isDeleted[idx]) {
                isDeleted[idx] = 0;
                int cnt = !isRedundant(idx);
                cnt += install(id(fanin0[idx]));
                cnt += install(id(fanin1[idx]));
                ref[id(fanin0[idx])]++;
                ref[id(fanin1[idx])]++;
                return cnt;
            } else {
                return 0;
            }
        };

        int replacedCount = 0, compromisedCount = 0, revertedCount = 0, preEvalRejectCount = 0, smallCutRejectCount = 0, newIdxRejectCount = 0;
        int replacedPosCount = 0, replacedZeroCount = 0;
        std::vector<RewriteAuditRecord> auditRecords;
        std::vector<int> firstDeletedByInternalRoot;
        std::vector<int> firstDeletedByExecutionOrder;
        if (rewriteAuditOptions.enabled) {
            auditRecords.reserve(nn - numInputs);
            firstDeletedByInternalRoot.assign(n + 1, -1);
            firstDeletedByExecutionOrder.assign(n + 1, -1);
            if (rewriteAuditOptions.preselectedInternalRoots != nullptr)
                rewriteAuditOptions.preselectedInternalRoots->clear();
        }

        std::vector<int> executionRoots;
        if (rewriteAuditOptions.enabled && !rewriteAuditOptions.executionOrder.empty()) {
            executionRoots = rewriteAuditOptions.executionOrder;
            if (static_cast<int>(executionRoots.size()) != nn - numInputs)
                throw std::runtime_error("Gate 3D execution order does not cover every original root");
        } else {
            executionRoots.reserve(nn - numInputs);
            for (int i = numInputs + 1; i <= nn; ++i)
                executionRoots.push_back(i);
        }

        auto appendAuditRecord = [&](RewriteAuditRecord && record) {
            if (rewriteAuditOptions.enabled)
                auditRecords.push_back(std::move(record));
        };

        for (int executionOrder = 0; executionOrder < static_cast<int>(executionRoots.size()); ++executionOrder) {
            int i = executionRoots[executionOrder];
            auto& cut = bestCut[i];
            int newIdx = id(replace[i]); // get the corresponding new choice node id
            RewriteAuditRecord auditRecord;
            if (rewriteAuditOptions.enabled) {
                auditRecord.internalRoot = i;
                auditRecord.root = rewriteAuditOriginalId(rewriteAuditOptions, i);
                auditRecord.regionId = rewriteAuditRegionId(rewriteAuditOptions, auditRecord.root);
                auditRecord.executionOrder = executionOrder;
                auditRecord.cutSlot = bestCutSlot[i];
                auditRecord.preselected = cut.used != 0;
                auditRecord.selectedSubgraph = cut.used ? bestOut[i] : -1;
                auditRecord.replacementLiteral = replace[i];
                auditRecord.replacementNodeInternal = newIdx;
                auditRecord.replacementNodeOriginal = rewriteAuditOriginalId(rewriteAuditOptions, newIdx);
                if (cut.used) {
                    int reportedLeafCount = cut.nLeaves;
                    auditRecord.cutLeavesInternal.assign(cut.leaves, cut.leaves + reportedLeafCount);
                    for (int leaf : auditRecord.cutLeavesInternal)
                        auditRecord.cutLeaves.push_back(rewriteAuditOriginalId(rewriteAuditOptions, leaf));
                    if (rewriteAuditOptions.preselectedInternalRoots != nullptr)
                        rewriteAuditOptions.preselectedInternalRoots->push_back(i);
                }
            }
            // if (cut.used == 0 || cut.nLeaves < 3 || newIdx == -1) {preEvalRejectCount++; continue;}
            if (cut.used == 0) {
                preEvalRejectCount++;
                auditRecord.status = "PRE_EVAL_REJECT";
                auditRecord.statusReason = "selected_cut_used_is_false";
                appendAuditRecord(std::move(auditRecord));
                continue;
            }
            if (cut.nLeaves < 3) {
                smallCutRejectCount++;
                auditRecord.status = "SMALL_CUT_REJECT";
                auditRecord.statusReason = "selected_cut_has_fewer_than_three_leaves";
                appendAuditRecord(std::move(auditRecord));
                continue;
            }
            if (newIdx == -1) {
                newIdxRejectCount++;
                auditRecord.status = replace[i] == -1 ? "FULL_CORRESPONDENCE" : "NEW_IDX_REJECT";
                auditRecord.statusReason = replace[i] == -1
                    ? "replacement_root_has_full_correspondence_in_original_graph"
                    : "replacement_node_id_is_minus_one";
                appendAuditRecord(std::move(auditRecord));
                continue;
            }

            if (newIdx == i || id(fanin1[newIdx]) == i) {
                auditRecord.status = "NO_OP";
                auditRecord.statusReason = "replacement_is_root_or_replacement_fanin_contains_root";
                appendAuditRecord(std::move(auditRecord));
                continue;
            }
            if (cut.nLeaves == 3) cut.leaves[cut.nLeaves++] = 0;

            if (rewriteAuditOptions.enabled) {
                std::vector<int> mffcLeaves(cut.leaves, cut.leaves + cut.nLeaves);
                RewriteAuditMFFCResult mffc = rewriteAuditCollectMFFC(
                    i, numInputs, n, fanin0, fanin1, ref, mffcLeaves
                );
                auditRecord.removableNodesBeforeInternal = mffc.nodes;
                for (int node : mffc.nodes) {
                    int originalNode = rewriteAuditOriginalId(rewriteAuditOptions, node);
                    if (originalNode >= 0)
                        auditRecord.removableNodesBefore.push_back(originalNode);
                    else
                        auditRecord.removableNodesBeforeGeneratedInternal.push_back(node);
                }
                for (int leaf : auditRecord.cutLeavesInternal) {
                    RewriteAuditLeafStatus leafStatus;
                    leafStatus.leafInternal = leaf;
                    leafStatus.leafOriginal = rewriteAuditOriginalId(rewriteAuditOptions, leaf);
                    leafStatus.deleted = isDeleted[leaf] != 0;
                    if (leaf >= 0 && leaf < static_cast<int>(firstDeletedByInternalRoot.size())) {
                        leafStatus.firstDeletedByInternalRoot = firstDeletedByInternalRoot[leaf];
                        leafStatus.firstDeletedByOriginalRoot = rewriteAuditOriginalId(
                            rewriteAuditOptions, firstDeletedByInternalRoot[leaf]
                        );
                        leafStatus.firstDeletedByExecutionOrder = firstDeletedByExecutionOrder[leaf];
                    }
                    auditRecord.cutLeafStatusBefore.push_back(leafStatus);
                    if (leafStatus.deleted) {
                        auditRecord.damagedCutLeaves.push_back(leafStatus.leafOriginal);
                        auditRecord.damagedByPriorRoot.push_back(leafStatus.firstDeletedByOriginalRoot);
                        auditRecord.damagedByPriorExecutionOrder.push_back(
                            leafStatus.firstDeletedByExecutionOrder
                        );
                    }
                }
            }
            bool compromised = false;  // if cut is damaged, do not perform replacement
            for (int j = 0; j < cut.nLeaves; j++) {
                if (isDeleted[cut.leaves[j]]) {
                    compromised = true;
                }
            }
            if (compromised) {
                compromisedCount++;
                auditRecord.status = "COMPROMISED";
                auditRecord.statusReason = "one_or_more_cut_leaves_are_deleted_before_install";
                appendAuditRecord(std::move(auditRecord));
                continue;
            }

            std::vector<unsigned char> deletedBeforeExecution;
            if (rewriteAuditOptions.enabled)
                deletedBeforeExecution.assign(isDeleted, isDeleted + n + 1);
            int added = install(newIdx);
            ref[newIdx]++;
            int saved = FastDelete(i);
            ref[newIdx]--;
            if (rewriteAuditOptions.enabled) {
                auditRecord.saved = saved;
                auditRecord.added = added;
                auditRecord.savedAddedEvaluated = true;
            }
            if (saved - added > 0 || (saved == added && fUseZeros)) {
                replacedCount++;
                if (saved - added > 0)
                    replacedPosCount++;
                else
                    replacedZeroCount++;

                expected += saved - added;
                isDeleted[i] = 0;
                fanin0[i] = 1;
                fanin1[i] = replace[i];
                ref[0]++;
                ref[newIdx]++;
                if (rewriteAuditOptions.enabled) {
                    auditRecord.status = "ACCEPTED";
                    for (int node = 0; node <= n; ++node) {
                        if (!deletedBeforeExecution[node] && isDeleted[node] && node != i) {
                            auditRecord.deletedNodesInternal.push_back(node);
                            int originalNode = rewriteAuditOriginalId(rewriteAuditOptions, node);
                            if (originalNode >= 0)
                                auditRecord.deletedNodes.push_back(originalNode);
                            if (node < static_cast<int>(firstDeletedByInternalRoot.size()) &&
                                firstDeletedByInternalRoot[node] == -1) {
                                firstDeletedByInternalRoot[node] = i;
                                firstDeletedByExecutionOrder[node] = executionOrder;
                            }
                        }
                    }
                    appendAuditRecord(std::move(auditRecord));
                }
            } else {
                revertedCount++;
                install(i); // revert
                ref[i]++;
                if(ref[newIdx] == 0) FastDelete(newIdx);
                ref[i]--;
                auditRecord.status = "REVERTED";
                auditRecord.statusReason = "saved_minus_added_not_positive_and_zero_gain_disabled_or_rejected";
                appendAuditRecord(std::move(auditRecord));
            }
        }
        if (rewriteAuditOptions.enabled)
            rewriteAuditWriteTrace(rewriteAuditOptions, auditRecords);
        free(replace);
        CHOICE_TIME += clock() - choiceStartTime;
        printf("successfully replaced %d cones (pos %d, zero %d), reverted %d cones, compromised %d cones, pre-eval rejected %d cones, small cut reject %d cones, new idx reject %d cones\n", 
               replacedCount, replacedPosCount, replacedZeroCount, revertedCount, compromisedCount, preEvalRejectCount, smallCutRejectCount, newIdxRejectCount);
    } else {
        gpuSolver->GetResults(n, bestOut, bestCut);
        gpuSolver->GetSelectedCutSlots(n, bestCutSlot);
        BuildTable();
        prt << "Finished building table" << endl;
        int nn = n;
        for(int i = numInputs + 1; i <= nn; i++) EvalAndReplace(i, bestCut[i]);
    }
    REPLACE_TIME += clock() - startTime;

    printf("after replace, n = %d\n", n);


    prt << "Finished eval and replace" << endl;
    startTime = clock();
    for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
        if(!isRedundant(i)) {
            fanin0[i] = Fanin(fanin0[i]);
            fanin1[i] = Fanin(fanin1[i]);
        } else
            isDeleted[i] = 1;
    }
    for(int i = 0; i < numOutputs; i++)
        output[i] = Fanin(output[i]);
    REDUNDANCY_TIME += clock() - startTime;
    startTime = clock();
    Reorder(); // remove deleted nodes
    prt << "Rewrite Iteration Ends" << endl;
    // printf("GPU expected: %d\n", GPUexpected);
    printf("real reduction: %d\n", expected);

    printf("** Total Time breakdown: ENUM %.2lf, EVAL %.2lf, REPLACE %.2lf, REORDER %.2lf, REDUNDANCY %.2lf\n",
           ENUM_TIME / CLOCKS_PER_SEC, EVAL_TIME / CLOCKS_PER_SEC, 
           REPLACE_TIME / CLOCKS_PER_SEC, REORDER_TIME / CLOCKS_PER_SEC,
           REDUNDANCY_TIME / CLOCKS_PER_SEC);
    printf("** Replace Time breakdown: COPYBACK %.2lf, REID %.2lf, CHOICE %.2lf\n", 
           COPYBACK_TIME / CLOCKS_PER_SEC, REID_TIME / CLOCKS_PER_SEC, CHOICE_TIME / CLOCKS_PER_SEC);
    printf("** CPU sequential time: %.2lf sec\n", (REDUNDANCY_TIME + CHOICE_TIME + (double)(clock() - startTime)) / CLOCKS_PER_SEC);

    delete gpuSolver;
    printf("after rewrite, n = %d\n", n);
}


int currWaveIndices(int n, int * vBuffer, const std::vector<int> & levelCount, int currIter, 
                    int waveWidth = 1, int waveStride = -1) {
    int nLevels = levelCount.size();
    std::vector<int> waveLevels;
    // level 0 are the PIs, the AND nodes starts at level 1
    for (int i = currIter * waveWidth + 1; i < nLevels; i += waveWidth + waveStride - 1) {
        for (int j = i; j < std::min(i + waveWidth, nLevels); j++)
            waveLevels.push_back(j);
        if (waveStride <= 0)
            break;
    }
    if (waveLevels.size() == 0)
        return -1;

    int nWaveNodes = 0;
    int maxLevel = -1;
    for (int level : waveLevels) {
        nWaveNodes += levelCount[level] - levelCount[level - 1];
        maxLevel = std::max(maxLevel, level);
    }
    assert(nWaveNodes > 0);

    int * vWaveMask = (int *) malloc((n + 1) * sizeof(int));
    memset(vWaveMask, 0, (n + 1) * sizeof(int));
    int curr = 0;
    for (int level : waveLevels) {
        for (int i = levelCount[level - 1]; i < levelCount[level]; i++) {
            vWaveMask[i + 1] = 1; // 1 stands for const 0/1
            curr++;
        }
    }
    assert(curr == nWaveNodes);

    cudaMemcpy(vBuffer, vWaveMask, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(vWaveMask);
    return maxLevel;
}

__global__ void EvaluateNodeWave(int sz, int *bestout, int *selectedCutSlots, int *fanin0, int *fanin1, int *isC0, int *isC1, int *nodeLevels, Cut *cuts, Cut* selectedCuts, int *nRef,
                                 Library * lib, TableNode* hashTable, const int * vWaveMask, int fUseZeros) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (!vWaveMask[id]) {
        // node not in this wave
        selectedCuts[id].used = 0;
        selectedCutSlots[id] = -1;
        return;
    }

    int reduction = -1, bestLevel = 99999999, bestCut = -1, bestOut;
    int match[54], tableSize, tableId[TABLE_SIZE], tableNum[TABLE_SIZE];
    int matchLevel[54];
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        if(cut->used == 0 || cut->nLeaves < 3) continue;
        int nleaves = cut->nLeaves;
        if(nleaves == 3)
            cut->leaves[cut->nLeaves++] = 0;
        tableSize = 0;
        int saved = CalcMFFC(id, cut, fanin0, fanin1, &tableSize, tableId, tableNum, nRef, id);
        int uPhase = lib->pPhases[cut->truthtable];
        int Class = lib->pMap[cut->truthtable];
        int *pPerm = lib->pPerms4[lib->pPerms[cut->truthtable]];
        for(int j = 0; j < 54; j++)
            match[j] = matchLevel[j] = -1;
	      uint64_t isC = 0;
        for(int j = 0; j < 4; j++) {
            match[j] = cut->leaves[pPerm[j]];
            matchLevel[j] = nodeLevels[match[j]];
            if(uPhase >> j & 1)
		            isC |= 1LL << j;
        }
        for(int j = 0; j < lib->nNodes[Class]; j++) {
            int num = j + 4;
            int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
            assert(matchLevel[in0] != -1 && matchLevel[in1] != -1);
            matchLevel[num] = 1 + (matchLevel[in0] > matchLevel[in1] ? matchLevel[in0] : matchLevel[in1]);

            if(match[in0] == -1 || match[in1] == -1 || match[in0] == id || match[in1] == id) continue;
            int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
            if(nodeId != -1 && !IsDeleted(nodeId, &tableSize, tableId, tableNum)) {
                match[num] = nodeId;
                matchLevel[num] = nodeLevels[nodeId];
            }
        }
        for(int out = 0; out < lib->nSubgr[Class]; out++) {
            int rt = lib->pSubgr[Class][out];
            if(match[rt] == id) continue;
            int nodesAdded = Eval(rt, match, Class, lib, -out - 2);
            int rtLevel = matchLevel[rt];
            assert(rtLevel != -1);
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded && bestCut == -1)) {
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded)) {
            //     reduction = saved - nodesAdded;
            //     bestCut = i;
            //     bestOut = out;
            // }
            if (saved - nodesAdded < 0 || (saved - nodesAdded == 0 && !fUseZeros))
                continue;
            if (saved - nodesAdded < reduction || (saved - nodesAdded == reduction && rtLevel >= bestLevel))
                continue;
            reduction = saved - nodesAdded;
            bestLevel = rtLevel;
            bestCut = i;
            bestOut = out;
        }
        cut->nLeaves = nleaves;
    }
    if(bestCut != -1) {
	atomicAdd(&GPUexpected, 1);
        selectedCuts[id] = cuts[ID(id, bestCut)];
        selectedCuts[id].used = 1;
        bestout[id] = bestOut;
        selectedCutSlots[id] = bestCut;
    } else
    {
        selectedCuts[id].used = 0;
        selectedCutSlots[id] = -1;
    }
}

int GPUSolver::EnumerateAndPreEvaluateWave(int currIter, int *level, const std::vector<int> &levelCount, 
                                           int n, int *CPUfanin0, int *CPUfanin1, int *CPUref, bool fUseZeros) {
    int waveWidth = 100, waveStride = -1;
    int * vWaveMask = replace; // note, replace has not been used yet, so use its memory for now
    int maxLevel = currWaveIndices(n, vWaveMask, levelCount, currIter, waveWidth, waveStride);
    if (maxLevel == -1)
        return 0;
    assert(maxLevel < levelCount.size());
    
    int * nodeLevels = phase; // note, phase has not been used yet, so use its memory for now
    cudaMemcpy(nodeLevels, level, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(fanin0, CPUfanin0, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fanin1, CPUfanin1, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(nRef, CPUref, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(hashTable, -1, sizeof(TableNode) * (2 * n + 1 + P));
    cudaMemset(cuts, 0, CUT_SET_SIZE * (n + 1) * sizeof(Cut));

    auto startTime = clock();
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin0, isComplement0, (n + 1));
    Convert<<<BLOCK_NUMBER(n + 1, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (fanin1, isComplement1, (n + 1));
    Inputs<<<BLOCK_NUMBER(levelCount[0], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> (nRef, cuts, levelCount[0]);

    for(int i = 1; i <= maxLevel; i++)
        CutEnumerate<<<BLOCK_NUMBER(levelCount[i] - levelCount[i - 1], LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>> 
            (fanin0, fanin1, isComplement0, isComplement1, nRef, cuts, levelCount[i - 1], levelCount[i] - levelCount[i - 1]);
    cudaDeviceSynchronize();
    ENUM_TIME += clock() - startTime;
    
    startTime = clock();
    BuildHashTable<<<BLOCK_NUMBER(n, LARGE_BLOCK_SIZE), LARGE_BLOCK_SIZE>>>(hashTable, n, fanin0, fanin1, isComplement0, isComplement1);
        EvaluateNodeWave<<<BLOCK_NUMBER(n, 768), 768>>>(n, bestSubgraph, selectedCutSlots, fanin0, fanin1, isComplement0, isComplement1,
                                                    nodeLevels, cuts, selectedCuts, nRef, lib, hashTable, vWaveMask, fUseZeros == true);
    gpuErrchk( cudaDeviceSynchronize() );
    std::cerr << "Error code " << cudaGetLastError() << " in EvaluateNode " << std::endl;
    EVAL_TIME += clock() - startTime;
    std::cout << "Finished GPU enumeration and pre-evaluation" << std::endl;
    return 1;
}

void CPUSolver::RewriteWave(bool fUseZeros) {
    ReadLibrary();

    for (int it = 0; ; it++) {
        printf("wave iter %d ...\n", it);
        LevelCount();
        expected = 0;
        gpuSolver = new GPUSolver(n);
        gpuSolver->CopyLib(lib);

        if (!gpuSolver->EnumerateAndPreEvaluateWave(it, level, levelCount, n, fanin0, fanin1, ref, fUseZeros)) {
            printf("end at iter %d\n", it);
            delete gpuSolver;
            break;
        }

        auto startTime = clock();
        gpuSolver->GetResults(n, bestOut, bestCut);
        int nn = n;
        int *replace = (int*)malloc((n + 1) * sizeof(int));
        n = gpuSolver->ReplaceSubgraphs(n, fanin0, fanin1, phase, replace);
        printf("after rebuild, n = %d\n", n);
        printf(" ** pre-eval good cones: %d, replace has full corrspondence: %d\n", GPUexpected, replaceHasFullCorrsp);
        auto choiceStartTime = clock();
        for (int i = nn + 1; i <= n; i++) { // initially, new nodes are marked as deleted
            isDeleted[i] = 1;
            ref[i] = 0;
        }

        function<int(int)> install = [&](int idx) {
            if (isDeleted[idx]) {
                isDeleted[idx] = 0;
                int cnt = !isRedundant(idx);
                cnt += install(id(fanin0[idx]));
                cnt += install(id(fanin1[idx]));
                ref[id(fanin0[idx])]++;
                ref[id(fanin1[idx])]++;
                return cnt;
            } else {
                return 0;
            }
        };

        int replacedCount = 0, compromisedCount = 0, revertedCount = 0, preEvalRejectCount = 0, smallCutRejectCount = 0, newIdxRejectCount = 0;
        int replacedPosCount = 0, replacedZeroCount = 0;
        for (int i = numInputs + 1; i <= nn; i++) {
            auto& cut = bestCut[i];
            int newIdx = id(replace[i]); // get the corresponding new choice node id
            // if (cut.used == 0 || cut.nLeaves < 3 || newIdx == -1) {preEvalRejectCount++; continue;}
            if (cut.used == 0) {preEvalRejectCount++; continue;}
            if (cut.nLeaves < 3) {smallCutRejectCount++; continue;}
            if (newIdx == -1) {newIdxRejectCount++; continue;}

            if (newIdx == i || id(fanin1[newIdx]) == i) continue;
            if (cut.nLeaves == 3) cut.leaves[cut.nLeaves++] = 0;
            bool compromised = false;  // if cut is damaged, do not perform replacement
            for (int j = 0; j < cut.nLeaves; j++) {
                if (isDeleted[cut.leaves[j]]) {
                    compromised = true;
                }
            }
            if (compromised) {compromisedCount++; continue;}

            int added = install(newIdx);
            ref[newIdx]++;
            int saved = FastDelete(i);
            ref[newIdx]--;
            if (saved - added > 0 || (saved == added && fUseZeros)) {
                replacedCount++;
                if (saved - added > 0)
                    replacedPosCount++;
                else
                    replacedZeroCount++;

                expected += saved - added;
                isDeleted[i] = 0;
                fanin0[i] = 1;
                fanin1[i] = replace[i];
                ref[0]++;
                ref[newIdx]++;
            } else {
                revertedCount++;
                install(i); // revert
                ref[i]++;
                if(ref[newIdx] == 0) FastDelete(newIdx);
                ref[i]--;
            }
        }
        free(replace);
        CHOICE_TIME += clock() - choiceStartTime;
        printf("successfully replaced %d cones (pos %d, zero %d), reverted %d cones, compromised %d cones, pre-eval rejected %d cones, small cut reject %d cones, new idx reject %d cones\n", 
               replacedCount, replacedPosCount, replacedZeroCount, revertedCount, compromisedCount, preEvalRejectCount, smallCutRejectCount, newIdxRejectCount);
        
        REPLACE_TIME += clock() - startTime;
        printf("after replace, n = %d\n", n);
        std::cout << "Finished eval and replace" << std::endl;

        startTime = clock();
        for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
            if(!isRedundant(i)) {
                fanin0[i] = Fanin(fanin0[i]);
                fanin1[i] = Fanin(fanin1[i]);
            } else
                isDeleted[i] = 1;
        }
        for(int i = 0; i < numOutputs; i++)
            output[i] = Fanin(output[i]);
        REDUNDANCY_TIME += clock() - startTime;
        
        Reorder(); // remove deleted nodes

        std::cout << "Rewrite Iteration Ends" << endl;
        // printf("GPU expected: %d\n", GPUexpected);
        printf("real reduction: %d\n", expected);

        printf("** Total Time breakdown: ENUM %.2lf, EVAL %.2lf, REPLACE %.2lf, REORDER %.2lf, REDUNDANCY %.2lf\n",
            ENUM_TIME / CLOCKS_PER_SEC, EVAL_TIME / CLOCKS_PER_SEC, 
            REPLACE_TIME / CLOCKS_PER_SEC, REORDER_TIME / CLOCKS_PER_SEC,
            REDUNDANCY_TIME / CLOCKS_PER_SEC);
        printf("** Replace Time breakdown: COPYBACK %.2lf, REID %.2lf, CHOICE %.2lf\n", 
            COPYBACK_TIME / CLOCKS_PER_SEC, REID_TIME / CLOCKS_PER_SEC, CHOICE_TIME / CLOCKS_PER_SEC);

        delete gpuSolver;
        printf("after rewrite iter, n = %d\n", n);
        printf("--------------------------------\n");
    }

    printf("End of rewrite wave, n = %d\n", n);
}

void CPUSolver::Init() {
    for(int i = numInputs + 1; i <= n; i++) {
        if(fanin0[i] > fanin1[i]) swap(fanin0[i], fanin1[i]);
    }
    memset(isDeleted, 0, sizeof(int) * (n + 1));
    memset(ref, 0, sizeof(int) * (n + 1));
    for(int i = numInputs + 1; i <= n; i++) {
        ref[id(fanin0[i])]++;
        ref[id(fanin1[i])]++;
        level[i] = max(level[id(fanin0[i])], level[id(fanin1[i])]) + 1;
        phase[i] = (phase[id(fanin0[i])] ^ isC(fanin0[i])) & (phase[id(fanin1[i])] ^ isC(fanin1[i]));
    }
    for(int i = 0; i < numOutputs; i++)
        ref[id(output[i])]++;
}

void CPUSolver::Reorder() {
    auto startTime = clock();

    level[0] = 0;
    order[0] = 0;

    for(int i = 1; i <= n; i++)
        level[i] = i <= numInputs ? 0 : -1;
    auto startTime2 = clock();
    
    // skipping the deleted nodes might cause errors, so do not skip them
    // the redundant nodes can be deleted by strash 
    for(int i = numInputs + 1; i <= n; i++) 
        isDeleted[i] = 0;

    for(int i = numInputs + 1; i <= n; i++) 
        if(!isDeleted[i]) TopoSort(i);
    printf(" *** Topo sort time: %.2lf sec\n", (clock() - startTime2) / (double) CLOCKS_PER_SEC);
    LevelCount();
    int newN = levelCount.back();
    for(int i = n; i >= 1; i--)
        if(!isDeleted[i]) order[i] = levelCount[level[i]]--;
    for(int i = numInputs + 1; i <= n; i++) {
        temp0[i] = fanin0[i];
        temp1[i] = fanin1[i];
    }
    for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
        fanin0[order[i]] = order[id(temp0[i])] * 2 + isC(temp0[i]);
    }
    for(int i = numInputs + 1; i <= n; i++) if(!isDeleted[i]) {
        fanin1[order[i]] = order[id(temp1[i])] * 2 + isC(temp1[i]);
    }
    for(int i = 0; i < numOutputs; i++)
        output[i] = order[id(output[i])] * 2 + isC(output[i]);
    n = newN;
    REORDER_TIME += clock() - startTime;
    Init();
}

void CPUSolver::WriteAig(const char * path) const {
    if (path == nullptr)
        throw std::runtime_error("Gate 3E output AIG path is null");
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("cannot open replay AIG: ") + path);

    file << "aig " << n << ' ' << numInputs << " 0 " << numOutputs << ' '
         << (n - numInputs) << '\n';
    for (int i = 0; i < numOutputs; ++i)
        file << static_cast<unsigned>(output[i]) << '\n';

    auto writeVarint = [&](unsigned value) {
        while (value >= 0x80U) {
            file.put(static_cast<char>((value & 0x7FU) | 0x80U));
            value >>= 7;
        }
        file.put(static_cast<char>(value));
    };
    for (int node = numInputs + 1; node <= n; ++node) {
        unsigned lhs = static_cast<unsigned>(2 * node);
        unsigned lit0 = static_cast<unsigned>(fanin0[node]);
        unsigned lit1 = static_cast<unsigned>(fanin1[node]);
        if (lit0 > lit1)
            std::swap(lit0, lit1);
        if (lhs < lit1 || lit1 < lit0)
            throw std::runtime_error("invalid topological literal while writing replay AIG");
        writeVarint(lhs - lit1);
        writeVarint(lit1 - lit0);
    }
}

void CPUSolver::ReadLibrary() {
    int cur = 0;		
    for(int i = 0; i < (1 << 16); i++)
        lib.pPhases[i] = LIBRARY[cur++];
    for(int i = 0; i < (1 << 16); i++)
        lib.pPerms[i] = LIBRARY[cur++];
    for(int i = 0; i < 2 * 3 * 4; i++)
        for(int j = 0; j < 4; j++)
            lib.pPerms4[i][j] = LIBRARY[cur++];
    for(int i = 0; i < (1 << 16); i++)
        lib.pMap[i] = LIBRARY[cur++];
    for(int i = 0; i < 222; i++)
        lib.nNodes[i] = LIBRARY[cur++];
    for(int i = 0; i < 222; i++)
        lib.nSubgr[i] = LIBRARY[cur++];
    for(int Class = 0; Class < 222; Class++) {

        for (int i = 0; i < lib.nNodes[Class]; i++) {
            lib.fanin0[Class][i] = LIBRARY[cur++];
            lib.fanin1[Class][i] = LIBRARY[cur++];
            lib.isC0[Class][i] = LIBRARY[cur++];
            lib.isC1[Class][i] = LIBRARY[cur++];
        }
        for(int j = 0; j < lib.nSubgr[Class]; j++)
            lib.pSubgr[Class][j] = LIBRARY[cur++];
    }
}

void CPUSolver::EvalAndReplace(int id, Cut &cut) {
    if(cut.used == 0 || cut.nLeaves < 3) return;
    if(cut.nLeaves == 3)
        cut.leaves[cut.nLeaves++] = 0;
    for(int i = 0; i < cut.nLeaves; i++)
        if(isDeleted[cut.leaves[i]]) return;
    visCnt++;
    //auto startTime = clock::now();
    int saved = MarkMFFC(id, cut);
    int uPhase = lib.pPhases[cut.truthtable];
    int Class = lib.pMap[cut.truthtable];
    int *pPerm = lib.pPerms4[lib.pPerms[cut.truthtable]];
    vector<int> match(54, -1), isComplement(54, 0);
    for(int i = 0; i < 4; i++) {
        match[i] = cut.leaves[pPerm[i]];
        isComplement[i] = uPhase >> i & 1;
    }
    int rt = lib.pSubgr[Class][bestOut[id]];
    vector<int> used(lib.nNodes[Class] + 4, 0);
    used[rt] = 1;
    for(int i = rt; i >= 4; i--) if(used[i]) {
        used[lib.fanin0[Class][i - 4]] = 1;
        used[lib.fanin1[Class][i - 4]] = 1;
    }
    for(int i = 0; i < lib.nNodes[Class]; i++) if(used[i + 4]) {
        int num = i + 4;
        int in0 = lib.fanin0[Class][i], in1 = lib.fanin1[Class][i];
        if(match[in0] == -1 || match[in1] == -1) continue;
        int nodeId = TableLookup(match[in0] * 2 + (isComplement[in0] ^ lib.isC0[Class][i]), match[in1] * 2 + (isComplement[in1] ^ lib.isC1[Class][i]));
        if(nodeId != -1)
            match[num] = nodeId;
    }
    int added = Eval(rt, Class, match);
    //buildTime += chrono::duration<double> (clock::now() - startTime).count();
    if(match[rt] == id) return;
    if(saved - added > 0) {
        expected += saved - added;
        //auto startTime = clock::now();
        Build(rt, Class, match, isComplement);
        Replace(match[rt], id, phase[match[rt]] ^ phase[id]);
        //replaceTime += chrono::duration<double> (clock::now() - startTime).count();
    }
}

namespace {

struct Gate3CRefState {
    std::vector<int> tableId;
    std::vector<int> tableNum;
};

std::string gate3cJsonEscape(const char * value) {
    if (value == nullptr)
        return "";
    std::string result;
    for (const char * p = value; *p != '\0'; ++p) {
        switch (*p) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += *p; break;
        }
    }
    return result;
}

void gate3cWriteIntArray(std::ostream & out, const std::vector<int> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out << ',';
        out << values[i];
    }
    out << ']';
}

int gate3cDecrease(int literal, const std::vector<int> & cutLeaves,
                  const std::vector<int> & refs, Gate3CRefState & state) {
    int nodeId = id(literal);
    for (int leaf : cutLeaves)
        if (nodeId == leaf)
            return 1;
    for (size_t i = 0; i < state.tableId.size(); ++i)
        if (state.tableId[i] == nodeId)
            return --state.tableNum[i];
    if (state.tableId.size() == TABLE_SIZE)
        return 1;
    state.tableId.push_back(nodeId);
    state.tableNum.push_back(refs[nodeId] - 1);
    return refs[nodeId] - 1;
}

int gate3cCalcMFFC(int cur, const std::vector<int> & cutLeaves,
                   const std::vector<int> & fanin0, const std::vector<int> & fanin1,
                   const std::vector<int> & refs, int nPIs,
                   Gate3CRefState & state, std::vector<int> & removable) {
    if (cur <= nPIs)
        throw std::runtime_error("GPU rewrite CalcMFFC attempted to recurse into a PI/constant");
    removable.push_back(cur);
    int saved = 1;
    if (gate3cDecrease(fanin0[cur], cutLeaves, refs, state) == 0)
        saved += gate3cCalcMFFC(id(fanin0[cur]), cutLeaves, fanin0, fanin1,
                                refs, nPIs, state, removable);
    if (gate3cDecrease(fanin1[cur], cutLeaves, refs, state) == 0)
        saved += gate3cCalcMFFC(id(fanin1[cur]), cutLeaves, fanin0, fanin1,
                                refs, nPIs, state, removable);
    return saved;
}

void gate3cLoadLibrary(rw::Library & lib) {
    int cur = 0;
    for (int i = 0; i < (1 << 16); ++i)
        lib.pPhases[i] = LIBRARY[cur++];
    for (int i = 0; i < (1 << 16); ++i)
        lib.pPerms[i] = LIBRARY[cur++];
    for (int i = 0; i < 2 * 3 * 4; ++i)
        for (int j = 0; j < 4; ++j)
            lib.pPerms4[i][j] = LIBRARY[cur++];
    for (int i = 0; i < (1 << 16); ++i)
        lib.pMap[i] = LIBRARY[cur++];
    for (int i = 0; i < 222; ++i)
        lib.nNodes[i] = LIBRARY[cur++];
    for (int i = 0; i < 222; ++i)
        lib.nSubgr[i] = LIBRARY[cur++];
    for (int Class = 0; Class < 222; ++Class) {
        for (int i = 0; i < lib.nNodes[Class]; ++i) {
            lib.fanin0[Class][i] = LIBRARY[cur++];
            lib.fanin1[Class][i] = LIBRARY[cur++];
            lib.isC0[Class][i] = LIBRARY[cur++];
            lib.isC1[Class][i] = LIBRARY[cur++];
        }
        for (int j = 0; j < lib.nSubgr[Class]; ++j)
            lib.pSubgr[Class][j] = LIBRARY[cur++];
    }
}

struct Gate3CReorderedGraph {
    std::vector<int> fanin0;
    std::vector<int> fanin1;
    std::vector<int> refs;
    std::vector<int> levels;
    std::vector<int> oldToNew;
    std::vector<int> newToOld;
    std::vector<int> levelCount;
};

Gate3CReorderedGraph gate3cReorderGraph(
    int n, int nPIs, int nPOs, const int * pFanin0, const int * pFanin1,
    const int * pOuts) {
    Gate3CReorderedGraph graph;
    std::vector<int> oldFanin0(n + 1, 0), oldFanin1(n + 1, 0);
    std::vector<int> oldLevels(n + 1, 0);

    // CPUSolver::Reset normalizes literals before its Reorder() call.  Keep
    // the same literal convention, but make the level-grouping permutation
    // explicit so the Gate 3C read-only path is deterministic on an AIG whose
    // original IDs are topological but not level-contiguous.
    for (int oldNode = nPIs + 1; oldNode <= n; ++oldNode) {
        oldFanin0[oldNode] = invertConstTrueFalse(pFanin0[oldNode]);
        oldFanin1[oldNode] = invertConstTrueFalse(pFanin1[oldNode]);
        if (oldFanin0[oldNode] > oldFanin1[oldNode])
            std::swap(oldFanin0[oldNode], oldFanin1[oldNode]);
        if (id(oldFanin0[oldNode]) >= oldNode || id(oldFanin1[oldNode]) >= oldNode)
            throw std::runtime_error("input AIG is not topologically ordered");
        oldLevels[oldNode] = 1 + std::max(
            oldLevels[id(oldFanin0[oldNode])], oldLevels[id(oldFanin1[oldNode])]
        );
    }

    int maxLevel = 0;
    for (int oldNode = 1; oldNode <= n; ++oldNode)
        maxLevel = std::max(maxLevel, oldLevels[oldNode]);

    graph.oldToNew.assign(n + 1, -1);
    graph.newToOld.assign(n + 1, -1);
    graph.oldToNew[0] = 0;
    graph.newToOld[0] = 0;
    int nextId = 1;
    for (int level = 0; level <= maxLevel; ++level) {
        // Ascending old IDs reproduce CPUSolver::Reorder's stable level order.
        for (int oldNode = 1; oldNode <= n; ++oldNode) {
            if (oldLevels[oldNode] != level)
                continue;
            graph.oldToNew[oldNode] = nextId;
            graph.newToOld[nextId] = oldNode;
            ++nextId;
        }
    }
    if (nextId != n + 1)
        throw std::runtime_error("failed to construct a complete rewrite reorder map");

    graph.fanin0.assign(n + 1, 0);
    graph.fanin1.assign(n + 1, 0);
    graph.refs.assign(n + 1, 0);
    graph.levels.assign(n + 1, 0);
    for (int newNode = 1; newNode <= n; ++newNode)
        graph.levels[newNode] = oldLevels[graph.newToOld[newNode]];
    for (int oldNode = nPIs + 1; oldNode <= n; ++oldNode) {
        int newNode = graph.oldToNew[oldNode];
        graph.fanin0[newNode] = dUtils::AigNodeLitCond(
            graph.oldToNew[id(oldFanin0[oldNode])], isC(oldFanin0[oldNode])
        );
        graph.fanin1[newNode] = dUtils::AigNodeLitCond(
            graph.oldToNew[id(oldFanin1[oldNode])], isC(oldFanin1[oldNode])
        );
        // Reorder() ends with Init(), and Init() canonicalizes each AND's
        // literal order before the GPU hash table is built.
        if (graph.fanin0[newNode] > graph.fanin1[newNode])
            std::swap(graph.fanin0[newNode], graph.fanin1[newNode]);
        ++graph.refs[id(graph.fanin0[newNode])];
        ++graph.refs[id(graph.fanin1[newNode])];
    }
    for (int po = 0; po < nPOs; ++po) {
        int output = invertConstTrueFalse(pOuts[po]);
        ++graph.refs[graph.oldToNew[id(output)]];
    }

    graph.levelCount.assign(maxLevel + 1, 0);
    for (int newNode = 1; newNode <= n; ++newNode)
        ++graph.levelCount[graph.levels[newNode]];
    for (int level = 1; level <= maxLevel; ++level)
        graph.levelCount[level] += graph.levelCount[level - 1];
    return graph;
}

} // namespace

int gate3cRewriteDecisionDump(const char * outputDir, const char * inputPath,
                              int nObjs, int nPIs, int nPOs, int nNodes,
                              const int * pFanin0, const int * pFanin1,
                              const int * pOuts, const int * pNumFanouts) {
    if (outputDir == nullptr || inputPath == nullptr ||
        pFanin0 == nullptr || pFanin1 == nullptr || pOuts == nullptr ||
        pNumFanouts == nullptr || nObjs != nNodes + nPIs + 1) {
        fprintf(stderr, "gate3c: invalid audit arguments\n");
        return 1;
    }

    try {
        std::filesystem::create_directories(outputDir);

        const int n = nObjs - 1;
        Gate3CReorderedGraph graph = gate3cReorderGraph(
            n, nPIs, nPOs, pFanin0, pFanin1, pOuts
        );
        std::vector<int> & fanin0 = graph.fanin0;
        std::vector<int> & fanin1 = graph.fanin1;
        std::vector<int> & refs = graph.refs;
        std::vector<int> & levels = graph.levels;
        std::vector<int> & levelCount = graph.levelCount;

        rw::Library lib{};
        gate3cLoadLibrary(lib);
        rw::GPUSolver gpuSolver(n);
        gpuSolver.CopyLib(lib);
        gpuSolver.EnumerateAndPreEvaluate(
            levels.data(), levelCount, n, fanin0.data(), fanin1.data(),
            refs.data(), false
        );

        std::vector<rw::Cut> cuts(static_cast<size_t>(n + 1) * CUT_SET_SIZE);
        gpuSolver.GetEnumeratedCuts(n, cuts.data());
        std::vector<unsigned char> candidateMasks(
            static_cast<size_t>(n + 1) * CUT_SET_SIZE * MAX_REWRITE_SUBGR
        );
        gpuSolver.GetCandidateMasks(n, candidateMasks.data());

        std::ofstream rawOut(std::filesystem::path(outputDir) / "gpu_rewrite_cuts.jsonl");
        if (!rawOut)
            throw std::runtime_error("cannot open gpu_rewrite_cuts.jsonl");

        int eligibleCutCount = 0;
        int decisionCount = 0;
        int candidateCount = 0;
        for (int gpuRoot = nPIs + 1; gpuRoot <= n; ++gpuRoot) {
            for (int cutSlot = 0; cutSlot < CUT_SET_SIZE; ++cutSlot) {
                const rw::Cut & storedCut = cuts[static_cast<size_t>(gpuRoot) * CUT_SET_SIZE + cutSlot];
                if (!storedCut.used || storedCut.nLeaves < 3)
                    continue;
                ++eligibleCutCount;

                rw::Cut cut = storedCut;
                int reportedLeafCount = cut.nLeaves;
                if (cut.nLeaves == 3)
                    cut.leaves[cut.nLeaves++] = 0;

                Gate3CRefState mffcState;
                std::vector<int> removable;
                int saved = gate3cCalcMFFC(
                    gpuRoot, std::vector<int>(cut.leaves, cut.leaves + cut.nLeaves),
                    fanin0, fanin1, refs, nPIs, mffcState, removable
                );
                std::sort(removable.begin(), removable.end());
                removable.erase(std::unique(removable.begin(), removable.end()), removable.end());

                int truth = cut.truthtable & 0xFFFF;
                int Class = lib.pMap[truth];
                std::vector<int> candidates;
                for (int subgraphIndex = 0; subgraphIndex < lib.nSubgr[Class]; ++subgraphIndex) {
                    if (candidateMasks[(static_cast<size_t>(ID(gpuRoot, cutSlot)) *
                                        MAX_REWRITE_SUBGR) + subgraphIndex])
                        candidates.push_back(subgraphIndex);
                }
                candidateCount += static_cast<int>(candidates.size());
                if (!candidates.empty())
                    ++decisionCount;

                std::vector<int> cutLeaves(cut.leaves, cut.leaves + reportedLeafCount);
                std::vector<int> originalCutLeaves;
                originalCutLeaves.reserve(cutLeaves.size());
                for (int leaf : cutLeaves)
                    originalCutLeaves.push_back(graph.newToOld[leaf]);
                std::vector<int> originalRemovable;
                originalRemovable.reserve(removable.size());
                for (int node : removable)
                    originalRemovable.push_back(graph.newToOld[node]);
                std::sort(originalRemovable.begin(), originalRemovable.end());

                int originalRoot = graph.newToOld[gpuRoot];
                rawOut << "{\"root\":" << originalRoot
                       << ",\"cut_slot\":" << cutSlot
                       << ",\"cut_leaves\":";
                gate3cWriteIntArray(rawOut, originalCutLeaves);
                rawOut << ",\"truth\":" << truth
                       << ",\"truth_class\":" << Class
                       << ",\"candidate_count\":" << candidates.size()
                       << ",\"candidate_subgraph_indices\":";
                gate3cWriteIntArray(rawOut, candidates);
                rawOut << ",\"removable_nodes\":";
                gate3cWriteIntArray(rawOut, originalRemovable);
                rawOut << ",\"removable_node_count\":" << originalRemovable.size()
                       << ",\"saved_count\":" << saved
                       << ",\"gpu_rewrite_node_id\":" << gpuRoot
                       << "}\n";
            }
        }

        std::ofstream manifest(std::filesystem::path(outputDir) / "gate3c_raw_manifest.json");
        if (!manifest)
            throw std::runtime_error("cannot open gate3c_raw_manifest.json");
        manifest << "{\"gate\":\"Gate 3C\","
                 << "\"input_path\":\"" << gate3cJsonEscape(inputPath) << "\","
                 << "\"n_objs\":" << nObjs
                 << ",\"n_pis\":" << nPIs
                 << ",\"n_pos\":" << nPOs
                 << ",\"n_nodes\":" << nNodes
                 << ",\"id_space\":\"original_aig_node_ids\""
                 << ",\"gpu_reorder\":\"stable_level_grouping_like_CPUSolver_Reorder\""
                 << ",\"cut_set_size\":" << CUT_SET_SIZE
                 << ",\"use_zeros\":false"
                 << ",\"eligible_cut_count\":" << eligibleCutCount
                 << ",\"executable_decision_count\":" << decisionCount
                 << ",\"candidate_count\":" << candidateCount
                 << ",\"candidate_rule\":\"saved_minus_nodes_added_gt_zero\""
                 << ",\"used_gpu_cut_enumeration\":true"
                 << ",\"used_gpu_candidate_mask_export\":true"
                 << ",\"used_gpu_rewrite_candidate_semantics\":true"
                 << ",\"replacement_executed\":false"
                 << ",\"litecut_executed\":false"
                 << ",\"gradient_executed\":false}\n";

        printf("gate3c: dumped %d eligible cuts, %d executable decisions, %d candidates to %s\n",
               eligibleCutCount, decisionCount, candidateCount, outputDir);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "gate3c: %s\n", e.what());
        return 1;
    }
}

namespace {

bool gate3dReadRegionMap(const char * path, std::vector<int> & originalToRegion) {
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line.rfind("node_id,", 0) == 0)
            continue;
        std::stringstream fields(line);
        std::string nodeText, regionText;
        if (!std::getline(fields, nodeText, ',') || !std::getline(fields, regionText, ','))
            continue;
        int node = std::stoi(nodeText);
        int region = std::stoi(regionText);
        if (node >= 0 && node < static_cast<int>(originalToRegion.size()))
            originalToRegion[node] = region;
    }
    return true;
}

void gate3dRunOnce(const char * outputTrace,
                   int nObjs, int nPIs, int nPOs,
                   const int * pFanin0, const int * pFanin1,
                   const int * pOuts,
                   const std::vector<int> & internalToOriginal,
                   const std::vector<int> & originalToRegion,
                   const std::vector<int> & executionOrder,
                   std::vector<int> & preselectedRoots) {
    rw::RewriteAuditOptions options;
    options.enabled = true;
    options.tracePath = outputTrace;
    options.internalToOriginal = internalToOriginal;
    options.originalToRegion = originalToRegion;
    options.executionOrder = executionOrder;
    options.preselectedInternalRoots = &preselectedRoots;

    rw::CPUSolver solver;
    solver.ConfigureRewriteAudit(options);
    solver.Reset(nPIs, nPOs, nObjs - 1, pFanin0, pFanin1, pOuts);
    solver.Rewrite(false, true);
}

} // namespace

int gate3dRewriteExecutionAudit(const char * outputDir, const char * inputPath,
                                const char * regionMapPath,
                                int nObjs, int nPIs, int nPOs, int nNodes,
                                const int * pFanin0, const int * pFanin1,
                                const int * pOuts, const int * pNumFanouts) {
    (void)pNumFanouts;
    if (outputDir == nullptr || inputPath == nullptr || regionMapPath == nullptr ||
        pFanin0 == nullptr || pFanin1 == nullptr || pOuts == nullptr ||
        nObjs != nNodes + nPIs + 1) {
        fprintf(stderr, "gate3d: invalid audit arguments\n");
        return 1;
    }

    try {
        std::filesystem::path root(outputDir);
        std::filesystem::path orderRoot = root / "order_sensitivity";
        std::filesystem::create_directories(orderRoot);

        const int n = nObjs - 1;
        Gate3CReorderedGraph graph = gate3cReorderGraph(
            n, nPIs, nPOs, pFanin0, pFanin1, pOuts
        );
        std::vector<int> originalToRegion(nObjs, -1);
        if (!gate3dReadRegionMap(regionMapPath, originalToRegion))
            throw std::runtime_error("cannot read Gate 3A region map: " + std::string(regionMapPath));

        std::vector<int> primaryRoots;
        gate3dRunOnce(
            (root / "decision_execution_trace.jsonl").c_str(),
            nObjs, nPIs, nPOs, pFanin0, pFanin1, pOuts,
            graph.newToOld, originalToRegion, {}, primaryRoots
        );

        std::map<int, std::vector<int>> rootsByRegion;
        for (int internalRoot : primaryRoots) {
            int originalRoot = rewriteAuditOriginalId(
                rw::RewriteAuditOptions{false, "", graph.newToOld, originalToRegion, {}, nullptr},
                internalRoot
            );
            int region = originalRoot >= 0 && originalRoot < nObjs
                ? originalToRegion[originalRoot] : -1;
            if (region >= 0)
                rootsByRegion[region].push_back(internalRoot);
        }

        std::vector<int> selectedRegions;
        for (const auto & entry : rootsByRegion)
            if (entry.second.size() >= 2)
                selectedRegions.push_back(entry.first);
        std::sort(selectedRegions.begin(), selectedRegions.end(), [&](int left, int right) {
            if (rootsByRegion[left].size() != rootsByRegion[right].size())
                return rootsByRegion[left].size() > rootsByRegion[right].size();
            return left < right;
        });
        if (selectedRegions.size() > 3)
            selectedRegions.resize(3);

        std::ofstream manifest(root / "gate3d_raw_manifest.json");
        if (!manifest)
            throw std::runtime_error("cannot open gate3d_raw_manifest.json");
        manifest << "{\"gate\":\"Gate 3D\",\"input_path\":\""
                 << gate3cJsonEscape(inputPath) << "\",\"region_map\":\""
                 << gate3cJsonEscape(regionMapPath) << "\",\"n_objs\":" << nObjs
                 << ",\"n_pis\":" << nPIs << ",\"n_pos\":" << nPOs
                 << ",\"n_nodes\":" << nNodes << ",\"primary_order\":\"ascending_internal_root\""
                 << ",\"selected_order_sensitivity_regions\":[";
        for (size_t i = 0; i < selectedRegions.size(); ++i) {
            if (i != 0)
                manifest << ',';
            manifest << selectedRegions[i];
        }
        manifest << "]}\n";

        for (int region : selectedRegions) {
            std::vector<int> order;
            order.reserve(n - nPIs);
            std::vector<int> regionRoots = rootsByRegion[region];
            std::reverse(regionRoots.begin(), regionRoots.end());
            bool inserted = false;
            for (int internalRoot = nPIs + 1; internalRoot <= n; ++internalRoot) {
                bool isRegionRoot = std::find(
                    rootsByRegion[region].begin(), rootsByRegion[region].end(), internalRoot
                ) != rootsByRegion[region].end();
                if (isRegionRoot) {
                    if (!inserted) {
                        order.insert(order.end(), regionRoots.begin(), regionRoots.end());
                        inserted = true;
                    }
                    continue;
                }
                order.push_back(internalRoot);
            }
            if (!inserted)
                throw std::runtime_error("failed to construct reverse order for region");

            std::ostringstream tracePath;
            tracePath << (orderRoot / ("region_" + std::to_string(region) + "_reverse.jsonl")).string();
            std::vector<int> replayRoots;
            gate3dRunOnce(
                (orderRoot / ("region_" + std::to_string(region) + "_original.jsonl")).c_str(),
                nObjs, nPIs, nPOs, pFanin0, pFanin1, pOuts,
                graph.newToOld, originalToRegion, {}, replayRoots
            );
            gate3dRunOnce(
                tracePath.str().c_str(),
                nObjs, nPIs, nPOs, pFanin0, pFanin1, pOuts,
                graph.newToOld, originalToRegion, order, replayRoots
            );
        }

        printf("gate3d: audited original GPU rewrite and %zu reverse-order region replays to %s\n",
               selectedRegions.size(), outputDir);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "gate3d: %s\n", e.what());
        return 1;
    }
}

namespace {

struct Gate3EConfig {
    std::string id;
    std::vector<int> roots;
    std::vector<int> cutSlots;
    std::vector<int> subgraphs;
};

bool gate3eReadConfigFile(const char * path, std::vector<Gate3EConfig> & configs) {
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    Gate3EConfig current;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream fields(line);
        std::string configId;
        int root = -1, cutSlot = -1, subgraph = -1;
        if (!(fields >> configId >> root >> cutSlot >> subgraph))
            return false;
        if (current.id != configId) {
            if (!current.id.empty())
                configs.push_back(std::move(current));
            current = Gate3EConfig{};
            current.id = configId;
        }
        if (root >= 0) {
            current.roots.push_back(root);
            current.cutSlots.push_back(cutSlot);
            current.subgraphs.push_back(subgraph);
        }
    }
    if (!current.id.empty())
        configs.push_back(std::move(current));
    return !configs.empty();
}

void gate3eWriteManifestLine(std::ofstream & out, const std::string & configId,
                             const std::filesystem::path & aigPath,
                             const std::filesystem::path & tracePath,
                             size_t requestedCount) {
    const std::filesystem::path aigRelative =
        aigPath.parent_path().filename() / aigPath.filename();
    const std::filesystem::path traceRelative =
        tracePath.parent_path().filename() / tracePath.filename();
    out << "{\"config_id\":\"" << gate3cJsonEscape(configId.c_str())
        << "\",\"requested_count\":" << requestedCount
        << ",\"aig_path\":\"" << gate3cJsonEscape(aigRelative.c_str())
        << "\",\"trace_path\":\"" << gate3cJsonEscape(traceRelative.c_str()) << "\"}\n";
}

} // namespace

int gate3eExecutionReplay(const char * outputDir, const char * inputPath,
                          const char * regionMapPath, const char * configPath,
                          int nObjs, int nPIs, int nPOs, int nNodes,
                          const int * pFanin0, const int * pFanin1,
                          const int * pOuts, const int * pNumFanouts) {
    (void)pNumFanouts;
    if (outputDir == nullptr || inputPath == nullptr || regionMapPath == nullptr ||
        configPath == nullptr || pFanin0 == nullptr || pFanin1 == nullptr ||
        pOuts == nullptr || nObjs != nNodes + nPIs + 1) {
        fprintf(stderr, "gate3e: invalid replay arguments\n");
        return 1;
    }

    try {
        std::filesystem::path root(outputDir);
        std::filesystem::create_directories(root / "aigs");
        std::filesystem::create_directories(root / "traces");

        std::vector<Gate3EConfig> configs;
        if (!gate3eReadConfigFile(configPath, configs))
            throw std::runtime_error("cannot read Gate 3E configuration file");

        const int n = nObjs - 1;
        Gate3CReorderedGraph graph = gate3cReorderGraph(
            n, nPIs, nPOs, pFanin0, pFanin1, pOuts
        );
        std::vector<int> originalToRegion(nObjs, -1);
        if (!gate3dReadRegionMap(regionMapPath, originalToRegion))
            throw std::runtime_error("cannot read Gate 3A region map: " + std::string(regionMapPath));

        std::ofstream manifest(root / "replay_manifest.jsonl");
        if (!manifest)
            throw std::runtime_error("cannot open Gate 3E replay manifest");

        for (const Gate3EConfig & config : configs) {
            if (config.roots.size() != config.cutSlots.size() ||
                config.roots.size() != config.subgraphs.size())
                throw std::runtime_error("malformed Gate 3E configuration");

            rw::RewriteAuditOptions options;
            options.enabled = true;
            options.replayMode = true;
            options.internalToOriginal = graph.newToOld;
            options.originalToRegion = originalToRegion;
            std::vector<bool> selectedInternal(static_cast<size_t>(n + 1), false);
            for (size_t i = 0; i < config.roots.size(); ++i) {
                int originalRoot = config.roots[i];
                if (originalRoot < 0 || originalRoot >= static_cast<int>(graph.oldToNew.size()) ||
                    graph.oldToNew[originalRoot] < 0)
                    throw std::runtime_error("Gate 3E configuration root is outside the input AIG");
                int internalRoot = graph.oldToNew[originalRoot];
                if (selectedInternal[internalRoot])
                    throw std::runtime_error("Gate 3E configuration contains duplicate roots");
                selectedInternal[internalRoot] = true;
                options.replayChoices.push_back({
                    internalRoot, config.cutSlots[i], config.subgraphs[i]
                });
                // The requested list is the controlled order. Unrequested roots
                // are appended below and remain PRE_EVAL_REJECT entries.
                options.executionOrder.push_back(internalRoot);
            }
            for (int internalRoot = nPIs + 1; internalRoot <= n; ++internalRoot)
                if (!selectedInternal[internalRoot])
                    options.executionOrder.push_back(internalRoot);

            std::filesystem::path tracePath = root / "traces" / (config.id + ".jsonl");
            std::filesystem::path aigPath = root / "aigs" / (config.id + ".aig");
            options.tracePath = tracePath.string();

            rw::CPUSolver solver;
            solver.ConfigureRewriteAudit(options);
            solver.Reset(nPIs, nPOs, nObjs - 1, pFanin0, pFanin1, pOuts);
            solver.Rewrite(false, true);
            solver.WriteAig(aigPath.c_str());
            gate3eWriteManifestLine(manifest, config.id, aigPath, tracePath, config.roots.size());
        }
        printf("gate3e: replayed %zu ordered configurations to %s\n", configs.size(), outputDir);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "gate3e: %s\n", e.what());
        return 1;
    }
}
