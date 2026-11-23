"""Apply the read-only rewrite-candidate export instrumentation to an ABC tree.

The instrumentation is inert unless CULS_RWR_DUMP points at an existing
directory. It never changes default behaviour: every edit is guarded by
`s_fDumpOn`, and the subgraph evaluation falls back to the untouched code path
when the variable is unset.

Each replacement asserts that its anchor occurs exactly once, so source drift
fails loudly instead of silently producing a partially patched tree.
"""
import argparse
import sys
from pathlib import Path

DUMP_BLOCK = r'''
/* ==== CULS rewrite-candidate export (research instrumentation) ============
   Enabled only when CULS_RWR_DUMP names an existing directory. In that mode
   every library subgraph of every four-input cut is evaluated with unbounded
   NodeMax/LevelMax, the network is left unchanged, and JSONL records are
   written. With the variable unset this code never runs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int Dec_GraphToNetworkCount( Abc_Obj_t * pRoot, Dec_Graph_t * pGraph, int NodeMax, int LevelMax );

#define CULS_MAX_GRAPH 256

static FILE * s_pDumpCuts  = NULL;
static FILE * s_pDumpNodes = NULL;
static char   s_DumpDir[2048] = {0};
static int    s_fDumpOn = 0;
static int    s_fDumpTried = 0;
static int    s_nCutCand = 0;

static int s_nNodeReq;
static int s_nNodeCutTotal, s_nNodeCutLt4, s_nNodeCutNull, s_nNodeCutFan, s_nNodeCutExp;
static int s_nNodeCand, s_nNodeCandAcc, s_nNodeSelf;

static int s_nAllNodesEval, s_nAllNodesPersist, s_nAllNodesFanout;
static int s_nAllCutTotal, s_nAllCutLt4, s_nAllCutNull, s_nAllCutFan, s_nAllCutExp, s_nAllCutTrivial;
static int s_nAllCand, s_nAllCandAcc, s_nAllSelf;

static void Rwr_DumpOpen( void )
{
    char File[2560];
    const char * pDir;
    if ( s_fDumpTried )
        return;
    s_fDumpTried = 1;
    pDir = getenv( "CULS_RWR_DUMP" );
    if ( pDir == NULL || pDir[0] == 0 )
        return;
    snprintf( s_DumpDir, sizeof(s_DumpDir), "%s", pDir );
    snprintf( File, sizeof(File), "%s/cuts.jsonl", s_DumpDir );
    s_pDumpCuts = fopen( File, "w" );
    snprintf( File, sizeof(File), "%s/nodes.jsonl", s_DumpDir );
    s_pDumpNodes = fopen( File, "w" );
    if ( s_pDumpCuts == NULL || s_pDumpNodes == NULL )
    {
        fprintf( stderr, "CULS_RWR_DUMP: cannot create output files under %s\n", s_DumpDir );
        exit( 1 );
    }
    s_fDumpOn = 1;
    fprintf( stderr, "CULS_RWR_DUMP: writing to %s\n", s_DumpDir );
}

int Rwr_DumpEnabled( void )
{
    Rwr_DumpOpen();
    return s_fDumpOn;
}

void Rwr_DumpNodeTable( Abc_Ntk_t * pNtk )
{
    char File[2560];
    FILE * pFile;
    Abc_Obj_t * pObj;
    int i;
    if ( !Rwr_DumpEnabled() )
        return;
    Abc_NtkLevel( pNtk );
    snprintf( File, sizeof(File), "%s/nodes.tsv", s_DumpDir );
    pFile = fopen( File, "w" );
    if ( pFile == NULL )
    {
        fprintf( stderr, "CULS_RWR_DUMP: cannot write nodes.tsv\n" );
        exit( 1 );
    }
    fprintf( pFile, "abc_id\tkind\tfanin0_lit\tfanin1_lit\tlevel\tfanout\n" );
    Abc_NtkForEachObj( pNtk, pObj, i )
    {
        if ( Abc_ObjIsNode(pObj) )
            fprintf( pFile, "%d\tAND\t%d\t%d\t%d\t%d\n", pObj->Id,
                Abc_ObjFaninId0(pObj) * 2 + Abc_ObjFaninC0(pObj),
                Abc_ObjFaninId1(pObj) * 2 + Abc_ObjFaninC1(pObj),
                (int)pObj->Level, Abc_ObjFanoutNum(pObj) );
        else if ( Abc_ObjIsPi(pObj) )
            fprintf( pFile, "%d\tPI\t-1\t-1\t%d\t%d\n", pObj->Id, (int)pObj->Level, Abc_ObjFanoutNum(pObj) );
        else if ( Abc_ObjIsPo(pObj) )
            fprintf( pFile, "%d\tPO\t%d\t-1\t%d\t0\n", pObj->Id,
                Abc_ObjFaninId0(pObj) * 2 + Abc_ObjFaninC0(pObj), (int)pObj->Level );
        else
            fprintf( pFile, "%d\tOTHER\t-1\t-1\t%d\t%d\n", pObj->Id, (int)pObj->Level, Abc_ObjFanoutNum(pObj) );
    }
    fclose( pFile );
}

void Rwr_DumpNodeBegin( int Required )
{
    s_nNodeReq = Required;
    s_nNodeCutTotal = s_nNodeCutLt4 = s_nNodeCutNull = s_nNodeCutFan = s_nNodeCutExp = 0;
    s_nNodeCand = s_nNodeCandAcc = s_nNodeSelf = 0;
}

void Rwr_DumpNodeRecord( Abc_Obj_t * pNode, int GainBest, int Reason )
{
    const char * pReason = Reason == 0 ? "evaluated" : Reason == 1 ? "persistent" : "high_fanout";
    if ( Reason == 1 )
        s_nAllNodesPersist++;
    else if ( Reason == 2 )
        s_nAllNodesFanout++;
    else
        s_nAllNodesEval++;
    if ( Reason != 0 )
        Rwr_DumpNodeBegin( -1 );
    if ( !s_fDumpOn )
        return;
    s_nAllCutTotal += s_nNodeCutTotal;
    s_nAllCutLt4 += s_nNodeCutLt4;
    s_nAllCutNull += s_nNodeCutNull;
    s_nAllCutFan += s_nNodeCutFan;
    s_nAllCutExp += s_nNodeCutExp;
    s_nAllCand += s_nNodeCand;
    s_nAllCandAcc += s_nNodeCandAcc;
    s_nAllSelf += s_nNodeSelf;
    fprintf( s_pDumpNodes,
        "{\"node\":%d,\"reason\":\"%s\",\"level\":%d,\"required\":%d,"
        "\"cuts_total\":%d,\"cuts_lt4_leaves\":%d,\"cuts_null_leaf\":%d,"
        "\"cuts_single_fanout_gt2\":%d,\"cuts_exported\":%d,"
        "\"candidates\":%d,\"candidates_accepted_by_default\":%d,"
        "\"candidates_self_replacement\":%d,\"gain_best\":%d}\n",
        pNode->Id, pReason, (int)Abc_ObjRegular(pNode)->Level,
        Reason == 0 ? s_nNodeReq : -1,
        s_nNodeCutTotal, s_nNodeCutLt4, s_nNodeCutNull, s_nNodeCutFan, s_nNodeCutExp,
        s_nNodeCand, s_nNodeCandAcc, s_nNodeSelf, GainBest );
}

void Rwr_DumpCutBegin( Rwr_Man_t * p, Abc_Obj_t * pRoot, Cut_Cut_t * pCut,
                       Vec_Ptr_t * vFaninsCur, int nNodesSaved, int LevelMax )
{
    Abc_Obj_t * pFanin;
    unsigned uTruth = 0xFFFF & *Cut_CutReadTruth(pCut);
    char * pPerm = p->pPerms4[ (int)p->pPerms[uTruth] ];
    int i, n = Vec_PtrSize( vFaninsCur );
    s_nCutCand = 0;
    s_nNodeCutTotal++;
    s_nNodeCutExp++;
    fprintf( s_pDumpCuts,
        "{\"node\":%d,\"node_level\":%d,\"required\":%d,\"n_nodes_saved\":%d,"
        "\"level_max\":%d,\"truth\":%u,\"truth_class\":%d,\"leaves\":[",
        pRoot->Id, (int)Abc_ObjRegular(pRoot)->Level, s_nNodeReq,
        nNodesSaved, LevelMax, uTruth, (int)p->pMap[uTruth] );
    for ( i = 0; i < n; i++ )
    {
        pFanin = (Abc_Obj_t *)Vec_PtrEntry( vFaninsCur, i );
        fprintf( s_pDumpCuts, "%s{\"id\":%d,\"compl\":%d,\"slot\":%d,\"level\":%d}",
            i ? "," : "", Abc_ObjId(Abc_ObjRegular(pFanin)), Abc_ObjIsComplement(pFanin),
            (int)pPerm[i], (int)Abc_ObjRegular(pFanin)->Level );
    }
    fprintf( s_pDumpCuts, "],\"candidates\":[" );
}

void Rwr_DumpCutEnd( void )
{
    if ( !s_fDumpOn )
        return;
    fprintf( s_pDumpCuts, "]}\n" );
}

static void Rwr_DumpGraphDepths( Dec_Graph_t * pGraph, int * pDepths )
{
    int i, nSize = pGraph->nSize, nLeaves = pGraph->nLeaves;
    int Dist[CULS_MAX_GRAPH];
    if ( nSize > CULS_MAX_GRAPH )
    {
        fprintf( stderr, "CULS_RWR_DUMP: graph with %d nodes exceeds the export limit\n", nSize );
        exit( 1 );
    }
    for ( i = 0; i < nLeaves; i++ )
        pDepths[i] = -1;
    if ( Dec_GraphIsConst(pGraph) || Dec_GraphIsVar(pGraph) )
    {
        if ( Dec_GraphIsVar(pGraph) )
            pDepths[(int)pGraph->eRoot.Node] = 0;
        return;
    }
    for ( i = 0; i < nSize; i++ )
        Dist[i] = -1;
    Dist[(int)pGraph->eRoot.Node] = 0;
    for ( i = nSize - 1; i >= nLeaves; i-- )
    {
        Dec_Node_t * pNode = pGraph->pNodes + i;
        if ( Dist[i] < 0 )
            continue;
        if ( Dist[(int)pNode->eEdge0.Node] < Dist[i] + 1 )
            Dist[(int)pNode->eEdge0.Node] = Dist[i] + 1;
        if ( Dist[(int)pNode->eEdge1.Node] < Dist[i] + 1 )
            Dist[(int)pNode->eEdge1.Node] = Dist[i] + 1;
    }
    for ( i = 0; i < nLeaves; i++ )
        pDepths[i] = Dist[i];
}

/* Returns the node count ABC itself would accept, or -1 when the default rule
   rejects the candidate. The unbounded call also fills the per-node levels
   that give ABC's own arrival level at the new root. */
int Rwr_DumpCandidate( Abc_Obj_t * pRoot, Dec_Graph_t * pGraph,
                       int nNodesSaved, int LevelMax, int iSub, int nLeaves )
{
    int Depths[CULS_MAX_GRAPH], AddedTrue, Accepted = 0, Self = 0, k;
    int Structural = 0, AbcLevel = -1, Lv;
    AddedTrue = Dec_GraphToNetworkCount( pRoot, pGraph, ABC_INFINITY, ABC_INFINITY );
    Rwr_DumpGraphDepths( pGraph, Depths );
    if ( Dec_GraphIsConst(pGraph) )
        AbcLevel = 0;
    else if ( Dec_GraphIsVar(pGraph) )
        AbcLevel = (int)Abc_ObjRegular((Abc_Obj_t *)pGraph->pNodes[(int)pGraph->eRoot.Node].pFunc)->Level;
    else
        AbcLevel = (int)Dec_GraphNode( pGraph, (int)pGraph->eRoot.Node )->Level;
    for ( k = 0; k < nLeaves; k++ )
    {
        if ( Depths[k] < 0 )
            continue;
        Lv = (int)Abc_ObjRegular((Abc_Obj_t *)pGraph->pNodes[k].pFunc)->Level;
        if ( Structural < Lv + Depths[k] )
            Structural = Lv + Depths[k];
    }
    if ( AddedTrue < 0 )
        Self = 1;
    else
        Accepted = ( AddedTrue <= nNodesSaved && AbcLevel <= LevelMax );
    s_nNodeCand++;
    if ( Self )
        s_nNodeSelf++;
    else if ( Accepted )
        s_nNodeCandAcc++;
    fprintf( s_pDumpCuts,
        "%s{\"subgraph\":%d,\"nodes_added\":%d,\"self\":%d,\"accepted_by_default\":%d,"
        "\"root_level\":%d,\"abc_level\":%d,\"leaf_depths\":[",
        s_nCutCand++ ? "," : "", iSub, AddedTrue, Self, Accepted, Structural, AbcLevel );
    for ( k = 0; k < nLeaves; k++ )
        fprintf( s_pDumpCuts, "%s%d", k ? "," : "", Depths[k] );
    fprintf( s_pDumpCuts, "]}" );
    return Accepted ? AddedTrue : -1;
}

void Rwr_DumpFinish( void )
{
    char File[2560];
    FILE * pFile;
    if ( !s_fDumpOn )
        return;
    if ( s_pDumpCuts )  { fclose( s_pDumpCuts );  s_pDumpCuts = NULL;  }
    if ( s_pDumpNodes ) { fclose( s_pDumpNodes ); s_pDumpNodes = NULL; }
    snprintf( File, sizeof(File), "%s/counters.json", s_DumpDir );
    pFile = fopen( File, "w" );
    if ( pFile == NULL )
    {
        fprintf( stderr, "CULS_RWR_DUMP: cannot write counters.json\n" );
        exit( 1 );
    }
    fprintf( pFile,
        "{\n"
        "  \"nodes_evaluated\": %d,\n"
        "  \"nodes_persistent\": %d,\n"
        "  \"nodes_high_fanout\": %d,\n"
        "  \"cuts_trivial_skipped\": %d,\n"
        "  \"cuts_total\": %d,\n"
        "  \"cuts_lt4_leaves\": %d,\n"
        "  \"cuts_null_leaf\": %d,\n"
        "  \"cuts_single_fanout_gt2\": %d,\n"
        "  \"cuts_exported\": %d,\n"
        "  \"candidates\": %d,\n"
        "  \"candidates_accepted_by_default\": %d,\n"
        "  \"candidates_self_replacement\": %d\n"
        "}\n",
        s_nAllNodesEval, s_nAllNodesPersist, s_nAllNodesFanout, s_nAllCutTrivial,
        s_nAllCutTotal, s_nAllCutLt4, s_nAllCutNull, s_nAllCutFan, s_nAllCutExp,
        s_nAllCand, s_nAllCandAcc, s_nAllSelf );
    fclose( pFile );
}
/* ==== end of CULS export instrumentation ==== */
'''

# --- rwrEva.c ---------------------------------------------------------------

RWR_PATCHES = [
    ("ABC_NAMESPACE_IMPL_START\n",
     "ABC_NAMESPACE_IMPL_START\n" + DUMP_BLOCK,
     'dump block'),
    ("""    Required = fUpdateLevel? Abc_ObjRequiredLevel(pNode) : ABC_INFINITY;
""",
     """    Required = fUpdateLevel? Abc_ObjRequiredLevel(pNode) : ABC_INFINITY;
    if ( s_fDumpOn )
        Rwr_DumpNodeBegin( Required );
""",
     'node begin'),
    ("""        if ( pCut->nLeaves < 4 )
            continue;
""",
     """        if ( pCut->nLeaves < 4 )
        {
            if ( s_fDumpOn )
                s_nNodeCutLt4++;
            continue;
        }
""",
     'lt4 counter'),
    ("""            p->nCutsBad++;
            continue;
""",
     """            p->nCutsBad++;
            if ( s_fDumpOn )
                s_nNodeCutNull++;
            continue;
""",
     'null-leaf counter'),
    ("""            if ( Counter > 2 )
                continue;
""",
     """            if ( Counter > 2 )
            {
                if ( s_fDumpOn )
                    s_nNodeCutFan++;
                continue;
            }
""",
     'fanout counter'),
    ("""    for ( pCut = pCut->pNext; pCut; pCut = pCut->pNext )
    {
""",
     """    if ( s_fDumpOn )
        s_nAllCutTrivial++;
    for ( pCut = pCut->pNext; pCut; pCut = pCut->pNext )
    {
""",
     'trivial-cut counter'),
    ("""    Vec_PtrForEachEntry( Rwr_Node_t *, vSubgraphs, pNode, i )
    {
""",
     """    if ( s_fDumpOn )
        Rwr_DumpCutBegin( p, pRoot, pCut, vFaninsCur, nNodesSaved, LevelMax );
    Vec_PtrForEachEntry( Rwr_Node_t *, vSubgraphs, pNode, i )
    {
""",
     'cut begin'),
    ("""        // detect how many unlabeled nodes will be reused
        nNodesAdded = Dec_GraphToNetworkCount( pRoot, pGraphCur, nNodesSaved, LevelMax );
        if ( nNodesAdded == -1 )
            continue;
        assert( nNodesSaved >= nNodesAdded );
""",
     """        // detect how many unlabeled nodes will be reused
        if ( s_fDumpOn )
            nNodesAdded = Rwr_DumpCandidate( pRoot, pGraphCur, nNodesSaved, LevelMax, i,
                                             Vec_PtrSize(vFaninsCur) );
        else
            nNodesAdded = Dec_GraphToNetworkCount( pRoot, pGraphCur, nNodesSaved, LevelMax );
        if ( nNodesAdded == -1 )
            continue;
        assert( nNodesSaved >= nNodesAdded );
""",
     'candidate evaluation'),
    ("""    if ( GainBest == -1 )
        return NULL;
    *pGainBest = GainBest;
""",
     """    if ( s_fDumpOn )
        Rwr_DumpCutEnd();
    if ( GainBest == -1 )
        return NULL;
    *pGainBest = GainBest;
""",
     'cut end'),
]

# --- abcRewrite.c -----------------------------------------------------------

REW_PATCHES = [
    ("""#include "bool/dec/dec.h"

ABC_NAMESPACE_IMPL_START
""",
     """#include "bool/dec/dec.h"

extern int  Rwr_DumpEnabled( void );
extern void Rwr_DumpNodeTable( Abc_Ntk_t * pNtk );
extern void Rwr_DumpNodeRecord( Abc_Obj_t * pNode, int GainBest, int Reason );
extern void Rwr_DumpFinish( void );

ABC_NAMESPACE_IMPL_START
""",
     'extern declarations'),
    ("""    int i, nNodes, nGain, fCompl, RetValue = 1;
""",
     """    int i, nNodes, nGain, fCompl, RetValue = 1, fDump;
""",
     'local flag declaration'),
    ("""    // cleanup the AIG
    Abc_AigCleanup((Abc_Aig_t *)pNtk->pManFunc);
""",
     """    // cleanup the AIG
    Abc_AigCleanup((Abc_Aig_t *)pNtk->pManFunc);
    fDump = Rwr_DumpEnabled();
    Rwr_DumpNodeTable( pNtk );
""",
     'node table'),
    ("""        if ( Abc_NodeIsPersistant(pNode) )
            continue;
""",
     """        if ( Abc_NodeIsPersistant(pNode) )
        {
            Rwr_DumpNodeRecord( pNode, -1, 1 );
            continue;
        }
""",
     'persistent skip'),
    ("""        if ( Abc_ObjFanoutNum(pNode) > 1000 )
            continue;
""",
     """        if ( Abc_ObjFanoutNum(pNode) > 1000 )
        {
            Rwr_DumpNodeRecord( pNode, -1, 2 );
            continue;
        }
""",
     'high-fanout skip'),
    ("""        nGain = Rwr_NodeRewrite( pManRwr, pManCut, pNode, fUpdateLevel, fUseZeros, fPlaceEnable );
        if ( !(nGain > 0 || (nGain == 0 && fUseZeros)) )
            continue;
""",
     """        nGain = Rwr_NodeRewrite( pManRwr, pManCut, pNode, fUpdateLevel, fUseZeros, fPlaceEnable );
        if ( fDump )
        {
            /* Export mode enumerates on the untouched network: never apply. */
            Rwr_DumpNodeRecord( pNode, nGain, 0 );
            continue;
        }
        if ( !(nGain > 0 || (nGain == 0 && fUseZeros)) )
            continue;
""",
     'no-apply hook'),
    ("""    Extra_ProgressBarStop( pProgress );
""",
     """    Extra_ProgressBarStop( pProgress );
    Rwr_DumpFinish();
""",
     'finish'),
]

TARGETS = {
    'rwrEva.c': ('src/opt/rwr/rwrEva.c', RWR_PATCHES),
    'abcRewrite.c': ('src/base/abci/abcRewrite.c', REW_PATCHES),
}


def patch_file(path, edits):
    text = path.read_text()
    for old, new, label in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f'{path.name}: anchor "{label}" matched {count} times, expected 1')
        text = text.replace(old, new, 1)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abc', required=True, type=Path, help='ABC source tree to patch in place')
    args = parser.parse_args()
    root = args.abc.resolve()
    for name, (rel, edits) in TARGETS.items():
        path = root / rel
        if not path.is_file():
            raise SystemExit(f'missing source file: {path}')
        patch_file(path, edits)
        print(f'patched {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
