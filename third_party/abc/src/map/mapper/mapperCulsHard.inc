/* Offline parity probe and hard forward on one prepared cut/match bank.
   The inputs, cut truth tables and supergate lists are shared across vectors.
   Each sweep starts with independent timing and matches; no covering pass runs. */
static void Map_CulsResetMatches( Map_Man_t * p )
{
    int i, f;
    for ( i = 0; i < p->vMapObjs->nSize; ++i )
    {
        Map_Node_t * node = p->vMapObjs->pArray[i];
        Map_Cut_t * cut;
        if ( !Map_NodeIsAnd(node) || node->pRepr ) continue;
        node->pCutBest[0] = node->pCutBest[1] = NULL;
        for ( f = 0; f < 2; ++f )
        {
            node->tArrival[f].Rise = node->tArrival[f].Fall = node->tArrival[f].Worst = MAP_FLOAT_LARGE;
            for ( cut = node->pCuts->pNext; cut; cut = cut->pNext )
            {
                Map_Match_t * match = &cut->M[f];
                match->pSuperBest = NULL;
                match->uPhaseBest = 0;
                match->tArrive = node->tArrival[f];
                match->AreaFlow = MAP_FLOAT_LARGE;
            }
        }
    }
}

int Map_CulsHardScores( Map_Man_t * p, const char * argument )
{
    int k, i, count;
    char * end;
    long parsed = strtol(argument, &end, 10);
    if ( *end || parsed < 1 || parsed > 64 ) return 0;
    count = (int)parsed;
    Map_MappingSetChoiceLevels(p);
    Map_MappingCuts(p);
    Map_MappingTruths(p);
    for ( k = 0; k < count; ++k )
    {
        int absent = 0, missingPO = 0;
        p->culsCondition = 1ULL << k;
        Map_CulsResetMatches(p);
        if ( !Map_MappingMatches(p) ) return 0;
        for ( i = 0; i < p->vMapObjs->nSize; ++i )
        {
            Map_Node_t * n = p->vMapObjs->pArray[i];
            if ( Map_NodeIsAnd(n) && !n->pRepr && !n->pCutBest[0] && !n->pCutBest[1] ) ++absent;
        }
        for ( i = 0; i < p->nOutputs; ++i )
        {
            Map_Node_t * out = Map_Regular(p->pOutputs[i]);
            if ( Map_NodeIsAnd(out) && !out->pCutBest[0] && !out->pCutBest[1] ) ++missingPO;
        }
        printf("CULS_HARD %d %.9g %d %d\n", k, missingPO ? -1.f : Map_TimeComputeArrivalMax(p), absent, missingPO);
    }
    return 1;
}
