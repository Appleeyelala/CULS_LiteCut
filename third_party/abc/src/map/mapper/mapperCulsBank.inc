/* Store each affected native match and its pin arcs before any global covering.
   Stable boundaries reuse mode0 pin arrivals from the paid original mapping. */
static int Map_CulsWriteAnchors( Map_Man_t * p, const char * path )
{
    int i;
    FILE * f = fopen(path, "w");
    if ( !f ) return 0;
    for ( i = 0; i < p->vMapObjs->nSize; ++i )
    {
        Map_Node_t * n = p->vMapObjs->pArray[i];
        if ( !n->pRepr )
            fprintf(f, "%d %.9g %.9g %.9g %.9g\n", Map_NodeReadAigId(n),
                    n->tArrival[0].Rise, n->tArrival[0].Fall, n->tArrival[1].Rise, n->tArrival[1].Fall);
    }
    return fclose(f) == 0;
}

int Map_CulsPrepareBank( Map_Man_t * p, const char * path, const char * anchors, int nAbcObjects )
{
    FILE * f, * a;
    int i, k, id, nMatch = 0;
    float t[4];
    unsigned char * known, * dirty, * seen;
    if ( !anchors || !(a = fopen(anchors,"r")) ) return 0;
    known = ABC_CALLOC(unsigned char,nAbcObjects);
    while ( fscanf(a,"%d%f%f%f%f",&id,t,t+1,t+2,t+3) == 5 )
    {
        if ( id < 0 || id >= nAbcObjects ) { fclose(a); ABC_FREE(known); return 0; }
        known[id] = 1;
    }
    fclose(a);
    if ( !(f = fopen(path,"w")) ) { ABC_FREE(known); return 0; }
    Map_MappingSetChoiceLevels(p);
    Map_MappingCuts(p);
    Map_MappingTruths(p);
    dirty = ABC_CALLOC(unsigned char,p->nNodes);
    seen = ABC_CALLOC(unsigned char,p->pSuperLib->nSupersAll);
    fprintf(f,"I %.9g %.9g\n",p->pSuperLib->tDelayInv.Rise,p->pSuperLib->tDelayInv.Fall);
    for ( i=0; i < p->vMapObjs->nSize; ++i )
    {
        Map_Node_t * n=p->vMapObjs->pArray[i];
        Map_Cut_t * c;
        id=Map_NodeReadAigId(n);
        if ( Map_NodeIsAnd(n) )
            dirty[n->Num]=!known[id] || n->culsVectors != ~0ULL || n->pNextE ||
                          dirty[Map_Regular(n->p1)->Num] || dirty[Map_Regular(n->p2)->Num];
        if ( n->pRepr ) continue;
        if ( !dirty[n->Num] ) { fprintf(f,"A %d %d\n",n->Num,id); continue; }
        fprintf(f,"N %d\n",n->Num);
        for ( c=n->pCuts->pNext; c; c=c->pNext )
        {
            int phase;
            if ( p->fSkipFanout && ((n->nRefs>3 && c->nLeaves>2) || (n->nRefs>1 && c->nLeaves>3)) ) continue;
            for ( phase=0; phase<2; ++phase )
            {
                int count=0;
                Map_Super_t * s;
                for ( s=c->M[phase].pSupers; s && count<30; s=s->pNext,++count )
                {
                    int j;
                    if ( s->Num < 0 || s->Num >= p->pSuperLib->nSupersAll ) goto invalid;
                    if ( !seen[s->Num] )
                    {
                        seen[s->Num]=1;
                        fprintf(f,"S %d",s->Num);
                        for ( k=0; k<6; ++k )
                            fprintf(f," %.9g %.9g %.9g %.9g",s->tDelaysR[k].Rise,s->tDelaysR[k].Fall,
                                    s->tDelaysF[k].Rise,s->tDelaysF[k].Fall);
                        fprintf(f,"\n");
                    }
                    for ( j=0; j<(int)s->nPhases; ++j )
                    {
                        fprintf(f,"M %d %llx %d %u %.9g",phase,c->culsVectors,s->Num,
                                c->M[phase].uPhase ^ s->uPhases[j],c->culsScale>0 ? c->culsScale : 1.f);
                        for ( k=0; k<c->nLeaves; ++k ) fprintf(f," %d",c->ppLeaves[k]->Num);
                        fprintf(f,"\n");
                        ++nMatch;
                    }
                }
            }
        }
    }
    for ( i=0; i<p->nOutputs; ++i )
    {
        Map_Node_t * n=Map_Regular(p->pOutputs[i]);
        fprintf(f,"P %d %d\n",Map_NodeIsConst(n) ? -1 : n->Num,!Map_IsComplement(p->pOutputs[i]));
    }
    fprintf(f,"E %d\n",nMatch);
    ABC_FREE(known); ABC_FREE(dirty); ABC_FREE(seen);
    return fclose(f)==0;
invalid:
    ABC_FREE(known); ABC_FREE(dirty); ABC_FREE(seen); fclose(f); return 0;
}
