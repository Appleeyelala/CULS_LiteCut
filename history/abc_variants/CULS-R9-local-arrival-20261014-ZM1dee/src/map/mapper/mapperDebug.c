/* Experimental observation only. No mapper fields, marks, cuts or refs are changed. */
#include "mapperInt.h"
ABC_NAMESPACE_IMPL_START
static int * dIds = NULL, * dPol = NULL, dSize = 0;
static Map_Node_t * dCutNode = NULL;
static int DEnabled(void) { return getenv("AIGREWRITE_MAP_DEBUG_NODE") != NULL; }
void Map_DebugRememberAiger(int var, int id, int pol, int size)
{
    if (!DEnabled()) return;
    if (var == 0) { ABC_FREE(dIds); ABC_FREE(dPol); dSize=size;
        dIds=ABC_ALLOC(int,size); dPol=ABC_ALLOC(int,size); }
    dIds[var]=id; dPol[var]=pol;
}
static int DVar(Map_Node_t * n)
{
    int i, id=Map_NodeReadAigId(n);
    for(i=0;i<dSize;i++) if(dIds[i]==id) return i;
    return -1;
}
static int DTarget(Map_Node_t * n)
{
    int v;
    if(!DEnabled() || !dIds) return 0;
    v=atoi(getenv("AIGREWRITE_MAP_DEBUG_NODE"));
    return v>=0 && v<dSize && Map_NodeReadAigId(n)==dIds[v];
}
static Map_Node_t * DFind(Map_Man_t * p)
{
    int i; for(i=0;i<p->vMapObjs->nSize;i++)
        if(DTarget(p->vMapObjs->pArray[i])) return p->vMapObjs->pArray[i];
    return NULL;
}
static void DStr(const char * s)
{
    const unsigned char * a=(const unsigned char *)(s?s:""); putchar('"');
    for(;*a;a++) { if(*a=='"'||*a=='\\') putchar('\\');
        if(*a>=32) putchar(*a); }
    putchar('"');
}
static void DLeaves(Map_Cut_t * c)
{
    int i; printf("\"leaves\":[");
    for(i=0;i<c->nLeaves;i++) printf("%s%d",i?",":"",DVar(c->ppLeaves[i]));
    printf("],\"leaf_mapper_ids\":[");
    for(i=0;i<c->nLeaves;i++) printf("%s%d",i?",":"",c->ppLeaves[i]->Num);
    printf("]");
}
static int DCutIdx(Map_Node_t * n, Map_Cut_t * c)
{
    int i=0; Map_Cut_t * a; for(a=n->pCuts;a;a=a->pNext,i++) if(a==c)return i;
    return -1;
}
static void DTime(Map_Time_t * t)
{ printf("{\"rise\":%.9g,\"fall\":%.9g,\"worst\":%.9g}",t->Rise,t->Fall,t->Worst); }
static void DTree(Map_Super_t * s)
{
    int i;if(!s->pRoot){printf("{\"leaf_pin\":%d}",s->Num);return;}
    printf("{\"cell\":");DStr(Mio_GateReadName(s->pRoot));printf(",\"fanins\":[");
    for(i=0;i<(int)s->nFanins;i++){if(i)printf(",");DTree(s->pFanins[i]);}printf("]}");
}
static void DSuper(Map_Super_t * s)
{
    int i; if(!s){printf("null");return;}
    printf("{\"id\":%d,\"root_cell\":",s->Num); DStr(s->pRoot?Mio_GateReadName(s->pRoot):"variable");
    printf(",\"formula\":"); DStr(s->pFormula);
    printf(",\"area\":%.9g,\"n_gates\":%u,\"n_fanins\":%u,\"fanout_limit\":%u,\"arcs\":[",s->Area,s->nGates,s->nFanins,s->nFanLimit);
    for(i=0;i<(int)s->nFanins;i++) printf("%s{\"pin\":%d,\"rr\":%.9g,\"fr\":%.9g,\"rf\":%.9g,\"ff\":%.9g}",
        i?",":"",i,s->tDelaysR[i].Rise,s->tDelaysR[i].Fall,s->tDelaysF[i].Rise,s->tDelaysF[i].Fall);
    printf("],\"implementation\":");DTree(s);printf("}");
}
void Map_DebugGraph(Map_Man_t * p)
{
    int i,k,v, wanted; const char * list=getenv("AIGREWRITE_MAP_DEBUG_IDS");
    Map_Node_t * n; if(!DEnabled())return;
    printf("MAPDBG {\"type\":\"config\",\"n_vars_max\":%d,\"target_aiger_var\":%d}\n",p->nVarsMax,atoi(getenv("AIGREWRITE_MAP_DEBUG_NODE")));
    for(i=0;i<p->vMapObjs->nSize;i++) {
        const char * a; n=p->vMapObjs->pArray[i];v=DVar(n);wanted=DTarget(n);
        for(a=list;a && *a;){ if(atoi(a)==v)wanted=1; a=strchr(a,',');if(a)a++; }
        if(!wanted)continue;
        printf("MAPDBG {\"type\":\"node\",\"aiger_var\":%d,\"abc_obj_id\":%d,\"mapper_id\":%d,\"node_type\":\"%s\",\"fanin_mapper_ids\":[",
               v,Map_NodeReadAigId(n),n->Num,Map_NodeIsAnd(n)?"AND":"PI");
        if(Map_NodeIsAnd(n))printf("%d,%d",Map_Regular(n->p1)->Num,Map_Regular(n->p2)->Num);
        printf("],\"fanin_complemented\":[");
        if(Map_NodeIsAnd(n))printf("%d,%d",Map_IsComplement(n->p1),Map_IsComplement(n->p2));
        printf("],\"aiger_to_abc_complemented\":%d}\n",v>=0?dPol[v]:-1);
    }
}
void Map_DebugCutBegin(Map_Node_t * n) { dCutNode=DTarget(n)?n:NULL; }
void Map_DebugCutEnd(void) { dCutNode=NULL; }
void Map_DebugCutArray(Map_Cut_t ** cs, int count)
{
    int i; if(!dCutNode)return;
    for(i=0;i<count;i++) {
        printf("MAPDBG {\"type\":\"cut_stage\",\"stage\":\"sorted_before_limit\",\"mapper_id\":%d,\"rank\":%d,\"limit_retained\":%s,",dCutNode->Num,i,i<249?"true":"false");
        DLeaves(cs[i]);printf("}\n");
    }
}
void Map_DebugDominated(Map_Node_t * n, Map_Cut_t * c)
{
    if(!DTarget(n))return;
    printf("MAPDBG {\"type\":\"cut_stage\",\"stage\":\"dominated\",\"mapper_id\":%d,",n->Num);
    DLeaves(c);printf("}\n");
}
void Map_DebugTruth(Map_Man_t * p, Map_Cut_t * c, unsigned * t)
{
    Map_Node_t * n; if(!DEnabled() || !(n=DFind(p)) || DCutIdx(n,c)<0)return;
    printf("MAPDBG {\"type\":\"truth\",\"cut_index\":%d,\"truth_lo\":%u,\"truth_hi\":%u,",DCutIdx(n,c),t[0],t[1]);
    DLeaves(c);printf("}\n");
}
void Map_DebugCuts(Map_Man_t * p)
{
    Map_Node_t * n;Map_Cut_t * c;Map_Super_t * s;int i,f,j,rank;
    if(!DEnabled() || !(n=DFind(p)))return;
    for(c=n->pCuts,i=0;c;c=c->pNext,i++){
        printf("MAPDBG {\"type\":\"cut\",\"cut_index\":%d,\"n_leaves\":%d,\"volume\":%d,\"cut_phase\":%d,",i,c->nLeaves,c->nVolume,c->Phase);
        DLeaves(c);printf("}\n");
        if(c->nLeaves==1)continue;
        for(f=0;f<2;f++)for(s=c->M[f].pSupers,rank=0;s;s=s->pNext,rank++)for(j=0;j<(int)s->nPhases;j++){
            printf("MAPDBG {\"type\":\"match_inventory\",\"cut_index\":%d,\"phase\":%d,\"library_rank\":%d,\"phase_index\":%d,\"input_phase_mask\":%u,\"within_match_limit\":%s,\"super\":",
                    i,f,rank,j,c->M[f].uPhase ^ s->uPhases[j],rank<30?"true":"false");
            DSuper(s);printf("}\n");
        }
    }
}
void Map_DebugMatch(Map_Node_t * n, Map_Cut_t * c, int f, float limit, const char * status)
{
    int i;Map_Match_t * m=c->M+f;if(!DTarget(n))return;
    printf("MAPDBG {\"type\":\"match_attempt\",\"mode\":%d,\"cut_index\":%d,\"phase\":%d,\"input_phase_mask\":%u,\"limit\":%.9g,\"status\":",
            n->p->fMappingMode,DCutIdx(n,c),f,m->uPhaseBest,limit); DStr(status);
    printf(",\"arrival\":");if(!strcmp(status,"area_limit"))printf("null");else DTime(&m->tArrive);
    printf(",\"area_cost\":%.9g,\"super\":",m->AreaFlow);DSuper(m->pSuperBest);
    printf(",\"leaf_arrivals\":[");for(i=0;i<c->nLeaves;i++){
        printf("%s",i?",":"");DTime(c->ppLeaves[i]->tArrival+((m->uPhaseBest&(1<<i))==0)); }
    printf("]}\n");
}
void Map_DebugSnapshot(Map_Man_t * p, const char * stage)
{
    int f;Map_Node_t * n;Map_Cut_t * c;Map_Match_t * m;
    if(!DEnabled() || !(n=DFind(p)))return;
    for(f=0;f<2;f++){
        c=n->pCutBest[f];m=c?c->M+f:NULL;
        printf("MAPDBG {\"type\":\"best\",\"stage\":");DStr(stage);
        printf(",\"mode\":%d,\"mapper_id\":%d,\"phase\":%d,\"refs\":[%d,%d,%d],\"node_arrival\":",p->fMappingMode,n->Num,f,n->nRefAct[0],n->nRefAct[1],n->nRefAct[2]);DTime(n->tArrival+f);
        printf(",\"cut_index\":%d,\"match_arrival\":",c?DCutIdx(n,c):-1);if(m)DTime(&m->tArrive);else printf("null");
        printf(",\"area_cost\":");if(m)printf("%.9g",m->AreaFlow);else printf("null");
        printf(",\"input_phase_mask\":%u,\"super\":",m?m->uPhaseBest:0);DSuper(m?m->pSuperBest:NULL);
        if(c){printf(",");DLeaves(c);}printf("}\n");
    }
}
static int DInside(Map_Node_t * n, Map_Cut_t * c, Map_Node_t * target, unsigned char * seen)
{
    int i,found;for(i=0;i<c->nLeaves;i++)if(c->ppLeaves[i]==n)return 0;
    if(n==target)return 1;if(!Map_NodeIsAnd(n)||seen[n->Num])return 0;seen[n->Num]=1;
    found=DInside(Map_Regular(n->p1),c,target,seen);
    return DInside(Map_Regular(n->p2),c,target,seen)||found;
}
void Map_DebugFinal(Map_Man_t * p)
{
    int i,f,inside;Map_Node_t * target,*n;Map_Cut_t * c;unsigned char * seen;
    if(!DEnabled() || !(target=DFind(p)))return;
    Map_DebugSnapshot(p,"final");seen=ABC_CALLOC(unsigned char,p->nNodes+2);
    for(i=0;i<p->vMapObjs->nSize;i++){
        n=p->vMapObjs->pArray[i];if(!Map_NodeIsAnd(n)||!n->nRefAct[2])continue;
        for(f=0;f<2;f++){
            c=n->pCutBest[f];if(!c || (!n->nRefAct[f] && n->pCutBest[!f]))continue;
            memset(seen,0,p->nNodes+2);inside=DInside(n,c,target,seen);
            if(!inside && n!=target)continue;
            printf("MAPDBG {\"type\":\"cover\",\"output_aiger_var\":%d,\"output_mapper_id\":%d,\"phase\":%d,\"refs\":[%d,%d,%d],\"contains_target_internal\":%s,",DVar(n),n->Num,f,n->nRefAct[0],n->nRefAct[1],n->nRefAct[2],inside&&n!=target?"true":"false");
            DLeaves(c);printf(",\"arrival\":");DTime(n->tArrival+f);printf(",\"super\":");DSuper(c->M[f].pSuperBest);printf("}\n");
        }
    }ABC_FREE(seen);
}
/* R9 experiment orchestration. Native mapping rules are not modified.
   A full subject graph is constructed for identity/order/structural refs,
   but cuts/truths/matches are computed ONLY for the root's PI-closed TFI.
   No candidate Map_Mapping(), recovery, covering, or netlist is called. */
#include "base/abc/abc.h"
#include "base/main/main.h"
extern Abc_Ntk_t * Io_ReadAiger( char *, int );
extern Map_Man_t * Abc_NtkToMap( Abc_Ntk_t *, double, int, float *, int );
static Map_Time_t r9Pi[64];
static int r9Refs[64], r9NumPi=0, r9Skip=0, r9Profile=0;
static float r9Baseline=0;
static double Abc_Clock2Sec( abctime ticks ) { return (double)ticks / (double)CLOCKS_PER_SEC; }
static void R9MarkTFI( Map_Node_t * n, unsigned char * marks )
{
    n=Map_Regular(n);
    if(n->Num<0 || marks[n->Num]) return;
    marks[n->Num]=1;
    if(Map_NodeIsAnd(n)) { R9MarkTFI(n->p1,marks); R9MarkTFI(n->p2,marks); }
}
void Map_R9CaptureFrozen( Map_Man_t * p )
{
    int i,f,frontier[5]={17,33,151,188,209}; Map_Node_t *n; Map_Cut_t *c;
    if(!getenv("AIGREWRITE_R9_LOCAL"))return;
    assert(p->fMappingMode==0 && p->nInputs<=64 && !p->pNodeDelays && !p->nChoices);
    n=DFind(p); assert(n && DVar(n)==213);
    r9Baseline=n->tArrival[1].Worst;
    r9NumPi=p->nInputs;r9Skip=p->fSkipFanout;r9Profile=p->fUseProfile;
    printf("R9CTX {\"type\":\"meta\",\"mode\":0,\"root\":213,\"root_phase\":1,\"root_arrival\":%.9g,\"epsilon\":%.9g,\"delay_target\":%.9g,\"skip_fanout\":%d,\"use_profile\":%d,\"n_vars_max\":%d,\"cuts_max_use\":250,\"matches_limit\":30,\"baseline_cut_cpu_s\":%.9g,\"baseline_truth_cpu_s\":%.9g,\"baseline_match_cpu_s\":%.9g,\"inverter_delay\":",r9Baseline,p->fEpsilon,p->DelayTarget,p->fSkipFanout,p->fUseProfile,p->nVarsMax,Abc_Clock2Sec(p->timeCuts),Abc_Clock2Sec(p->timeTruth),Abc_Clock2Sec(p->timeMatch));
    DTime(&p->pSuperLib->tDelayInv);printf("}\n");
    for(i=0;i<p->nInputs;i++){
        n=p->pInputs[i];r9Pi[i]=n->tArrival[1];r9Refs[i]=n->nRefs;
        printf("R9CTX {\"type\":\"pi\",\"function_point\":%d,\"mapper_id\":%d,\"n_refs\":%d,\"positive\":",DVar(n),n->Num,n->nRefs);DTime(n->tArrival+1);printf(",\"negative\":");DTime(n->tArrival);printf("}\n");
    }
    for(i=0;i<p->vMapObjs->nSize;i++){
        int wanted=0,k;n=p->vMapObjs->pArray[i];for(k=0;k<5;k++)if(DVar(n)==frontier[k])wanted=1;if(!wanted)continue;
        printf("R9CTX {\"type\":\"logical_frontier\",\"function_point\":%d,\"mapper_id\":%d,\"n_refs\":%d,\"n_ref_est\":[%.9g,%.9g,%.9g],\"positive\":",DVar(n),n->Num,n->nRefs,n->nRefEst[0],n->nRefEst[1],n->nRefEst[2]);DTime(n->tArrival+1);printf(",\"negative\":");DTime(n->tArrival);
        printf(",\"cuts\":[");k=0;for(c=n->pCuts;c;c=c->pNext){if(k++)printf(",");printf("{");DLeaves(c);printf("}");}printf("],\"best_phases\":[");
        for(f=0;f<2;f++){if(f)printf(",");c=n->pCutBest[f];printf("{\"phase\":%d,\"available\":%s",f,c?"true":"false");if(c){printf(",\"area_flow\":%.9g,\"input_phase_mask\":%u,\"super\":",c->M[f].AreaFlow,c->M[f].uPhaseBest);DSuper(c->M[f].pSuperBest);}printf("}");}printf("]}\n");
    }
}
static int R9LocalCalculate( char * path, int rootVar, const char *variant, float *result )
{
    Abc_Ntk_t *ntk;Map_Man_t *q;Map_NodeVec_t *all,*active;Map_Node_t *root=NULL,*n;Map_Cut_t *c;
    unsigned char *marks;int i,f,nAnd=0,nCuts=0,nUsedPi=0,phase,fullNodes;
    abctime start=Abc_Clock(), matchStart;
    ntk=Io_ReadAiger(path,1);if(!ntk)return 0;
    q=Abc_NtkToMap(ntk,-1,1,NULL,0);if(!q){Abc_NtkDelete(ntk);return 0;}
    if(q->nInputs!=r9NumPi || q->nChoices){Map_ManFree(q);Abc_NtkDelete(ntk);return 0;}
    q->fSkipFanout=r9Skip;q->fUseProfile=r9Profile;
    all=q->vMapObjs;fullNodes=all->nSize;
    for(i=0;i<all->nSize;i++)if(Map_NodeReadAigId(all->pArray[i])==dIds[rootVar])root=all->pArray[i];
    if(!root){Map_ManFree(q);Abc_NtkDelete(ntk);return 0;}
    phase=1-dPol[rootVar];marks=ABC_CALLOC(unsigned char,q->nNodes+2);R9MarkTFI(root,marks);
    active=Map_NodeVecAlloc(all->nSize);
    for(i=0;i<all->nSize;i++){
        n=all->pArray[i];if(Map_NodeIsVar(n) || marks[n->Num])Map_NodeVecPush(active,n);
        if(marks[n->Num]){if(Map_NodeIsAnd(n))nAnd++;else if(Map_NodeIsVar(n))nUsedPi++;}
    }
    q->vMapObjs=active;
    q->pInputArrivals=ABC_ALLOC(Map_Time_t,q->nInputs);
    for(i=0;i<q->nInputs;i++){q->pInputArrivals[i]=r9Pi[i];q->pInputs[i]->nRefs=r9Refs[i];}
    Map_MappingSetChoiceLevels(q);
    Map_MappingCuts(q);Map_MappingTruths(q);
    for(i=0;i<active->nSize;i++){n=active->pArray[i];if(Map_NodeIsAnd(n))for(c=n->pCuts;c;c=c->pNext)if(c->nLeaves>1)nCuts++;}
    matchStart=Abc_Clock();q->fMappingMode=0;
    if(!Map_MappingMatches(q)){q->vMapObjs=all;Map_NodeVecFree(active);ABC_FREE(marks);Map_ManFree(q);Abc_NtkDelete(ntk);return 0;}
    *result=root->tArrival[phase].Worst;
    printf("R9LOCAL {\"variant\":\"%s\",\"function_point\":%d,\"mapper_id\":%d,\"phase\":%d,\"rise\":%.9g,\"fall\":%.9g,\"arrival\":%.9g,\"local_ands\":%d,\"frontier_pis\":%d,\"subject_graph_nodes_constructed\":%d,\"evaluated_cuts\":%d,\"evaluated_matches\":%d,\"evaluated_phases\":%d,\"elapsed_cpu_s\":%.9g,\"match_cpu_s\":%.9g,\"candidate_full_mapping_called\":false,\"downstream_matched\":false,\"best\":[",variant,rootVar,root->Num,phase,root->tArrival[phase].Rise,root->tArrival[phase].Fall,*result,nAnd,nUsedPi,fullNodes,nCuts,q->nMatches,q->nPhases,Abc_Clock2Sec(Abc_Clock()-start),Abc_Clock2Sec(Abc_Clock()-matchStart));
    for(f=0;f<2;f++){if(f)printf(",");c=root->pCutBest[f];printf("{\"phase\":%d",f);if(c){printf(",");DLeaves(c);printf(",\"input_phase_mask\":%u,\"super\":",c->M[f].uPhaseBest);DSuper(c->M[f].pSuperBest);}printf("}");}printf("]}\n");
    q->vMapObjs=all;Map_NodeVecFree(active);ABC_FREE(marks);Map_ManFree(q);Abc_NtkDelete(ntk);return 1;
}
int Abc_CommandLocalR9Arrival( Abc_Frame_t *frame, int argc, char **argv )
{
    float base,cand;(void)frame;
    if(argc!=3 || !r9NumPi){printf("R9STOP missing baseline mode0 context or arguments\n");return 1;}
    if(!R9LocalCalculate(argv[1],213,"baseline",&base)){printf("R9STOP baseline local calculation failed\n");return 1;}
    if(fabs(base-r9Baseline)>0.00002f){printf("R9STOP baseline mismatch predicted=%.9g actual=%.9g\n",base,r9Baseline);return 1;}
    printf("R9CHECK baseline_local=PASS\n");
    if(!R9LocalCalculate(argv[2],322,"candidate",&cand)){printf("R9STOP candidate local calculation failed\n");return 1;}
    return 0;
}
ABC_NAMESPACE_IMPL_END
