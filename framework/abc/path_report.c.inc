/* Optional read-only arrival traceback in the existing SC_Man.
 * Uses the same Liberty interpolation and float arithmetic as sclLib.h.
 * IDs are local to this mapped network. This does not align logical AIG IDs.
 */
static void CulsTraceArrivalPath( SC_Man * p, Abc_Obj_t * pPo, int iPo )
{
    Abc_Obj_t * pNode = Abc_ObjFanin0(pPo), * pFanin, * pBest, * pPi;
    int Rise = Abc_SclObjTimeOne(p, pPo, 1) >= Abc_SclObjTimeOne(p, pPo, 0);
    int Sequence = 0, k, InRise, BestRise, BestPin, iPi;
    float Arrival = (float)Abc_SclObjTimeOne(p, pPo, Rise);
    printf( "GT_PATH_BEGIN %d %d %.9g\n", iPo, Rise, Arrival );
    while ( pNode && Sequence <= Abc_NtkObjNum(p->pNtk) )
    {
        SC_Cell * pCell = Abc_ObjIsNode(pNode) ? Abc_SclObjCell(pNode) : NULL;
        SC_Pair Load = *Abc_SclObjLoad(p, pNode);
        float Best = -ABC_INFINITY, BestArc = 0, BestSlew = 0, BestLoad = 0, Extra = 0;
        float Actual = (float)Abc_SclObjTimeOne(p, pNode, Rise), Predicted;
        printf( "GT_PATH_NODE %d %d %d %d %.9g %s %s\n", iPo, Sequence,
                Abc_ObjId(pNode), Rise, Actual, Abc_ObjIsCi(pNode) ? "PI" :
                Abc_ObjFaninNum(pNode) == 0 ? "CONST" : "CELL", pCell ? pCell->pName : "-" );
        if ( Abc_ObjIsCi(pNode) || Abc_ObjFaninNum(pNode) == 0 )
        {
            iPi = -1;
            if ( Abc_ObjIsCi(pNode) )
                Abc_NtkForEachPi( p->pNtk, pPi, iPi )
                    if ( pPi == pNode ) break;
            printf( "GT_PATH_END %d %d %d\n", iPo, Sequence + 1, iPi );
            return;
        }
        if ( pCell == NULL || p->pFuncFanin != NULL )
        {
            printf( "GT_PATH_ERROR %d UNSUPPORTED_CELL_OR_FANIN_CALLBACK\n", iPo );
            return;
        }
        if ( p->EstLoadMax )
        {
            float Value = (0.5 * Load.fall + 0.5 * Load.rise) / (p->EstLoadAve * p->EstLoadMax);
            if ( Value > 1 )
            {
                Load.rise = Load.fall = p->EstLoadAve * p->EstLoadMax;
                Extra = p->EstLinear * log( Value );
            }
        }
        pBest = NULL; BestRise = BestPin = -1;
        Abc_ObjForEachFanin( pNode, pFanin, k )
        {
            SC_Timing * pTiming = Scl_CellPinTime( pCell, k );
            SC_Pair * pSlew = Abc_SclObjSlew(p, pFanin);
            for ( InRise = 0; InRise <= 1; InRise++ )
            {
                float Slew, OutputLoad, Arc, Value;
                if ( pTiming->tsense != sc_ts_Non &&
                     !((pTiming->tsense == sc_ts_Pos && Rise == InRise) ||
                       (pTiming->tsense == sc_ts_Neg && Rise != InRise)) )
                    continue;
                Slew = InRise ? pSlew->rise : pSlew->fall;
                OutputLoad = Rise ? Load.rise : Load.fall;
                Arc = Scl_LibLookup(Rise ? &pTiming->pCellRise : &pTiming->pCellFall, Slew, OutputLoad);
                Value = (float)Abc_SclObjTimeOne(p, pFanin, InRise) + Arc;
                if ( Value > Best )
                {
                    Best = Value; pBest = pFanin; BestRise = InRise; BestPin = k;
                    BestArc = Arc; BestSlew = Slew; BestLoad = OutputLoad;
                }
            }
        }
        Predicted = Best + Extra;
        if ( pBest == NULL || fabs(Predicted - Actual) > 1e-5 + 1e-6 * fabs(Actual) )
        {
            printf( "GT_PATH_ERROR %d ARRIVAL_RECONSTRUCTION %.9g %.9g\n", iPo, Predicted, Actual );
            return;
        }
        printf( "GT_PATH_ARC %d %d %d %d %d %d %d %.9g %.9g %.9g %.9g %.9g %.9g\n",
                iPo, Sequence, Abc_ObjId(pNode), Abc_ObjId(pBest), BestPin, BestRise, Rise,
                Abc_SclObjTimeOne(p, pBest, BestRise), BestArc, Extra, Actual, BestSlew, BestLoad );
        pNode = pBest; Rise = BestRise; Sequence++;
    }
    printf( "GT_PATH_ERROR %d INVALID_OR_CYCLIC_CHAIN\n", iPo );
}

static void CulsGlobalPathReport( SC_Man * p )
{
    Abc_Obj_t * pPo;
    int i;
    const char * enabled = getenv("CULS_GLOBAL_TIMING_PATH_REPORT");
    if ( enabled == NULL || strcmp(enabled, "1") != 0 ) return;
    printf( "GT_PATH_META 0.95 %d\n", Abc_NtkPoNum(p->pNtk) );
    Abc_NtkForEachPo( p->pNtk, pPo, i )
        if ( Abc_SclObjTimeMax(p, pPo) >= 0.95 * p->ReportDelay )
            CulsTraceArrivalPath(p, pPo, i);
}
