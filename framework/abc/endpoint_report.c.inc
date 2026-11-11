/* Read-only instrumentation of ABC's existing Liberty timing manager.
 * Included immediately before Abc_SclTimePerformInt in sclSize.c.
 * No delay/area calculations or mapping decisions are changed.
 * GT_TRACE_NATIVE is ABC's min-slack global traceback, not an AIG ID mapping.
 */
static void CulsGlobalTimingReport( SC_Man * p )
{
    Abc_Obj_t * pPo;
    int i;
    const char * enabled = getenv( "CULS_GLOBAL_TIMING_REPORT" );
    if ( enabled == NULL || strcmp( enabled, "1" ) != 0 )
        return;
    printf( "GT_META %.9g %.9g %d %d\n", Abc_SclGetTotalArea(p->pNtk),
            (double)p->ReportDelay, Abc_NtkPoNum(p->pNtk), Abc_NtkNodeNum(p->pNtk) );
    Abc_NtkForEachPo( p->pNtk, pPo, i )
        printf( "GT_PO %d %.9g %.9g\n", i,
                Abc_SclObjTimeOne(p, pPo, 1), Abc_SclObjTimeOne(p, pPo, 0) );
}
