/* Export the exact in-memory, unit-converted Liberty tables used by stime. */
static void CulsPrintSurface(FILE *f, SC_Surface *t)
{
    int i, k;
    fprintf(f, "T %d %d", Vec_FltSize(&t->vIndex0), Vec_FltSize(&t->vIndex1));
    for (i=0; i<Vec_FltSize(&t->vIndex0); i++) fprintf(f," %.9g",Vec_FltEntry(&t->vIndex0,i));
    for (k=0; k<Vec_FltSize(&t->vIndex1); k++) fprintf(f," %.9g",Vec_FltEntry(&t->vIndex1,k));
    for (i=0; i<Vec_PtrSize(&t->vData); i++) {
        Vec_Flt_t *row=(Vec_Flt_t *)Vec_PtrEntry(&t->vData,i);
        for (k=0; k<Vec_FltSize(row); k++) fprintf(f," %.9g",Vec_FltEntry(row,k));
    }
    fputc('\n', f);
}
static void CulsExportLiberty(SC_Man *p)
{
    const char *path=getenv("CULS_LIB_TABLES");
    SC_Cell *c;
    int i,k;
    FILE *f;
    if (!path || !(f=fopen(path,"w"))) return;
    if (p->pWLoadUsed || p->EstLoadMax || p->vInDrive) {
        fprintf(f,"UNSUPPORTED_EXTERNAL_CONSTRAINTS\n"); fclose(f); return;
    }
    SC_LibForEachCell(p->pLib,c,i) {
        SC_Pin *out;
        if (c->seq || c->unsupp || c->n_outputs!=1) continue;
        out=SC_CellPin(c,c->n_inputs);
        if (Vec_PtrSize(&out->vRTimings)!=c->n_inputs) continue;
        for (k=0;k<c->n_inputs;k++) {
            SC_Timings *rt=(SC_Timings *)Vec_PtrEntry(&out->vRTimings,k);
            if (Vec_PtrSize(&rt->vTimings)!=1) break;
        }
        if(k!=c->n_inputs) continue;
        fprintf(f,"C %s %.9g %s\n",c->pName,c->area,out->pName);
        for (k=0;k<c->n_inputs;k++) {
            SC_Pin *pin=SC_CellPin(c,k);
            SC_Timing *t=Scl_CellPinTime(c,k);
            fprintf(f,"P %s %.9g %.9g %d\n",pin->pName,pin->rise_cap,pin->fall_cap,t->tsense);
            CulsPrintSurface(f,&t->pCellRise); CulsPrintSurface(f,&t->pCellFall);
            CulsPrintSurface(f,&t->pRiseTrans); CulsPrintSurface(f,&t->pFallTrans);
        }
    }
    fprintf(f,"END\n"); fclose(f);
}
