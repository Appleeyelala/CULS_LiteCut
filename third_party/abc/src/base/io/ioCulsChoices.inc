#include <math.h>
/* Research-only input for one combinational read_aiger/map process.
   Literal indices refer to the reader's actual vNodes, including strashing.
   The caller supplies locally truth-checked candidates and frozen vector masks. */
extern Abc_Obj_t * Abc_AigAndCreate( Abc_Aig_t *, Abc_Obj_t *, Abc_Obj_t * );

typedef struct {
    Abc_Obj_t * repr, * alt;
    unsigned long long original, allowed;
    int decision, choice, invert;
    float scale;
} CulsAlternative;

static Abc_Obj_t * CulsReadLit( Vec_Ptr_t * nodes, int lit )
{
    if ( lit < 0 || lit / 2 >= Vec_PtrSize(nodes) ) return NULL;
    return Abc_ObjNotCond( (Abc_Obj_t *)Vec_PtrEntry(nodes, lit / 2), lit & 1 );
}

static int CulsReadChoices( Abc_Ntk_t * ntk, Vec_Ptr_t * nodes, const char * path )
{
    FILE * file = fopen(path, "r");
    char line[256];
    CulsAlternative a[256];
    int count = 0, left, right, i, j;
    if ( !file ) return 0;
    while ( fgets(line, sizeof(line), file) )
    {
        if ( line[0] == 'g' && sscanf(line, "g %d %d", &left, &right) == 2 )
        {
            Abc_Obj_t * x = CulsReadLit(nodes, left), * y = CulsReadLit(nodes, right);
            if ( !x || !y ) goto invalid;
            Vec_PtrPush(nodes, Abc_AigAnd((Abc_Aig_t *)ntk->pManFunc, x, y));
        }
        else if ( line[0] == 'e' )
        {
            Abc_Obj_t * x, * y;
            if ( sscanf(line, "e %d %d", &left, &right) != 2 ) goto invalid;
            x = CulsReadLit(nodes, left); y = CulsReadLit(nodes, right);
            if ( !x || x != y ) goto invalid;
        }
        else if ( line[0] == 't' )
        {
            float factor;
            Abc_Obj_t * x;
            if ( sscanf(line, "t %d %f", &left, &factor) != 2 || !isfinite(factor) || factor <= 0 )
                goto invalid;
            x = CulsReadLit(nodes, left);
            if ( !x || !Abc_ObjIsNode(Abc_ObjRegular(x)) ) goto invalid;
            x = Abc_ObjRegular(x);
            x->culsScale = x->culsTagged ? 1.f : factor;
            x->culsTagged = 1;
            x->culsVectors = ~0ULL;
        }
        else if ( line[0] == 'r' && count < 256 )
        {
            CulsAlternative * e = a + count++;
            Abc_Obj_t * x, * y;
            if ( sscanf(line, "r %d %d %d %d %llx %llx %f", &e->decision, &e->choice,
                  &left, &right, &e->original, &e->allowed, &e->scale) != 7 ) goto invalid;
            x = CulsReadLit(nodes, left); y = CulsReadLit(nodes, right);
            if ( !x || !y || e->scale <= 0 || !isfinite(e->scale) ) goto invalid;
            e->repr = Abc_ObjRegular(x); e->alt = Abc_ObjRegular(y);
            e->invert = Abc_ObjIsComplement(x) ^ Abc_ObjIsComplement(y);
            if ( e->invert != (e->repr->fPhase ^ e->alt->fPhase) ) goto invalid;
        }
        else goto invalid;
    }
    if ( ferror(file) ) goto invalid;
    fclose(file);
    for ( i = 0; i < count; ++i )
    {
        CulsAlternative * e = a + i;
        if ( !Abc_ObjIsNode(e->repr) ) return 0;
        e->repr->culsTagged = 1;
        e->repr->culsVectors |= e->original;
    }
    for ( i = 0; i < count; ++i )
    {
        CulsAlternative * e = a + i;
        const char * status;
        if ( e->alt == e->repr ) {
            e->repr->culsVectors |= e->allowed;
            status = "alias";
        }
        else if ( !Abc_ObjIsNode(e->alt) ) status = "terminal";
        else {
            for ( j = 0; j < i && a[j].alt != e->alt; ++j ) {}
            if ( j < i && a[j].repr != e->repr ) status = "shared_class";
            else {
                if ( j == i ) {
                    if ( Abc_ObjFanoutNum(e->alt) ) {
                        Abc_Obj_t * original = e->alt;
                        e->alt = Abc_AigAndCreate((Abc_Aig_t *)ntk->pManFunc,
                                                  Abc_ObjChild0(original), Abc_ObjChild1(original));
                        e->alt->culsPrivate = 1;
                        for ( j = i+1; j < count; ++j )
                            if ( a[j].repr == e->repr && a[j].alt == original ) a[j].alt = e->alt;
                    }
                    e->alt->pData = e->repr->pData;
                    e->repr->pData = e->alt;
                    e->alt->culsScale = e->scale;
                } else e->alt->culsScale = 1.0f;
                e->alt->culsTagged = 1;
                e->alt->culsVectors |= e->allowed;
                status = "linked";
            }
        }
        printf("CULS_OPTION %d %d %d %d %d %s\n", e->decision, e->choice, e->repr->Id, e->alt->Id, e->invert, status);
    }
    if ( count == 0 ) Abc_AigCleanup((Abc_Aig_t *)ntk->pManFunc);
    return 1;
invalid:
    fclose(file);
    return 0;
}
