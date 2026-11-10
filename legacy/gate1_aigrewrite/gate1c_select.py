import json,itertools
p='gate1c_out/c1355_probe.json'; d=json.load(open(p))['records']
# deterministic compatible pair: disjoint saved nodes, no root in other's saved, no leaves in other's deleted
for a,b in itertools.combinations(d,2):
    ra=set(a['saved_region_nodes']); rb=set(b['saved_region_nodes'])
    da=set(a['cut_leaves']); db=set(b['cut_leaves'])
    if ra.isdisjoint(rb) and a['root'] not in rb and b['root'] not in ra and da.isdisjoint(rb) and db.isdisjoint(ra):
        print(json.dumps({'a':a,'b':b})); break
