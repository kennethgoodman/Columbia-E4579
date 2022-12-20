import json

cb = open('services/backend/output/cg_cb_recs.json')
recs_cb = json.load(cb)

cf = open('services/backend/output/cg_cf_recs.json')
recs_cf = json.load(cf)

for k, v in recs_cf.items():
    recs_cf[k] = v + recs_cb[k]

with open('services/backend/output/cg_recs.json', 'w') as fp:
    json.dump(recs_cf, fp)