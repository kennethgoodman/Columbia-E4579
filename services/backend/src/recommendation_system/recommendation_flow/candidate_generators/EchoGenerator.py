import json
from .AbstractGenerator import AbstractGenerator

cb = open('src/echo_space/output/cg_cb_recs.json')
recs_loaded = json.load(cb)
cf = open('src/echo_space/output/cg_cf_recs.json')
recs_loaded = json.load(cf)

class EchoGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit=None, offset=None, seed=None, starting_point=None):
        recs_cb, recs_cf = recs_loaded[str(user_id)], recs_loaded[str(user_id)]
        if not limit:
            return recs_cf + recs_cb
        ratio = len(recs_cf) / (len(recs_cf) + len(recs_cb))
        cf_limit = int(ratio * limit)
        return recs_cf[:cf_limit] + recs_cb[:(limit - cf_limit)]
