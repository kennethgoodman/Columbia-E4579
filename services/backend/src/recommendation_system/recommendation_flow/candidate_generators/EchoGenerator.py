import json
from .AbstractGenerator import AbstractGenerator


class EchoGenerator(AbstractGenerator):

    def get_content_ids(self, user_id, limit=None, offset=None, seed=None, starting_point=None):

        cb = open('src/echo_space/output/cg_cb_recs.json')
        recs_cb = json.load(cb)[str(user_id)]

        cf = open('src/echo_space/output/cg_cf_recs.json')
        recs_cf = json.load(cf)[str(user_id)]

        if not limit:
            return recs_cf + recs_cb

        ratio = len(recs_cf) / (len(recs_cf) + len(recs_cb))

        cf_limit = int(ratio * limit)

        return recs_cf[:cf_limit] + recs_cb[:(limit - cf_limit)]
