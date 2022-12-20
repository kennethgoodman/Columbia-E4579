import json


class EchoGenerator:

    def get_content_ids(self, user_id, limit, offset, _seed, starting_point):

        f = open('services/backend/output/cg_recs.json')
        recs = json.load(f)[str(user_id)]

        return recs
