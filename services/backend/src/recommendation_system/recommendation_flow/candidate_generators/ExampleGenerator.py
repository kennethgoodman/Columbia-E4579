from .AbstractGenerator import AbstractGenerator
import json


class ExampleGenerator(AbstractGenerator):

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

    # def get_content_ids(self, _, limit, offset, _seed, starting_point):
    #     if starting_point is None:
    #         # TODO: should discount by creation_time so closer events have more weight
    #         results = (
    #             Engagement.query.with_entities(
    #                 Engagement.content_id, func.count()
    #             )
    #             .filter_by(
    #                 engagement_type=EngagementType.Like,
    #             )
    #             .group_by(Engagement.content_id)
    #             .order_by(func.count().desc())
    #             .limit(limit)
    #             .offset(offset)
    #             .all()
    #         )
    #         return list(map(lambda x: x[0], results)), None
    #     elif starting_point.get("content_id", False):
    #         content_ids, scores = ann_with_offset(
    #             starting_point["content_id"], 0.9, limit, offset, return_distances=True
    #         )
    #         return content_ids, scores
    #     raise NotImplementedError("Need to provide a key we know about")
