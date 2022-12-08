from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
from .AbstractGenerator import AbstractGenerator
from src.api.utils.auth_utils import get_user
import json

class AlphaGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        print("user id from alpha generator:",user_id)
        # instead of using json file, just query engagement data for this user?
        # all images that have not been engaged by this user can be a candidate
        # select content_ids from content where content id not in (select content id from engagement where user id == )

        with open("/usr/src/app/src/queue.json", "r") as openfile:
            try:
                data = json.load(openfile)
                print("data from queue.json",data,type(data))
                queue = [i['queue'] for i in data if i['id']==user_id][0]
            except: # if json file is empty, the case at initialization. or if it doesn't contain data for the current user
                # data = []
                queue = []

        print('queue',queue)

        print('starting_point',starting_point)

        if starting_point is None:
            results = (
                Content.query.with_entities(Content.id)
            )
            # print("results[:10]",results[:10])

            # results = (
            #     Content.query.with_entities(Content.id)
            #     .order_by(func.random(seed))
            #     .limit(limit)
            #     .offset(offset)
            #     .all()
            # )
            # print("results[:10]",results[:10])
            print("if starting_point is None") #,list(map(lambda x: x[0], results)))
            print("len(results)",len(list(map(lambda x: x[0], results))))
            # print("results[:10]",list(map(lambda x: x[0], results))[:10])

            results = list(map(lambda x: x[0], results))
            results = [i for i in results if i not in queue]
            # return list(map(lambda x: x[0], results)), None
            return results, None

        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            # print("elif starting_point.get("content_id", False)")
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
