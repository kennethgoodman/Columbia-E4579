from src import db

from .AbstractModel import AbstractModel

class RuleBasedModel(AbstractModel):
    def liked_same_style(self, con, content_id, user_id):
        # If the user has liked images from this category +1 to the score, else 0

        # first get the style
        style_query = 'SELECT artist_style '\
                'FROM generated_content_metadata '\
                f'WHERE content_id={content_id};'
        style = con.execute(style_query).one()[0]

        liked_same_style_query = 'SELECT COUNT(engagement.content_id) '\
                                'FROM engagement LEFT JOIN generated_content_metadata '\
                                'ON engagement.content_id=generated_content_metadata.content_id '\
                                f'WHERE engagement.user_id={user_id} AND '\
                                'engagement.engagement_type="Like" AND '\
                                'engagement.engagement_value=1 AND '\
                                f'generated_content_metadata.artist_style="{style}";'
        result = con.execute(liked_same_style_query).one()[0]

        return result > 0
    
    def popularity_score(self, con, content_id):
        # If ratio of likes/(likes+dislikes) > threshold  +1 to the score, else 0
        likes_query = 'SELECT COUNT(*) '\
                'FROM engagement '\
                f'WHERE content_id={content_id} AND '\
                'engagement_type="Like" AND '\
                'engagement_value=1;'

        dislikes_query = 'SELECT COUNT(*) '\
                'FROM engagement '\
                f'WHERE content_id={content_id} AND '\
                'engagement_type="Like" AND '\
                'engagement_value=-1;'

        num_of_likes = con.execute(likes_query).one()[0]
        num_of_dislikes = con.execute(dislikes_query).one()[0]
        total_score = num_of_likes+num_of_dislikes
        return num_of_likes*1.0/total_score > 0.5 if total_score > 0 else False

    def calculate_score(self, content_id, user_id):
        score = 0
        with db.engine.connect() as con:
            score += int(self.liked_same_style(con, content_id, user_id))
            score += int(self.popularity_score(con, content_id))
        return score

    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):

        # content id string
        content_id_str = '(' + ','.join(list(map(str, content_ids))) + ')'

        # liked same style
        p_engage = dict()

        with db.engine.connect() as con:

            for content_id in content_ids:
                p_engage[content_id] = 0

            likes_dislikes = con.execute(
                    f"select content_id,"\
                    f"SUM(CASE WHEN (engagement.engagement_value = -1 and engagement.engagement_type = 'Like') THEN 1 ELSE 0 END)," \
                    f"SUM(CASE WHEN (engagement.engagement_value = 1 and engagement.engagement_type = 'Like') THEN 1 ELSE 0 END)" \
                    f"from engagement where engagement.content_id in {content_id_str} group by engagement.content_id;"
                    ).all()

            for (content_id, dislikes, likes) in likes_dislikes:
                likes = int(likes) + 1
                dislikes = int(dislikes) + 1
                total = likes + dislikes

                if likes / total > 0.5:
                    p_engage[content_id] = p_engage[content_id] + 1

            same_style_total_likes = con.execute(
                     f"with cte1 as (select count(*) as count,artist_style " \
                     f"from engagement left join generated_content_metadata " \
                     f"on engagement.content_id = " \
                     f"generated_content_metadata.content_id where " \
                     f"engagement.user_id = {user_id} and " \
                     f"engagement.engagement_type = 'Like' " \
                     f"and engagement.engagement_value = 1 group by " \
                     f"generated_content_metadata.artist_style) " \
                     f"select content_id,cte1.count from " \
                     f"generated_content_metadata left join cte1 on " \
                     f"generated_content_metadata.artist_style = " \
                     f"cte1.artist_style where " \
                     f"generated_content_metadata.content_id in {content_id_str};"
                     ).all()


            for (content_id, total_likes) in same_style_total_likes:
                total_likes = 0 if total_likes is None else int(total_likes)

                if total_likes > 0:
                    p_engage[content_id] = p_engage[content_id] + 1


        # popularity score
        return list(
            map(
                lambda content_id: {
                    "content_id": content_id,
                    "p_engage": p_engage[content_id],
                    "score": kwargs.get("scores", {}).get(content_id, {}).get("score", None),
                },
                content_ids,
            )
        )
