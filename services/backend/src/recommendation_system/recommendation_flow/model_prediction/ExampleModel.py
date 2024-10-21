import random
from src import db
from sqlalchemy.sql import text

from .AbstractModel import AbstractModel


class ExampleModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        if len(content_ids) == 0:
            return [], [], [], []
        get_styles_for_content_ids = text(f"""
            SELECT content_id, COALESCE(artist_style, '')
            from generated_content_metadata
            where content_id in ({','.join(map(str, content_ids))})
        """)
        get_styles_user_has_engaged_with = text(f"""
            select DISTINCT COALESCE(artist_style, '')
            from engagement
            inner join generated_content_metadata
                using (content_id)
                where user_id = {user_id};
        """)
        with db.engine.connect() as con:
            styles_for_content_ids = list(con.execute(get_styles_for_content_ids))
            content_id_to_style_map = {}
            for (content_id, artist_style) in styles_for_content_ids:
                content_id_to_style_map[content_id] = artist_style
            results = con.execute(get_styles_user_has_engaged_with)
            styles_user_has_engaged_with = set(map(lambda x: x[0], results))

        content_ids = list(content_ids)
        like = [float(content_id_to_style_map[content_id] in styles_user_has_engaged_with) for _ in content_ids]
        dislike = [1 - float(content_id_to_style_map[content_id] in styles_user_has_engaged_with) for _ in content_ids]
        eng = [float(content_id_to_style_map[content_id] in styles_user_has_engaged_with) for _ in content_ids]

        return (like, dislike, eng, content_ids)

    def _get_name(self):
        return "ExampleModel"

class ExampleModelWithForcedText(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        if len(content_ids) == 0:
            return [], [], [], []
        get_styles_for_content_ids = text(f"""
            SELECT content_id, COALESCE(artist_style, ''), media_type
            from generated_content_metadata
                inner join content on generated_content_metadata.content_id = content.id
            where content_id in ({','.join(map(str, content_ids))})
        """)
        get_styles_user_has_engaged_with = text(f"""
            select DISTINCT COALESCE(artist_style, '')
            from engagement
            inner join generated_content_metadata
                using (content_id)
                where user_id = {user_id};
        """)
        with db.engine.connect() as con:
            styles_for_content_ids = list(con.execute(get_styles_for_content_ids))
            content_id_to_style_map = {}
            content_id_to_is_text = {}
            for (content_id, artist_style, media_type) in styles_for_content_ids:
                content_id_to_style_map[content_id] = artist_style
                content_id_to_is_text[content_id] = media_type == "Text"
            results = con.execute(get_styles_user_has_engaged_with)
            styles_user_has_engaged_with = set(map(lambda x: x[0], results))

        content_ids = list(content_ids)
        like = [
            float(content_id_to_style_map[content_id] in styles_user_has_engaged_with) 
            for _ in content_ids
        ]
        dislike = [
            1 - float(content_id_to_style_map[content_id] in styles_user_has_engaged_with) 
            for _ in content_ids
        ]
        eng = [
            1 if content_id_to_is_text[content_id] and random.random() > 0.5
            else float(content_id_to_style_map[content_id] in styles_user_has_engaged_with) 
            for _ in content_ids
        ]

        return (like, dislike, eng, content_ids)

    def _get_name(self):
        return "ExampleModelWithForcedText"
