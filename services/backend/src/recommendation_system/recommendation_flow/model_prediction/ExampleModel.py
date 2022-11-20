import random
from src import db
from sqlalchemy.sql import text

from .AbstractModel import AbstractModel


class ExampleModel(AbstractModel):
    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        get_styles_for_content_ids = text(f"""
            SELECT content_id, artist_style 
            from generated_content_metadata 
            where content_id in ({','.join(map(str, content_ids))})
        """)
        get_styles_user_has_engaged_with = text(f"""
            select DISTINCT artist_style 
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

        return list(
            map(
                lambda content_id: {
                    "content_id": content_id,
                    "p_engage": float(content_id_to_style_map[content_id] in styles_user_has_engaged_with),
                    "score": kwargs.get("scores", {})
                    .get(content_id, {})
                    .get("score", None),
                },
                content_ids,
            )
        )
