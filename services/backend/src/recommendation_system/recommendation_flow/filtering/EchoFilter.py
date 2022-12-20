from src.api.engagement.crud import get_dislike_count_by_content_id, get_like_count_by_content_id
from .AbstractFilter import AbstractFilter


class EchoFilter(AbstractFilter):
    def filter(self, content_ids):

        filtered = []
        for c_id in content_ids:
            like = get_like_count_by_content_id(c_id)
            dislike = get_dislike_count_by_content_id(c_id)
            like_ratio = like / (dislike + like)

            if like_ratio >= 0.8:
                filtered.append(c_id)

        return filtered
