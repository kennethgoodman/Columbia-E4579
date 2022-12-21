from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

from implicit.als import AlternatingLeastSquares
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


class ALSEstimator:
    """Alternating Least Squares model from Implicit library.
    https://github.com/benfred/implicit

    Attributes:
        user_col (str): Name of column that indicates user ID. Defaults to "user_id".
        item_col (str): Name of column that indicates item ID. Defaults to "item_id".
        rating_col (str): Name of column that indicates rating. Defaults to "rating".
        preprocess_func (Callable): The callable to use for preprocessing the train set. Defaults to None.
        confidence_func (Callable): The callable to use for the confidence. Defaults confidence_log with alpha 40.
    """

    _seen: pd.DataFrame
    _conf_sprs: sparse.csr_matrix

    def __init__(
        self,
        user_col: str = 'user_id',
        item_col: str = 'content_id',
        rating_col: str = 'engagement_value',
        factors: int = 20,
        regularization: float = 0.1,
        iterations: int = 15,
        random_state: Optional[int] = None,
    ) -> None:
        """Initializes ALSEstimator class."""

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        self._user_encoder = LabelEncoder()
        self._item_encoder = LabelEncoder()
        self._conf_sprs: Union[None, sparse.csr_matrix] = None

    def _preprocess(self, rating: pd.DataFrame) -> sparse.csr_matrix:

        self._seen = deepcopy(rating)[[self.user_col, self.item_col]]

        rating[self.user_col] = self._user_encoder.fit_transform(rating[self.user_col])
        rating[self.item_col] = self._item_encoder.fit_transform(rating[self.item_col])

        rating_sprs = sparse.csr_matrix((rating[self.rating_col], (rating[self.user_col], rating[self.item_col])))

        return rating_sprs

    def fit(self, rating: pd.DataFrame) -> None:
        """Trains the model by the given data.

        Args:
            data (DataClass): Input data.
        """

        self._conf_sprs = self._preprocess(rating)
        self._model.fit(self._conf_sprs)

    # pylint: disable=too-many-locals
    def recommend(
        self,
        users: Union[None, List[Union[str, int]]] = None,
        n_recommendations: int = 10
    ):
        """Generates a list of recommended items with for each user.

        Args:
            data: (Optional[DataClass]): Currently has no effects for Collaborative-
            Filtering based models.
            users: (Union[None, List[Union[str, int]]]): List of users to make recommendation. If None, it will output
                predictions to all users. Default is None.
            n_recommendations (int): Number of items to recommend per user. Defaults to 10.
            to_exclude (pyspark.sql.dataframe.DataFrame): Dataframe of user-item interactions to be excluded from
                recommendation lists.
        Returns:
            pyspark.sql.dataframe.DataFrame: Dataframe of recommended items for each user.

        Raises:
            Exception: "The model has not been fitted."
        """

        # Prepare users list
        if users:
            user_idx = self._user_encoder.transform(list(set(users) & set(self._user_encoder.classes_)))
        else:
            user_idx = list(range(0, self._conf_sprs.shape[0]))

        recs, scores = self._model.recommend(
            userid=user_idx,
            user_items=self._conf_sprs,
            N=n_recommendations,
            filter_already_liked_items=True,
        )

        # Convert users/items back to original ids
        user_list = self._user_encoder.inverse_transform(user_idx).tolist()
        recs_inv = self._item_encoder.inverse_transform(recs.flatten()).reshape(recs.shape)

        recs = dict(zip(user_list, recs_inv.tolist()))

        return recs
