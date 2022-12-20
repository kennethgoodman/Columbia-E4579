from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_target(engagement_path: str, output_path: str, save: bool = True) -> Optional[pd.DataFrame]:
    """Generates data with target variable (like).

    Args:
        engagement_path (str): Path of engagement.csv
        output_path (str): Output path.
        save (bool): Whether to save the output to the output path or not.

    Returns:
        Optional[pd.DataFrame]: Dataframe in the following columns:
            - user_id (int): user ID
            - content_id (int): content ID
            - engagement_value (int): 0 or 1. 0 represents the user has seen the content but did not click like (
                including when the user click dislike). 1 represents the user clicked liked on the item.
    """

    engagement = pd.read_csv(engagement_path)

    # get unique user-item pairs from engagement data
    user_item = engagement[['user_id', 'content_id']].drop_duplicates()

    # Get the latest like of each user-item pair
    action = engagement[engagement.engagement_type == 'Like'].sort_values('created_date').groupby(
        ['user_id', 'content_id']).tail(1)
    action = action[['user_id', 'content_id', 'engagement_value']]

    user_item_like = pd.merge(user_item, action, how='left', left_on=['user_id', 'content_id'],
                              right_on=['user_id', 'content_id'])

    # Turn dislike (-1) into 0
    user_item_like['engagement_value'] = user_item_like['engagement_value'].clip(0)

    # Fill the user-item pair that has no like/dislike with engagement value of 0.
    user_item_like = user_item_like.fillna(0)

    if save:
        user_item_like.to_csv(output_path, index=False)

    return user_item_like


def split_data(target_df: pd.DataFrame,
               output_folder: str,
               test_size: float = 0.2,
               random_state: int = 4579,
               save: bool = True) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Splits the data into train and test set with stratification on user_id.

    Args:
        target_df (pd.DataFrame): Dataframe with user_id, content_id, and target value.
        output_folder (str): Output folder.
        test_size (float): Proportion of test set.
        random_state (int): Random state (keep fix for reproducibility)
        save (bool): Whether to save the output to the output path or not.

    Returns:
        Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Tuple of dataframes (train, test) in the same format as
            the input dataframe.
    """

    # Keep only the users who have interacted with more than 1 content.
    user_niteratacts = target_df.user_id.value_counts()
    target_df = target_df[target_df.user_id.isin(user_niteratacts[user_niteratacts > 1].index)]

    # Split the user-item pairs randomly into train, test with stratification on user_id
    train, test = train_test_split(target_df, test_size=test_size, random_state=random_state,
                                   stratify=target_df[['user_id']])

    if save:
        train.to_csv(f'{output_folder}/train.csv', index=False)
        test.to_csv(f'{output_folder}/test.csv', index=False)

    return train, test


target_df = generate_target('../../../seed_data/data/nov_19_dump/engagement.csv',
                            '../../echo_space/processed_data/target.csv')

train, test = split_data(target_df, '../../../../../services/backend/processed_data')
