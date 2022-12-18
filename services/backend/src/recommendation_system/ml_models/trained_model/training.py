

import os
import tensorflow_decision_forests as tfdf
import pandas as pd
import pickle
import sqlalchemy
import numpy as np


"""

Data source

- content.csv
- content metadata.csv
- engagement.csv

Model:  Tensorflow Gradient Boosted Decision Forest

Label:
- 1: engagement time > 3 seconds, or liked
- 0: otherwise

Target variable: probability of engagement (user -> content)

"""

def preprocess_data(data_path):

    """
    Before training the model, process the data for faster speed
    """
    # convert list to dict format of id_embed
    with open(data_path + 'id_to_embedding.pkl', 'rb') as f:
        id_embed = pickle.load(f)

    engage = pd.read_csv(data_path + 'engagement.csv')
    engaged_contents = engage.content_id.unique()
    id_embedding_dict = {k:embed for (k,embed) in id_embed if k in engaged_contents}
    with open(data_path + 'dic_id_to_embedding.pickle', 'wb') as f:
        pickle.dump(id_embedding_dict, f)

    # save the needed content metadata
    content_meta = pd.read_csv(data_path + 'generated_content_metadata.csv')
    content_artist_style_dic = dict(zip(content_meta.content_id, content_meta.artist_style))
    with open(data_path + 'content_artist_style_dic.pickle', 'wb') as f:
        pickle.dump(content_artist_style_dic, f)


def split_dataset(dataset, test_ratio=0.30):
    """
    Split training and testing dataset
    """
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def feature_generation(data_path):
    
    # data_path = '/Users/zhongsitong/Desktop/Columbia/RecommendationSys/Columbia-E4579-main/data/local_data' 
    # os.chdir(data_path)

    # content = pd.read_csv('content.csv')

    engage = pd.read_csv(data_path + 'engagement.csv') # user_id	content_id	engagement_type	engagement_value	created_date

    with open(data_path + 'dic_id_to_embedding.pickle', 'rb') as f:
        dic_id_embed = pickle.load(f)

    with open(data_path + 'content_artist_style_dic.pickle', 'rb') as f:
        dic_id_style = pickle.load(f)

    train = engage[['user_id','content_id','engagement_type','engagement_value']]
    train['style'] = train['content_id'].apply(lambda x: dic_id_style[x])
    train['like'] = train.apply(lambda row: pd.Series(1 if row['engagement_type'] == 'Like' and
                                                      row['engagement_value'] == 1 else 0), axis=1)
    train['dislike'] = train.apply(lambda row: pd.Series(1 if row['engagement_type'] == 'Like' and
                                                         row['engagement_value'] == -1 else 0), axis=1)
    train['engage_time'] = train.apply(lambda row: pd.Series(row['engagement_value']
                                                             if row['engagement_type']=='MillisecondsEngagedWith' else 0), axis=1)
    train['content_total_likes'] = train.groupby('content_id')['like'].transform('sum')
    train['content_total_dislikes'] = train.groupby('content_id')['dislike'].transform('sum')
    train['user_total_likes'] = train.groupby('user_id')['like'].transform('sum')
    
    # Generate embed matrix
    train_content_ids = train.content_id.tolist()
    embed_matrix = []
    for content_id in train_content_ids:
        embed_matrix.append(dic_id_embed[content_id])
    embed_matrix = pd.DataFrame(embed_matrix)

    train = train.reset_index(drop=True)
    embed_matrix = embed_matrix.reset_index(drop=True)
    train_df = pd.concat([train, embed_matrix], axis=1)

    Y = train_df.apply(lambda row: pd.Series(1 if row['like'] or row['engage_time'] > 3000 else 0), axis=1)
    train_df = train_df.drop(['like','dislike','engage_time','content_id','engagement_type','engagement_value'], axis=1)
    train_df['label'] = Y
    train_df['user_id'] = train_df['user_id'].astype(str)
    train_df['style'] = train_df['style'].astype(str)
    train_df.columns = [str(c) for c in train_df.columns]

    return train_df


def model_training_GBDT(dataset, save_path):

    train_df, test_df = split_dataset(dataset)

    # tf_train = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='label',task=tfdf.keras.Task.CLASSIFICATION)
    tf_train = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='label',task=tfdf.keras.Task.REGRESSION)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='label',task=tfdf.keras.Task.REGRESSION)

    # model = tfdf.keras.GradientBoostedTreesModel(
    #     task=tfdf.keras.Task.CLASSIFICATION
    #     num_trees=500,
    #     growing_strategy="BEST_FIRST_GLOBAL",
    #     max_depth=8,
    #     categorical_algorithm="RANDOM"
    #     )

    model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
    model.fit(tf_train)

    # print(model.evaluate(tf_test, return_dict=True))
    # save the model
    with open(save_path + 'gbdt_model_v3.pickle', 'wb') as f:
        pickle.dump(model, f)
    return model


def debug(dataset):

    with open(data_path + 'gbdt_model_v3.pickle', 'rb') as f:
        model = pickle.load(f)
    
    dataset = dataset.drop('label',axis=1)
    dt = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, task=tfdf.keras.Task.REGRESSION)
    predicted = model.predict(dt)
    print(predicted)
    print(predicted.max())
    print(sum(predicted))
    print(len(predicted))


# if __name__ == "__main__":
    
#     base_path = os.getcwd() # need to locate at main foler - base path
#     data_path = base_path + '/data/local_data/'

#     # preprocess_data(data_path)  # update when new data came 
    
#     print("Feature Generation Started...")
#     train_df = feature_generation(data_path)
#     print("shape of training data:", train_df) 

#     print("Model Training Started...")
#     model = model_training_GBDT(train_df, data_path)  # saved the trained model
#     print("Model Saved...")





    



