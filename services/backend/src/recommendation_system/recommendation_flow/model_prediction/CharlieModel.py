import random
from src import db
from sqlalchemy.sql import text
from sqlalchemy.sql.expression import bindparam
import pandas as pd

from .AbstractModel import AbstractModel


class CharlieModel(AbstractModel):
    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        ## use the user_id to get all engagement of this

        #use crud?

        ## pandas dataframe

        ## artist_style = {}
        ## for i in dataframe (only the ones with engagement time), this is for each content id:
            ## if key of artist style not in artist_style:
                ## artist_style[key] = { Like : 0, Dislike : 0, Neutral: 0}
            ## else:
                ## if like and engagement value then artist_style[key][Like] += 1
                ## elif like then artist_style[key][Dislike] += 1
                ## else artist_style[key][Neutral] += 1
            

        ## results is {artist_style: {like: INT, dislike: INT, Neutral: INT}}

        ## {aritst_style: score}

        ## if we are doing similarity, get all the users, get the dictionary for all the users

        ## rows are users, columns are artist style

        ## combind the score for like: 1, dislike -1, neutral, 0.5 for each artist type

        ## create the same thing for source

        ## for content_ids, get artist style and source, 
        ## map 50/50 score based on like and dislike INT

        with db.engine.connect() as con:

            all_sql_statement = """
            SELECT c.content_id, user_id, engagement_type, engagement_value, artist_style, source
            FROM engagement as e
            JOIN generated_content_metadata as c
            ON c.content_id = e.content_id
            """
            
            table_df = pd.read_sql_query(all_sql_statement, con=con)
            
            mapping_sql_statement = """
            SELECT content_id, artist_style, source
            FROM generated_content_metadata
            WHERE content_id in :contentids
            """ 

            params = {'contentids': tuple(content_ids)}
            mapping_sql_statement = text(mapping_sql_statement)
            mapping_sql_statement = mapping_sql_statement.bindparams(bindparam('contentids', expanding=True))

            mapping_df = pd.read_sql_query(
            mapping_sql_statement,
            con=con,
            params=params
            )
            
            engagementtime = table_df[table_df['engagement_type']=='MillisecondsEngagedWith']
            like_list = table_df[(table_df['engagement_type']=='Like')&(table_df['engagement_value']==1)]['content_id'].tolist() 
            dislike_list = table_df[(table_df['engagement_type']=='Like')&(table_df['engagement_value']==-1)]['content_id'].tolist()
            
            artist_style = {}
            source_dict = {} 
            for index, row in engagementtime.iterrows():
                user = row['user_id']
                style = row['artist_style']
                source = row['source']
                ## when doing the aggregation, check if user_id is the same as input, if yes, then keep the code
                ## if not, += weight * 1

                if user ==  user_id:
                    
                    if style not in artist_style:
                        artist_style[style] = { 'Like' : 0, 'Dislike' : 0, 'Neutral': 0, 'Agg': 0}
                    if source not in source_dict: 
                        source_dict[source] = { 'Like' : 0, 'Dislike' : 0, 'Neutral': 0, 'Agg': 0} 
                    
                    if row['content_id'] in like_list:  
                        artist_style[style]['Like'] += 1
                        artist_style[style]['Agg'] += 1
                        source_dict[source]['Like'] += 1 
                        source_dict[source]['Agg'] += 1
                    elif row['content_id'] in dislike_list:  
                        artist_style[style]['Dislike'] += 1
                        artist_style[style]['Agg'] -= 1
                        source_dict[source]['Dislike'] += 1
                        source_dict[source]['Agg'] -= 1
                    else: 
                        artist_style[style]['Neutral'] += 1
                        artist_style[style]['Agg'] += 0.5
                        source_dict[source]['Neutral'] += 1
                        source_dict[source]['Agg'] += 0.5
                else:
                    if row['content_id'] in like_list:  
                        artist_style[style]['Like'] += 1
                        artist_style[style]['Agg']+=0.5
                        source_dict[source]['Like'] += 1 
                        source_dict[source]['Agg'] += 0.5
                    elif row['content_id'] in dislike_list:  
                        artist_style[style]['Dislike'] += 1
                        artist_style[style]['Agg'] -= 0.5
                        source_dict[source]['Dislike'] += 1
                        source_dict[source]['Agg'] -= 0.5
                    else: 
                        artist_style[style]['Neutral'] += 1
                        artist_style[style]['Agg'] += 0.25
                        source_dict[source]['Neutral'] += 1
                        source_dict[source]['Agg'] += 0.25 

            artist_style_final = {}
            for i in artist_style.keys():
                artist_style_final[i] = artist_style[i]['Agg']

            source_final = {}
            for i in source_dict.keys():
                source_final[i] = source_dict[i]['Agg']

            ## we are here
            scores = []
            for index, row in mapping_df.iterrows():
                ## map this content_id to its artist_style and source
                artist_style = row['artist_style']
                source = row['source']
                score = 0

                ## get artist_style score for this artist style
                ## get source score for this source
                if artist_style in artist_style_final:
                    score += 0.5 * artist_style_final[artist_style]
                    
                if source in source_final:
                    score += 0.5 * source_final[source]

                ## append to scores
                scores.append(score)

        return list(
            map(
                lambda content_id: {
                    "content_id": content_id,
                    "p_engage": scores, ## what is this?
                    "score": kwargs.get("scores", {scores})
                    .get(content_id, {})
                    .get("score", None),
                },
                content_ids,
            )
        )
