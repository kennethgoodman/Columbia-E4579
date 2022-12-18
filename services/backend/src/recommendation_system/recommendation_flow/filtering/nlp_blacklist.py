import pandas as pd

nlp_data = pd.read_pickle('df_prompt_labeled.pkl')
print(nlp_data.describe())
# print(nlp_data)
print(len(nlp_data))
# print('nlp_data:',list(nlp_data[nlp_data.offensive==False].content_id))
print('nlp_data.columns:',nlp_data.columns)
print('min(nlp_data.content_id)',min(nlp_data.content_id))
# print('nlp_data:',nlp_data.)
