import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import pickle
import random
from two_tower import TwoTowerModel, ContrastiveLoss, EngagementDataset
from two_tower import preprocessing

# Load data and preprocess (use your preprocessing code here)
df = pd.read_csv("C:/Users/tanis/Desktop/Columbia/Coursework/Fall 2023/03. Recommendation Systems/Data/columbia_data.tsv", sep='\t')
with open("C:/Users/tanis/Desktop/Columbia/Coursework/Fall 2023/03. Recommendation Systems/Data/id_to_pickle_dict.pkl", "rb") as f:
    prompt_to_embedding = pickle.load(f)
prompt_embedding = pd.DataFrame(prompt_to_embedding.items(), columns=["content_id", "prompt_embedding"])
df = df.merge(prompt_embedding, on="content_id")
del prompt_embedding

df = preprocessing(df)
# print(df[cont_avg_eng])

# Create overall features
user_columns = (
    ['user_avg_eng'] +
    list(filter(lambda x: 'ms_engaged_' in x, df.columns)) +
    list(filter(lambda x: 'like_vector_' in x, df.columns)) +
    list(filter(lambda x: 'dislike_vector_' in x, df.columns)) 
    )
user_columns = list(set(user_columns))
user_features = df[user_columns]
user_features_tensor = torch.FloatTensor(user_features.values)
del user_features

content_columns = (
    # [cont_avg_eng] +
    list(filter(lambda x: 'artist_style_' in x, df.columns)) +
    list(filter(lambda x: 'model_version_' in x, df.columns)) +
    list(filter(lambda x: 'source_' in x, df.columns)) +
    list(filter(lambda x: 'seed_' in x, df.columns)) +
    list(filter(lambda x: 'prompt_embedding_' in x, df.columns)) +
    ['content_id', 'cont_avg_eng','guidance_scale', 'num_inference_steps']
)
content_features = df[content_columns]
content_features_tensor = torch.FloatTensor(content_features.values)
del content_features

DISLIKE_ENGAGEMENT_TYPE_VALUE = 0
LIKE_ENGAGEMENT_TYPE_VALUE = 1
MS_ENGAGEMENT_TYPE_VALUE = 2

engagement_type_value = df[['engagement_type', 'engagement_value']]
engagement_type_value['engagement_type'] = engagement_type_value.apply(
    lambda x: (
       LIKE_ENGAGEMENT_TYPE_VALUE    if x['engagement_type'] == 'Like' and x['engagement_value'] ==  1 else # Like
       DISLIKE_ENGAGEMENT_TYPE_VALUE    if x['engagement_type'] == 'Like' and x['engagement_value'] == -1 else # Dislike
       MS_ENGAGEMENT_TYPE_VALUE                                                                          # MsEngagementType
    ), axis=1
)
engagement_type_tensor = torch.FloatTensor(engagement_type_value[['engagement_type']].values)
engagement_value_tensor = torch.FloatTensor(engagement_type_value[['engagement_value']].values)
del engagement_type_value
del df

# Specify the input and output dimensions
user_dim = len(user_columns)
content_dim = len(content_columns)
output_dim = 64

# Create the model
model = TwoTowerModel(user_dim, content_dim, output_dim)

# Specify the loss function
loss_function = ContrastiveLoss(margin=1.0, lambda_reg=0.01, lambda_orthog=0.001)

# Calculate targets
targets = loss_function.calculate_targets(engagement_type_tensor, engagement_value_tensor)

# Create dataset and DataLoader instances
engagement_data = EngagementDataset(user_features_tensor, content_features_tensor, targets)
train_data, test_data = train_test_split(engagement_data, test_size=0.2, random_state=28)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=28)

batch_size = 512
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Create optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                               max_lr=0.005,
                                               steps_per_epoch=len(train_loader),
                                               epochs=50,
                                               anneal_strategy='linear',
                                               pct_start=0.3,
                                               div_factor=25.0,
                                               final_div_factor=10000.0)

# Training loop
for epoch in range(20):
    model.train()
    for batch in train_loader:
        user_features_batch, content_features_batch, targets_batch = batch
        optimizer.zero_grad()
        user_embedding, content_embedding = model(user_features_batch, content_features_batch)
        loss = loss_function(user_embedding, content_embedding, targets_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            user_features_batch, content_features_batch, targets_batch = batch
            user_embedding, content_embedding = model(user_features_batch, content_features_batch)
            loss = loss_function(user_embedding, content_embedding, targets_batch)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation - Epoch {epoch+1}, Loss: {val_loss}')

    # Save the model after each epoch
    # torch.save(model.state_dict(), f"path/to/save/model_epoch_{epoch + 1}.pth")
    # print(f'Model saved after Epoch {epoch+1}')

# Save the final trained model
torch.save(model.state_dict(), "C:/Users/tanis/Columbia-E4579/services/backend/src/recommendation_system/ml_models/charlie/two_tower_trained_saved.pth")
print('Final trained model saved.')
