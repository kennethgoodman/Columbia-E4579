import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import pickle

#from src.recommendation_system.ml_models.delta.two_tower import TwoTowerModel

# Read Data
df = pd.read_csv('columbia_data.tsv', sep='\t')
with open("id_to_pickle_dict.pkl", "rb") as f:
    prompt_to_embedding = pickle.load(f)
prompt_embedding = pd.DataFrame(prompt_to_embedding.items(), columns=["content_id", "prompt_embedding"])
df = df.merge(prompt_embedding, on="content_id")
del prompt_embedding
del prompt_to_embedding

# Configuration options
TOP_ARTIST_STYLES = 30
TOP_SOURCES = 30
TOP_SEEDS = 14
TOP_CONTENT = 251
PROMPT_EMBEDDING_LENGTH = 512

# Get the top artist styles, sources, and seeds
top_artist_styles = df['artist_style'].value_counts().nlargest(TOP_ARTIST_STYLES).index.tolist()
top_sources = df['source'].value_counts().nlargest(TOP_SOURCES).index.tolist()
top_seeds = df['seed'].value_counts().nlargest(TOP_SEEDS).index.tolist()

# Replace less frequent artist styles, sources, and seeds with 'other'
df['artist_style'] = df['artist_style'].apply(lambda x: x if x in top_artist_styles else 'other')
df['source'] = df['source'].apply(lambda x: x if x in top_sources else 'other')
df['seed'] = df['seed'].apply(lambda x: str(x) if x in top_seeds else 'other')

# One-hot encode categorical features
encoder = OneHotEncoder()
content_onehot = encoder.fit_transform(df[['artist_style', 'model_version', 'seed', 'source']])
content_onehot_df = pd.DataFrame(content_onehot.toarray(), columns=encoder.get_feature_names_out(
    ['artist_style', 'model_version', 'seed', 'source']))
df = pd.concat([df, content_onehot_df], axis=1)

# Normalize linear features
scaler = StandardScaler()
df[['guidance_scale', 'num_inference_steps']] = scaler.fit_transform(df[['guidance_scale', 'num_inference_steps']])

# Compute top N content pieces based on engagement_value
from collections import defaultdict

top_n_content = df.groupby('content_id')['engagement_value'].count().nlargest(TOP_CONTENT).index.tolist()
user_vector_dict = defaultdict(lambda: {
    'millisecond_engaged_vector': np.zeros(len(top_n_content)),
    'like_vector': np.zeros(len(top_n_content)),
    'dislike_vector': np.zeros(len(top_n_content))
})


# Initialize vectors for each user
def aggregate_engagement(group):
    # Summing millisecond engagement values
    millisecond_engagement_sum = group.loc[group['engagement_type'] != 'Like', 'engagement_value'].sum()

    # Counting likes and dislikes
    likes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == 1)].shape[0]
    dislikes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == -1)].shape[0]

    return pd.Series({
        'millisecond_engagement_sum': millisecond_engagement_sum,
        'likes_count': likes_count,
        'dislikes_count': dislikes_count
    })


# Group by user_id and content_id, then apply the function
engagement_aggregate = df[df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id']).apply(
    aggregate_engagement).reset_index()

# Now, populate your user_vector_dict
for _, row in engagement_aggregate.iterrows():
    user_id = row['user_id']
    content_id = row['content_id']
    idx = top_n_content.index(content_id)

    user_vector_dict[user_id]['millisecond_engaged_vector'][idx] = row['millisecond_engagement_sum']
    user_vector_dict[user_id]['like_vector'][idx] = row['likes_count']
    user_vector_dict[user_id]['dislike_vector'][idx] = row['dislikes_count']

# Convert to DataFrame
user_vector_df = pd.DataFrame.from_dict(user_vector_dict, orient='index')
del user_vector_dict

# Unpack vector columns into individual columns
millisecond_columns = [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
like_columns = [f"like_vector_{i}" for i in range(TOP_CONTENT)]
dislike_columns = [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]

user_vector_df[millisecond_columns] = pd.DataFrame(user_vector_df['millisecond_engaged_vector'].tolist(),
                                                   index=user_vector_df.index)
user_vector_df[like_columns] = pd.DataFrame(user_vector_df['like_vector'].tolist(), index=user_vector_df.index)
user_vector_df[dislike_columns] = pd.DataFrame(user_vector_df['dislike_vector'].tolist(), index=user_vector_df.index)

# Drop the original vector columns
user_vector_df.drop(['millisecond_engaged_vector', 'like_vector', 'dislike_vector'], axis=1, inplace=True)

# Join User Vector To Df
df = df.merge(
    user_vector_df.reset_index().rename(columns={'index': 'user_id'}),
    on='user_id'
)
del user_vector_df

# Unpack prompt embedding
prompt_columns = [f"prompt_embedding_{i}" for i in range(PROMPT_EMBEDDING_LENGTH)]
prompt_df = pd.DataFrame(df['prompt_embedding'].tolist(), columns=prompt_columns, index=df.index)
df = pd.concat([df, prompt_df], axis=1)
df = df.drop('prompt_embedding', axis=1)

# Create overall features
user_columns = (
        [f'ms_engaged_{i}' for i in range(TOP_CONTENT)] +
        [f'like_vector_{i}' for i in range(TOP_CONTENT)] +
        [f'dislike_vector_{i}' for i in range(TOP_CONTENT)]
)
user_features = df[user_columns]
user_features_tensor = torch.FloatTensor(user_features.values)
del user_features

content_columns = (
        list(filter(lambda x: 'artist_style_' in x, df.columns)) +
        list(filter(lambda x: 'model_version_' in x, df.columns)) +
        list(filter(lambda x: 'source_' in x, df.columns)) +
        list(filter(lambda x: 'seed_' in x, df.columns)) +
        list(filter(lambda x: 'prompt_embedding_' in x, df.columns)) +
        ['content_id', 'guidance_scale', 'num_inference_steps']
)
content_features = df[content_columns]
content_features_tensor = torch.FloatTensor(content_features.values)
del content_features

DISLIKE_ENGAGEMENT_TYPE_VALUE = 0
LIKE_ENGAGEMENT_TYPE_VALUE = 1
MS_ENGAGEMENT_TYPE_VALUE = 2

# Convert engagement type to numerical values
engagement_type_value = df[['engagement_type', 'engagement_value']]
engagement_type_value['engagement_type'] = engagement_type_value.apply(
    lambda x: (
        LIKE_ENGAGEMENT_TYPE_VALUE if x['engagement_type'] == 'Like' and x['engagement_value'] == 1 else  # Like
        DISLIKE_ENGAGEMENT_TYPE_VALUE if x['engagement_type'] == 'Like' and x['engagement_value'] == -1 else  # Dislike
        MS_ENGAGEMENT_TYPE_VALUE  # MsEngagementType
    ), axis=1
)
engagement_type_tensor = torch.FloatTensor(engagement_type_value[['engagement_type']].values)
engagement_value_tensor = torch.FloatTensor(engagement_type_value[['engagement_value']].values)
del engagement_type_value
del df

import torch
import torch.nn as nn


# Ideally define this in your two_tower.py file and import
# if there are issues with model training, you can open the annotation and use this
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, content_dim, output_dim):
        super(TwoTowerModel, self).__init__()
        # Define the layers for content and user towers here if needed
        self.fc0 = nn.Linear(user_dim, output_dim)
        self.fc0_1= nn.Linear(content_dim, output_dim)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, user_tensor, content_tensor):
         return self.forward_user(user_tensor), self.forward_content(content_tensor)

    def forward_content(self, content_tensor):
        # Forward pass for content
        content_tensor = F.relu(self.fc0_1(content_tensor))
        content_tensor = F.relu(self.fc1(content_tensor))
        content_tensor = F.relu(self.fc2(content_tensor))
        content_tensor = self.fc3(content_tensor)

        return content_tensor


    def forward_user(self, user_tensor):
        # Forward pass for user
        user_tensor = F.relu(self.fc0(user_tensor))
        user_tensor = F.relu(self.fc1(user_tensor))
        user_tensor = F.relu(self.fc2(user_tensor))
        user_tensor = self.fc3(user_tensor)

        return user_tensor



# Specify the input and output dimensions
user_dim = len(user_columns)
content_dim = len(content_columns)
output_dim = 64

# Create the model
model = TwoTowerModel(user_dim, content_dim, output_dim)

# Define the loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_reg=0.01, lambda_orthog=0.01):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.lambda_orthog = lambda_orthog

    def calculate_targets(self, engagement_type_vector, engagement_value_vector):
        # Conditions for 0=dislike, 1=like, and 2=milliseconds engaged
        return torch.where(
            engagement_type_vector == DISLIKE_ENGAGEMENT_TYPE_VALUE,
            torch.zeros_like(engagement_type_vector),  # dislike
            torch.where(
                engagement_type_vector == LIKE_ENGAGEMENT_TYPE_VALUE,
                torch.ones_like(engagement_type_vector),  # like
                torch.where(
                    engagement_value_vector < 500,
                    torch.zeros_like(engagement_type_vector),  # bad engagement
                    torch.where(
                        engagement_value_vector <= 2500,
                        torch.ones_like(engagement_type_vector),  # engaged 500ms => 2.5s
                        torch.zeros_like(engagement_type_vector),  # bad engagement
                    )
                )  # millisecond engaged with
            )
        )

    def forward(self, user_embedding, content_embedding, targets, with_debug=False):
        noise_factor = 0.08 # was 0.0005
        user_embedding += noise_factor * torch.randn(*user_embedding.shape)
        content_embedding += noise_factor * torch.randn(*content_embedding.shape)

        cosine_sim = F.cosine_similarity(user_embedding, content_embedding, dim=1)

        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - targets) * torch.pow(cosine_sim, 2) +
            (targets) * torch.pow(
                torch.clamp(self.margin - cosine_sim, min=0.0),
                2
            )
        )

        # Regularization terms
        reg_user = torch.norm(user_embedding, p=2) / user_embedding.numel()
        reg_content = torch.norm(content_embedding, p=2) / content_embedding.numel()
        regularization_loss = (reg_user + reg_content)

        # orthognal loss of content
        ortho_reg = torch.norm(
            torch.mm(content_embedding, content_embedding.t()) -
            torch.eye(content_embedding.size(0))
        ) / content_embedding.numel()

        total_loss = (
                loss_contrastive +
                self.lambda_reg * regularization_loss +
                self.lambda_orthog * ortho_reg
        )
        if with_debug:
            print(f"""losses are:
              Contrastive = {loss_contrastive},
              lambda * regularization = {self.lambda_reg * regularization_loss},
              lambda * orthoginal = {self.lambda_orthog * ortho_reg},
          """)

        return total_loss


# Some Plotting
def random_cosine_similarity(content_embedding, n=10):
    similarity = torch.nn.functional.cosine_similarity(content_embedding.unsqueeze(0), content_embedding.unsqueeze(1),
                                                       dim=2)
    # Ensure that the diagonal contains -inf so it won't interfere with the upper triangular part
    similarity = similarity - 2 * torch.eye(similarity.shape[
                                                0])  # Subtracting 2 will give -2 for the diagonal, which is smaller than any possible cosine similarity (-1 to 1).

    # Get the upper triangular part without the diagonal
    indices = torch.triu_indices(row=similarity.shape[0], col=similarity.shape[1], offset=1)
    upper_triangular_part = similarity[indices[0], indices[1]]
    return upper_triangular_part[torch.randperm(upper_triangular_part.size(0))[:n]]


def plot_bins(content_embedding):
    similarity = torch.nn.functional.cosine_similarity(content_embedding.unsqueeze(0), content_embedding.unsqueeze(1),
                                                       dim=2)
    # Ensure that the diagonal contains -inf so it won't interfere with the upper triangular part
    similarity = similarity - 2 * torch.eye(similarity.shape[
                                                0])  # Subtracting 2 will give -2 for the diagonal, which is smaller than any possible cosine similarity (-1 to 1).

    # Get the upper triangular part without the diagonal
    indices = torch.triu_indices(row=similarity.shape[0], col=similarity.shape[1], offset=1)
    upper_triangular_part = similarity[indices[0], indices[1]]
    bins = 100
    x = range(-100, 100, 2)
    actual_hist = torch.histc(upper_triangular_part, bins=100, min=-1, max=1)
    actual_hist /= actual_hist.sum()  # normalized
    plt.bar(x, actual_hist, align='center')
    plt.xlabel('Bins')
    plt.title('Cosine Similarity Bins Of All Pairs Of Validation Set Content Embeddings')
    plt.show()


from torch.utils.data import Dataset, DataLoader


class EngagementDataset(Dataset):
    def __init__(self, user_features, content_features, targets):
        self.user_features = user_features
        self.content_features = content_features
        self.targets = targets

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, index):
        return self.user_features[index], self.content_features[index], self.targets[index]


batch_size = 512

loss_function = ContrastiveLoss(
    margin=1.1,  # may be changed not sure # Maybe normalize this by dimensions/batch_size?
    lambda_reg=0.01,
    lambda_orthog=0.001
)

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

targets = loss_function.calculate_targets(engagement_type_tensor, engagement_value_tensor)
engagement_data = EngagementDataset(user_features_tensor, content_features_tensor, targets)

train_data, test_data = train_test_split(engagement_data, test_size=0.2, random_state=28)

# Further split the training data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=28)

# Create DataLoader instances
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

data_loader = DataLoader(engagement_data, batch_size=batch_size, shuffle=True)

from torch.optim.lr_scheduler import StepLR

max_lr = 0.005  # upper bound
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Scheduler - One Cycle Learning Rate Policy
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=max_lr,
                                                steps_per_epoch=len(train_loader),
                                                # Assuming train_loader is your DataLoader
                                                epochs=50,  # The total number of epochs you're planning to train
                                                anneal_strategy='linear',
                                                pct_start=0.3,
                                                # The fraction of epochs increasing LR, adjust based on your needs
                                                div_factor=25.0,  # max_lr divided by this factor gives the starting LR
                                                final_div_factor=10000.0)  # max_lr divided by this factor gives the ending LR

# Training loop
from tqdm import tqdm
import matplotlib.pyplot as plt

for epoch in range(20):
    model.train()
    for batch in tqdm(train_loader, desc=f'Training {epoch + 1}: '):
        user_features_batch, content_features_batch, targets_batch = batch
        optimizer.zero_grad()
        user_embedding, content_embedding = model(user_features_batch, content_features_batch)
        loss = loss_function(user_embedding, content_embedding, targets_batch)
        loss.backward()
        optimizer.step()
        # scheduler.step() -- depending on your scheduler
    loss = loss_function(user_embedding, content_embedding, targets_batch, with_debug=True)
    print(f'Epoch {epoch + 1}, Last Batch Single Loss: {loss.item()}')

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation {epoch + 1}: '):
            user_features_batch, content_features_batch, targets_batch = batch
            user_embedding, content_embedding = model(user_features_batch, content_features_batch)
            loss = loss_function(user_embedding, content_embedding, targets_batch)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation - Epoch {epoch + 1}, Average Loss: {val_loss}')
    model.train()
    # scheduler.step(val_loss)
    print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]}')

    plot_bins(content_embedding)

    # Second validation
    print(f'Epoch {epoch + 1}, Avg Cosine-Sims: {random_cosine_similarity(content_embedding, n=10)}')

# model save address for model loading
save_path = "./twotower_model.dict"
#torch.save(model, save_path)
torch.save(model.state_dict(), save_path)
