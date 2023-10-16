import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import pickle
import random
from two_tower import TwoTowerModel, ContrastiveLoss
from engagement_dataset import EngagementDataset

# Load data and preprocess (use your preprocessing code here)
# ...
df = pd.read_csv("C:/Users/127ri/Columbia-E4579/data/columbia_data.tsv", sep="\t")
with open("C:/Users/127ri/Columbia-E4579/data/id_to_pickle_dict.pkl", "rb") as f:
    prompt_to_embedding = pickle.load(f)
prompt_embedding = pd.DataFrame(
    prompt_to_embedding.items(), columns=["content_id", "prompt_embedding"]
)
df = df.merge(prompt_embedding, on="content_id")
del prompt_embedding


# Configuration options
TOP_ARTIST_STYLES = 30
TOP_SOURCES = 30
TOP_SEEDS = 14
TOP_CONTENT = 251
PROMPT_EMBEDDING_LENGTH = 512

# Get the top artist styles, sources, and seeds
top_artist_styles = (
    df["artist_style"].value_counts().nlargest(TOP_ARTIST_STYLES).index.tolist()
)
top_sources = df["source"].value_counts().nlargest(TOP_SOURCES).index.tolist()
top_seeds = df["seed"].value_counts().nlargest(TOP_SEEDS).index.tolist()

# Replace less frequent artist styles, sources, and seeds with 'other'
df["artist_style"] = df["artist_style"].apply(
    lambda x: x if x in top_artist_styles else "other"
)
df["source"] = df["source"].apply(lambda x: x if x in top_sources else "other")
df["seed"] = df["seed"].apply(lambda x: str(x) if x in top_seeds else "other")

# One-hot encode categorical features
encoder = OneHotEncoder()
content_onehot = encoder.fit_transform(
    df[["artist_style", "model_version", "seed", "source"]]
)
content_onehot_df = pd.DataFrame(
    content_onehot.toarray(),
    columns=encoder.get_feature_names_out(
        ["artist_style", "model_version", "seed", "source"]
    ),
)
df = pd.concat([df, content_onehot_df], axis=1)

# Normalizing linear features
scaler = StandardScaler()
df[["guidance_scale", "num_inference_steps"]] = scaler.fit_transform(
    df[["guidance_scale", "num_inference_steps"]]
)

# Compute top N content pieces based on engagement_value
from collections import defaultdict

top_n_content = (
    df.groupby("content_id")["engagement_value"]
    .count()
    .nlargest(TOP_CONTENT)
    .index.tolist()
)
user_vector_dict = defaultdict(
    lambda: {
        "millisecond_engaged_vector": np.zeros(len(top_n_content)),
        "like_vector": np.zeros(len(top_n_content)),
        "dislike_vector": np.zeros(len(top_n_content)),
    }
)


# Initialize vectors for each user
def aggregate_engagement(group):
    # Summing millisecond engagement values
    millisecond_engagement_sum = group.loc[
        group["engagement_type"] != "Like", "engagement_value"
    ].sum()

    # Counting likes and dislikes
    likes_count = group.loc[
        (group["engagement_type"] == "Like") & (group["engagement_value"] == 1)
    ].shape[0]
    dislikes_count = group.loc[
        (group["engagement_type"] == "Like") & (group["engagement_value"] == -1)
    ].shape[0]

    return pd.Series(
        {
            "millisecond_engagement_sum": millisecond_engagement_sum,
            "likes_count": likes_count,
            "dislikes_count": dislikes_count,
        }
    )


# Group by user_id and content_id, then apply the function
engagement_aggregate = (
    df[df["content_id"].isin(top_n_content)]
    .groupby(["user_id", "content_id"])
    .apply(aggregate_engagement)
    .reset_index()
)

# Now, populate your user_vector_dict
for _, row in engagement_aggregate.iterrows():
    user_id = row["user_id"]
    content_id = row["content_id"]
    idx = top_n_content.index(content_id)

    user_vector_dict[user_id]["millisecond_engaged_vector"][idx] = row[
        "millisecond_engagement_sum"
    ]
    user_vector_dict[user_id]["like_vector"][idx] = row["likes_count"]
    user_vector_dict[user_id]["dislike_vector"][idx] = row["dislikes_count"]

# Convert to DataFrame
user_vector_df = pd.DataFrame.from_dict(user_vector_dict, orient="index")
del user_vector_dict

# Unpack vector columns into individual columns
millisecond_columns = [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
like_columns = [f"like_vector_{i}" for i in range(TOP_CONTENT)]
dislike_columns = [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]

user_vector_df[millisecond_columns] = pd.DataFrame(
    user_vector_df["millisecond_engaged_vector"].tolist(), index=user_vector_df.index
)
user_vector_df[like_columns] = pd.DataFrame(
    user_vector_df["like_vector"].tolist(), index=user_vector_df.index
)
user_vector_df[dislike_columns] = pd.DataFrame(
    user_vector_df["dislike_vector"].tolist(), index=user_vector_df.index
)

# Drop the original vector columns
user_vector_df.drop(
    ["millisecond_engaged_vector", "like_vector", "dislike_vector"],
    axis=1,
    inplace=True,
)

# Join User Vector To Df
df = df.merge(
    user_vector_df.reset_index().rename(columns={"index": "user_id"}), on="user_id"
)
del user_vector_df

# Unpack prompt embedding
prompt_columns = [f"prompt_embedding_{i}" for i in range(PROMPT_EMBEDDING_LENGTH)]
df[prompt_columns] = pd.DataFrame(df["prompt_embedding"].tolist(), index=df.index)
df = df.drop("prompt_embedding", axis=1)

# Create overall features
user_columns = (
    [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
    + [f"like_vector_{i}" for i in range(TOP_CONTENT)]
    + [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]
)
user_features = df[user_columns]
user_features_tensor = torch.FloatTensor(user_features.values)
del user_features

content_columns = (
    list(filter(lambda x: "artist_style_" in x, df.columns))
    + list(filter(lambda x: "model_version_" in x, df.columns))
    + list(filter(lambda x: "source_" in x, df.columns))
    + list(filter(lambda x: "seed_" in x, df.columns))
    + list(filter(lambda x: "prompt_embedding_" in x, df.columns))
    + ["content_id", "guidance_scale", "num_inference_steps"]
)
content_features = df[content_columns]
content_features_tensor = torch.FloatTensor(content_features.values)
del content_features

DISLIKE_ENGAGEMENT_TYPE_VALUE = 0
LIKE_ENGAGEMENT_TYPE_VALUE = 1
MS_ENGAGEMENT_TYPE_VALUE = 2

engagement_type_value = df[["engagement_type", "engagement_value"]]
engagement_type_value["engagement_type"] = engagement_type_value.apply(
    lambda x: (
        LIKE_ENGAGEMENT_TYPE_VALUE
        if x["engagement_type"] == "Like" and x["engagement_value"] == 1
        else DISLIKE_ENGAGEMENT_TYPE_VALUE  # Like
        if x["engagement_type"] == "Like" and x["engagement_value"] == -1
        else MS_ENGAGEMENT_TYPE_VALUE  # Dislike  # MsEngagementType
    ),
    axis=1,
)
engagement_type_tensor = torch.FloatTensor(
    engagement_type_value[["engagement_type"]].values
)
engagement_value_tensor = torch.FloatTensor(
    engagement_type_value[["engagement_value"]].values
)
del engagement_type_value
del df


# Specify the input and output dimensions
user_dim = len(user_columns)
content_dim = len(content_columns)
output_dim = 64

# Create the model
model = TwoTowerModel(user_dim, content_dim, output_dim)

# Specify the loss function
loss_function = ContrastiveLoss(margin=1.0, lambda_reg=0.02, lambda_orthog=0.01)

# Calculate targets
targets = loss_function.calculate_targets(
    engagement_type_tensor, engagement_value_tensor
)

# Create dataset and DataLoader instances
engagement_data = EngagementDataset(
    user_features_tensor, content_features_tensor, targets
)
train_data, test_data = train_test_split(
    engagement_data, test_size=0.2, random_state=28
)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=28)

batch_size = 512
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Create optimizer and scheduler
optimizer = optim.Adam(
    model.parameters(),
    lr=0.000000001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.00001,
    steps_per_epoch=len(train_loader),
    epochs=50,
    anneal_strategy="linear",
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=10000.0,
)

pos_weight = torch.tensor(2.0)  # Example: Positives are twice as important
neg_weight = torch.tensor(1.0)

# Training loop
for epoch in range(20):
    model.train()
    for batch in train_loader:
        user_features_batch, content_features_batch, targets_batch = batch
        optimizer.zero_grad()
        user_embedding, content_embedding = model(
            user_features_batch, content_features_batch
        )
        loss = loss_function(
            user_embedding, content_embedding, targets_batch, pos_weight, neg_weight
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            user_features_batch, content_features_batch, targets_batch = batch
            user_embedding, content_embedding = model(
                user_features_batch, content_features_batch
            )
            loss = loss_function(
                user_embedding, content_embedding, targets_batch, pos_weight, neg_weight
            )
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation - Epoch {epoch+1}, Loss: {val_loss}")

    # Save the model after each epoch

# Save the final trained model
torch.save(
    model.state_dict(),
    "C:/Users/127ri/Columbia-E4579/services/backend/src/recommendation_system/ml_models/alpha/two_tower_trained_saved.pth",
)
print("Final trained model saved.")
