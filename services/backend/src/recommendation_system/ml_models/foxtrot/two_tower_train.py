import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, StepLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from src.recommendation_system.ml_models.foxtrot.utils import create_training_tensors
from src.recommendation_system.ml_models.foxtrot.train_utils import (
    random_cosine_similarity,
    plot_bins,
)

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, content_dim, hidden_dim, output_dim):
        super(TwoTowerModel, self).__init__()

        self.dropout = nn.Dropout(0.5)
        # User tower
        self.user_linear1 = nn.Linear(user_dim, hidden_dim)
        self.user_relu = nn.LeakyReLU()
        self.user_batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.user_batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.user_linear2 = nn.Linear(hidden_dim, output_dim)

        # Content tower
        self.content_linear1 = nn.Linear(content_dim, hidden_dim)
        self.content_relu = nn.LeakyReLU()
        self.content_batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.content_batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.content_linear2 = nn.Linear(hidden_dim, output_dim)

        # initialization
        nn.init.kaiming_normal_(self.user_linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.content_linear1.weight, mode='fan_in', nonlinearity='relu')

    def forward_content(self, content_tensor):
        # Content tower
        content_embedding = self.content_linear1(content_tensor)
        content_embedding = self.content_relu(content_embedding)
        content_embedding = self.content_batchnorm1(content_embedding)
        content_embedding = self.dropout(content_embedding)
        content_embedding = self.content_linear2(content_embedding)

        return content_embedding

    def forward_user(self, user_tensor):
        # User tower
        user_embedding = self.user_linear1(user_tensor)
        user_embedding = self.user_relu(user_embedding)
        user_embedding = self.user_batchnorm1(user_embedding)
        user_embedding = self.dropout(user_embedding)
        user_embedding = self.user_linear2(user_embedding)

        return user_embedding

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_reg=0.01, lambda_orthog=0.001, noise_factor=0.0005):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.lambda_orthog = lambda_orthog
        self.noise_factor = noise_factor
        self.DISLIKE_ENGAGEMENT_TYPE_VALUE = 0
        self.LIKE_ENGAGEMENT_TYPE_VALUE = 1
        self.MS_ENGAGEMENT_TYPE_VALUE = 2

    def calculate_targets(self, engagement_type_vector, engagement_value_vector):
        # Conditions for 0=dislike, 1=like, and 2=milliseconds engaged
        return torch.where(
              engagement_type_vector == self.DISLIKE_ENGAGEMENT_TYPE_VALUE,
              torch.zeros_like(engagement_type_vector), # dislike
              torch.where(
                  engagement_type_vector == self.LIKE_ENGAGEMENT_TYPE_VALUE,
                  torch.ones_like(engagement_type_vector), # like
                  torch.where(
                      engagement_value_vector < 700,
                      torch.zeros_like(engagement_type_vector), # bad engagement
                      torch.where(
                        engagement_value_vector <= 2500,
                        torch.ones_like(engagement_type_vector), # engaged 500ms => 2.5s
                        torch.zeros_like(engagement_type_vector), # bad engagement
                      )
                  ) # millisecond engaged with
              )
        )

    def forward(self, user_embedding, content_embedding, targets, with_debug=False):
        user_embedding += self.noise_factor * torch.randn(*user_embedding.shape)
        content_embedding += self.noise_factor * torch.randn(*content_embedding.shape)

        cosine_sim = F.cosine_similarity(user_embedding, content_embedding, dim=1)

        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - targets) * torch.pow(cosine_sim, 2) +
            (targets)     * torch.pow(
                torch.clamp(self.margin - cosine_sim, min=0.0),
                2
            )
        )

        # Regularization terms
        reg_user = torch.norm(user_embedding, p=2)
        reg_content = torch.norm(content_embedding, p=2)
        regularization_loss = (reg_user + reg_content)

        # orthognal loss of content
        ortho_reg = torch.norm(
            torch.mm(content_embedding, content_embedding.t()) -
            torch.eye(content_embedding.size(0))
        )

        total_loss = (
            loss_contrastive +
            self.lambda_reg * regularization_loss +
            self.lambda_orthog * ortho_reg
        )
        if with_debug:
          print(f"""losses are:
              {loss_contrastive},
              {self.lambda_reg * regularization_loss},
              {self.lambda_orthog * ortho_reg},
          """)

        return total_loss

class EngagementDataset(Dataset):
    def __init__(self, user_features, content_features, targets):
        self.user_features = user_features
        self.content_features = content_features
        self.targets = targets

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, index):
        return self.user_features[index], self.content_features[index], self.targets[index]

def train(epochs=50, hidden_dim=512, batch_size=512, margin=1.0, lambda_reg=0.03, lambda_orthog=0.005, noise_factor=0.0001):
    user_features_tensor, content_features_tensor, user_columns, content_columns, engagement_type_tensor, engagement_value_tensor = create_training_tensors()

    # Specify the input and output dimensions
    user_dim = len(user_columns)
    content_dim = len(content_columns)
    output_dim = 64
    hidden_dim = hidden_dim

    print(f"user_dim: {user_dim}")
    print(f"content_dim: {content_dim}")

    # Create the model
    model = TwoTowerModel(user_dim, content_dim, hidden_dim, output_dim)

    # Batch Definition
    batch_size = batch_size
    total_dim = user_dim + content_dim
    norm_factor = total_dim/batch_size
    loss_function = ContrastiveLoss(
        margin=margin / norm_factor, # Maybe normalize this by dimensions/batch_size?
        lambda_reg=lambda_reg / norm_factor,  # Maybe normalize this by dimensions/batch_size?
        lambda_orthog=lambda_orthog / norm_factor, # Maybe normalize this by dimensions/batch_size?
        noise_factor=noise_factor
    )

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

    max_lr = 0.005   # upper bound
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Scheduler - One Cycle Learning Rate Policy
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                   max_lr=max_lr,
                                                   steps_per_epoch=len(train_loader),  # Assuming train_loader is your DataLoader
                                                   epochs=epochs,  # The total number of epochs you're planning to train
                                                   anneal_strategy='linear',
                                                   pct_start=0.3,  # The fraction of epochs increasing LR, adjust based on your needs
                                                   div_factor=30.0,  # max_lr divided by this factor gives the starting LR
                                                   final_div_factor=10000.0)  # max_lr divided by this factor gives the ending LR

    # Directory to save the best models
    save_dir = "../recommendation_system/ml_models/foxtrot/best_models"
    os.makedirs(save_dir, exist_ok=True)

    # List to keep track of the best models
    best_models = []


    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Training {epoch + 1}: '):
            user_features_batch, content_features_batch, targets_batch = batch
            optimizer.zero_grad()
            #user_embedding, content_embedding = model(user_features_batch, content_features_batch)
            user_embedding = model.forward_user(user_features_batch)
            content_embedding = model.forward_content(content_features_batch)
            loss = loss_function(user_embedding, content_embedding, targets_batch)
            loss.backward()
            optimizer.step()
            #scheduler.step() #-- depending on your scheduler
        loss = loss_function(user_embedding, content_embedding, targets_batch, with_debug=True)
        print(f"user_features_batch: {user_features_batch.shape}")
        print(f"content_features_batch: {content_features_batch.shape}")
        print(f"user_embedding: {user_embedding.shape}")
        print(f"content_embedding: {content_embedding.shape}")
        print(f'Epoch {epoch+1}, Last Batch Single Loss: {loss.item()}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation {epoch + 1}: '):
                user_features_batch, content_features_batch, targets_batch = batch
                #user_embedding, content_embedding = model(user_features_batch, content_features_batch)
                user_embedding = model.forward_user(user_features_batch)
                content_embedding = model.forward_content(content_features_batch)
                loss = loss_function(user_embedding, content_embedding, targets_batch)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f'Validation - Epoch {epoch+1}, Loss: {val_loss}')

        cosine_similarity_list = random_cosine_similarity(content_embedding, n=20).tolist()

        # Save the best 5 models
        save = True
        for i in cosine_similarity_list:
            if i < 0.5:
                save = False
        if save:
            best_models.append({'epoch': epoch, 'val_loss': val_loss, 'model_state': model.state_dict()})
            best_models.sort(key=lambda x: x['val_loss'])

        # Keep only the top 5 models and delete the rest
        while len(best_models) > 10:
            removed_model = best_models.pop()
            file_path = os.path.join(save_dir, f"model_val_loss_{removed_model['val_loss']}.pth")
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save the current best models
        for best_model in best_models:
            torch.save(best_model['model_state'], os.path.join(save_dir, f"model_val_loss_{best_model['val_loss']}.pth"))

        model.train()
        #scheduler.step(val_loss)
        print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]}')

        plot_bins(content_embedding)

        # Second validation
        print(f'Epoch {epoch+1}, Avg Cosine-Sims: {random_cosine_similarity(content_embedding, n=20)}')
