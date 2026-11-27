import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset, random_split
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, transforms
import itertools
import argparse
import os
import re
import json
import scipy.io.arff as arff
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Argument parser
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--csv', type=str, required=True, help='Path to the dataset (csv / arff / data)')
parser.add_argument('--save_dir', type=str, required=False, default=None, help='Path to save the final model')
# ---- added for the MV experiments ----
parser.add_argument('--mv_level', type=str, default=None, help='e.g. mr10 (default: taken from the path)')
parser.add_argument('--output_root', type=str, default="/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/table2image",
                    help='Root for the per-dataset result json')
parser.add_argument('--image_data_root', type=str, default="/home/gkianfar/scratch/Amin/ICC/Unzippeddata/Image",
                    help='Folder with FashionMNIST and MNIST (already downloaded)')
args = parser.parse_args()

# Parameters
EPOCH = 50
BATCH_SIZE = 64

csv_path = args.csv
file_name = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))   # dataset folder name

# MV level (mr00..mr50) from the path, so adult@mr10 and adult@mr30 never share names
_mv = args.mv_level
if _mv is None:
    for _part in reversed(os.path.abspath(csv_path).split(os.sep)[:-1]):
        if re.match(r'^mr\d\d$', _part):
            _mv = _part
            break
if _mv:
    file_name = f"{file_name}_{_mv}"          # e.g. adult_mr10
saving_path = (args.save_dir + '.pt') if args.save_dir else None

# Load and preprocess the tabular data
def load_dataset(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.arff':
        try:
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            return df
        except Exception:
            import arff as arff_lib            # sparse ARFF: scipy fails, liac-arff reads it
            with open(file_path, 'r') as f:
                dataset = arff_lib.load(f)
            return pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    elif file_ext == '.data':
        for sep in [',', ' ', '\t', ';']:
            try:
                d = pd.read_csv(file_path, sep=sep, header=None)
                if d.shape[1] > 1:
                    return d
            except Exception:
                continue
        raise Exception("Could not determine delimiter for .data file")
    raise ValueError(f"Unsupported file format: {file_ext}")

df = load_dataset(csv_path)
df = df.replace(['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-'], np.nan)   # missing markers
target_col_candidates = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival']
target_col = next((col for col in df.columns if str(col).lower() in target_col_candidates), None)
if target_col is None:
    target_col = df.columns[-1]

# rare classes (<10 samples) removed, >20 classes skipped  (same rules as main_mv.py)
_vc = df[target_col].value_counts()
_valid = _vc[_vc >= 10].index.tolist()
if len(_valid) < 2:
    print("[ERROR] insufficient samples per class. Skipping...")
    exit(1)
df = df[df[target_col].isin(_valid)]
if df[target_col].nunique() > 20:
    print(f"[ERROR] Dataset has {df[target_col].nunique()} classes (>20). Skipping...")
    exit(1)
# (no dropping of feature columns with >50% missing values)

y = df[target_col].values
X_df = df.drop(columns=[target_col])
for col in X_df.columns:
    if not pd.api.types.is_numeric_dtype(X_df[col]):
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    else:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
X = SimpleImputer(strategy='median').fit_transform(X_df)      # median imputation, as in main_mv.py
y = LabelEncoder().fit_transform(pd.Series(y).astype(str)) if not pd.api.types.is_numeric_dtype(pd.Series(y)) else y.astype(int)

# Mapping labels for classes
unique_values = sorted(set(y))
value_map = {unique_values[i]: i for i in range(len(unique_values))}
y = [value_map[val] for val in y]
y = np.array(y)

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FashionMNIST
fashionmnist_dataset = datasets.FashionMNIST(
    root=args.image_data_root,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

# Load MNIST
mnist_dataset = datasets.MNIST(
    root=args.image_data_root,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

# Target + 10 (MNIST)
class ModifiedLabelDataset(Dataset):
    def __init__(self, dataset, label_offset=10):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset

modified_mnist_dataset = ModifiedLabelDataset(mnist_dataset, label_offset=10)

# # Combine labels for FashionMNIST (0~9) and MNIST (10~19)
# selected_labels_fashion = list(range(10))
# selected_labels_mnist = list(range(10, 20))

# # Filter FashionMNIST dataset
# fashion_indices = [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in selected_labels_fashion]
# filtered_fashion = Subset(fashionmnist_dataset, fashion_indices)

# # Filter MNIST dataset and remap labels
# mnist_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label in range(10)]
# filtered_mnist = Subset(mnist_dataset, mnist_indices)



# Normalize tabular features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split tabular data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create TensorDatasets
train_tabular_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_tabular_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# Calculate number of samples needed for each label
train_tabular_label_counts = torch.bincount(train_tabular_dataset.tensors[1], minlength=int(len(unique_values)))
test_tabular_label_counts = torch.bincount(test_tabular_dataset.tensors[1], minlength=int(len(unique_values)))

num_samples_needed = train_tabular_label_counts.tolist()
num_samples_needed_test = test_tabular_label_counts.tolist()


valid_labels = {i for i in range(len(unique_values))}  # Only keep the labels present in unique_values

# Filter FashionMNIST dataset
filtered_fashion = Subset(fashionmnist_dataset, 
                          [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in valid_labels])


# Filter MNIST dataset and remap labels
filtered_mnist = Subset(modified_mnist_dataset, 
                        [i for i, (_, label) in enumerate(modified_mnist_dataset) if label in valid_labels])



# Combine FashionMNIST and MNIST
combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

# Integrity check
indices_by_label = {label: [] for label in range(int(len(unique_values)))}

for i, (_, label) in enumerate(combined_dataset):
    if label not in indices_by_label:
        print(f"Unexpected label {label} at index {i}")  # If anything goes wrong...
    indices_by_label[label].append(i)


# Generate repeated indices for balanced dataset
repeated_indices = {
    label: list(itertools.islice(itertools.cycle(indices_by_label[label]),
                                 num_samples_needed[label] + num_samples_needed_test[label]))
    for label in indices_by_label
}



# Align the train and test indices for both tabular and image datasets by label
aligned_train_indices = []
aligned_test_indices = []

for label in valid_labels:
    # Matching indices by label for tabular and image data
    train_tab_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
    test_tab_indices = [i for i, lbl in enumerate(y_test) if lbl == label]

    train_img_indices = repeated_indices[label][:num_samples_needed[label]]
    test_img_indices = repeated_indices[label][num_samples_needed[label]:num_samples_needed[label] + num_samples_needed_test[label]]

    # Ensure alignment of indices
    if len(train_tab_indices) == len(train_img_indices) and len(test_tab_indices) == len(test_img_indices):
        aligned_train_indices.extend(list(zip(train_tab_indices, train_img_indices)))
        aligned_test_indices.extend(list(zip(test_tab_indices, test_img_indices)))
    else:
        raise ValueError(f"Mismatch in train/test counts for label {label}")

# Create final filtered subsets with aligned indices
train_filtered_tab_set = Subset(train_tabular_dataset, [idx[0] for idx in aligned_train_indices])
train_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_train_indices])

test_filtered_tab_set = Subset(test_tabular_dataset, [idx[0] for idx in aligned_test_indices])
test_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_test_indices])

# Define synchronized dataset class with consistency check
class SynchronizedDataset(Dataset):
    def __init__(self, tabular_dataset, image_dataset):
        self.tabular_dataset = tabular_dataset
        self.image_dataset = image_dataset
        assert len(self.tabular_dataset) == len(self.image_dataset), "Datasets must have the same length."

    def __len__(self):
        return len(self.tabular_dataset)

    def __getitem__(self, index):
        tab_data, tab_label = self.tabular_dataset[index]
        img_data, img_label = self.image_dataset[index]
        assert tab_label == img_label, f"Label mismatch: tab_label={tab_label}, img_label={img_label}"
        return tab_data, tab_label, img_data, img_label

# Create synchronized datasets
train_synchronized_dataset = SynchronizedDataset(train_filtered_tab_set, train_filtered_img_set)
test_synchronized_dataset = SynchronizedDataset(test_filtered_tab_set, test_filtered_img_set)

# Create data loaders
train_synchronized_loader = DataLoader(dataset=train_synchronized_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_synchronized_loader = DataLoader(dataset=test_synchronized_dataset, batch_size=BATCH_SIZE)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# d_embedding = 24
# m_cont_embeddings = PeriodicEmbeddings(n_cont_features, lite=False)

# Compute bins
# Using quantile-based bins
# quantile_bins = compute_bins(X_train_tabular_tensor)

# # Using target-aware tree-based bins
# tree_bins = compute_bins(
#     X_train_tabular_tensor,
#     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
#     y=y_train_tabular_tensor,
#     regression=True,
# )

# Define MLP-Q / MLP-T model
mlp_config = {
    'd_out': tab_latent_size,  # For example, a single regression task.
    'n_blocks': 2,
    'd_block': 256,
    'dropout': 0.1,
}

# model_with_embeddings = nn.Sequential(
#     m_cont_embeddings,
#     nn.Flatten(),
#     MLP(d_in=n_cont_features * d_embedding, **mlp_config)
# )

class SimpleMLP(nn.Module):
    def __init__(self, tab_latent_size = tab_latent_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)  # Input layer to hidden layer (12 neurons)
        self.fc2 = nn.Linear(tab_latent_size, int(len(unique_values)))  # Hidden layer to output layer
        self.relu = nn.ReLU()        # ReLU activation function

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, torch.sigmoid(x)  # Sigmoid activation for binary classification

model_with_embeddings = SimpleMLP(tab_latent_size)

# mlp_q_model = nn.Sequential(
#     PiecewiseLinearEncoding(quantile_bins),
#     nn.Flatten(),
#     MLP(d_in=sum(len(b) - 1 for b in quantile_bins), **mlp_config)
# )

# mlp_t_model = nn.Sequential(
#     PiecewiseLinearEncoding(tree_bins),
#     nn.Flatten(),
#     MLP(d_in=sum(len(b) - 1 for b in tree_bins), **mlp_config)
# )

# # Define MLP-QLR / MLP-TLR model
# mlp_qlr_model = nn.Sequential(
#     PiecewiseLinearEmbeddings(quantile_bins, d_embedding, activation=True),
#     nn.Flatten(),
#     MLP(d_in=n_cont_features * d_embedding, **mlp_config)
# )

# mlp_tlr_model = nn.Sequential(
#     PiecewiseLinearEmbeddings(tree_bins, d_embedding, activation=True),
#     nn.Flatten(),
#     MLP(d_in=n_cont_features * d_embedding, **mlp_config)
# )


class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, tab_latent_size=8, latent_size=8):
        super(CVAEWithTabEmbedding, self).__init__()
        
        self.mlp = model_with_embeddings
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size, 128),
#             nn.Linear(tab_latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        
#         self.tab_classifier = nn.Linear(tab_latent_size, int(len(unique_values)))
        self.final_classifier = SimpleCNN(num_classes=int(len(unique_values)))  # Custom ResNet-based classifier for images

    def encode(self, x, tab_embedding):
        return self.encoder(torch.cat([x, tab_embedding], dim=1))
#         return self.encoder(tab_embedding)
    
    def decode(self, z, tab_embedding):
        return self.decoder(torch.cat([z, tab_embedding], dim=1))
    
    def forward(self, x, tab_data):
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding)
        recon_x = self.decode(z, tab_embedding)
#         tab_pred = tab_embedding
#         tab_pred = self.tab_classifier(tab_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))  # Reshape recon_x for CNN input
        return recon_x, tab_pred, img_pred


cvae = CVAEWithTabEmbedding(tab_latent_size).to(DEVICE)
optimizer = optim.AdamW(cvae.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR by a factor of 0.1 every 10 epochs



def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    BCE = F.mse_loss(recon_x, x)  # Reconstruction loss
    tab_loss = F.binary_cross_entropy_with_logits(tab_pred, tab_labels)  # Use BCE for multi-label target
    img_loss = F.binary_cross_entropy_with_logits(img_pred, img_labels)
    return BCE + tab_loss + img_loss


def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    
    num_classes=int(len(unique_values))
    
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        img_label = img_label.to(DEVICE)
        tab_label = tab_label.to(DEVICE)
        
        optimizer.zero_grad()
        
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)
        tab_pred = tab_pred.squeeze(-1)
        img_pred = img_pred.squeeze(-1)
        
        tab_label = F.one_hot(tab_label.long(), num_classes=num_classes).float()
        img_label = F.one_hot(img_label.long(), num_classes=num_classes).float()

        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
#     print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_data_loader):.4f}')
#     scheduler.step()

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def test(model, test_data_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path='best_model.pth'):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    num_classes = int(len(unique_values))  # Adjust this for your actual number of classes
    correct_tab = {i: 0 for i in range(num_classes)}
    total_tab = {i: 0 for i in range(num_classes)}
    correct_img = {i: 0 for i in range(num_classes)}
    total_img = {i: 0 for i in range(num_classes)}
    
    all_tab_labels = []
    all_tab_preds = []
    all_img_labels = []
    all_img_preds = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE)
            tab_label = tab_label.to(DEVICE)
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            
            # Squeeze the last dimension if it's of size 1 (e.g., for binary classification)
            tab_pred = tab_pred.squeeze(-1)
            img_pred = img_pred.squeeze(-1)

            # Convert labels to float (not long, as per your request)
            tab_label = F.one_hot(tab_label.long(), num_classes=num_classes).float()
            img_label = F.one_hot(img_label.long(), num_classes=num_classes).float()
            
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            # Store predictions and labels for AUC calculation
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_pred.cpu().numpy())  # Probabilities for each class
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_pred.cpu().numpy())  # Probabilities for each class
            
            

            # Handle binary or multi-class cases
            if tab_pred.dim() == 1:  # Binary case (single output per sample)
                tab_predicted = (tab_pred > 0.5).long()  # Use a threshold for binary classification
            else:  # Multi-class case (probabilities for each class)
                tab_predicted = torch.argmax(tab_pred, dim=1)
            
            for i in range(len(tab_label)):
                label = torch.argmax(tab_label[i]).item()  # Convert one-hot to class index
                correct_tab[label] += (tab_predicted[i] == label).item()
                total_tab[label] += 1

            # Calculate accuracy for image classification
            if img_pred.dim() == 1:  # Binary classification for images
                img_predicted = (img_pred > 0.5).long()  # Use a threshold for binary classification
            else:  # Multi-class case (probabilities for each class)
                img_predicted = torch.argmax(img_pred, dim=1)
            
            for i in range(len(img_label)):
                label = torch.argmax(img_label[i]).item()  # Convert one-hot to class index
                correct_img[label] += (img_predicted[i] == label).item()
                total_img[label] += 1
                
            tab_label_indices = torch.argmax(tab_label, dim=1)  # Convert one-hot encoded labels back to indices
            correct_tab_total += (tab_predicted == tab_label_indices).sum().item()

            img_label_indices = torch.argmax(img_label, dim=1)
            correct_img_total += (img_predicted == img_label_indices).sum().item()

            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total
    tab_accuracy = {cls: (correct_tab[cls] / total_tab[cls]) * 100 if total_tab[cls] > 0 else 0 for cls in range(num_classes)}
    img_accuracy = {cls: (correct_img[cls] / total_img[cls]) * 100 if total_img[cls] > 0 else 0 for cls in range(num_classes)}

    # Calculate AUC for tabular and image classification
#     tab_auc = roc_auc_score(all_tab_labels, all_tab_preds, multi_class="ovr", average="macro")
#     print(all_img_labels)
#     print(all_img_preds)
    
    all_img_preds = F.softmax(torch.tensor(all_img_preds), dim=1)
    all_img_preds = np.array(torch.argmax(all_img_preds, dim=1)).reshape(-1, 1)
#     print(all_img_preds)
    img_auc = roc_auc_score(all_img_labels, all_img_preds, multi_class="ovr", average="macro")

    # Save the best model based on image classification accuracy
    if img_accuracy_total > best_accuracy:
        best_accuracy = img_accuracy_total
        best_epoch = epoch
#         torch.save(model.state_dict(), best_model_path)
        
    # Compare AUC after calculation
    if img_auc > best_auc:
        best_auc = img_auc

    return best_accuracy, best_auc, best_epoch


best_accuracy = 0
best_auc = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train(cvae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_auc, best_epoch = test(cvae, test_synchronized_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path=saving_path)

print(f'Best model image classification accuracy: {best_accuracy:.4f} at epoch: {best_epoch}, Best AUC: {best_auc:.4f}')

# ---- added for the MV experiments: save the result of this run ----
os.makedirs(os.path.join(args.output_root, 'per_dataset'), exist_ok=True)
with open(os.path.join(args.output_root, 'per_dataset', f'{file_name}.json'), 'w') as f:
    json.dump({'run_tag': file_name, 'model': 'table2image', 'mv_level': _mv, 'num_samples': int(X.shape[0]),
               'num_features': int(X.shape[1]), 'num_classes': int(len(unique_values)),
               'best_accuracy': float(best_accuracy), 'best_auc': float(best_auc), 'best_epoch': int(best_epoch)}, f, indent=2)
