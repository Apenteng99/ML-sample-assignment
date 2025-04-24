import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from glob import glob
import time
import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# Helper Functions
def str2bool(v):
    return v.lower() in ('true')

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]

# Model Definitions
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network with reduced complexity."""
    def __init__(self, conv_dim=32, c_dim=5, repeat_num=3):  # Reduced conv_dim and repeat_num
        super(Generator, self).__init__()
        layers = []
        # Initial convolution with reflection padding
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=0, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers with reflection padding
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=0, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers with attention
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            if i == repeat_num // 2:  # Add attention layer in the middle
                layers.append(SelfAttention(curr_dim))

        # Up-sampling layers with reflection padding
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Final convolution with reflection padding
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SelfAttention(nn.Module):
    """Self-attention layer for feature maps."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Compute query, key, value
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Compute attention scores
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return out + x  # Residual connection

class Discriminator(nn.Module):
    """Discriminator network with reduced complexity."""
    def __init__(self, image_size=128, conv_dim=32, c_dim=5, repeat_num=3):  # Reduced conv_dim and repeat_num
        super(Discriminator, self).__init__()
        layers = []
        
        # Initial convolution with reflection padding
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=0))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Dropout2d(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=0))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout2d(0.2))
            curr_dim = curr_dim * 2

        # Add attention layer
        layers.append(SelfAttention(curr_dim))

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        
        # Final convolution with reflection padding
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return h, out_src

# Data Loader
class MedicalData(data.Dataset):
    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

    def preprocess(self):
        if self.mode == 'train':
            self.datasetA = glob(os.path.join(self.image_dir, 'train', 'pos', '*png')) + glob(os.path.join(self.image_dir, 'train', 'neg_mixed', '*png'))
            self.datasetB = glob(os.path.join(self.image_dir, 'train', 'neg', '*png'))
        else:
            self.datasetA = glob(os.path.join(self.image_dir, 'test', 'pos', '*png'))
            self.datasetB = glob(os.path.join(self.image_dir, 'test', 'neg', '*png'))

    def __getitem__(self, index):
        filenameA = self.datasetA[index % len(self.datasetA)]
        filenameB = self.datasetB[index % len(self.datasetB)]
        imageA = Image.open(filenameA).convert("RGB")
        imageB = Image.open(filenameB).convert("RGB")
        return self.transform(imageA), self.transform(imageB)

    def __len__(self):
        return max(len(self.datasetA), len(self.datasetB))

def get_loader(image_dir, image_size=128, batch_size=16, dataset='MedicalData', mode='train', num_workers=1):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.RandomRotation(10))  # Add slight rotation
        transform.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))  # Add color augmentation
        transform.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1)))  # Add slight translation
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = MedicalData(image_dir, transform, mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)
    return data_loader

# Solver
class Solver(object):
    """Solver for training and testing Brainomaly."""
    def __init__(self, data_loader, config):
        self.config = config
        self.data_loader = data_loader
        self.device = torch.device('cpu')  # Force CPU usage
        self.build_model()
        self.best_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print(f'Loading the best trained models...')
        G_path = os.path.join(self.config.model_save_dir, 'best-G.ckpt')
        D_path = os.path.join(self.config.model_save_dir, 'best-D.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=self.device))
        self.D.load_state_dict(torch.load(D_path, map_location=self.device))

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.config.g_conv_dim, 0, self.config.g_repeat_num).to(self.device)
        self.D = Discriminator(self.config.image_size, self.config.d_conv_dim, 0, self.config.d_repeat_num).to(self.device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        
        # Add learning rate schedulers
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    def validate(self):
        """Validate the model on a subset of the data."""
        self.G.eval()
        self.D.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                if i >= 10:  # Use only 10 batches for validation
                    break
                x_realA = x_realA.to(self.device)
                x_realB = x_realB.to(self.device)

                # Compute validation loss
                mask = self.G(x_realA)
                x_fakeB = torch.tanh(x_realA + mask)
                _, out_src = self.D(x_fakeB)
                loss = -torch.mean(out_src)
                total_loss += loss.item()

        self.G.train()
        self.D.train()
        return total_loss / 10

    def train(self):
        """Train the GAN model with early stopping."""
        data_iter = iter(self.data_loader)
        start_time = time.time()
        for i in range(self.config.num_iters):
            try:
                x_realA, x_realB = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_realA, x_realB = next(data_iter)

            x_realA = x_realA.to(self.device)
            x_realB = x_realB.to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()
            _, out_src = self.D(x_realB)
            d_loss_real = -torch.mean(out_src)

            mask = self.G(x_realA)
            x_fakeB = torch.tanh(x_realA + mask)
            _, out_src2 = self.D(x_fakeB.detach())
            d_loss_fake = torch.mean(out_src2)

            alpha = torch.rand(x_realB.size(0), 1, 1, 1).to(self.device)
            x_hat2 = (alpha * x_realB.data + (1 - alpha) * x_fakeB.data).requires_grad_(True)
            _, out_src2 = self.D(x_hat2)
            d_loss_gp = self.gradient_penalty(out_src2, x_hat2)

            # Add feature matching loss
            _, real_features = self.D(x_realB)
            _, fake_features = self.D(x_fakeB.detach())
            d_loss_feat = F.mse_loss(real_features, fake_features)

            d_loss = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp + 0.1 * d_loss_feat
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            if (i + 1) % self.config.n_critic == 0:
                self.g_optimizer.zero_grad()
                maskOT = self.G(x_realA)
                x_fakeB2 = torch.tanh(x_realA + maskOT)
                _, out_src2 = self.D(x_fakeB2)
                g_loss_fake = -torch.mean(out_src2)

                maskOO = self.G(x_realB)
                x_fakeB3 = torch.tanh(x_realB + maskOO)
                g_loss_id = torch.mean(torch.abs(x_realB - x_fakeB3))

                # Add perceptual loss
                _, real_features = self.D(x_realB)
                _, fake_features = self.D(x_fakeB2)
                g_loss_perceptual = F.mse_loss(real_features, fake_features)

                # Add cycle consistency loss
                mask_cycle = self.G(x_fakeB2)
                x_cycle = torch.tanh(x_fakeB2 + mask_cycle)
                g_loss_cycle = torch.mean(torch.abs(x_realA - x_cycle))

                g_loss = g_loss_fake + self.config.lambda_id * g_loss_id + 0.1 * g_loss_perceptual + 10.0 * g_loss_cycle
                g_loss.backward()
                self.g_optimizer.step()

            # Validation and early stopping
            if (i + 1) % self.config.log_step == 0:
                val_loss = self.validate()
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    G_path = os.path.join(self.config.model_save_dir, 'best-G.ckpt')
                    D_path = os.path.join(self.config.model_save_dir, 'best-D.ckpt')
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f'Early stopping at iteration {i + 1}')
                        break
                
                # Update learning rates based on validation loss
                self.g_scheduler.step(val_loss)
                self.d_scheduler.step(val_loss)

                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = f"Elapsed [{et}], Iteration [{i + 1}/{self.config.num_iters}], Val Loss: {val_loss:.4f}"
                print(log)

            if (i + 1) % self.config.model_save_step == 0:
                G_path = os.path.join(self.config.model_save_dir, f'{i + 1}-G.ckpt')
                D_path = os.path.join(self.config.model_save_dir, f'{i + 1}-D.ckpt')
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print(f'Saved model checkpoints into {self.config.model_save_dir}...')

    def gradient_penalty(self, y, x):
        """Compute gradient penalty."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1)**2)

    def testAUCInductive(self):
        """Test the model in inductive mode."""
        self.restore_model(self.config.test_iters)
        self.G.eval()
        self.D.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                x_realA = x_realA.to(self.device)
                x_realB = x_realB.to(self.device)
                
                # Generate anomaly scores
                mask = self.G(x_realA)
                x_fakeB = torch.tanh(x_realA + mask)
                _, out_src = self.D(x_fakeB)
                scores = -out_src.cpu().numpy().flatten()
                
                # Collect scores and labels
                all_scores.extend(scores)
                all_labels.extend([1] * len(scores))  # 1 for positive (anomaly) samples
                
                # Also get scores for negative samples
                mask = self.G(x_realB)
                x_fakeB = torch.tanh(x_realB + mask)
                _, out_src = self.D(x_fakeB)
                scores = -out_src.cpu().numpy().flatten()
                
                all_scores.extend(scores)
                all_labels.extend([0] * len(scores))  # 0 for negative (normal) samples
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_score = auc(fpr, tpr)
        
        print(f'Inductive AUC: {auc_score:.4f}')
        
        # Save results
        results = pd.DataFrame({
            'score': all_scores,
            'label': all_labels
        })
        results.to_csv(os.path.join(self.config.result_dir, 'inductive_results.csv'), index=False)

    def testAUCTransductive(self):
        """Test the model in transductive mode."""
        self.restore_model(self.config.test_iters)
        self.G.eval()
        self.D.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                x_realA = x_realA.to(self.device)
                x_realB = x_realB.to(self.device)
                
                # Generate anomaly scores for both domains
                mask_A = self.G(x_realA)
                x_fakeA = torch.tanh(x_realA + mask_A)
                _, out_src_A = self.D(x_fakeA)
                scores_A = -out_src_A.cpu().numpy().flatten()
                
                mask_B = self.G(x_realB)
                x_fakeB = torch.tanh(x_realB + mask_B)
                _, out_src_B = self.D(x_fakeB)
                scores_B = -out_src_B.cpu().numpy().flatten()
                
                # Combine scores from both domains
                all_scores.extend(scores_A)
                all_scores.extend(scores_B)
                all_labels.extend([1] * len(scores_A))  # 1 for positive (anomaly) samples
                all_labels.extend([0] * len(scores_B))  # 0 for negative (normal) samples
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_score = auc(fpr, tpr)
        
        print(f'Transductive AUC: {auc_score:.4f}')
        
        # Save results
        results = pd.DataFrame({
            'score': all_scores,
            'label': all_labels
        })
        results.to_csv(os.path.join(self.config.result_dir, 'transductive_results.csv'), index=False)

    def testAUCp(self):
        """Test the model using AUCp."""
        self.restore_model(self.config.test_iters)
        self.G.eval()
        self.D.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                x_realA = x_realA.to(self.device)
                x_realB = x_realB.to(self.device)
                
                # Generate anomaly scores for positive samples
                mask = self.G(x_realA)
                x_fakeB = torch.tanh(x_realA + mask)
                _, out_src = self.D(x_fakeB)
                scores = -out_src.cpu().numpy().flatten()
                all_scores.extend(scores)
                all_labels.extend([1] * len(scores))  # 1 for positive (anomaly) samples
                
                # Generate anomaly scores for negative samples
                mask = self.G(x_realB)
                x_fakeB = torch.tanh(x_realB + mask)
                _, out_src = self.D(x_fakeB)
                scores = -out_src.cpu().numpy().flatten()
                all_scores.extend(scores)
                all_labels.extend([0] * len(scores))  # 0 for negative (normal) samples
        
        # Calculate AUCp
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        aucp_score = auc(fpr, tpr)
        
        print(f'AUCp: {aucp_score:.4f}')
        
        # Save results
        results = pd.DataFrame({
            'score': all_scores,
            'label': all_labels
        })
        results.to_csv(os.path.join(self.config.result_dir, 'aucp_results.csv'), index=False)

class SupervisedCNN(nn.Module):
    """Simple CNN classifier for supervised anomaly detection."""
    def __init__(self, input_channels=3, num_classes=2):
        super(SupervisedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SupervisedSolver:
    """Solver for supervised learning model."""
    def __init__(self, data_loader, config):
        self.config = config
        self.data_loader = data_loader
        self.device = torch.device('cpu')
        self.build_model()
        self.best_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0

    def build_model(self):
        """Create and initialize the supervised model."""
        self.model = SupervisedCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.supervised_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    def restore_model(self):
        """Restore the trained supervised model."""
        print('Loading the best trained supervised model...')
        model_path = os.path.join(self.config.model_save_dir, 'best-supervised.ckpt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def train(self):
        """Train the supervised model."""
        start_time = time.time()
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                # Combine positive and negative samples
                x = torch.cat([x_realA, x_realB], dim=0)
                y = torch.cat([torch.ones(x_realA.size(0)), torch.zeros(x_realB.size(0))], dim=0).long()
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(self.data_loader)
            accuracy = 100. * correct / total
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                model_path = os.path.join(self.config.model_save_dir, 'best-supervised.ckpt')
                torch.save(self.model.state_dict(), model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
            
            # Log progress
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = f"Elapsed [{et}], Epoch [{epoch + 1}/{self.config.num_epochs}], "
            log += f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, "
            log += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            print(log)

    def validate(self):
        """Validate the supervised model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                if i >= 10:  # Use only 10 batches for validation
                    break
                    
                x = torch.cat([x_realA, x_realB], dim=0)
                y = torch.cat([torch.ones(x_realA.size(0)), torch.zeros(x_realB.size(0))], dim=0).long()
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        self.model.train()
        return total_loss / 10, 100. * correct / total

    def test(self):
        """Test the supervised model."""
        self.restore_model()
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, (x_realA, x_realB) in enumerate(self.data_loader):
                x = torch.cat([x_realA, x_realB], dim=0)
                y = torch.cat([torch.ones(x_realA.size(0)), torch.zeros(x_realB.size(0))], dim=0).long()
                
                x = x.to(self.device)
                outputs = self.model(x)
                scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                all_scores.extend(scores)
                all_labels.extend(y.numpy())
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_score = auc(fpr, tpr)
        
        print(f'Supervised Model AUC: {auc_score:.4f}')
        
        # Save results
        results = pd.DataFrame({
            'score': all_scores,
            'label': all_labels
        })
        results.to_csv(os.path.join(self.config.result_dir, 'supervised_results.csv'), index=False)

# Main Function
def main(config):
    # For fast training.
    torch.backends.cudnn.benchmark = True

    # Create directories if they don't exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    data_loader = get_loader(config.image_dir, config.image_size, config.batch_size,
                             config.dataset, config.mode, config.num_workers)

    # Solver for training and testing Brainomaly.
    solver = Solver(data_loader, config)
    supervised_solver = SupervisedSolver(data_loader, config)

    if config.mode == 'train':
        solver.train()
        supervised_solver.train()
    elif config.mode == 'testAUCInductive':
        solver.testAUCInductive()
        supervised_solver.test()
    elif config.mode == 'testAUCTransductive':
        solver.testAUCTransductive()
        supervised_solver.test()
    elif config.mode == 'testAUCp':
        solver.testAUCp()
        supervised_solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=3, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=3, help='number of strided conv layers in D')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity loss')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='MedicalData', choices=['MedicalData'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=2, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=400000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'testAUCInductive', 'testAUCTransductive', 'testAUCp'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/MedicalData')
    parser.add_argument('--log_dir', type=str, default='MedicalData/logs')
    parser.add_argument('--model_save_dir', type=str, default='MedicalData/models')
    parser.add_argument('--sample_dir', type=str, default='MedicalData/samples')
    parser.add_argument('--result_dir', type=str, default='MedicalData/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    # Supervised learning configuration
    parser.add_argument('--supervised_lr', type=float, default=0.001, help='learning rate for supervised model')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for supervised training')

    config = parser.parse_args()
    print(config)
    main(config)