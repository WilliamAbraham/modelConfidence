import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    """Simple CNN baseline model"""
    def __init__(self, input_channels=1, output_channels=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*7*7, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    """Multi-Layer Perceptron (Fully Connected Network)"""
    def __init__(self, input_channels=1, output_channels=10, hidden_sizes=[512, 256], dropout=0.2):
        super(MLP, self).__init__()
        input_size = 28 * 28 * input_channels  # MNIST is 28x28
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)

class LeNet5(nn.Module):
    """LeNet-5 architecture"""
    def __init__(self, input_channels=1, output_channels=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

class ResNetBlock(nn.Module):
    """Residual block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """Lightweight ResNet for MNIST"""
    def __init__(self, input_channels=1, output_channels=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.layer1 = self._make_layer(8, 8, 1, stride=1)
        self.layer2 = self._make_layer(8, 16, 1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, output_channels)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VGG(nn.Module):
    """VGG-style network for MNIST"""
    def __init__(self, input_channels=1, output_channels=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DenseNetBlock(nn.Module):
    """DenseNet block"""
    def __init__(self, in_channels, growth_rate):
        super(DenseNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseNet(nn.Module):
    """DenseNet for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, growth_rate=12, num_blocks=4):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks
        self.dense1 = self._make_dense_block(16, growth_rate, num_blocks)
        in_channels = 16 + growth_rate * num_blocks
        self.trans1 = self._make_transition(in_channels, in_channels // 2)
        in_channels = in_channels // 2
        
        self.dense2 = self._make_dense_block(in_channels, growth_rate, num_blocks)
        in_channels = in_channels + growth_rate * num_blocks
        self.trans2 = self._make_transition(in_channels, in_channels // 2)
        in_channels = in_channels // 2
        
        self.dense3 = self._make_dense_block(in_channels, growth_rate, num_blocks)
        in_channels = in_channels + growth_rate * num_blocks
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, output_channels)

    def _make_dense_block(self, in_channels, growth_rate, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(DenseNetBlock(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EfficientNet(nn.Module):
    """Lightweight EfficientNet for MNIST"""
    def __init__(self, input_channels=1, output_channels=10):
        super(EfficientNet, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # MBConv blocks (very simplified)
        self.blocks = nn.Sequential(
            self._make_mbconv(16, 16, 1, 1),
            self._make_mbconv(16, 32, 2, 1),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, output_channels)
        )

    def _make_mbconv(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Projection
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class Transformer(nn.Module):
    """Lightweight Vision Transformer for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, patch_size=7, 
                 embed_dim=64, num_heads=4, num_layers=2, mlp_ratio=2):
        super(Transformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (28 // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder (much smaller)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_channels)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Take class token
        return self.head(cls_output)

# Model registry for easy access
MODELS = {
    'cnn': CNN,
    'mlp': MLP,
    'lenet5': LeNet5,
    'resnet': ResNet,
    'vgg': VGG,
    'densenet': DenseNet,
    'efficientnet': EfficientNet,
    'transformer': Transformer
}

def get_model(model_name, input_channels=1, output_channels=10, **kwargs):
    """Get model by name with specified parameters"""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    return model_class(input_channels=input_channels, output_channels=output_channels, **kwargs)