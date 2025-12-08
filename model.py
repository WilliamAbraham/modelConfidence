import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EAFNO(nn.Module): ## epimesdic error awareness FNO
    #def __init__(self,batchsize,device, num_properties,input_dim, hidden_dim, latent_dim,im_x,im_y,modes1, modes2):
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None, image_size=(28, 28)):
        super(EAFNO, self).__init__()
        self.im_x, self.im_y = image_size
        # Adjust modes based on image size (keep similar ratio to original)
        self.modes1 = min(self.im_x, 28)
        self.modes2 = min(self.im_y // 2 + 1, 15)
        self.hidden_dim = 8
        self.epi_channels = 10
        self.output_channels = output_channels  # output channels for classification + epi error
        self.activation_function = nn.LeakyReLU(0.2)
        self.p = nn.Linear(3, self.hidden_dim) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.hidden_dim, self.output_channels, self.modes1, self.modes2)
        self.mlp0 = FNO_MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.mlp1 = FNO_MLP(self.output_channels, self.output_channels, self.hidden_dim)
        self.w0 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv2d(self.hidden_dim, self.output_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        x = x.view(-1,self.im_x,self.im_y,1)
        grid = _get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.activation_function(self.p(x))
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation_function(x)
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x_class = x
        x_output = self.avgpool(x_class)
        x_output = x_output.view(x_output.size(0), -1)
        x_epi = x
        x_epi = torch.sigmoid(x_epi)+1
        mid_value = x_epi[:,:,self.im_x//2,self.im_x//2]
        mid_value = mid_value[:,:,None,None]
        sph_err = _calculate_spherical_error(x_epi,mid_value, self.epi_channels, self.im_x, self.im_y)
        return x_output, sph_err

class EACNN(nn.Module):
    """Simple CNN baseline model"""
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None, image_size=(28, 28)):
        super(EACNN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = self._make_norm_layer(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = self._make_norm_layer(16)
        self.fc1 = nn.Linear(16*7*7, output_channels)

        ### uncertainty block
        self.im_x, self.im_y = image_size
        # Adjust modes based on image size (keep similar ratio to original)
        self.modes1 = min(self.im_x // 2, 14)
        self.modes2 = min(self.im_y // 4 + 1, 8)
        self.epi_hidden_dim = 8
        self.epi_channels = 10
        self.output_channels = output_channels  # output channels for classification + epi error
        self.activation_function = nn.LeakyReLU(0.2)
        self.epi_p = nn.Linear(3, self.epi_hidden_dim) # input channel is 3: (a(x, y), x, y)
        self.epi_conv0 = SpectralConv2d(self.epi_hidden_dim, self.epi_hidden_dim, self.modes1, self.modes2)
        self.epi_conv1 = SpectralConv2d(self.epi_hidden_dim, self.epi_channels, self.modes1, self.modes2)
        self.epi_mlp0 = FNO_MLP(self.epi_hidden_dim, self.epi_hidden_dim, self.epi_hidden_dim)
        self.epi_mlp1 = FNO_MLP(self.epi_channels, self.epi_channels, self.epi_hidden_dim)
        self.epi_w0 = nn.Conv2d(self.epi_hidden_dim, self.epi_hidden_dim, 1)
        self.epi_w1 = nn.Conv2d(self.epi_hidden_dim, self.epi_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

    def uncertainty_forward(self, x,mid_value):
        x = x.view(-1,self.im_x,self.im_y,1)
        grid = _get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.activation_function(self.epi_p(x))
        x = x.permute(0, 3, 1, 2)
        x1 = self.epi_conv0(x)
        x1 = self.epi_mlp0(x1)
        x2 = self.epi_w0(x)
        x = x1 + x2
        x = self.activation_function(x)
        x1 = self.epi_conv1(x)
        x1 = self.epi_mlp1(x1)
        x2 = self.epi_w1(x)
        x = x1 + x2
        x_epi = x
        mid_value = mid_value[:,:,None,None]
        sph_err = _calculate_spherical_error(x_epi,mid_value, self.epi_channels, self.im_x, self.im_y)
        return sph_err

    def forward(self, x0):
        x = F.relu(self.bn1(self.conv1(x0)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.softmax(x, dim=1)+1
        sph_err = self.uncertainty_forward(x0,x)
        return x, sph_err
    
def _get_grid(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device) 

def get_spherical_grid(batchsize, channels, size_x, size_y, device):
    x_coords = torch.linspace(-1, 1, size_x).to(device)
    y_coords = torch.linspace(-1, 1, size_y).to(device)
    l0 = np.sqrt(2*np.sqrt(1)-1)/np.sqrt(2)
    x_coords = x_coords * l0
    y_coords = y_coords * l0
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
    x_grid = x_grid.unsqueeze(0).unsqueeze(0).repeat(batchsize,channels,1,1)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0).repeat(batchsize,channels,1,1)
    dx = x_coords[1]-x_coords[0]
    dy = y_coords[1]-y_coords[0]
    return x_grid, y_grid, dx, dy

def _calculate_spherical_error(x,mid_value, num_channels, im_x, im_y):
    batchsize = x.shape[0]
    device = x.device
    x_grid, y_grid, dx, dy = get_spherical_grid(batchsize, num_channels, im_x, im_y, device)
    x = x/mid_value
    sph_r = np.sqrt(1)
    z0 = 1 - sph_r
    r0_sq = sph_r**2
    K0 = 1/r0_sq
    ## assume the sphere center is (0,0,1-np.sqrt(3))
    z_grid = x - z0
    radius_sq = x_grid**2 + y_grid**2 + z_grid**2
    radical_error = torch.mean((radius_sq - r0_sq)**2,dim=[2,3],keepdim=True)
    K = _principal_curvatures_heightfield(z_grid, dx, dy)
    curvature_err = torch.mean((K[:,:,3:-3,3:-3] - K0)**2,dim=[2,3],keepdim=True)
    sph_err = radical_error + curvature_err*0.1
    return sph_err

def _principal_curvatures_heightfield(z_grid,dx,dy):
    # First derivatives
    fx = torch.gradient(z_grid, spacing=dx, dim=2)[0]   
    fy = torch.gradient(z_grid, spacing=dy, dim=3)[0]
    # Second derivatives
    fxx = torch.gradient(fx, spacing=dx, dim=2)[0]
    fyy = torch.gradient(fy, spacing=dy, dim=3)[0]
    fxy = torch.gradient(fx, spacing=dy, dim=3)[0]
    eps = 1e-8
    K = (fxx*fyy - fxy*fxy)/((1 + fx*fx + fy*fy)**2 + eps)
    return K

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(FNO_MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class LocalMLP(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(LocalMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float32))
    
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        out = self.compl_mul2d(x, self.weights1)
        return out
    

class CNN(nn.Module):
    """Simple CNN baseline model"""
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None):
        super(CNN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = self._make_norm_layer(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = self._make_norm_layer(16)
        self.fc1 = nn.Linear(16*7*7, output_channels)

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x,torch.zeros(x.shape[0],1,1,1).to(x.device)  # Placeholder for epi_error

class MLP(nn.Module):
    """Multi-Layer Perceptron (Fully Connected Network)"""
    def __init__(self, input_channels=1, output_channels=10, hidden_sizes=[512, 256], dropout=0.2, norm_layer=None):
        super(MLP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        input_size = 28 * 28 * input_channels  # MNIST is 28x28
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size, bias=False),
                self._make_norm_layer(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_channels))
        self.network = nn.Sequential(*layers)

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)

class LeNet5(nn.Module):
    """LeNet-5 architecture"""
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None):
        super(LeNet5, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, bias=False)
        self.bn1 = self._make_norm_layer(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        self.bn2 = self._make_norm_layer(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, output_channels)

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

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
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None):
        super(ResNetBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = self._make_norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self._make_norm_layer(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                self._make_norm_layer(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

class ResNet(nn.Module):
    """Lightweight ResNet for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._make_norm_layer(8)
        
        self.layer1 = self._make_layer(8, 8, 1, stride=1)
        self.layer2 = self._make_layer(8, 16, 1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, output_channels)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, norm_layer=self._norm_layer))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, torch.zeros(x.shape[0],1,1,1).to(x.device)  # Placeholder for epi_error

class VGG(nn.Module):
    """VGG-style network for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None):
        super(VGG, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            self._make_norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            self._make_norm_layer(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            self._make_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            self._make_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            self._make_norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            self._make_norm_layer(128),
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

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DenseNetBlock(nn.Module):
    """DenseNet block"""
    def __init__(self, in_channels, growth_rate, norm_layer=None):
        super(DenseNetBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.bn1 = self._make_norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = self._make_norm_layer(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

class DenseNet(nn.Module):
    """DenseNet for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, growth_rate=12, num_blocks=4, norm_layer=None):
        super(DenseNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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
        
        self.bn = self._make_norm_layer(in_channels)
        self.fc = nn.Linear(in_channels, output_channels)

    def _make_dense_block(self, in_channels, growth_rate, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(DenseNetBlock(in_channels, growth_rate, norm_layer=self._norm_layer))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            self._make_norm_layer(in_channels),
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

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)

class EfficientNet(nn.Module):
    """Lightweight EfficientNet for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, norm_layer=None):
        super(EfficientNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            self._make_norm_layer(16),
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
                self._make_norm_layer(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),
            self._make_norm_layer(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Projection
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            self._make_norm_layer(out_channels)
        ])
        
        return nn.Sequential(*layers)

    def _make_norm_layer(self, num_features):
        return self._norm_layer(num_features)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class Transformer(nn.Module):
    """Lightweight Vision Transformer for MNIST"""
    def __init__(self, input_channels=1, output_channels=10, patch_size=7, 
                 embed_dim=64, num_heads=4, num_layers=2, mlp_ratio=2, norm_layer=None):
        super(Transformer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self._norm_layer = norm_layer

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
        self.norm = self._make_norm_layer(embed_dim)
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

    def _make_norm_layer(self, normalized_shape):
        return self._norm_layer(normalized_shape)

# Model registry for easy access
MODELS = {
    'cnn': CNN,
    'mlp': MLP,
    'lenet5': LeNet5,
    'resnet': ResNet,
    'vgg': VGG,
    'densenet': DenseNet,
    'efficientnet': EfficientNet,
    'transformer': Transformer,
    'eafno': EAFNO,
    'eacnn': EACNN,
}

def get_model(model_name, input_channels=1, output_channels=10, **kwargs):
    """Get model by name with specified parameters.

    Additional keyword arguments like `norm_layer` are forwarded to the model constructor.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    return model_class(input_channels=input_channels, output_channels=output_channels, **kwargs)