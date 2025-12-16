import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# ==========================================
# 1. Helper Blocks
# ==========================================

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = (1 / in_features) ** 0.5

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.padding = kernel_size // 2
        
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / (in_channels * kernel_size ** 2) ** 0.5
        
        self.modulation = EqualizedLinear(style_dim, in_channels, bias=True)
        self.noise_scaler = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, style, noise_map=None):
        batch, in_channels, height, width = x.shape
        style = self.modulation(style).view(batch, 1, in_channels, 1, 1) + 1.0
        weights = self.weight * self.scale * style 

        if self.demodulate:
            demod = torch.rsqrt(weights.pow(2).sum([2, 3, 4]) + 1e-8)
            weights = weights * demod.view(batch, self.out_channels, 1, 1, 1)

        x = x.view(1, -1, height, width)
        weights = weights.view(-1, in_channels, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, weights, padding=self.padding, groups=batch)
        x = x.view(batch, self.out_channels, height, width)

        if noise_map is not None:
            if noise_map.shape[2:] != x.shape[2:]:
                noise_map = F.interpolate(noise_map, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + (noise_map * self.noise_scaler)
        
        return F.leaky_relu(x + self.bias.view(1, -1, 1, 1), 0.2)

class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)
        
    def forward(self, x, style):
        return self.conv(x, style, noise_map=None)

# ==========================================
# 2. Encoder
# ==========================================

class Encoder(nn.Module):
    def __init__(self, style_dim=512):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            backbone = resnet50(pretrained=True)
        
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1 
        self.layer2 = backbone.layer2 
        self.layer3 = backbone.layer3 
        self.layer4 = backbone.layer4 
        
        self.style_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            EqualizedLinear(2048, style_dim),
            nn.LeakyReLU(0.2),
            EqualizedLinear(style_dim, style_dim)
        )
        
        self.noise_head_1 = nn.Conv2d(256, 1, 1)
        self.noise_head_2 = nn.Conv2d(512, 1, 1)
        self.noise_head_3 = nn.Conv2d(1024, 1, 1)
        self.noise_head_4 = nn.Conv2d(2048, 1, 1)

    def forward(self, x):
        x0 = self.stem(x)       
        x1 = self.layer1(x0)    
        x2 = self.layer2(x1)    
        x3 = self.layer3(x2)    
        x4 = self.layer4(x3)    
        
        w = self.style_head(x4)
        
        noise_maps = {
            'low': self.noise_head_4(x4),  
            'mid': self.noise_head_3(x3),  
            'high': self.noise_head_2(x2), 
            'super': self.noise_head_1(x1) 
        }
        
        return w, noise_maps

# ==========================================
# 3. Generator (Fixed 256 Config)
# ==========================================

class Generator(nn.Module):
    def __init__(self, size=256, style_dim=512):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.const_style = EqualizedLinear(style_dim, 512)
        
        # CORRECTED LAYERS -> GUARANTEES 256 OUTPUT
        # Start 4x4. Each step after first upsamples.
        # Steps: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.layers_config = [
            (512, 'low'),   # 4x4 (No upsample)
            (512, 'low'),   # 8x8
            (256, 'mid'),   # 16x16
            (128, 'high'),  # 32x32
            (64, 'super'),  # 64x64
            (32, 'super'),  # 128x128
            (16, 'super')   # 256x256 (Added this layer)
        ]
        
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        
        in_ch = 512
        for out_ch, noise_key in self.layers_config:
            self.convs.append(ModulatedConv2d(in_ch, out_ch, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_ch, style_dim))
            in_ch = out_ch
            
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, w, noise_maps):
        batch = w.shape[0]
        x = self.const_input.repeat(batch, 1, 1, 1)
        rgb = None
        
        for i, (conv, to_rgb) in enumerate(zip(self.convs, self.to_rgbs)):
            noise_key = self.layers_config[i][1]
            specific_noise = noise_maps.get(noise_key)
            
            # Upsample on every layer except the first one
            if i > 0:
                x = self.upsample(x)
                
            x = conv(x, w, noise_map=specific_noise)
            
            if rgb is not None:
                rgb = self.upsample(rgb)
                rgb = rgb + to_rgb(x, w)
            else:
                rgb = to_rgb(x, w)
                
        return rgb

# ==========================================
# 4. The Complete Model
# ==========================================

class GAN(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Generator(size=resolution)
        
    def forward(self, x):
        # Safety Resize to 256 for Encoder (if input is oddly sized)
        if x.shape[2] != 256 or x.shape[3] != 256:
            x_in = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            x_in = x
            
        w, noise_maps = self.encoder(x_in)
        reconstruction = self.decoder(w, noise_maps)
        return reconstruction

    def inference(self, x_rgba):
        has_alpha = x_rgba.shape[1] == 4
        if has_alpha:
            x_rgb = x_rgba[:, :3, :, :]
            x_alpha = x_rgba[:, 3:4, :, :]
        else:
            x_rgb = x_rgba
            x_alpha = None
            
        with torch.no_grad():
            rec_rgb = self.forward(x_rgb)
        
        if has_alpha:
            target_h, target_w = rec_rgb.shape[2], rec_rgb.shape[3]
            rec_alpha = F.interpolate(x_alpha, size=(target_h, target_w), mode='bilinear', align_corners=False)
            result = torch.cat([rec_rgb, rec_alpha], dim=1)
        else:
            result = rec_rgb
        return result