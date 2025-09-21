import torch
import torch.nn as nn

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
#         super(ConvBlock, self).__init__()

#         self.conv_block = nn.Sequential(
#             nn.ReflectionPad2d(1), # Вообще по статье ReflectionPad используется один раз в начале, а затем, видимо, в Conv2d padding=1
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv_block(x)

class CFRWDPatchDisBranch(nn.Module):
    def __init__(self, input_channels=4, condition_channels=3, ndf=64, return_features=True):
        super(CFRWDPatchDisBranch, self).__init__()
        self.return_features = return_features

        # Определение слоёв (пример на основе типичной PatchGAN структуры, как в статье)
        # Здесь 4 conv-блока + финальный conv, признаки извлекаем после каждого LeakyReLU (кроме первого, если без нормы)
        self.main = nn.Sequential(
            nn.Conv2d(input_channels + condition_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),  # После этого можно извлекать feat1
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # feat2
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # feat3
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # feat4
            
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)  # Финальный output
        )
    
    def forward(self, x, condition):
        features = []
        # Concatenate input (SAR + optical) with condition (real optical) along channel dimension
        x = torch.cat((x, condition), dim=1)
        # Проходим через sequential, но извлекаем признаки только после активаций (LeakyReLU)
        for i, layer in enumerate(self.main):
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU) and self.return_features:
                features.append(x)  # Добавляем после каждой ReLU (обычно 4 штуки)
        
        if self.return_features:
            return x, features  # Возвращаем output и list[features]
        return x  # Только output

class CFRWDPatchDis(nn.Module):
    def __init__(self, input_channels=4, condition_channels=3, ndf=64, return_features=True):
        super(CFRWDPatchDis, self).__init__()
        self.return_features = return_features
        # Large-scale branch
        self.large_scale_branch = CFRWDPatchDisBranch(input_channels, condition_channels, ndf, return_features)
        # Downsample for small-scale branch
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Small-scale branch
        self.small_scale_branch = CFRWDPatchDisBranch(input_channels, condition_channels, ndf, return_features)
    
    def forward(self, sar, fake_optical, real_optical):
        input_img = torch.cat((sar, fake_optical), dim=1)
        
        # Large-scale branch output
        if self.return_features:
            large_output, large_features = self.large_scale_branch(input_img, real_optical)
        else:
            large_output = self.large_scale_branch(input_img, real_optical)
        
        # Small-scale branch output
        small_input = self.downsample(input_img)
        small_condition = self.downsample(real_optical)
        if self.return_features:
            small_output, small_features = self.small_scale_branch(small_input, small_condition)
        else:
            small_output = self.small_scale_branch(small_input, small_condition)

        outputs = (large_output, small_output)

        if self.return_features:
            # Собираем все признаки: список из large_features + small_features (каждый — list из 4 feat)
            all_features = large_features + small_features  # Плоский список из 8 признаков
            return outputs, all_features
        
        return outputs

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    height, width = 256, 256
    sar = torch.randn(batch_size, 1, height, width)  # SAR image (1 channel)
    fake_optical = torch.randn(batch_size, 3, height, width)  # Generated optical image (3 channels)
    real_optical = torch.randn(batch_size, 3, height, width)  # Real optical image (3 channels, as condition)
    
    discriminator = CFRWDPatchDis(input_channels=4, condition_channels=3, return_features=False)
    output, features = discriminator(sar, fake_optical, real_optical)
    large_output, small_output = output
    
    print(f"Large scale output shape: {large_output.shape}")  # Expected: [batch_size, 1, height/16, width/16]
    print(f"Small scale output shape: {small_output.shape}")  # Expected: [batch_size, 1, height/32, width/32]
    print(f"Number of feature maps extracted: {len(features)}")  # Expected: 8 (4 from each branch)