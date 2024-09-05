import torch
import torch.nn as nn
    
# ====================================
# Encoder class for UNet
# ====================================
class EncoderUNet(nn.Module):
    def __init__(self, n0):
        super(EncoderUNet, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        self.conv1_1 = nn.Conv2d(1, n1, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn1_1 = nn.BatchNorm2d(n1)
        self.bn1_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        self.conv2_1 = nn.Conv2d(n1, n2, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn2_1 = nn.BatchNorm2d(n2)
        self.bn2_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        self.conv3_1 = nn.Conv2d(n2, n3, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(n3)
        self.bn3_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # 4th Conv block
        # ====================================
        self.conv4_1 = nn.Conv2d(n3, n4, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(n4, n4, kernel_size = 3, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(n4)
        self.bn4_2 = nn.BatchNorm2d(n4)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # ====================================
        # 1st Conv block - two conv layers, followed by batch normalization and max-pooling
        # ====================================
        conv1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        conv1_2 = self.relu(self.bn1_2(self.conv1_2(conv1_1)))
        pool1 = self.max_pool(conv1_2)
        
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2_2 = self.relu(self.bn2_2(self.conv2_2(conv2_1)))
        pool2 = self.max_pool(conv2_2)
        
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3_2 = self.relu(self.bn3_2(self.conv3_2(conv3_1)))
        pool3 = self.max_pool(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4_2 = self.relu(self.bn4_2(self.conv4_2(conv4_1)))
        
        return conv1_2, conv2_2, conv3_2, conv4_2
    

    
# ====================================
# Attention class
# ====================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
    
# ====================================
# Attention Decoder class
# ====================================    
class AttentionDecoder(nn.Module):
    def __init__(self, n0, num_classes):
        super(AttentionDecoder, self).__init__()
        n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0 # number of channels
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv3 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.attn5 = AttentionBlock(F_g = n4, F_l = n3, F_int = n3)
        self.conv5_1 = nn.Conv2d(n4 + n3, n3, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(n3, n3, kernel_size = 3, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(n3)
        self.bn5_2 = nn.BatchNorm2d(n3)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.attn6 = AttentionBlock(F_g = n3, F_l = n2, F_int = n2)
        self.conv6_1 = nn.Conv2d(n3 + n2, n2, kernel_size = 3, padding = 1)
        self.conv6_2 = nn.Conv2d(n2, n2, kernel_size = 3, padding = 1)
        self.bn6_1 = nn.BatchNorm2d(n2)
        self.bn6_2 = nn.BatchNorm2d(n2)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        self.deconv1 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.attn7 = AttentionBlock(F_g = n2, F_l = n1, F_int = n1)
        self.conv7_1 = nn.Conv2d(n2 + n1, n1, kernel_size = 3, padding = 1)
        self.conv7_2 = nn.Conv2d(n1, n1, kernel_size = 3, padding = 1)
        self.bn7_1 = nn.BatchNorm2d(n1)
        self.bn7_2 = nn.BatchNorm2d(n1)
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        self.conv_pred = nn.Conv2d(n1, num_classes, kernel_size = 3, padding = 1, bias = False)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, zs):
        conv1_2, conv2_2, conv3_2, conv4_2 = zs
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = self.deconv3(conv4_2)
        attn5 = self.attn5(deconv3, conv3_2)
        skip3 = torch.cat([deconv3, attn5], axis = 1)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(skip3)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = self.deconv2(conv5_2)
        attn6 = self.attn6(deconv2, conv2_2)
        skip2 = torch.cat([deconv2, attn6], axis = 1)
        conv6_1 = self.relu(self.bn6_1(self.conv6_1(skip2)))
        conv6_2 = self.relu(self.bn6_2(self.conv6_2(conv6_1)))
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = self.deconv1(conv6_2)
        attn7 = self.attn7(deconv1, conv1_2)
        skip1 = torch.cat([deconv1, attn7], axis = 1)
        conv7_1 = self.relu(self.bn7_1(self.conv7_1(skip1)))
        conv7_2 = self.relu(self.bn7_2(self.conv7_2(conv7_1)))
        
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = self.conv_pred(conv7_2)
        
        return pred