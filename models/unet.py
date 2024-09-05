from models.model_zoo import *
import pdb

# ====================================
# UNet class
# ====================================
class UNet(nn.Module):
    def __init__(self, n0):
        super(UNet, self).__init__()
        self.encoder_unet = EncoderUNet(n0)
        self.decoder_unet = DecoderUNet(n0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        zs = self.encoder_unet(x)
        logits = self.decoder_unet(zs)
        probs = self.sigmoid(logits)
        
        return logits, probs





