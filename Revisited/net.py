import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(p=0.5)
        self.sigmod = nn.Sigmoid ()
        self.label_embedding = nn.Embedding(10, 512)

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding= 1),
            nn.ReLU(inplace=True),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding= 1),
            nn.ReLU(inplace=True),
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding= 1),
            nn.ReLU(inplace=True),
        )

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding= 1),
            nn.ReLU(inplace=True),
        )
        
        self.middle_1_0 = nn.Conv2d(1024, 1024, 3, padding= 1)
        self.middle_1_1 = nn.Conv2d(1024, 1024, 3, padding= 1)
        
       
        self.deconv4_0 = nn.ConvTranspose2d(1536, 512, 3, stride=(2,2), padding = 1, output_padding = 1)
        self.uconv4_1 = nn.Conv2d(1024, 512, 3, padding= 1) 
        self.uconv4_2 = nn.Conv2d(512, 512, 3, padding= 1)

        self.deconv3_0 = nn.ConvTranspose2d(512, 512, 3, stride=(2,2), padding = 1, output_padding = 1)
        self.uconv3_1 = nn.Conv2d(768, 256, 3, padding= 1) 
        self.uconv3_2 = nn.Conv2d(256, 256, 3, padding= 1)

        self.deconv2_0 = nn.ConvTranspose2d(256, 512, 3, stride=(2,2), padding = 1, output_padding = 1)
        self.uconv2_1 = nn.Conv2d(640, 128, 3, padding= 1) 
        self.uconv2_2 = nn.Conv2d(128, 128, 3, padding= 1)

        self.deconv1_0 = nn.ConvTranspose2d(128, 512, 3, stride=(2,2), padding = 1, output_padding = 1)
        self.uconv1_1 = nn.Conv2d(576, 192, 3, padding= 1) 
        self.uconv1_2 = nn.Conv2d(192, 192, 3, padding= 1)

  
        self.out_layer = nn.Conv2d(192, 1, 1)

 

    def forward(self, x, input_labels, target_labels):
        conv1 = self.encoder_1(x)
        pool1 = self.pool(conv1)
        pool1 = self.dropout(pool1)

        conv2 = self.encoder_2(pool1)
        pool2 = self.pool(conv2)
        pool2 = self.dropout(pool2)

        conv3 = self.encoder_3(pool2)
        pool3 = self.pool(conv3)
        pool3 = self.dropout(pool3)

        conv4 = self.encoder_4(pool3)
        pool4 = self.pool(conv4)
        encoder_out = self.dropout(pool4)

        input_label_embedding = self.label_embedding(input_labels).view(input_labels.size(0), 512, 1, 1)
        x1 = torch.cat([encoder_out, input_label_embedding.expand_as(encoder_out)], dim=1)

        convm = self.middle_1_0(x1)
        convm = self.activate(convm)
        convm = self.middle_1_1(convm)
        x2 = self.activate(convm)

        target_label_embedding = self.label_embedding(target_labels).view(target_labels.size(0), 512, 1, 1)
        x2 = torch.cat([x2, target_label_embedding.expand(x2.size(0), 512, x2.size(2), x2.size(3))], dim=1)

        deconv4 = self.deconv4_0(x2)
        uconv4 = torch.cat([deconv4, conv4], 1)   # (None, 4, 4, 1024)
        uconv4 = self.dropout(uconv4)
        uconv4 = self.uconv4_1(uconv4)            # (None, 4, 4, 512)
        uconv4 = self.activate(uconv4)
        uconv4 = self.uconv4_2(uconv4)            # (None, 4, 4, 512)
        uconv4 = self.activate(uconv4)

        deconv3 = self.deconv3_0(uconv4)          # (None, 8, 8, 512)
        uconv3 = torch.cat([deconv3, conv3], 1)   # (None, 8, 8, 768)
        uconv3 = self.dropout(uconv3)
        uconv3 = self.uconv3_1(uconv3)            # (None, 8, 8, 256)
        uconv3 = self.activate(uconv3)
        uconv3 = self.uconv3_2(uconv3)            # (None, 8, 8, 256)
        uconv3 = self.activate(uconv3)
        
        deconv2 = self.deconv2_0(uconv3)          # (None, 16, 16, 512)
        uconv2 = torch.cat([deconv2, conv2], 1)   # (None, 16, 16, 640)
        uconv2 = self.dropout(uconv2)
        uconv2 = self.uconv2_1(uconv2)            # (None, 16, 16, 128)
        uconv2 = self.activate(uconv2)
        uconv2 = self.uconv2_2(uconv2)            # (None, 16, 16, 128)
        uconv2 = self.activate(uconv2)

        deconv1 = self.deconv1_0(uconv2)          # (None, 32, 32, 512)
        uconv1 = torch.cat([deconv1, conv1], 1)   # (None, 32, 32, 576)
        uconv1 = self.dropout(uconv1)
        uconv1 = self.uconv1_1(uconv1)            # (None, 32, 32, 192)
        uconv1 = self.activate(uconv1)
        uconv1 = self.uconv1_2(uconv1)            # (None, 32, 32, 192)
        uconv1 = self.activate(uconv1)

        out = self.out_layer(uconv1)
        out = self.sigmod(out)

        return out

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_size, image_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 128 * 8 * 8),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)