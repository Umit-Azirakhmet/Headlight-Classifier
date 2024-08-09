import torch
import torch.nn as nn
import timm
import torchvision.models as models 


class EffnetHeadlightClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=0)

        for param in self.base_model.parameters():
            param.requires_grad = False
        
        for param in self.base_model.blocks[-1].parameters():
            param.requires_grad = True

        in_features = self.base_model.num_features
        self.additional_layers = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.additional_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class MobilenetHeadlightClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = models.mobilenet_v3_large(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False
        
        for param in self.base_model.features[-1].parameters():
            param.requires_grad = True

        in_features = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Identity()

        self.additional_layers = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.additional_layers(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x
    

class ResnetHeadlightClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False
        
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x