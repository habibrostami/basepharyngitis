import torch
import torch.nn as nn
import torchvision
from torchvision.models import swin_b


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class JointDenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(JointDenseNet121, self).__init__()
        in_features = 21
        mid_layer = 16
        out_features = 8
        self.feature_classifier = nn.Sequential(nn.Linear(in_features,mid_layer),
                                                nn.Linear(mid_layer, out_features=out_features))
        self.feature_classifier.apply(self.init_normal)

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = Identity()

        l = nn.Linear(kernelCount, 1) #kernelCount+out_features
        # self.init_normal(l)
        #self.densenet121.classifier = nn.Sequential(l,  nn.Softmax())
        self.fc = nn.Sequential(l,  nn.Sigmoid())


    def init_normal(self,m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    def forward(self, x, features):
        # print(f'{features[:,1:]}     count', torch.count_nonzero(features[1:]))
        # if torch.count_nonzero(features[:,1:]) == 0:
        #     print("-------------------------------------------------")
        #     return torch.Tensor([1.0,0.0]).cuda()
        x = self.densenet121(x)
        x_f = self.feature_classifier(features)
        #print(features)
        #return (x + x_f) / 2
        res = torch.cat(tensors=(x,x_f), dim=1)
        res = x
        #print(f"res shape ====== {res.shape}    , x shape = {x.shape}  x_f shape = {x_f.shape}")
        res = self.fc(res)
        #return (x_f + x)/2
        return res

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        in_features = 34
        mid_layer = 16
        self.feature_classifier = nn.Sequential(nn.Linear(in_features,mid_layer), nn.Linear(mid_layer, classCount), nn.Softmax())
        self.feature_classifier.apply(self.init_normal)

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        l = nn.Linear(kernelCount, kernelCount//2)
        self.init_normal(l)
        self.densenet121.classifier = nn.Sequential(l,  nn.Softmax())
        l2 = nn.Linear(classCount * 2, classCount)
        self.init_normal(l2)
        self.joint = nn.Sequential(l2, nn.Softmax())

    def init_normal(self,m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    def forward(self, x, features):
        # print(f'{features[:,1:]}     count', torch.count_nonzero(features[1:]))
        # if torch.count_nonzero(features[:,1:]) == 0:
        #     print("-------------------------------------------------")
        #     return torch.Tensor([1.0,0.0]).cuda()
        x = self.densenet121(x)
        x_f = self.feature_classifier(features)
        #print(features)
        #return (x + x_f) / 2
        #res = torch.cat(tensors=(x,x_f), dim=1)
        # print(res.shape)
        #res = self.joint(res)
        return (x_f + x)/2
        #return x_f


# class model_densenet121(nn.Module):
#
#     def __init__(self, classCount, isTrained):
#         super(model_densenet121, self).__init__()
#
#         self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
#
#         kernelCount = self.densenet121.classifier.in_features
#
#         self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
#
#     def forward(self, x):
#         x = self.densenet121(x)
#         return x

class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet169, self).__init__()

        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features

        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet169(x)
        return x


class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet201, self).__init__()

        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features

        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet201(x)
        return x


class ms_cam(nn.Module):
    def __init__(self, in_c, r, BatchNorm2d=nn.BatchNorm2d):
        super(ms_cam, self).__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(in_c, in_c // r, kernel_size=1, stride=1, bias=False),
                                     BatchNorm2d(in_c // r),
                                     nn.ReLU(),
                                     nn.Conv2d(in_c // r, in_c, kernel_size=1, stride=1, bias=False),
                                     BatchNorm2d(in_c),
                                     )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c, in_c // r, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(in_c // r),
            nn.ReLU(),
            nn.Conv2d(in_c // r, in_c, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(in_c)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = x1 + x2
        x3 = torch.sigmoid(x3)

        return x3


class iAFF(nn.Module):
    def __init__(self, in_c, r, BatchNorm2d=nn.BatchNorm2d):
        super(iAFF, self).__init__()
        self.ms_cam1 = ms_cam(in_c, r, BatchNorm2d=BatchNorm2d)
        self.ms_cam2 = ms_cam(in_c, r, BatchNorm2d=BatchNorm2d)

    def forward(self, x_1, x_2):
        x1_sum = x_1 + x_2
        amap1 = self.ms_cam1(x1_sum)

        x_1_a = amap1 * x_1
        x_2_a = (1 - amap1) * x_2

        x2_sum = x_1_a + x_2_a
        amap2 = self.ms_cam2(x2_sum)

        x_1 = amap2 * x_1
        x_2 = (1 - amap2) * x_2

        x = x_1 + x_2
        return x


class ms_fusion(nn.Module):
    def __init__(self, in_c, out_c, r, BatchNorm2d=nn.BatchNorm2d):
        super(ms_fusion, self).__init__()
        self.iaff = iAFF(out_c, r, BatchNorm2d=BatchNorm2d)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x_1, x_2):
        x_1 = self.down_sample(x_1)
        x = self.iaff(x_1, x_2)
        return x


class model_densenet121(nn.Module):
    def __init__(self, classCount, isTrained, BatchNorm2d=nn.BatchNorm2d):
        super(model_densenet121, self).__init__()
        import densenet4
        self.backbone1 = densenet4.densenet169(pretrained=isTrained)  # 1024
        self.backbone2 = densenet4.densenet169(pretrained=isTrained)
        # torch.Size([2, 256, 56, 56])
        # torch.Size([2, 512, 28, 28])
        # torch.Size([2, 1024, 14, 14])
        # torch.Size([2, 1024, 7, 7])
        self.ms_fusion1 = ms_fusion(256, 512, 8, BatchNorm2d=BatchNorm2d)
        self.ms_fusion2 = ms_fusion(512, 1280, 8, BatchNorm2d=BatchNorm2d)
        self.ms_fusion3 = ms_fusion(1280, 1664, 8, BatchNorm2d=BatchNorm2d)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(2048, classCount)
        self.fc1 = nn.Linear(1664 * 2, classCount)
        self.fc2 = nn.Linear(1664, classCount)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone1._forward(x)
        _, _, _, x4_2 = self.backbone2._forward(x)
        x_f = self.ms_fusion1(x1, x2)
        x_f = self.ms_fusion2(x_f, x3)
        x_f = self.ms_fusion3(x_f, x4)
        feat1 = self.pool(x4)
        feat1 = feat1.squeeze(-1).squeeze(-1)
        feat2 = self.pool(x_f)
        feat2 = feat2.squeeze(-1).squeeze(-1)
        feat3 = self.pool(x4_2)
        feat3 = feat3.squeeze(-1).squeeze(-1)
        y1 = self.fc1(torch.cat([feat1, feat2], dim=1))
        y2 = self.fc2(feat3)
        y = (y2 + y1) / 2.0
        return torch.sigmoid(y)


class model_densenet121_att(nn.Module):
    def __init__(self, classCount, isTrained, BatchNorm2d=nn.BatchNorm2d):
        super(model_densenet121_att, self).__init__()
        import densenet4

        self.backbone1 = densenet4.densenet169(pretrained=isTrained)  # 1024
        self.backbone2 = densenet4.densenet169(pretrained=isTrained)
        # torch.Size([2, 256, 56, 56])
        # torch.Size([2, 512, 28, 28])
        # torch.Size([2, 1024, 14, 14])
        # torch.Size([2, 1024, 7, 7])
        self.ms_fusion1 = ms_fusion(256, 512, 8, BatchNorm2d=BatchNorm2d)
        self.ms_fusion2 = ms_fusion(512, 1280, 8, BatchNorm2d=BatchNorm2d)
        self.ms_fusion3 = ms_fusion(1280, 1664, 8, BatchNorm2d=BatchNorm2d)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(2048, classCount)
        self.fc1 = nn.Linear(1664 * 2, classCount)
        self.fc2 = nn.Linear(1664, classCount)

    def getacm(self, y, classifier):
        b, c, h, w = y.size()
        y = y.permute([0, 2, 3, 1]).contiguous().view(-1, c)
        y = classifier(y)
        y = y.view(b, h, w, -1).permute([0, 3, 1, 2])
        return y

    def forward(self, x):
        x_flip = torch.flip(x, [3])
        x1, x2, x3, x4 = self.backbone1._forward(x)
        _, _, _, x4_flip = self.backbone2._forward(x_flip)
        x4_2 = torch.flip(x4_flip, [3])
        x_f = self.ms_fusion1(x1, x2)
        x_f = self.ms_fusion2(x_f, x3)
        x_f = self.ms_fusion3(x_f, x4)
        feat1 = torch.cat([x4, x_f], dim=1)
        cam1 = self.getacm(feat1, self.fc1)
        feat1 = self.pool(feat1)
        feat1 = feat1.squeeze(-1).squeeze(-1)
        cam2 = self.getacm(x4_2, self.fc2)
        feat2 = self.pool(x4_2)
        feat2 = feat2.squeeze(-1).squeeze(-1)

        y1 = self.fc1(feat1)
        y2 = self.fc2(feat2)
        y = (y2 + y1) / 2.0

        return torch.sigmoid(y), cam1, cam2


class CustomSwin_T(nn.Module):
    def __init__(self, classNum = 3):
        super(CustomSwin_T, self).__init__()
        weights = 'IMAGENET1K_V1'
        self.model_pre_trained = swin_b (weights, progress=bool)
        self.in_features = self.model_pre_trained.head.in_features
        swin_model = torch.nn.Sequential(*list(self.model_pre_trained.children())[:-1])
        self.swin_features = swin_model
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(in_features),#added by rostami
            # nn.Dropout(0.5), #added by rostami
            nn.Linear(self.in_features, classNum),
            nn.Softmax(dim=1)
        )

    def forward(self, x,features):

        # print(f' x shape before features = {x.shape}')
        # print(f'x min = {x.min()} x max = {x.max()}')
        # import matplotlib.pyplot as plt
        # t = x.detach().cpu()
        # image_array = t[0].permute(1, 2, 0).numpy()
        # plt.imshow(image_array)
        # plt.axis('off')  # Turn off axis
        # plt.show()
        x = self.swin_features(x)
        x = x.view(x.size(0), -1)
        #print(f'shape of x after swin feature viwew = {x.shape}')

        x = self.fc(x)
        return x



def swin_t_pretrained():
    model = CustomSwin_T()
    return model
##-----------------------------------------------------------


class JointMobileNet(nn.Module):

    def __init__(self, classCount, isTrained):
        super(JointMobileNet, self).__init__()
        in_features = 21
        mid_layer = 16
        out_features = 8
        self.feature_classifier = nn.Sequential(nn.Linear(in_features,mid_layer),
                                                nn.Linear(mid_layer, out_features=out_features))

        self.mobilenet = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        print(self.mobilenet.classifier)
        kernelCount = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = Identity()

        l = nn.Linear( kernelCount, out_features=1)
        self.fc = nn.Sequential(l)


    def init_normal(self,m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    def manual_prediction(self, features):
        pass

    def forward(self, x, features):
        x = self.mobilenet(x)
        x = torch.squeeze(x)
        #print(f' x shape = {x.shape}')
        res = self.fc(x)
        return res



class JointConvNextNet(nn.Module):

    def __init__(self, classCount, isTrained):
        super(JointConvNextNet, self).__init__()
        in_features = 21
        mid_layer = 16
        out_features = 8
        self.feature_classifier = nn.Sequential(nn.Linear(in_features,mid_layer),
                                                nn.Linear(mid_layer, out_features=out_features))
        #self.feature_classifier.apply(self.init_normal)

        self.mobilenet = torchvision.models.convnext_small(weights = torchvision.models.ConvNeXt_Small_Weights)
        print(self.mobilenet.classifier)
        kernelCount = self.mobilenet.classifier[2].in_features
        self.mobilenet.classifier = Identity()

        l = nn.Linear( kernelCount, out_features=1) # kernelCount + out_features
        #self.init_normal(l)
        #self.densenet121.classifier = nn.Sequential(l,  nn.Softmax())
        self.fc = nn.Sequential(l)


    def init_normal(self,m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    def manual_prediction(self, features):
        pass

    def forward(self, x, features):
        #print(f' x = {x}')
        # print(f'{features[:,1:]}     count', torch.count_nonzero(features[1:]))
        # if torch.count_nonzero(features[:,1:]) == 0:
        #     print("-------------------------------------------------")
        #     return torch.Tensor([1.0,0.0]).cuda()
        x = self.mobilenet(x)
        x = torch.squeeze(x)
        #print(f'x = {x}')
        #x_f = self.feature_classifier(features)
        #print(features)
        #return (x + x_f) / 2
        #print(f'x = {x}   x_f = {x_f}')
        #res = torch.cat(tensors=(x,x_f), dim=1)
        res = x
        #print(f'res {res.shape}')
        # print(f'x {x.shape}')
        # print(f'x_f {x_f.shape}')
        #print(f"res shape ====== {res.shape}    , x shape = {x.shape}  x_f shape = {x_f.shape}")
        res = self.fc(res)
        #print(f' res ====== {res}')
        #return (x_f + x)/2
        return res



class SWIN(nn.Module):
    def __init__(self, classCount, isTrained):
        super(SWIN, self).__init__()
        in_features = 21
        mid_layer = 16
        out_features = 8
        self.feature_classifier = nn.Sequential(nn.Linear(in_features,mid_layer),
                                                nn.Linear(mid_layer, out_features=out_features))

        self.swin = torchvision.models.swin_t(weights = torchvision.models.Swin_T_Weights)
        kernelCount = self.swin.head.in_features
        l = nn.Linear(kernelCount, out_features=1)  # kernelCount + out_features
        self.swin.head = l


    def manual_prediction(self, features):
        pass

    def forward(self, x, features):
        x = self.swin(x)
        return x


