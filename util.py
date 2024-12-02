import numpy
import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix as skleanconfusion_matrix
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from functools import partial
from typing import Any, Callable, List, Optional, TypeVar

#from pytorch_histogram_matching import Histogram_Matching
from skimage.exposure import match_histograms

from PIL import Image
import torchvision
from settings import trBatchSize

class histo_match(torch.nn.Module):
    def __init__(self):
        super(histo_match, self).__init__()
        #self.HM = Histogram_Matching(differentiable=True)
        REF_IMG = "/mnt/2T/Shojaei/pharyngitis/data/histo_ref/9ijhgfd3efdv.JPG"
        self.ref_imageData = Image.open(REF_IMG)
        self.ref_imageData = self.ref_imageData.convert('RGB')
        self.ref_imageData = torchvision.transforms.functional.pil_to_tensor(self.ref_imageData)
        # self.ref_imageData = self.ref_imageData.unsqueeze(dim=0)

        print("in init ------------------------------------------------")
    def forward(self,  img):  # we assume inputs are always structured like this
        if img.ndim == 4:
             channel_axis = 1
        else:
            channel_axis = 0

        print(
            f"I'm transforming an image of shape  type of image = {type(img)}  type of ref = {type(self.ref_imageData)}",
            f"shape of image = {img.shape}  shape of ref  = {self.ref_imageData.shape}"
        )
        rst = match_histograms(img, self.ref_imageData, channel_axis=channel_axis)
        print(f"rst type = type{rst}")

        #rst = self.HM(img, self.ref_imageData)
        return rst


device = torch.cpu
if torch.cuda.is_available():
    device = torch.cuda


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_val_test_preprocess(transResize):
    val_transformList = []
    val_transformList.append(transforms.Resize(transResize))
    val_transformList.append(transforms.ToTensor())
    val_transformList.append(normalize)
    valtransfromSequence = transforms.Compose(val_transformList)
    return valtransfromSequence

def get_train_preprocess(transResize):
    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)
    return transformSequence


def accuracy (y_True, y_Pred):
    acc = []
    #print(f'y true = {y_True}  ')
    #print(f'y Pred = {y_Pred}  ')
    threshold = 0.5
    y_Pred = numpy.where(y_Pred >= threshold, 1, 0)

    # print(f' y true  = {y_True}')
    # print(f' y pred  = {y_Pred}')

    acc = accuracy_score(y_True,y_Pred)

    # for i in range (len(y_True)):
    #     acc.append (accuracy_score (y_True[i], y_Pred[i]))
    return acc

def specificity_score (y_True , y_pred):
    specificity = []
    threshold = 0.5
    y_pred = numpy.where(y_pred >= threshold, 1, 0)
    try:
        #print(y_pred)
        #print(y_True)
        tn, fp, _, _ = skleanconfusion_matrix(y_True, y_pred).ravel()
        specificity = tn / (tn + fp)
    except Exception as e:
        print(f' specificity_score - exception = {e}')
        specificity = 0
    return specificity


def confusion_matrix (y_True , y_pred):
    specificity = []
    threshold = 0.5
    y_pred = numpy.where(y_pred >= threshold, 1, 0)
    res = multilabel_confusion_matrix(y_True, y_pred)

    return res.tolist()


def f1 (y_True , y_pred):

    f1 = []
    threshold = 0.5
    y_pred = numpy.where(y_pred >= threshold, 1, 0)
    f1 = f1_score (y_True , y_pred, average='macro',)
    return f1

def precision(y_True , y_pred):
    threshold = 0.5
    y_pred = numpy.where(y_pred >= threshold, 1, 0)
    p = precision_score(y_True , y_pred, average='macro')
    return p

def recall(y_True , y_pred):
    threshold = 0.5
    y_pred = numpy.where(y_pred >= threshold, 1, 0)
    r = recall_score(y_True , y_pred, average='macro')
    return r

def sensitivity (y_True, y_pred):

    sens = []
    threshold = 0.5
    y_pred = numpy.where(y_pred >= threshold, 1, 0)
    sens = recall_score (y_True, y_pred, average='macro')
    return sens

def auc (y_True, y_pred):

    auc = []
    threshold = 0.5
    #y_pred = numpy.where(y_pred >= threshold, 1, 0)
    try:
        auc = roc_auc_score(y_True, y_pred, average='macro', multi_class="ovr")
    except:
        auc = 0.0

    return auc


def compute_loss(loss, varOutput, varTarget, weight = None):
    varOutput, varTarget = curate_values_for_index(varOutput, varTarget)
    lossvalue = loss(varOutput, varTarget)

    # if weight is not None:
    #     pos_weight = weight[1]
    #     neg_weight = weight[0]
    #     varOutput = torch.clamp(varOutput, min=1e-8, max=1 - 1e-8)
    #     lossvalue = pos_weight * (varTarget * torch.log(varOutput)) + neg_weight * ((1 - varTarget) * torch.log(1 - varOutput))
    #     lossvalue =  torch.neg(torch.mean(lossvalue))
    # else:
    #     lossvalue = loss(varOutput, varTarget)


    #print(f'loss value = {lossvalue}')
    return lossvalue

def curate_values_for_index(pred,label):
    return pred, label


def load_pretrained_weights():
    return None