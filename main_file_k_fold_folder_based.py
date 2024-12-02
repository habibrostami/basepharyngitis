import json
import logging
import numpy
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import util
from DatasetGen import FolderIntervalDatasetGenerator
from models import DenseNet121, DenseNet169, DenseNet201, swin_t_pretrained, JointDenseNet121, \
    JointMobileNet, JointConvNextNet, SWIN
from settings import MAIN_DIR, EXCEL_FILE, ROOT_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, LOG_FILE, PRETRAINED_KOREA_MODEL_PATH, \
    PRETRAINED_KOREA_MODEL_PATH_MOBILE_NET, PRETRAINED_KOREA_MODEL_PATH_SWIN


def base_augment():
    transformList = []
    transformList.append(transforms.RandomRotation(90))  # Rotate image 90 degrees to the right
    transformList.append(transforms.RandomHorizontalFlip())  # Horizontal flip
    transformList.append(transforms.RandomAffine((13, 28)))  # Translate image
    transformList.append(transforms.RandomAutocontrast())
    # transformList.append(transforms.RandomEqualize(p=1))
    return transformList


def train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
          trMaxEpoch, transResize, checkpoint, model_out_path_dir, num_folds=5):
    # -------------------- SETTINGS: NETWORK ARCHITECTURE

    # -------------------- SETTINGS: DATA TRANSFORMS
    transformSequence = util.get_train_preprocess(transResize)
    valtransfromSequence = util.get_val_test_preprocess(transResize)

    # -------------------- SETTINGS: DATASET BUILDERS
    print(f'MAIN Data PATH = {pathDirData}')
    main_dataset = FolderIntervalDatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
                                                  transform=transformSequence)

    main_dataset_len = main_dataset.main_size
    # remove old file

    for k in range(num_folds):
        from_index = k * (main_dataset_len // num_folds)
        to_index = min(from_index + main_dataset_len // num_folds, main_dataset_len)

        print(f'from_index = {from_index} to_index = {to_index}')
        TrainDataset = FolderIntervalDatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
                                                      transform=transformSequence)
        if num_folds == 1:
            exclude = False
        else:
            exclude = True

        TrainDataset.set_interval(from_index, to_index, exclude=exclude)
        #TrainDataset.up_sample(class_folder="/bact/", up_sample_rate=5)
        dataLoaderTrain = DataLoader(dataset=TrainDataset, batch_size=trBatchSize, shuffle=True, num_workers=3,
                                     pin_memory=True, drop_last=True)

        weights = TrainDataset.get_weights()
        # weights = None
        print(f'weights = {weights}')
        ValDataSet = FolderIntervalDatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
                                                    transform=transformSequence)

        ValDataSet.set_interval(from_index, to_index, exclude=False)

        dataLoaderVal = DataLoader(dataset=ValDataSet, batch_size=trBatchSize, shuffle=True, num_workers=3,
                                   pin_memory=True, drop_last=True)

        # val_normal = 0
        # val_viral = 0
        # val_bact = 0
        # for batchID, (img, features, target) in enumerate(dataLoaderVal):
        #     #print(f' target = {target}')
        #     if torch.argmax(target).item() == NORMAL-1:
        #         val_normal += 1
        #     elif torch.argmax(target).item() == VIRAL-1:
        #         val_viral += 1
        #     elif torch.argmax(target).item() == BACTERIAL-1:
        #         val_bact += 1
        # print(f' -- val_normal = {val_normal}, val_viral = {val_viral} , val_bact = {val_bact}')
        if nnArchitecture == 'JointDenseNet121':
            model = JointDenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-121':
            model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169':
            model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201':
            model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'SWIN-NET':
            model = swin_t_pretrained().cuda()
        elif nnArchitecture == 'MOBILE_NET':
            model = JointMobileNet(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'JointConvNextNet':
            model = JointConvNextNet(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == "SWIN":
            model = SWIN(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'RESNET-34':
        #     model = models.resnet34(pretrained=nnIsTrained)
        #     num_ftrs = model.fc.in_features
        #     model.fc = nn.Linear(num_ftrs, nnClassCount)  # Assuming nnClassCount is 3 for your case

        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
        if checkpoint is not None:
            model = load_pretrained_weights(model, checkpoint)
            print(f'pre trained model loaded----')
        k_fold_train(model, dataLoaderTrain, dataLoaderVal, nnClassCount=3, weights=weights)


# --------------------------------------------------------------------------------

def load_pretrained_weights(model, model_address):
    print(f"pre trained model address checkpoint = {model_address}")
    modelCheckpoint = torch.load(model_address)
    model.load_state_dict(modelCheckpoint['state_dict'])
    return model


def k_fold_train(model, dataLoaderTrain, dataLoaderVal, nnClassCount=3, weights=None):
    # -------------------- SETTINGS: OPTIMIZER & SCHEDULER

    optimizer = optim.Adam(model.parameters(), lr=0.0001, )  # betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, threshold=0.01, mode='min')

    label_smoothing = 0.3
    if weights is not None:
        weights = torch.Tensor(weights)
        weights = weights.cuda()
        print(f'in loss weight = {weights}')
        loss = torch.nn.BCELoss(weight=weights[1])
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1])

    else:
        loss = torch.nn.BCEWithLogitsLoss()

    print(f' loss after definition = {loss}')
    # ---- Load checkpoint

    # ---- TRAIN THE NETWORK
    lossMIN = 100000

    for epochID in range(0, trMaxEpoch):

        print(f'loss in for loop = {loss}')
        print('epoch_train:', epochID)
        epochTrain(model=model, dataLoader=dataLoaderTrain, optimizer=optimizer, scheduler=scheduler,
                   epochMax=trMaxEpoch, loss=loss, epochID=epochID, weights=weights)

        print('val epoch ID :', epochID)
        with torch.no_grad():
            lossVal, losstensor, _ = epochVal(model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount,
                                              loss)

        # scheduler.step(losstensor.item()) # uncomment -- habib rostami
        print(f' loss val = {lossVal} , loss min = {lossMIN} --------')
        if lossVal < lossMIN:
            # if 1:

            lossMIN = lossVal
            # /mnt/2T/ssajed/peerj/Correlation_consolidation.pth.tar
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                        'optimizer': optimizer.state_dict()}, model_out_path_dir + "model" + ".pth.tar")
        elif epochID % 5 == 0:
            # torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + str(epochID) + '.pth.tar')
            print('Epoch [' + str(epochID + 1) + ']   loss= ' + str(lossVal))
        else:
            print('Epoch [' + str(epochID + 1) + ']   loss= ' + str(lossVal))

    lossVal, losstensor, log_dict = epochVal(model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
    with open(LOG_FILE, "a") as log_file:
        log_file.write("-------------------------------\r\n")
        log_file.write(json.dumps(log_dict))


def epochTrain(model, dataLoader, optimizer, scheduler, epochMax, loss, epochID, weights):
    c = 0
    model.train()
    true_labels = []
    pred_labels = []

    print(f' loss in epoch train = {loss}')

    for batchID, (img, features, target, _) in enumerate(dataLoader):
        c = c + 1
        # print(f'target type = {target.dtype}')
        target = target.to(torch.float32)
        target = target.cuda()
        varInput1 = torch.autograd.Variable(img).cuda()
        varInput2 = torch.autograd.Variable(features).cuda()
        varTarget = torch.autograd.Variable(target).cuda()
        # imagePath = torch.autograd.Variable(imagePath).cuda()

        # varOutput, cor_loss = model(varInput1, varInput2)

        optimizer.zero_grad()

        varOutput = model(varInput1, varInput2)
        varOutput = torch.squeeze(varOutput)
        # print(f' varTarget = {varTarget} varTarget type = {varTarget.dtype}  varOutput = {varOutput} varoutType = {varOutput.dtype}')

        lossvalue = util.compute_loss(loss, varOutput, varTarget, weight=weights)
        # print(f'lossvalue = {lossvalue} , optimizer = {optimizer} loss func = {loss}')
        # check this as double check habib rostami
        lossvalue.backward()
        optimizer.step()
        # scheduler.step(lossvalue)

        true_labels.extend(varTarget.detach().cpu().numpy().tolist())
        pred_labels.extend(varOutput.detach().cpu().numpy().tolist())

    true_labels = numpy.array(true_labels)
    pred_labels = numpy.array(pred_labels)

    # print(f'pred label ={pred_labels}')
    acc_Train = util.accuracy(true_labels, pred_labels)
    print('accuracy_Train:', acc_Train)

    precision_train = util.precision(true_labels, pred_labels)
    print(f'Precision Train: {precision_train}')

    recall_train = util.recall(true_labels, pred_labels)
    print(f'Recall Train: {recall_train}')

    conf = util.confusion_matrix(true_labels, pred_labels)
    print('confusion = ', conf)
    # results['conf'] = conf

    f1_Train = util.f1(true_labels, pred_labels)
    print('F1 Score_Train:', f1_Train)

    specificity_Train = util.specificity_score(true_labels, pred_labels)
    print('Specificity_Train:', specificity_Train)

    # print(f'true label = {true_labels}')
    AUC_Train = util.auc(true_labels, pred_labels)
    print('AUC_Train:', AUC_Train)

    Sensitivity_Train = util.sensitivity(true_labels, pred_labels)
    print('Sensitivity_Train:', Sensitivity_Train)

    logging.info(f"Epoch [{epochID}] - "
                 f"Train Accuracy: {acc_Train:.4f}, "
                 f"Train Precision: {precision_train:.4f}, "
                 f"Train Recall: {recall_train:.4f}, "
                 f"Train F1 Score: {f1_Train:.4f}, "
                 f"Specificity_Train: {specificity_Train:.4f}, "
                 f"Sensitivity_Train: {Sensitivity_Train:.4f}, "
                 f"AUC_Train: {AUC_Train:.4f}, ")


# --------------------------------------------------------------------------------
def epochVal(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
    model.eval()
    true_labels = []
    pred_labels = []

    lossVal = 0

    lossValNorm = 0

    losstensorMean = 0
    incorrect_predictions = []
    # image_paths  = []

    for i, (input1, input2, target, _) in enumerate(dataLoader):
        # print('val',i,'/',len(dataLoader))

        # print(f'input1 shape = {input1.shape} , input2 shape = {input2.shape}')
        target = target.to(torch.float32)
        target = target.cuda()

        # varInput1 = torch.autograd.Variable(input1)
        # varInput2 = torch.autograd.Variable(input2)

        varTarget = torch.autograd.Variable(target)
        # Path= torch.autograd.Variable(imagePath)

        # varOutput = model(varInput1, varInput2)
        # varOutput, varTarget = curate_values_for_index(varOutput, varTarget, CLASS_INDEX)
        # print(f'before calling model')

        varOutput = model(input1, input2)

        # print(f'after calling model')

        varOutput, varTarget = util.curate_values_for_index(varOutput, varTarget)

        # image_paths = DatasetGenerator.get_image_paths()

        # incorrect prediction
        # if varOutput != varTarget:
        #     incorrect_predictions.append(varInput1, varTarget, varOutput, Path)
        #     print(f"varInput1: {varInput1} , varTarget: {varTarget} , varOutput: {varOutput}, image_paths: {Path}")

        # if loss is not None:
        #     varOutput = torch.squeeze(varOutput)
        #     losstensor = loss(varOutput, varTarget)
        #
        #     alpha = 0.25
        #     gamma = 2
        #     #print(f' losstensor = {losstensor}')
        #     pt = torch.exp(-losstensor)
        #     losstensor = (alpha * (1 - pt) ** gamma * losstensor).mean()  # mean over the batch
        #
        #     # cor_loss = cor_loss * 0.0000001
        #     # print(f' lossvalue = {lossvalue} , cor_loss = {cor_loss}')
        #     # losstensor = losstensor + cor_loss
        #
        #     losstensorMean += losstensor
        #     # print(f'var outttput = {varOutput}   ')
        #     # print(f'target = {varTarget}')
        #     lossVal += losstensor.item()
        #     lossValNorm += 1

        true_labels.extend(varTarget.detach().cpu().numpy().tolist())
        pred_labels.extend(varOutput.detach().cpu().numpy().tolist())

    # if loss is not None:
    #     outLoss = lossVal / lossValNorm
    #     losstensorMean = losstensorMean / lossValNorm
    outLoss = lossVal
    true_labels = numpy.array(true_labels)
    pred_labels = numpy.array(pred_labels)

    # print(f'out of acc true label = {true_labels}')
    # print(f'out of acc pred label = {pred_labels}')

    acc_Validation = util.accuracy(true_labels, pred_labels)
    print('Accuracy_Validation:', acc_Validation)

    precision_val = util.precision(true_labels, pred_labels)
    print(f'Precision Validation: {precision_val}')

    recall_val = util.recall(true_labels, pred_labels)
    print(f'Recall Validation: {recall_val}')

    f1_Validation = util.f1(true_labels, pred_labels)
    print('F1 Score_Validation:', f1_Validation)
    specificity_Validation = util.specificity_score(true_labels, pred_labels)
    print('Specificity_Validation:', specificity_Validation)

    AUC_val = util.auc(true_labels, pred_labels)
    print('AUC_Validation:', AUC_val)

    conf = util.confusion_matrix(true_labels, pred_labels)
    print('confusion validation = ', conf)

    specificity_Validation = util.specificity_score(true_labels, pred_labels)
    print('Specificity_Validation:', specificity_Validation)

    AUC_val = util.auc(true_labels, pred_labels)
    print('AUC_Validation:', AUC_val)

    Sensitivity_Validation = util.sensitivity(true_labels, pred_labels)
    print('Sensitivity_Validation:', Sensitivity_Validation)

    log_dict = {"acc_Validation": acc_Validation, "precision_val": precision_val, "recall_val": recall_val,
                "f1_Validation": f1_Validation,
                "AUC_val": AUC_val, "conf_val": conf}

    logging.info(f"Validation Accuracy: {acc_Validation:.4f},"
                 f"Precision_Validation: {precision_val :.4f},"
                 f"Recall_Validation: {recall_val :.4f},"
                 f"F1 Score_Validation: {f1_Validation :.4f},"
                 f"Specificity_Validation: {specificity_Validation :.4f},"
                 f"Sensitivity_Validation: {Sensitivity_Validation:.4f},"
                 f"AUC_Validation: {AUC_val:.4f}.")

    return outLoss, losstensorMean, log_dict


if __name__ == "__main__":
    #nnArchitecture = 'DENSE-NET-121'
    #nnArchitecture = 'JointDenseNet121'
    # nnArchitecture = 'MOBILE_NET'
    # nnArchitecture = "JointConvNextNet"
    nnArchitecture = "SWIN"
    trBatchSize = 20
    trMaxEpoch = 3
    transResize = (IMAGE_WIDTH, IMAGE_HEIGHT)
    checkpoint = None
    # checkpoint = PRETRAINED_KOREA_MODEL_PATH
    # checkpoint = PRETRAINED_KOREA_MODEL_PATH_MOBILE_NET
    checkpoint = PRETRAINED_KOREA_MODEL_PATH_SWIN
    model_out_path_dir = ROOT_DIR + "models/"

    train(pathDirData=MAIN_DIR, pathFileTrain=EXCEL_FILE, pathFileVal=EXCEL_FILE, nnArchitecture=nnArchitecture,
          nnIsTrained=True, nnClassCount=2, trBatchSize=trBatchSize, trMaxEpoch=trMaxEpoch, transResize=transResize,
          checkpoint=checkpoint, model_out_path_dir=model_out_path_dir, num_folds=1)



