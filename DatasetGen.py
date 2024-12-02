import os
import random

#import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from settings import BACTERIAL_FOLDER, NON_BACTERIAL_FOLDER
from settings import MAIN_DIR, EXCEL_FILE,  BACTERIAL, NON_BACTERIAL, FEMALE, MALE, IMAGE_WIDTH, \
    IMAGE_HEIGHT
from util import get_train_preprocess
from collections import Counter




class DatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):

        self.pathImageDirectory = pathImageDirectory
        self.pathDatasetFile = pathDatasetFile

        self.listImagePaths = []
        # self.listImageLabels = []
        self.transform = transform

        lst = self.read_xcel_file(pathDatasetFile)
        self.listImagePaths = lst
        #self.listImageLabels =

    def read_xcel_file(self,path):
        lst = []
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O','P','Q','R','S','T','U','V','W','X']
        # columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V']
        # read the excel file into a pandas dataframe, specifying the columns to be read
        df = pd.read_excel(path, header=None, names=columns)
        df = df.fillna("-1")
        # df = df.replace('normal', NORMAL)
        # df = df.replace('viral' , VIRAL)
        # df = df.replace('bacterial', BACTERIAL)
        # df = df.replace('fungal', FUNGAL)
        df = df.replace('female', FEMALE)
        df = df.replace('male', MALE)
        # display the first five rows of the dataframe
        # print(df.head())
        first_row = True
        c = 0
        invalid_labels = 0
        for index, row in df.iterrows():
            # access the values of each column for the current row
            # print(row)
            if first_row:
                first_row = False
                continue
            path = self.pathImageDirectory + str(index)
            if not os.path.exists(path):
                #print(f'folder does not exist = {path}')
                continue
            if len(os.listdir(path)) < 1:
                #print(f"folder is empty = {path}")
                continue
            # print(index)
            # c_value = row['A']
            # print(f' c value = {c_value}')
            item = df.loc[index, :].values.flatten().tolist()
            item_lst = [index]
            item = [float(x) for x in item]
            #item[0] = item[0] / 100 # normalize age
            #print(f'label = {item[-1]}, item = {item}')

            if item[-1] == NON_BACTERIAL or  item[-1] == BACTERIAL:
                # print(f' item 3 = {item[1]}')
                item[1] = item[1] / 100 # normalize age
                item_lst.append(item[1])
                item_lst.extend(item[3:])
                lst.append(item_lst)
                c = c + 1

            else:
                print("invalid label = ",index)
                invalid_labels = invalid_labels + 1
                #print(f' no label for item = {index}')

        print(f'dict values len = {len(lst)}')
        print(f'c = {c} , invalid_labels= {invalid_labels}')
        #print(lst)
        #random.shuffle(lst)

        #seed (e.g. 4) is important to be fixed. It assures that validation and train data are disjoint and consistent
        random.Random(4).shuffle(lst)

        normal_c = 0
        viral_c = 0
        bac_c = 0
        others = 0
        for item in lst:
            if item[-1] == NON_BACTERIAL:
                normal_c += 1
            elif item[-1] == BACTERIAL:
                bac_c += 1
            else:
                others += 1
        print(f' non bacterial count = {normal_c}    bact count = {bac_c} others = {others}')
        return lst

    def get_item(self, index):
        item = self.listImagePaths[index]
        #print(f'item in first row = {item}')
        #print(f'item = {item}')
        features = item[1:-1]
        features = torch.FloatTensor(features)
        folder_path = self.pathImageDirectory + str(item[0]) + '/'
        files = os.listdir(folder_path)
        if len(files) < 1:
            print(f'no file exists in  {folder_path}')

        if len(files) < 1:
            print("folde is empty: ", folder_path)
        try:
            imagePath = folder_path + files[0]
            imageData = Image.open(imagePath)
            #imageData = imageData.convert('RGB')
            imageLabel = item[-1]
            #t= [0.0,0.0,0.0]
            #print(f'imageLabel = {imageLabel}')
            #t[int(imageLabel)] = 1.0
            #imageLabel = torch.FloatTensor(imageLabel)
            #print("imageLabel ======= ",imageLabel)
        except Exception as e:
            print(e)


        if self.transform != None:
            imageData = self.transform(imageData)

        #print(f'features = {features} , len feature = {len(features)}')
        #print(f' label = {imageLabel}')

        #print(f'imagePath = {imagePath}  features  = {features}  label = {imageLabel}')
        return (imageData, features, imageLabel)

    def __getitem__(self, index):
        return self.get_item(index)
    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.listImagePaths)



class IntervalDatasetGenerator (DatasetGenerator):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):
        super(IntervalDatasetGenerator, self).__init__(pathImageDirectory, pathDatasetFile, transform)
        self.main_list = self.listImagePaths

        print(f'self.main_list first item = {self.main_list[0]}')
        # self.main_label_list = self.listImageLabels
        # print(f' self.main_label_list = {self.main_label_list}  len self.listImageLabels = {len(self.listImageLabels)}')
        self.main_size = len(self.listImagePaths)
        self.size = self.main_size
        #print(f'main size = {self.main_size}')
    def set_interval(self, from_index, to_index, exclude= False):
        #print(f'in set interval')
        if from_index != to_index:
            if exclude == False:
                self.listImagePaths = self.main_list[from_index:to_index]
                #print(f'self.listImagePaths = {self.listImagePaths}')
                # self.listImageLabels = self.main_label_list[from_index:to_index]
                self.size = len(self.listImagePaths)
            else:
                self.listImagePaths = []
                self.listImageLabels = []
                #print(f' len self.main_list = {len(self.main_list)}')
                for i in range(len(self.main_list)):
                    #print(f' i = {i} from_index = {from_index} to index = {to_index}')
                    if i < from_index or i >= to_index:
                        self.listImagePaths.append(self.main_list[i])
                        # self.listImageLabels.append(self.main_label_list[i])
                    self.size = len(self.listImagePaths)

        return self


    def get_weights(self):
        import numpy
        labels = []
        dataLoaderTrain = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=1,
                                     pin_memory=True, drop_last=False)
        for batchID, (img, lst, target) in enumerate(dataLoaderTrain):
            #print(f' target in loop = {target}')
            labels.append(target.item())


        #print(f'targets = {labels}')
        di = Counter(labels)
        weights = di.values()  # counts the elements' frequency
        print(f'counter weights = {weights}')
        keys = di.keys()  # equals to list(set(words))
        print(f' keys = {keys}')
        sorted_key = sorted(keys)
        new_weights = []
        for k in sorted_key:
            new_weights.append(self.__len__()/di[k])


        return new_weights

    def __getitem__(self, index):
        #print('in get item ')
        return self.get_item(index)

    def __len__(self):
        #print(f'self size = {self.size}')
        return self.size








class FolderDatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------
    def __init__(self, pathImageDirectory, transform):

        self.pathImageDirectory = pathImageDirectory

        self.listImagePaths = []
        # self.listImageLabels = []
        self.transform = transform

        lst = self.read_image_file(self.pathImageDirectory)
        self.listImagePaths = lst
        #self.listImageLabels =

    def read_image_file(self,path):
        lst = []

        bact_list = os.listdir(path + BACTERIAL_FOLDER)
        non_bact_list = os.listdir(path + NON_BACTERIAL_FOLDER)
        for item in bact_list:
            f_path = path + BACTERIAL_FOLDER +  item
            lst.append(f_path)
        for item in non_bact_list:
            f_path = path + NON_BACTERIAL_FOLDER + item
            lst.append(f_path)


        #seed (e.g. 4) is important to be fixed. It assures that validation and train data are disjoint and consistent
        random.Random(4).shuffle(lst)

        nonbac_c = 0
        bac_c = 0
        others = 0
        for item in lst:
            if "/"+BACTERIAL_FOLDER in item:
                bac_c += 1
            elif  "/" + NON_BACTERIAL_FOLDER in item:
                nonbac_c += 1
            else:
                others += 1
        print(f' non bacterial count = {nonbac_c}    bact count = {bac_c} others = {others}')

        return lst

    def get_item(self, index):
        item = self.listImagePaths[index]
        #print(f'item in first row = {item}')
        #print(f'item = {item}')
        features = []
        features = torch.FloatTensor(features)
        try:
            imagePath = item
            image_data = Image.open(imagePath)
            #image_data = image_data.convert('RGB')
            if self.transform != None:
                image_data = self.transform(image_data)

            if "/"+BACTERIAL_FOLDER in item:
                imageLabel = BACTERIAL
            elif  "/" + NON_BACTERIAL_FOLDER in item:
                imageLabel = NON_BACTERIAL
            else:
                print("------------------------error in reading images in get_item")
            return (image_data, features, imageLabel, imagePath)
        except Exception as e:
            print(e)
            return None, None, None



        #print(f'features = {features} , len feature = {len(features)}')
        #print(f' label = {imageLabel}')

        #print(f'imagePath = {imagePath}  features  = {features}  label = {imageLabel}')

    def __getitem__(self, index):
        return self.get_item(index)
    # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.listImagePaths)




class FolderIntervalDatasetGenerator (FolderDatasetGenerator):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):
        super(FolderIntervalDatasetGenerator, self).__init__(pathImageDirectory, transform)
        self.main_list = self.listImagePaths

        print(f'self.main_list first item = {self.main_list[0]}')
        self.main_size = len(self.listImagePaths)
        self.size = self.main_size
    def set_interval(self, from_index, to_index, exclude= False):
        if from_index != to_index:
            if exclude == False:
                self.listImagePaths = self.main_list[from_index:to_index]
                self.size = len(self.listImagePaths)
            else:
                self.listImagePaths = []
                self.listImageLabels = []
                for i in range(len(self.main_list)):
                    #print(f' i = {i} from_index = {from_index} to index = {to_index}')
                    if i < from_index or i >= to_index:
                        self.listImagePaths.append(self.main_list[i])
                        # self.listImageLabels.append(self.main_label_list[i])
                    self.size = len(self.listImagePaths)

        return self
    def up_sample(self, class_folder = '/bact/', up_sample_rate= 5):
        temp_list = []
        for item in self.listImagePaths:
            if class_folder in item:
                for i in range(up_sample_rate):
                    temp_list.append(item)
        self.listImagePaths.extend(temp_list)
        self.size = len(self.listImagePaths)

    def get_weights(self):
        import numpy
        labels = []
        dataLoaderTrain = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=1,
                                     pin_memory=True, drop_last=False)
        for batchID, (img, lst, target, _) in enumerate(dataLoaderTrain):
            #print(f' target in loop = {target}')
            labels.append(target.item())


        #print(f'targets = {labels}')
        di = Counter(labels)
        weights = di.values()  # counts the elements' frequency
        print(f'counter weights = {weights}')
        keys = di.keys()  # equals to list(set(words))
        print(f' keys = {keys}')
        sorted_key = sorted(keys)
        new_weights = []
        for k in sorted_key:
            new_weights.append(self.__len__()/di[k])


        return new_weights

    def __getitem__(self, index):
        #print('in get item ')
        return self.get_item(index)

    def __len__(self):
        #print(f'self size = {self.size}')
        return self.size





if __name__ == "__main__":

    datasetTrain = IntervalDatasetGenerator(pathImageDirectory=MAIN_DIR, pathDatasetFile=EXCEL_FILE,
                                    transform=get_train_preprocess((IMAGE_WIDTH, IMAGE_HEIGHT)))

    datasetTrain = datasetTrain.set_interval(10,21)

    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=2, shuffle=True, num_workers=3,
                                 pin_memory=True, drop_last=True)

    for batchID, (img, lst, target) in enumerate(dataLoaderTrain):
        print(target)
        print("\n")
