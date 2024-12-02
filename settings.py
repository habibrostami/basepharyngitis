# MAIN_DIR = "/mnt/2T/Shojaei/pharyngitis/data/images/"
MAIN_DIR= "/mnt/2T/Shojaei/pharyngitis/data/histomatched_images_folders/"
ROOT_DIR = "/mnt/2T/Shojaei/pharyngitis/pharyngitis/"
EXCEL_FILE = "/mnt/2T/Shojaei/pharyngitis/pharyngitis/last_excel.xlsx"

BACTERIAL_FOLDER = "bact/"
NON_BACTERIAL_FOLDER = "anonbact/"
PRETRAINED_KOREA_MODEL_PATH = "/mnt/2T/Shojaei/pharyngitis/pharyngitis/models/JointConvNextNet_korea.pth.tar"
PRETRAINED_KOREA_MODEL_PATH_MOBILE_NET = "/mnt/2T/Shojaei/pharyngitis/pharyngitis/models/MobileNet_model.pth.tar"
PRETRAINED_KOREA_MODEL_PATH_SWIN = "/mnt/2T/Shojaei/pharyngitis/pharyngitis/models/swin_model_korea.pth.tar"

#for histogram matching
REF_IMG = "/mnt/2T/Shojaei/pharyngitis/data/histomatched_images_folders/ref.jpg"

IMAGE_WIDTH = 224  # Width of your images
IMAGE_HEIGHT = 224  # Height of your images
NON_BACTERIAL = 0
BACTERIAL = 1
TEST_FILE = "/mnt/2T/Shojaei/pharyngitis/test.txt"
LOG_FILE = "/mnt/2T/Shojaei/pharyngitis/log.txt"
FEMALE = 1
MALE = 0

trBatchSize = 30