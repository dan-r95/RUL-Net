# %autoreload 2
# %autoreload

#import importlib
# importlib.reload(analyse_Data)


from data_processing import RESCALE, test_engine_id
from model import CNNLSTM
from matplotlib import pyplot as plt
import time
import datetime
from utils_laj import *
from data_processing import get_CMAPSSData, get_PHM08Data, data_augmentation, analyse_Data

today = datetime.date.today()


dataset = "cmaps"
file = 1  # represent the sub-dataset for cmapss
TRAIN = True
TRJ_WISE = True
PLOT = True
path = "data/CMaps/"

analyse_Data(path=path, dataset=dataset, files=[
             file], plot=False, min_max=False)

if TRAIN:
    data_augmentation(path=path,
                      files=file,
                      low=[10, 35, 50, 70, 90, 110, 130, 150, 170,
                           190, 210, 230, 250, 270, 290, 310, 330],
                      high=[35, 50, 70, 90, 110, 130, 150, 170, 190,
                            210, 230, 250, 270, 290, 310, 330, 350],
                      plot=False,
                      combine=False)


epochs = 500
lr = 000.1

CNNLSTM(dataset=dataset,path=path, file_no=file, Train=TRAIN, trj_wise=TRJ_WISE, plot=PLOT, epochs= epochs,  learning_rate = lr)
