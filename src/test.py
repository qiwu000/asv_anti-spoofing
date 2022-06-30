import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from feature import calc_cqt, calc_stft
from metrics import calculate_eer
from model.lcnn import build_lcnn


import pandas as pd
a=pd.read_csv("src\protocol\\train_protocol.csv")

print(a)