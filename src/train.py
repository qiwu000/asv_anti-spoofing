"""

Train the LCNN and predict.
训练LCNN以及预测

Note: This is optimized for ASVspoof2019's competition.
      If you wnat to use for your own data  change the database path.


Todo:
    * Select 'feature_type'(fft or cqt).
    * Set the path to 'saving_path' for saving your model.
    * Set the Database path depends on your enviroment.

注意：这是针对 ASVspoof2019 的比赛进行了优化。
       如果您想为自己的数据使用更改数据库路径。


要做的：
     * 选择“feature_type”（fft 或 cqt）。
     * 将路径设置为“saving_path”以保存模型。
     * 设置数据库路径取决于您的环境。

"""


import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from feature import calc_cqt, calc_stft
from metrics import calculate_eer
from model.lcnn import build_lcnn

# ---------------------------------------------------------------------------------------------------------------------------------------
# model parameters
epochs = 100
batch_size = 256
lr = 0.00001

# We can use 2 types of spectrogram that extracted by using FFT or CQT.
# Set cqt of stft.
# 我们可以使用通过 FFT 或 CQT 提取 2 种频谱图。
# 设置 stft 的 cqt。
#   特征类型
feature_type = "cqt"

# The path for saving model
# This is used for ModelChecking callback.
# 模型保存路径
# 用于 ModelChecking 回调。
#    保存的路径
saving_path = "lcnn.h5"
# ---------------------------------------------------------------------------------------------------------------------------------------


# Replace the path to protcol of ASV2019 depending on your environment.
# 根据环境替换 ASV2019 协议的路径。
#       csv为标签
protocol_tr = "./protocol/train_protocol.csv"
protocol_dev = "./protocol/dev_protocol.csv"
protocol_eval = "./protocol/eval_protocol.csv"

# Choose access type PA or LA.
# Replace 'asvspoof_database/ to your database path.
# 选择访问类型 PA 或 LA。
# 将 'asvspoof_database/ 替换为您的数据库路径。
#      。。。asvspoof_database/"PA"/ASVspoof2019_PA_train/flac/。。。
#

access_type = "PA"#通道选择
path_to_database = "asvspoof_database/" + access_type
path_tr = path_to_database + "/ASVspoof2019_" + access_type + "_train/flac/"
path_dev = path_to_database + "/ASVspoof2019_" + access_type + "_dev/flac/"
path_eval = path_to_database + "/ASVspoof2019_" + access_type + "_eval/flac/"

if __name__ == "__main__":

    #读入标签
    #    tr 训练集  
    df_tr = pd.read_csv(protocol_tr)
    #    dev 开发集（用于调参、选择特征）
    df_dev = pd.read_csv(protocol_dev)


    #提取stft特征
    if feature_type == "stft":
        print("正在提取训练数据...")
        x_train, y_train = calc_stft(df_tr, path_tr)
        print("正在提取开发数据...")
        x_val, y_val = calc_stft(df_dev, path_dev)

    #提取cqt特征
    elif feature_type == "cqt":
        print("正在提取训练数据...")
        x_train, y_train = calc_cqt(df_tr, path_tr)
        print("正在提取开发数据...")
        x_val, y_val = calc_cqt(df_dev, path_dev)


    input_shape = x_train.shape[1:]
    lcnn = build_lcnn(input_shape)

    lcnn.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    #回调函数
    es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    cp_cb = ModelCheckpoint(
        filepath="./model",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
    )

    # Train LCNN
    history = lcnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=[x_val, y_val],
        callbacks=[es, cp_cb],
    )
    del x_train, x_val

    print("Extracting eval data（提取评估数据）")
    df_eval = pd.read_csv(protocol_eval)

    if feature_type == "stft":
        x_eval, y_eval = calc_stft(df_eval, path_eval)

    elif feature_type == "cqt":
        x_eval, y_eval = calc_cqt(df_eval, path_eval)

    # predict
    preds = lcnn.predict(x_eval)

    score = preds[:, 0] - preds[:, 1]  # Get likelihood   获取似然比
    eer = calculate_eer(y_eval, score)  # Get EER score     获取EER
    print(f"EER : {eer*100} %")
