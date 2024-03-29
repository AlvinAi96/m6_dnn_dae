"""
autoencoder_dnn_classifier.py
@author: aihongfeng
@date: 20220420
function:
    This script trains a DNN model with DAE based on the given training, validation, and testing sets.

    Prediction task: Multi-class classification
    Prediction target: Rank (class_num=5)
    Training target: All assets are trained together, resulting in only one model.
"""


import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
import time
import datetime
import warnings
warnings.filterwarnings('ignore')


from utils import read_dataset, symbol2assetID, plot_feat_importances, set_seed
from feature import feature_selection
from eval import multi_class_eval, cal_overallRPS, prepare_sub, sub_checker


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from sklearn.metrics import recall_score, f1_score, accuracy_score
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import random

# force to cpu running
cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpu)
print(tf.config.list_logical_devices())

def make_dataset(feature, asset_id, y, seed, batch_size=800, mode="train"):
    '''Prepare train dataset'''
    # transfer multi-D tensor -> 1d tensor
    ds = tf.data.Dataset.from_tensor_slices(((asset_id, feature), (feature, y, y)))
    if mode == "train":
        # in avoid of insufficient memory ，shuffle for each buffer_size data
        ds = ds.shuffle(buffer_size=4096, seed=seed)
    # Improve the training process by loading batches into memory, 
    # allowing for GPU training while the CPU prepares the data for the next training iteration.
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE) 
    return ds

def preprocess_test(asset, feature):
    return (asset, feature), (0,0,0)

def make_test_dataset(feature, asset_id, batch_size=800):
    '''Prepare test dataset'''
    ds = tf.data.Dataset.from_tensor_slices(((asset_id, feature)))
    ds = ds.map(preprocess_test)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE) 
    return ds

def get_model(feat_size, dr, id_size, id_df):
    """
    feat_size: int feature number (drop the asset column)
    dr: float dropout rate
    id_size: int asset number 
    id_df: asset's dataframe，used for lookup_layer
    ref: Kaggle Jane Street 1st on LeaderBoard Notebook: https://www.kaggle.com/code/vikazrajpurohit/jane-street-1st-on-leaderboard-notebook
    """
    asset_id_inputs = tf.keras.Input((1, ))
    feature_inputs = tf.keras.Input((feat_size,))

    # Build an index layer for IDs -> id embedding
    asset_id_lookup_layer = layers.IntegerLookup(max_tokens=id_size)
    asset_id_lookup_layer.adapt(id_df)
    asset_id_x = asset_id_lookup_layer(asset_id_inputs)
    asset_id_x = layers.Embedding(id_size, 32, input_length=1)(asset_id_x)
    asset_id_x = layers.Reshape((-1, ))(asset_id_x)
    asset_id_x = layers.Dense(64, activation='swish')(asset_id_x)
    asset_id_x = layers.Dense(64, activation='swish')(asset_id_x)
    asset_id_x = layers.Dense(64, activation='swish')(asset_id_x)

    # auto-encoder backbone layer (feat_size->35->feat_size)
    encoder = layers.GaussianNoise(0.3)(feature_inputs)
    encoder = layers.Dense(35)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('swish')(encoder)

    decoder = layers.Dropout(0.3)(encoder)
    decoder = layers.Dense(feat_size, name='decoder')(decoder)

    # auto-encoder decision layer
    x_ae = layers.Dense(35)(decoder)
    x_ae = layers.BatchNormalization()(x_ae)
    x_ae = layers.Activation('swish')(x_ae)
    x_ae = layers.Dropout(0.3)(x_ae)
    x_ae = layers.Concatenate(axis=1)([asset_id_x, x_ae])
    out_ae = layers.Dense(5, activation='softmax', name='ae_action')(x_ae)
    
    # id embedding + feature embedding -> decision making
    x = layers.Concatenate(axis=1)([asset_id_x, encoder])
    x = layers.Dropout(dr)(x)
    x = layers.Dense(64, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(dr)(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(dr)(x)
    x = layers.Dense(16, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(dr)(x)
    output = layers.Dense(5, activation='softmax', name='action')(x)

    model = tf.keras.Model(inputs=[asset_id_inputs, feature_inputs], outputs=[decoder, out_ae, output])
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                 loss={'decoder':tf.keras.losses.MeanSquaredError(),
                      'ae_action':'categorical_crossentropy',
                      'action':'categorical_crossentropy'},
                 metrics={'decoder':'mae', 'ae_action':'accuracy', 'action':'accuracy'})
    return model


def run(model_params, 
        root_path, train_fname, valid_fname, test_fname, pred_fname, meta_fname, seed,
        infer_flag=False, scale_flag=True,
        remove_unstable_flag=True, top_fnum=None, corr_thresh=0.10,
        save_model_flag=True):
    """run dnn"""
    # load data
    train_df, valid_df, test_df, pred_df, meta_df = read_dataset(root_path, train_fname, valid_fname, test_fname, pred_fname, meta_fname)
    print('Init | train_df:{}, valid_df:{}, test_df:{}, pred_df:{}'.format(train_df.shape, valid_df.shape, test_df.shape, pred_df.shape))

    #  If making inference and submitting results, train using the entire dataset
    if infer_flag == True:
        train_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    train_df.fillna(0, inplace=True)
    valid_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    # define feature column and target column
    NUM_CLASS = model_params['num_class']
    TARGET_COL = 'Rank'                                                 # gt hard Rank label
    PRED_RANK_COL = 'pred_'+TARGET_COL                                  # pred hard Rank label
    RANK_DIST_COLS = [TARGET_COL+str(i) for i in range(1, NUM_CLASS+1)] # gt soft Rank label
    PRED_RANK_DIST_COLS = ['pred_'+f for f in RANK_DIST_COLS]           # pred soft Rank label
    SELF_DEFINE_USELESS_COLS = ['symbol', 'GICS_sector/ETF_type', 'type_id', 'subtype_id', 'data_type', 'return'] 
    USE_COLS = []
    for f in list(train_df):
        if f not in ['asset', 'Date'] and \
                f != TARGET_COL and \
                f not in RANK_DIST_COLS and \
                f not in SELF_DEFINE_USELESS_COLS and \
                'ind_mean' not in f and 'ind_var' not in f:   # drop industry-agg feature
            USE_COLS.append(f) 
            
    print('init feature number:', len(USE_COLS))
    asset_id_df = pd.DataFrame({'asset':list(train_df['asset'].unique())})

    # standardlization
    if scale_flag == True:
        scaler = StandardScaler()
        not_id_use_cols = [f for f in USE_COLS if f != 'asset']
        scaler.fit(train_df[not_id_use_cols].values)
        train_df[not_id_use_cols] = scaler.transform(train_df[not_id_use_cols].values)
        valid_df[not_id_use_cols] = scaler.transform(valid_df[not_id_use_cols].values)
        test_df[not_id_use_cols] = scaler.transform(test_df[not_id_use_cols].values)
        pred_df[not_id_use_cols] = scaler.transform(pred_df[not_id_use_cols].values)

    # feature selection
    if remove_unstable_flag == True:
        unstable_feats = feature_selection.get_corr_unstable_feats(train_df, USE_COLS, top_fnum, corr_thresh)
        train_df.drop(unstable_feats, axis=1, inplace=True)
        valid_df.drop(unstable_feats, axis=1, inplace=True)
        test_df.drop(unstable_feats, axis=1, inplace=True)
        pred_df.drop(unstable_feats, axis=1, inplace=True)
        print('Feature Selection | train_df:{}, valid_df:{}, test_df:{}, pred_df:{}'.format(train_df.shape, valid_df.shape, test_df.shape, pred_df.shape))
        USE_COLS = [f for f in USE_COLS if f not in unstable_feats]
        
    USE_COLS.append('asset') 
    FEAT_SIZE = len(USE_COLS)                                       # feature number
    ID_SIZE = 100                                                   # asset number
    print('feature_size:', FEAT_SIZE)
    # print model
    model = get_model(feat_size=FEAT_SIZE, dr=model_params['dropout_rate'], id_size=ID_SIZE, id_df=asset_id_df)
    model.summary()
    # keras.utils.plot_model(model, show_shapes=True, to_file=root_path+'result/dnn_model_structure.png')
    del model

    # prepare dataset
    train_ds = make_dataset(train_df[USE_COLS], train_df['asset'], train_df[RANK_DIST_COLS], mode='train', seed=seed)
    valid_ds = make_dataset(valid_df[USE_COLS], valid_df['asset'], valid_df[RANK_DIST_COLS], mode='valid', seed=seed)
    test_ds = make_dataset(test_df[USE_COLS], test_df['asset'], test_df[RANK_DIST_COLS], mode='test', seed=seed)
    pred_ds = make_test_dataset(pred_df[USE_COLS], pred_df[['asset']])

    # train, valid, save model
    model = get_model(feat_size=FEAT_SIZE, dr=model_params['dropout_rate'], id_size=ID_SIZE, id_df=asset_id_df)
    if save_model_flag == True:
        model_path = root_path + 'model/autoencoder_dnn/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        checkpoint = keras.callbacks.ModelCheckpoint(model_path + "autoencoder_dnn_model2_2_featexp2", save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(patience=model_params['patience'])
    if infer_flag == True:
        history = model.fit(train_ds, epochs=model_params['epoch'], validation_data=test_ds, callbacks=[checkpoint, early_stop])
    else:
        history = model.fit(train_ds, epochs=model_params['epoch'], validation_data=valid_ds, callbacks=[checkpoint, early_stop])
    model = keras.models.load_model(model_path + "autoencoder_dnn_model2_2_featexp2")

    # predict
    train_df.loc[:, PRED_RANK_DIST_COLS] = model.predict(train_ds)[2]
    valid_df.loc[:, PRED_RANK_DIST_COLS] = model.predict(valid_ds)[2]
    test_df.loc[:, PRED_RANK_DIST_COLS] = model.predict(test_ds)[2]
    pred_df.loc[:, RANK_DIST_COLS] = model.predict(pred_ds)[2]
    train_df.loc[:, PRED_RANK_COL] = np.argmax(train_df[PRED_RANK_DIST_COLS].values, axis=1) # based on distribution，gain hard-rank prediciton
    valid_df.loc[:, PRED_RANK_COL] = np.argmax(valid_df[PRED_RANK_DIST_COLS].values, axis=1)
    test_df.loc[:, PRED_RANK_COL] = np.argmax(test_df[PRED_RANK_DIST_COLS].values, axis=1) 
    
    print('==========Total Result Analysis==========')
    # Calculate classification metrics for multiple-day assets.
    print('\n')
    trn_acc, trn_micro_recall, trn_macro_recall, trn_micro_f1, trn_macro_f1 = multi_class_eval(train_df['Rank'], train_df['pred_Rank'], 'train')
    val_acc, val_micro_recall, val_macro_recall, val_micro_f1, val_macro_f1 = multi_class_eval(valid_df['Rank'], valid_df['pred_Rank'], 'valid')
    tst_acc, tst_micro_recall, tst_macro_recall, tst_micro_f1, tst_macro_f1 = multi_class_eval(test_df['Rank'], test_df['pred_Rank'], 'test')

    # Calculate rps metrics for multiple-day assets
    trn_overallRPS = cal_overallRPS(train_df)
    val_overallRPS = cal_overallRPS(valid_df)
    tst_overallRPS = cal_overallRPS(test_df)
    print('RPS | train:%.3f, valid:%.3f, test:%.3f' % (trn_overallRPS, val_overallRPS, tst_overallRPS))

    # save result
    collect_cols = ['Date', 'asset'] + [TARGET_COL] + [PRED_RANK_COL] + RANK_DIST_COLS + PRED_RANK_DIST_COLS
    result_path = root_path + 'result/autoencoder_dnn/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    train_df[collect_cols].to_csv(result_path + 'dnn_pred_train_rank_df.csv', index=False)
    valid_df[collect_cols].to_csv(result_path + 'dnn_pred_valid_rank_df.csv', index=False)
    test_df[collect_cols].to_csv(result_path + 'dnn_pred_test_rank_df.csv', index=False)

    # prepare submission
    pred_df = pred_df.rename(columns={'symbol':'ID'})
    sub_df = prepare_sub(pred_df) 
    sub_df = sub_checker(sub_df, meta_df)  # check submission
    sub_fname = result_path + 'autoencoder_dnn_sub_{}_final.csv'.format(str(datetime.datetime.today()).split(' ')[0].replace('-',''))
    sub_df.to_csv(sub_fname, index=False)


if __name__=="__main__":
    seed = 1996
    set_seed(seed)    
    model_params = {'num_class':5,
                'dropout_rate':0.3,
                'patience':10,
                'epoch':500}

    root_path = './'
    feat_type = 'BF_TZS_AMN_WSA'
    train_fname = f'pp_data2/train_rank_df_{feat_type}.csv'
    valid_fname = f'pp_data2/valid_rank_df_{feat_type}.csv'
    test_fname = f'pp_data2/test_rank_df_{feat_type}.csv'
    pred_fname = f'pp_data2/predict_rank_df_{feat_type}.csv'
    meta_fname = f'pp_data2/M6_Universe.csv'

    # feature selection
    top_fnum = None
    corr_thresh = 0.155

    infer_flag = False               # Is it the inference stage (i.e., training using the entire dataset)?
    scale_flag = True               # scale or not
    remove_unstable_flag = True     # feature selection or not 
    save_model_flag = True          # save best model ckpt or not

    run(model_params, 
        root_path, train_fname, valid_fname, test_fname, pred_fname, meta_fname, seed,
        infer_flag, scale_flag,
        remove_unstable_flag, top_fnum, corr_thresh,
        save_model_flag)
        