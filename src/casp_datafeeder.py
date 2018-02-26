import numpy as np
import glob
import sys
import os
import pandas as pd




def read_csv_file_w_mids(csv_fn, feature_columns, label_column , validation_trjs=[], train_trjs=[]):
    trjranges_train=[]
    trjranges_test=[]
    train_x=[]
    train_y=[]
    train_mids=[]
    test_x=[]
    test_y=[]
    test_mids=[]

    csv_df = pd.read_csv(csv_fn, index_col=0)
    
    trs = csv_df.TR.unique()
    
    # loop over all trj ...
    for tr in trs:
        runtypes = csv_df.loc[csv_df.TR == tr, "RUNTYPE"].unique()
        
        for rt in runtypes:
            runs = csv_df.loc[(csv_df.TR == tr) & (csv_df.RUNTYPE == rt), "RUN"].unique()
            runs = runs[~np.isnan(runs)] # remove nans
            
            for r in sorted(runs):
                trj_name = str(tr)+"_"+str(rt)+"_"+str(r)
                
                
                trj_df = csv_df.loc[(csv_df.TR == tr) & (csv_df.RUNTYPE == rt) & (csv_df.RUN == r), ["TR", "RUNTYPE", "RUN", "TIME_PS", "MODELID"]+feature_columns+[label_column]].sort_values("TIME_PS")

                
                if trj_name in validation_trjs: # put trj into test set
                    print "reading trj :", trj_name, "(TEST)"
                    start = len(test_x)
                    test_x.extend(trj_df[feature_columns].values)
                    test_y.extend(np.array(trj_df[label_column].values, dtype=np.int64))
                    
                    mid_str_arr = [ "TR:"+str(tr)+"$RT:"+str(rt)+"$R:"+str(int(r))+"$M:"+str(int(mid))+"$T:"+str(int(t)) for tr, rt, r, mid, t in trj_df[["TR", "RUNTYPE", "RUN", "MODELID", "TIME_PS"]].values]
                    test_mids.extend(mid_str_arr)

                    end = len(test_x)
                
                    trjranges_test.append({"start": start, "end":end, "name":trj_name})
                elif trj_name in train_trjs or len(train_trjs) == 0: # put trj into train set
                    print "reading trj :", trj_name, "(TRAIN)"
                    start = len(train_x)
                    train_x.extend(trj_df[feature_columns].values)
                    train_y.extend(np.array(trj_df[label_column].values, dtype=np.int64))

                    mid_str_arr = [ "TR:"+str(tr)+"$RT:"+str(rt)+"$R:"+str(int(r))+"$M:"+str(int(mid))+"$T:"+str(int(t)) for tr, rt, r, mid, t in trj_df[["TR", "RUNTYPE", "RUN", "MODELID", "TIME_PS"]].values]
                    train_mids.extend(mid_str_arr)
                    end = len(train_x)
                
                    trjranges_train.append({"start": start, "end":end, "name":trj_name})
                else:
                    print "reading trj :", trj_name, "(SKIP)"

    #check if we have entries
    if len(trjranges_train) == 0:
        sys.exit("No training data has been found. Aborting.")
        
    return train_x, train_y, train_mids, trjranges_train, test_x, test_y, test_mids, trjranges_test
#--



def read_csv_file_predict(csv_fn, feature_columns, trjs=[]):
    trjranges=[]

    x=[]
    trj_names=[]
    mids=[]

    csv_df = pd.read_csv(csv_fn, index_col=0)
    
    trs = csv_df.TR.unique()
    
    # loop over all trj ...
    for trj_name in trjs:
        print "reading trj :", trj_name
        #example: TR759_point_rst_8.0
        tr = trj_name.split("_")[0]
        r = float(trj_name.split("_")[-1])

        rt = "_".join(trj_name.split("_")[1:-1])

        trj_df = csv_df.loc[(csv_df.TR == tr) & (csv_df.RUNTYPE == rt) & (csv_df.RUN == r), ["TR", "RUNTYPE", "RUN", "TIME", "MODELID"]+feature_columns].sort_values("TIME")

        x.extend(trj_df[feature_columns].values)
        mids.extend(trj_df["MODELID"].values)
        trj_names.extend([trj_name for _ in trj_df["MODELID"].values])
                
                


    #check if we have entries
    if len(mids) == 0:
        sys.exit("No data has been found. Aborting.")
        
    return x, trj_names, mids
#--



def rnn_minibatch_sequencer(raw_data_x, raw_data_y, batch_size, sequence_size, nb_epochs):
    data = np.array(raw_data_x)
    labels = np.array(raw_data_y)
    data_len = data.shape[0]
    nb_features = data.shape[1]


    nb_batches = data_len // (sequence_size * batch_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], (batch_size, nb_batches*sequence_size,nb_features))
    ydata = np.reshape(labels[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    
    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            
            yield x, y, epoch
#--


def rnn_minibatch_sequencer_w_mids(raw_data_x, raw_data_y, raw_data_mids, batch_size, sequence_size, nb_epochs):
    data = np.array(raw_data_x)
    labels = np.array(raw_data_y)
    mids = np.array(raw_data_mids)
    data_len = data.shape[0]
    nb_features = data.shape[1]


    nb_batches = data_len // (sequence_size * batch_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], (batch_size, nb_batches*sequence_size,nb_features))
    ydata = np.reshape(labels[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    mdata = np.reshape(mids[0:rounded_data_len], [batch_size, nb_batches * sequence_size])    

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            m = mdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            m = np.roll(m, -epoch, axis=0)
            
            yield x, y, m, epoch
#--









def get_class_size(csv_fn, label_column):
    csv_df = pd.read_csv(csv_fn, index_col=0)

    unq_labels = csv_df[label_column].unique()
    nb_unq_labels = len(unq_labels)
    #print unq_labels
    return nb_unq_labels


def get_validation_set(fold, cv_csv_fn):
    df = pd.read_csv(cv_csv_fn, index_col=0)
    return list(df.loc[df.FOLD == fold].TRJ_NAME.values)

def get_train_vali_set(fold, cv_csv_fn):
    df = pd.read_csv(cv_csv_fn, index_col=0)
    
    train_trjs = list(df.loc[(df.FOLD == fold) & (df.TRAIN_VALI == "TRAIN")].TRJ_NAME.values)

    vali_trjs = list(df.loc[(df.FOLD == fold) & (df.TRAIN_VALI == "VALI")].TRJ_NAME.values)


    return train_trjs, vali_trjs
