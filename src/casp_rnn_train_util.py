import numpy as np
import glob
import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  
import time
import math
import platform
from tensorflow.core.framework import summary_pb2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import h5py 



import casp_datafeeder as df


#################
#### METRICS #### 
#################

# parse confusion matrix into pandas df
def compute_metrics_from_cm(cmtx_list, names_list=[]):
    label_metrics_df = pd.DataFrame(columns=["NAME", "SET", "LABEL", "STEP", "PRECISION", "RECALL", 
                                         "F1","NUM_TP", "NUM_FN", "NUM_FP", "NUM_TOTAL"])

    result_rows=[]
    for idx, cm_fn in enumerate(cmtx_list):
        
        if len(names_list) == len(cmtx_list):
            name = names_list[idx]
        else:
            name = cm_fn.split("/")[-2].split("-")[0]
            name = "_".join(name.split("_")[-3:])
        print "processing", cm_fn, "...",

        #parse vali cm
        for step, cmtx in parse_confusion_matrix_log(cm_fn):
            num_labels = len(cmtx)
            #compute metrics for each label 
            for label in range(num_labels):
                precision_score = precision(label, cmtx)
                recall_score = recall(label, cmtx)
                f1_score = f1(label, cmtx)
                n_tp = num_tp(label, cmtx)
                n_fn = num_fn(label, cmtx)
                n_fp = num_fp(label, cmtx)
                n_tot = num_total(label, cmtx)




                result_rows.append({"NAME":name, "SET":"VALI", 
                                                           "LABEL":label, "STEP": step,
                                                           "PRECISION":precision_score, "RECALL":recall_score,
                                                           "F1": f1_score, "NUM_TP":n_tp, "NUM_FN":n_fn, 
                                                           "NUM_FP":n_fp, "NUM_TOTAL":n_tot})
        print "vali done!"

    label_metrics_df = label_metrics_df.append(result_rows, ignore_index=True)
    
    label_metrics_df = label_metrics_df.sort_values(["NAME","STEP"])
    
    return label_metrics_df


def movingaverage(interval, window_size):
    #print len(interval)
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


#Mean Validation Precision of label 0 for all folds at their respective best validation precision
def mvp0_best_each_fold_runavg(cmtx_list, windowsize=30):
    mean_precision = 0.0

    cm_fn2precision_arr = dict()
    cm_fn2run_avg_precision_arr = dict()
    cm_fn2step_arr = dict()
    cm_fn2best_step_precision = dict()
    
    #parse confusion matrix and compute precision for label 0 at each step
    for cm_fn in cmtx_list:
        for step, cm in parse_confusion_matrix_log(cm_fn):
            if cm_fn not in cm_fn2precision_arr.keys():
                cm_fn2precision_arr[cm_fn] = []
            if cm_fn not in cm_fn2step_arr.keys():
                cm_fn2step_arr[cm_fn] = []
                           
            precision_s = precision(0, cm)
            cm_fn2precision_arr[cm_fn].append(precision_s)
            cm_fn2step_arr[cm_fn].append(step)

    
    #compute running avg over precision
    for cm_fn in cmtx_list:
        p_arr = cm_fn2precision_arr[cm_fn]
        runavg_p_arr = movingaverage(p_arr,windowsize)
        cm_fn2run_avg_precision_arr[cm_fn] = runavg_p_arr
    
    #get best precision for each fold   
    for cm_fn in cmtx_list:
        best_precision = 0.0
        best_step = -1
        for step, precision_s in zip(cm_fn2step_arr[cm_fn], cm_fn2run_avg_precision_arr[cm_fn]):
            
            if precision_s > best_precision:
                best_precision = precision_s
                best_step = step
        cm_fn2best_step_precision[cm_fn] = (best_step, best_precision)
            

    #compute mean_precision
    p_arr = []
    for cm_fn in cmtx_list:
        p_arr.append(cm_fn2best_step_precision[cm_fn][1])
        
    mean_precision = np.mean(p_arr)
    
    #return step and negative best mean precison
    return cm_fn2best_step_precision, -1*mean_precision



def group_confusion_matrix(cm, grouping):
    #e.g. grouping = [[0,1],[2]]
    dim= len(grouping)
    new_cm = np.zeros((dim,dim))

    length_old_cm = len(cm)
    
    #get TP values
    for idx, g in enumerate(grouping):

        tp = 0
        for i in g:
            for j in g:
                if i < length_old_cm and j < length_old_cm:
                    tp=tp+cm[i,j]
        new_cm[idx,idx]=tp
        
        
    #get FN values
    for idx, g in enumerate(grouping):

        for jdx, h in enumerate(grouping):
            if idx != jdx:
                fn=0
                for i in g:
                    for j in h:
                        if i < length_old_cm and j < length_old_cm:
                            fn=fn+cm[i,j]

                new_cm[idx,jdx]=fn
                
    return new_cm

def parse_confusion_matrix_log(cm_log_fn):
    with open(cm_log_fn) as f:
        step = -1
        mtx_str = ""
        for line in f:
            if line.startswith("STEP"):
                step=int(line.split()[-1])
                continue
            elif line.startswith("ENDSTEP"):
                mtx_str = mtx_str.strip()
                np_mtx = np.array(np.matrix(mtx_str.replace("\n", "; ").replace("[", "").replace("]", "")))
                mtx_str=""
                yield step, np_mtx
            elif step != -1:
                mtx_str = mtx_str + line
#--

def precision(label, confusion_mtx):
    confusion_mtx = np.array(confusion_mtx)
    tp = confusion_mtx[label, label]
    fp = np.sum(confusion_mtx[:,label])-tp

    precision = -1
    if tp == 0:
        precision = 0
    else:
        precision = float(tp) / (float(tp)+float(fp))
    
    return precision
#--

def recall(label, confusion_mtx):
    confusion_mtx = np.array(confusion_mtx)
    tp = confusion_mtx[label, label]
    fn = np.sum(confusion_mtx[label,:])-tp

    recall = -1
    if tp == 0:
        recall = 0
    else:
        recall = float(tp) / (float(tp)+float(fn))
    
    return recall
#--

def f1(label, confusion_mtx):
    p = precision(label, confusion_mtx)
    r = recall(label, confusion_mtx)

    f1 = -1
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = float(2) * ((p*r)/(p+r))
    
    return f1
#--

def num_tp(label, confusion_mtx):
    confusion_mtx = np.array(confusion_mtx)
    
    num_tp = -1
    if label < len(confusion_mtx):
        num_tp = confusion_mtx[label,label]

    return num_tp
#--

def num_fn(label, confusion_mtx):
    confusion_mtx = np.array(confusion_mtx)

    num_fn = -1
    if label < len(confusion_mtx):
        num_tp = confusion_mtx[label, label]
        num_fn = np.sum(confusion_mtx[label,:])-num_tp


    return num_fn
#--

def num_fp(label, confusion_mtx):
    confusion_mtx = np.array(confusion_mtx)

    num_fp = -1
    if label < len(confusion_mtx):
        num_tp = confusion_mtx[label, label]
        num_fp = np.sum(confusion_mtx[:,label])-num_tp


    return num_fp
#--

def num_total(label, confusion_mtx):
    confusion_mtx = np.array(confusion_mtx)

    num_tot = -1
    if label < len(confusion_mtx):
        num_tot = np.sum(confusion_mtx[label,:])


    return num_tot
#--


def write_header(log_fh=None, metrics=[]):
    if log_fh != None:
        log_fh.write("\t".join(["step"]+metrics))
        log_fh.write("\n")
#--

def metric_log_writer(step, metrics, metrics_results_dic, log_fh=None):
    values=[]

    
    if log_fh != None:
        #add step
        values.append(str(step))

        for m in metrics:
            values.append(str(metrics_results_dic[m]))

        log_fh.write("\t".join(values))
        log_fh.write("\n")
#--

def vali_predictions_log_writer_header(log_fh):
    if log_fh != None:
        log_fh.write("STEP,TR,RUNTYPE,RUN,MID,TIME_PS,Y_TRUE,Y_PRED,YP_0, YP_1,YP_2\n")
#--

def vali_predictions_log_writer(step, mids , Y_, pred_y, yo, log_fh, compression_level=9):

    mids = np.array(mids)
    mids = mids.flatten()

    Y_ = np.array(Y_)
    Y_ = Y_.flatten()

    pred_y = np.array(pred_y)
    pred_y = pred_y.flatten()

    step_data = []

    if log_fh != None:
        for mid, y_true, y, yo in  zip(mids , Y_, pred_y, yo):

            row = []

            tr = mid.split("$")[0].split(":")[1]
            rt = mid.split("$")[1].split(":")[1]
            run = mid.split("$")[2].split(":")[1]
            m = mid.split("$")[3].split(":")[1]
            t = mid.split("$")[4].split(":")[1]


            row.append(step)
            #write mid fields
            row.append(tr)
            row.append(rt)
            row.append(run)
            row.append(m)
            row.append(t)

            #write ground truth
            row.append(y_true)

            #write prediction
            row.append(y)


            #write probability vector for prediction
            row.append(yo[0])
            row.append(yo[1])
            row.append(yo[2])

            step_data.append(row)

        if compression_level == 0:
            log_fh.create_dataset('step_'+str(step), data=np.array(step_data))
        else:
            log_fh.create_dataset('step_'+str(step), data=np.array(step_data), compression="gzip", compression_opts=compression_level)
#--


def yr_log_writer(step, mids , yrs, log_fh, compression_level=9):
    mids = np.array(mids)
    print "shape mids (before flatten):", mids.shape
    mids = mids.flatten()
    print "shape mids (after flatten):", mids.shape



    yrs = np.array(yrs)
    hidden_size = yrs.shape[-1]
    print "shape yr (before flatten):", yrs.shape
    yrs = yrs.reshape([-1,hidden_size])
    print "shape yr (after flatten):", yrs.shape

    step_data = []

    if log_fh != None:
        for mid, yr in  zip(mids , yrs):

            row = []

            tr = mid.split("$")[0].split(":")[1]
            rt = mid.split("$")[1].split(":")[1]
            run = mid.split("$")[2].split(":")[1]
            m = mid.split("$")[3].split(":")[1]
            t = mid.split("$")[4].split(":")[1]


            row.append(step)
            #write mid fields
            row.append(tr)
            row.append(rt)
            row.append(run)
            row.append(m)
            row.append(t)

            #write out of  hidden state of last layer
            row.extend(yr)


            step_data.append(row)


        if compression_level == 0:
            log_fh.create_dataset('step_'+str(step), data=np.array(step_data))
        else:
            log_fh.create_dataset('step_'+str(step), data=np.array(step_data), compression="gzip", compression_opts=compression_level)



        

def get_confusion_matrix(y_true, y_pred, label=None):
    y_true = np.array(y_true) # [ BATCHSIZE, SEQLEN ]
    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]

    y_true_flat = y_true.flatten() # [ BATCHSIZE X SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]



    cm=[]
    if label!=None:
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=label)
    else:
        cm = confusion_matrix(y_true_flat, y_pred_flat)

    return cm
#--


def confusion_matrix_log_writer(step, y_true, y_pred, log_fh=None, label=None):

    y_true = np.array(y_true) # [ BATCHSIZE, SEQLEN ]
    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]

    y_true_flat = y_true.flatten() # [ BATCHSIZE X SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]



    if log_fh != None:
        cm=[]
        if label!=None:
            cm = confusion_matrix(y_true_flat, y_pred_flat, labels=label)
        else:
            cm = confusion_matrix(y_true_flat, y_pred_flat)
        cm_str = str(cm)
        log_fh.write("STEP "+str(step)+"\n")
        log_fh.write(cm_str)
        log_fh.write("\n")
        log_fh.write("ENDSTEP\n")
#--


def tf_summary_metrics(metrics, metrics_results_dic):
    tf_smm_values=[]

    for m in metrics:
        smm_val = summary_pb2.Summary.Value(tag=m, simple_value=metrics_results_dic[m])
        tf_smm_values.append(smm_val)

    tf_smm = summary_pb2.Summary(value=tf_smm_values)

    return tf_smm

    

def compute_metrics(y_true, y_pred, metrics=[], labels=None):
    results={}

    for m in metrics:
        if m == "precision_micro":
            results[m] = batch_precision(y_true, y_pred, average="micro")
        elif m == "precision_macro":
            results[m] = batch_precision(y_true, y_pred, average="macro")
        elif m == "precision_weighted":
            results[m] = batch_precision(y_true, y_pred, average="weighted")
        elif m == "recall_micro":
            results[m] = batch_recall(y_true, y_pred, average="micro")
        elif m == "recall_macro":
            results[m] = batch_recall(y_true, y_pred, average="macro")
        elif m == "recall_weighted":
            results[m] = batch_recall(y_true, y_pred, average="weighted")
        elif m == "f1_micro":
            results[m] = batch_f1(y_true, y_pred, average="micro")
        elif m == "f1_macro":
            results[m] = batch_f1(y_true, y_pred, average="macro")
        elif m == "f1_weighted":
            results[m] = batch_f1(y_true, y_pred, average="weighted")
        elif m == "accuracy":
            results[m] = batch_accuracy(y_true, y_pred)
        elif m.startswith("class_") and m.endswith("_precision"):
            cname = int(m.split("_")[1])
            cm = get_confusion_matrix(y_true, y_pred, label=labels)
            results[m] = precision(cname, cm)
        elif m.startswith("class_") and m.endswith("_recall"):
            cname = int(m.split("_")[1])
            cm = get_confusion_matrix(y_true, y_pred, label=labels)
            results[m] = recall(cname, cm)
        elif m.startswith("class_") and m.endswith("_f1"):
            cname = int(m.split("_")[1])
            cm = get_confusion_matrix(y_true, y_pred, label=labels)
            results[m] = f1(cname, cm)
        elif m.startswith("gclass_") and m.endswith("_precision"):
            #e.g. gclass_0_grouping_0-1:2-3-4:5-6_precision

            grouping_str = m.split("_")[3]
            groups = [map(int, g.split("-")) for g in grouping_str.split(":")]
            cname = int(m.split("_")[1])

            cm = get_confusion_matrix(y_true, y_pred, label=labels)
            g_cm = group_confusion_matrix(cm, groups)

            results[m] = precision(cname, g_cm)
        elif m.startswith("gclass_") and m.endswith("_recall"):
            #e.g. gclass_0_grouping_0-1:2-3-4:5-6_recall

            grouping_str = m.split("_")[3]
            groups = [map(int, g.split("-")) for g in grouping_str.split(":")]
            cname = int(m.split("_")[1])

            cm = get_confusion_matrix(y_true, y_pred, label=labels)
            g_cm = group_confusion_matrix(cm, groups)

            results[m] = recall(cname, g_cm)

        elif m.startswith("gclass_") and m.endswith("_f1"):
            #e.g. gclass_0_grouping_0-1:2-3-4:5-6_f1

            grouping_str = m.split("_")[3]
            groups = [map(int, g.split("-")) for g in grouping_str.split(":")]
            cname = int(m.split("_")[1])

            cm = get_confusion_matrix(y_true, y_pred, label=labels)
            g_cm = group_confusion_matrix(cm, groups)

            results[m] = f1(cname, g_cm)

    return results
#--


def batch_precision(y_true, y_pred, average="micro"):
    y_true = np.array(y_true) # [ BATCHSIZE, SEQLEN ]
    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]

    y_true_flat = y_true.flatten() # [ BATCHSIZE X SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]

    #print y_true_flat

    if average == "micro":
        return precision_score(y_true_flat, y_pred_flat, average="micro")
    elif average == "macro":
        return precision_score(y_true_flat, y_pred_flat, average="macro")
    elif average == "weighted":
        return precision_score(y_true_flat, y_pred_flat, average="weighted")
    else:
        return np.nan

def batch_recall(y_true, y_pred, average="micro"):
    y_true = np.array(y_true) # [ BATCHSIZE, SEQLEN ]
    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]

    y_true_flat = y_true.flatten() # [ BATCHSIZE X SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]

    #print y_true_flat

    if average == "micro":
        return recall_score(y_true_flat, y_pred_flat, average="micro")
    elif average == "macro":
        return recall_score(y_true_flat, y_pred_flat, average="macro")
    elif average == "weighted":
        return recall_score(y_true_flat, y_pred_flat, average="weighted")
    else:
        return np.nan


def batch_f1(y_true, y_pred, average="micro"):
    y_true = np.array(y_true) # [ BATCHSIZE, SEQLEN ]
    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]

    y_true_flat = y_true.flatten() # [ BATCHSIZE X SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]

    #print y_true_flat

    if average == "micro":
        return f1_score(y_true_flat, y_pred_flat, average="micro")
    elif average == "macro":
        return f1_score(y_true_flat, y_pred_flat, average="macro")
    elif average == "weighted":
        return f1_score(y_true_flat, y_pred_flat, average="weighted")
    else:
        return np.nan


def batch_accuracy(y_true, y_pred):
    y_true = np.array(y_true) # [ BATCHSIZE, SEQLEN ]
    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]

    y_true_flat = y_true.flatten() # [ BATCHSIZE X SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]

    return accuracy_score(y_true_flat, y_pred_flat, normalize=True)

################
#### MODELS #### 
################







def model_rnn_in_out_dropout_const_lr_weighted_loss(  FEATURESIZE, ALPHASIZE, model_name,
                                        SEQLEN=10, 
                                        BATCHSIZE=100, 
                                        INTERNALSIZE=512, 
                                        NLAYERS=3,
                                        learning_rate=0.001,
                                        dropout_pkeep=1.0,
                                        class_weights=None):


    #check if len(class_weights) != ALPHASIZE
    if len(class_weights) != ALPHASIZE:
        sys.exit("len(class_weights) != ALPHASIZE")
    
    #
    # the model
    #
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    # inputs
    X = tf.placeholder(tf.float32, [None, None, FEATURESIZE], name='X')    # [ BATCHSIZE, SEQLEN ]
    # expected outputs
    Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
    # input state
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    # using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=10 times
    # dynamic_rnn infers SEQLEN from the size of the inputs Xo

    onecell = rnn.GRUCell(INTERNALSIZE)
    dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
    multicell = rnn.MultiRNNCell([dropcell]*NLAYERS, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=Hin)

    # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
    # H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
    H = tf.identity(H, name='H')  # just to give it a name

    # Softmax layer implementation:
    # Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
    # From the readout point of view, a value coming from a cell or a minibatch is the same thing

    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
 
    class_weights = tf.constant(np.array(class_weights, dtype=np.float32)) # [ ALPHASIZE ]
    weight_map = tf.multiply(Yflat_, class_weights) # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    weight_map = tf.reduce_sum(weight_map, axis=1) # [ BATCHSIZE x SEQLEN ]


    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.multiply(loss, weight_map) # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
    
    
    Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
    Y = tf.bitcast(tf.cast(Y, tf.int8), tf.uint8) # [ BATCHSIZE, SEQLEN ]
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    return {    "model_name":model_name,
                "train_step":train_step,
                "lr":lr,
                "pkeep":pkeep,
                "batchsize":batchsize, 
                "H":H,
                "X":X,
                "Y":Y,
                "Y_":Y_,
                "Yo_":Yo_,
                "Yo":Yo,
                "Yr":Yr,
                "Hin":Hin,
                "loss":loss,
                "FEATURESIZE":FEATURESIZE,
                "ALPHASIZE":ALPHASIZE,
                "SEQLEN":SEQLEN,
                "BATCHSIZE":BATCHSIZE,
                "INTERNALSIZE":INTERNALSIZE,
                "NLAYERS":NLAYERS,
                "learning_rate":learning_rate,
                "dropout_pkeep":dropout_pkeep}

#--






##################
#### TRAINING #### 
##################




def train_rnn(  model_desc, csv_fn, FEATURES, LABEL, VALIDATION_TRJS, log, checkpoint_dir,
                nb_epochs=1000, 
                vis_progress_nbatches=100, 
                save_checkpoint_nbatches=500,
                checkpoint_fn=None, num_save_checkpoint=1,
                metrics=["precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro", "recall_weighted", "f1_micro", "f1_macro", "f1_weighted", "accuracy"],
                monitoring_metric="accuracy", track_vali_predictions=False, TRAIN_TRJS=[], track_last_hidden_state=False, compression_level=9, with_gap_between_trj=False):

    
    #check if len(FEATURES) != FEATURESIZE
    if len(FEATURES) != model_desc["FEATURESIZE"]:
        sys.exit("len(FEATURES) != FEATURESIZE")


    BATCHSIZE=model_desc["BATCHSIZE"]
    batchsize=model_desc["batchsize"]
    SEQLEN=model_desc["SEQLEN"]
    loss=model_desc["loss"]
    Y_=model_desc["Y_"]
    Y=model_desc["Y"]
    Yo=model_desc["Yo"]
    Yr=model_desc["Yr"]
    INTERNALSIZE=model_desc["INTERNALSIZE"]
    NLAYERS=model_desc["NLAYERS"]
    X=model_desc["X"]
    Hin=model_desc["Hin"]
    lr=model_desc["lr"]
    learning_rate=model_desc["learning_rate"]
    pkeep=model_desc["pkeep"]
    dropout_pkeep=model_desc["dropout_pkeep"]
    train_step=model_desc["train_step"]
    H=model_desc["H"]
    model_name=model_desc["model_name"]
    alphasize=model_desc["ALPHASIZE"]

    labels=range(alphasize)



    # visualisation progress and checkpoint save points
    VIS_PROGRESS_N_BATCHES = vis_progress_nbatches * BATCHSIZE * SEQLEN
    CHECKPOINT_N_BATCHES = save_checkpoint_nbatches * BATCHSIZE * SEQLEN


    #load data
    train_x, train_y,  train_mids, trjranges_train, test_x, test_y, test_mids, trjranges_test = df.read_csv_file_w_mids(csv_fn, FEATURES, LABEL, VALIDATION_TRJS, TRAIN_TRJS)


    # display some stats on the data
    epoch_size = len(train_x) // (BATCHSIZE * SEQLEN)
    print "EPOCH_SIZE:", epoch_size

    print "TRAINING DATA SIZE:", len(train_x)
    print "TEST DATA SIZE:", len(test_x)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))



    

    m_accuracy = tf.contrib.metrics.accuracy(Y,Y_)

    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    m_acc_summary = tf.summary.scalar("batch_m_accuracy", m_accuracy)

    summaries = tf.summary.merge([  loss_summary, 
                                    acc_summary,
                                    m_acc_summary
                                 ])

    # Init Tensorboard. 
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter(log + "-training")
    validation_writer = tf.summary.FileWriter(log + "-validation")

    #log for seperate metrics computed with sklearn, e.g. precision, recall, etc
    train_log = open(log + "-training/metrics.tab",'w',0)
    vali_log = open(log + "-validation/metrics.tab",'w',0)
    
    train_cm_log = open(log + "-training/confusion_matrix.log",'w',0)
    vali_cm_log = open(log + "-validation/confusion_matrix.log",'w',0)

    vali_pred_log_hf = h5py.File(log + "-validation/validation_pred_log.h5", 'w')
    Yr_log_hf = h5py.File(log + "-validation/Yr_pred_log.h5", 'w')




    write_header(train_log, metrics)
    write_header(vali_log, metrics)

    # Init for saving models. They will be saved into a directory named 'checkpoints'.
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver = tf.train.Saver(max_to_keep=num_save_checkpoint)
    best_model_saver = tf.train.Saver(max_to_keep=10) # saves the last 10 best models

    #resume from last checkpoint if checkpoint_fn == -1
    if checkpoint_fn == -1 or checkpoint_fn == "-1":
        all_meta=set(glob.glob(checkpoint_dir+"/*.meta"))
        bestmodel_meta=set(glob.glob(checkpoint_dir+"/*bestmodel*.meta"))

        latest_checkpoint_meta = all_meta - bestmodel_meta
        last_checkpoint = latest_checkpoint_meta.pop().rstrip(".meta")

        checkpoint_fn = last_checkpoint

    # init
    sess = tf.Session()
    if checkpoint_fn != None: #load checkpoint
        print "RESUMING FROM CHECKPOINT", checkpoint_fn
        saver.restore(sess, checkpoint_fn)

        # if we can find the last istate then load the values
        print "TRY LOADING LAST ISTATE ..."
        if os.path.exists(checkpoint_fn+".npy"):
            istate = np.load(checkpoint_fn+".npy")
            print "ISTATE FOUND, reusing last values..."
        else:
            istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
            print "NO ISTATE FOUND, setting ISTATE to zeros..."

        checkpoint_step = int(checkpoint_fn.split('-')[-1])
    else:
        istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


    step = 0
    best_validation_val = 0.0

    # training loop
    for x, y_, m, epoch in df.rnn_minibatch_sequencer_w_mids(train_x, train_y, train_mids, BATCHSIZE, SEQLEN, nb_epochs):


        #when restarting from checkpoint iterate through the data till step == checkpoint_step 
        if checkpoint_fn != None and step <= checkpoint_step:
            if step == 0:
                print "skipping forward..."
            elif step == checkpoint_step:
                print "Done! Resuming from step", step

            step += BATCHSIZE * SEQLEN
            continue
            

        # train on one minibatch
        feed_dict = {   X: x, Y_: y_, Hin: istate, 
                        lr: learning_rate, 
                        pkeep: dropout_pkeep, 
                        batchsize: BATCHSIZE}
        _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

        #compute additional metrics, e.g. recall, precision, f1, ...
        train_metrics_dict = compute_metrics(y_, y, metrics, labels=labels)

        # save training data for Tensorboard and seperate log file
        summary_writer.add_summary(smm, step)
        additional_train_metrics_smm = tf_summary_metrics(metrics, train_metrics_dict)
        summary_writer.add_summary(additional_train_metrics_smm, step)
        metric_log_writer(step, metrics, train_metrics_dict, train_log)
        confusion_matrix_log_writer(step, y_, y, train_cm_log, label=labels)



        # display a visual validation of progress (every n batches)
        if step % VIS_PROGRESS_N_BATCHES == 0:
            feed_dict =     {X: x, Y_: y_, Hin: istate, 
                            pkeep: 1.0, # no dropout for validation
                            batchsize: BATCHSIZE}  
            y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)

            monitoring_metric_val = float("{0:.4f}".format(train_metrics_dict[monitoring_metric]))
            print "TRAIN: "+monitoring_metric+"=", monitoring_metric_val, "EPOCH=", str(epoch)+"/"+str(nb_epochs), "STEP=", step, "MNAME=", model_name

        #validation on external test set
        if step % VIS_PROGRESS_N_BATCHES == 0 and len(test_x) > 0:
            VALI_SEQLEN = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
            bsize = len(test_y) // VALI_SEQLEN
            vali_x, vali_y, vali_m, _ = next(df.rnn_minibatch_sequencer_w_mids(test_x, test_y, test_mids, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
            feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,  # no dropout for validation
                     batchsize: bsize}
            ls, y, yo, ostate_vali, acc, smm, yr = sess.run([batchloss, Y, Yo, H, accuracy, summaries, Yr], feed_dict=feed_dict)


            if track_vali_predictions:
                vali_predictions_log_writer(step, vali_m , vali_y, y, yo, vali_pred_log_hf, compression_level=compression_level)

            if track_last_hidden_state:
                yr_log_writer(step, vali_m , yr, Yr_log_hf,  compression_level=compression_level)
            


            #compute additional metrics, e.g. recall, precision, f1, ...
            validation_metrics_dict = compute_metrics(vali_y, y, metrics, labels=labels)

            #print validation results
            monitoring_metric_val = float("{0:.4f}".format(validation_metrics_dict[monitoring_metric]))
            print "VALI: "+monitoring_metric+"=", monitoring_metric_val, "STEP=", step, "MNAME=", model_name


            # save validation data for Tensorboard and metric.log file
            validation_writer.add_summary(smm, step)
            additional_vali_metrics_smm = tf_summary_metrics(metrics, validation_metrics_dict)
            validation_writer.add_summary(additional_vali_metrics_smm, step)
            metric_log_writer(step, metrics, validation_metrics_dict, vali_log)
            confusion_matrix_log_writer(step, vali_y, y, vali_cm_log, label=labels)

            #save checkpoint if monitoring_metric_va > best_validation_val
            if monitoring_metric_val > best_validation_val:
                val_str = str(monitoring_metric_val).replace(".", "_")
                epoch_str = str(epoch)
                step_str = str(step)
                best_model_saver.save(sess, checkpoint_dir + '/' + model_name + '-bestmodel_acc_' + val_str + '_epoch_' + epoch_str + '_step_' + step_str + '-' + timestamp, global_step=step)

                np.save(checkpoint_dir + '/' + model_name + '-bestmodel_val_' + val_str + '_epoch_' + epoch_str + '_step_' + step_str + '-' + timestamp + '-' + str(step), ostate_vali)

                print "SAVING BEST MODEL CHECKPOINT.", "EPOCH="+str(epoch), "STEP="+str(step), monitoring_metric+"="+str(monitoring_metric_val)
                best_validation_val = monitoring_metric_val



        # save a checkpoint 
        if step % CHECKPOINT_N_BATCHES == 0:
            saver.save(sess, checkpoint_dir + '/' + model_name + '-' + timestamp, global_step=step)
            np.save(checkpoint_dir + '/' + model_name + '-' + timestamp + '-' + str(step), ostate)
            print "SAVING CHECKPOINT.", "EPOCH="+str(epoch), "STEP="+str(step)


        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN


    vali_pred_log_hf.close()
    train_log.close()
    vali_log.close()
#--





##################
#### PREDICT  #### 
##################


def predict_rnn(model_desc, csv_fn, checkpoint_fn,  FEATURES, TRJ_NAMES):

    saver = tf.train.Saver(max_to_keep=0)

    # init
    sess = tf.Session()
    if checkpoint_fn != None: #load checkpoint
        print "RESUMING FROM CHECKPOINT", checkpoint_fn
        saver.restore(sess, checkpoint_fn)





    Y=model_desc["Y"]
    Yo=model_desc["Yo"]
    INTERNALSIZE=model_desc["INTERNALSIZE"]
    NLAYERS=model_desc["NLAYERS"]
    X=model_desc["X"]
    Hin=model_desc["Hin"]
    H=model_desc["H"]
    pkeep=model_desc["pkeep"]
    batchsize=model_desc["batchsize"]
    model_name=model_desc["model_name"]
    alphasize=model_desc["ALPHASIZE"]




    #load data
    x_raw, trj_names, mids = df.read_csv_file_predict(csv_fn, FEATURES, TRJ_NAMES)    

    mock_y = [-1 for _ in trj_names]

    SEQLEN = len(x_raw)
    bsize = 1
    vali_x, _, _ = next(df.rnn_minibatch_sequencer(x_raw, mock_y, bsize, SEQLEN, 1))  # all data in 1 batch
    nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])


    feed_dict = {X: vali_x, Hin: nullstate, pkeep: 1.0,  # no dropout for prediction
             batchsize: bsize}
    y_pred, ostate_vali, yo = sess.run([Y, H, Yo], feed_dict=feed_dict)


    y_pred = np.array(y_pred) # [ BATCHSIZE, SEQLEN ]
    y_pred_flat = y_pred.flatten() # [ BATCHSIZE X SEQLEN ]

    yo = np.array(yo) # # [ BATCHSIZE x SEQLEN, ALPHASIZE ]

    return y_pred_flat, yo, trj_names, mids
#--

