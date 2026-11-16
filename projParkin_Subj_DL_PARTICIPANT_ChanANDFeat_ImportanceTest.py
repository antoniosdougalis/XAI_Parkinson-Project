# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:09:31 2025
@author: Antonios
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import time

os.chdir(r'C:\Users\anton\Documents\Python Scripts\JournalOfPersonalised_Code')
import jpersMed_myReducedWeights_TransformerUtil as myRT

# os.chdir(r'C:\Users\anton\Documents\Python Scripts\JournalOfPersonalised_Code')
# import jpersMed_myPosEncoding_TransformerUtil as myRT

# import scipy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sklearn.metrics as skm
from scipy import stats


def remove_nans(train, test):
    
    mean = torch.nanmean(train, dim=0)
    
    train = torch.where(torch.isnan(train), mean, train)
    test  = torch.where(torch.isnan(test),  mean, test)
    
    return train, test

def zscore_train_test(train, test):
    
    mean = torch.mean(train, dim=0)
    std = torch.std(train, dim=0)
    
    std[std == 0] = 1
    
    train_z = (train - mean) / std
    test_z  = (test  - mean) / std
    
    return train_z, test_z

def top_interpretable_channels(testW, ch_names, top=5):
   
    n_chans, n_subjs = testW.shape
    
    top_counts = np.zeros(n_chans)
    
    for subji in range(n_subjs):
        
        # get top-N channel indices for this subject
        top_idx = np.argsort(testW[:, subji])[::-1][:top]
        
        # count votes
        top_counts[top_idx] += 1
    
    # make it fractional
    top_counts = top_counts/n_subjs
    
    # rank channels by how often they appear in top-N
    ranked_idx = np.argsort(top_counts)[::-1] # fractional agreement
    
    ranked_channels = [(ch_names[i], top_counts[i]) for i in ranked_idx]
    
    return top_counts, ranked_channels

#% load the Essential Common parameter data
os.chdir(r'C:\Users\anton\Documents\Python Scripts\Project_Parkinon_EEG_Analysis')
fileName = 'projParkin_EEG_DL_Data.npz'

PD_Feature_EEGdata = np.load(fileName)
data_content = list(PD_Feature_EEGdata.keys())

subj_data        = PD_Feature_EEGdata['subj_data_DL']  
feature_labels   = PD_Feature_EEGdata['subj_feature_labels']  

# reduce the dataset to something smaller (exclude some parameters)

# array(['del_welch_PW', 'the_welch_PW', 'alp_welch_PW', 'bet_welch_PW',
#        'gam_welch_PW', 'ap_exp', 'ap_off', 'irasa_exp', 'peak_CF',
#        'peak_BP', 'peak_BW', 'del_DFA_exp', 'the_DFA_exp', 'alp_DFA_exp',
#        'bet_DFA_exp', 'gam_DFA_exp', 'del_fEI_exp', 'the_fEI_exp',
#        'alp_fEI_exp', 'bet_fEI_exp', 'gam_fEI_exp', 'del_relP_exp',
#        'the_relP_exp', 'alp_relP_exp', 'bet_relP_exp', 'gam_relP_exp',
#        'myPAC', 'del_PLV', 'the_PLV', 'alp_PLV', 'bet_PLV', 'gam_PLV',
#        'del_PLI', 'the_PLI', 'alp_PLI', 'bet_PLI', 'gam_PLI',
#        'del/alp_bicPAC', 'the/alp_bicPAC', 'del/bet_bicPAC',
#        'the/bet_bicPAC', 'del/gam_bicPAC', 'the/gam_bicPAC',
#        'del/alp_bicPPC', 'the/alp_bicPPC', 'del/bet_bicPPC',
#        'the/bet_bicPPC', 'del/gam_bicPPC', 'the/gam_bicPPC', 'tdeTau',
#        'tdeStr', 'the_rng_freqSld', 'alp_rng_freqSld', 'gam_rng_freqSld',
#        'the_dif_freqSld', 'alp_dif_freqSld', 'gam_dif_freqSld',
#        'the_mod_freqSld', 'alp_mod_freqSld', 'gam_mod_freqSld', 'tLZS',
#        'the_cor_freqSld', 'alp_cor_freqSld', 'gam_cor_freqSld',
#        'the_p_freqSld', 'alp_p_freqSld', 'gam_p_freqSld',
#        'alp/the_HLockCount', 'gam/the_HLockCount', 'alp/the_HLockPerT',
#        'gam/the_HLockPerT', 'avalSize', 'avalDur', 'kappaSize',
#        'timeMean', 'timeVar', 'timeIQR'], dtype='<U18')


featCOI = ['ap_exp', 'ap_off', 'peak_CF', 'peak_BP', 'peak_BW',
'alp_DFA_exp', 'alp_fEI_exp', 'the_DFA_exp', 'the_fEI_exp', 
'del/gam_bicPAC', 'the/gam_bicPAC', 
'alp_rng_freqSld', 'alp_dif_freqSld', 
'the_rng_freqSld', 'the_dif_freqSld',
'the_cor_freqSld', 'alp_cor_freqSld',
'the_mod_freqSld', 'alp_mod_freqSld',
'alp/the_HLockPerT','gam/the_HLockPerT',
'avalSize',  'kappaSize',
'del_relP_exp','the_relP_exp', 'alp_relP_exp', 'bet_relP_exp', 'gam_relP_exp',
'del_PLV', 'the_PLV', 'alp_PLV', 'bet_PLV', 'gam_PLV',
'del_PLI', 'the_PLI', 'alp_PLI', 'bet_PLI', 'gam_PLI',
'timeMean','timeVar']

# featCOI = ['ap_exp', 'ap_off', 'peak_CF', 'peak_BP', 'peak_BW',
# 'alp_DFA_exp', 'alp_fEI_exp', 'the_DFA_exp', 'the_fEI_exp', 
# 'del/gam_bicPAC', 'the/gam_bicPAC', 
# 'alp_rng_freqSld', 'alp_dif_freqSld', 
# 'the_rng_freqSld', 'the_dif_freqSld',
# 'the_cor_freqSld', 'alp_cor_freqSld',
# 'the_mod_freqSld', 'alp_mod_freqSld',
# 'alp/the_HLockPerT','gam/the_HLockPerT',
# 'avalSize',  'kappaSize']

# without ALL the highly correlated time domain features (keep Var)
# featCOI = ['del_relP_exp','the_relP_exp', 'alp_relP_exp', 'bet_relP_exp', 'gam_relP_exp',
#            'del_PLV', 'the_PLV', 'alp_PLV', 'bet_PLV', 'gam_PLV',
#            'del_PLI', 'the_PLI', 'alp_PLI', 'bet_PLI', 'gam_PLI',
#            'timeMean','timeVar']

featidx = []
for fti, fName in enumerate( featCOI ):
    idx = np.where(feature_labels == fName)[0][0]
    featidx.append(idx)
 
# for random reductions 50%, used seeds 69, 7, 666, 777
# randFile = 3
# seedNum = [69, 7, 666, 777]
# np.random.seed(seedNum[randFile])
# randCOI = np.random.choice(len(featCOI), int(len(featCOI)/2), replace=False)
# featidx = [featidx[i] for i in randCOI]
    
subj_data = subj_data[:, :,featidx]
feature_labels = feature_labels[featidx]

# get all common featured data as numpies
subj_labels        = PD_Feature_EEGdata['subj_labels']  
group_labels       = PD_Feature_EEGdata['group_labels'] 
labels             = PD_Feature_EEGdata['labels'] 
epochs_PerSubj     = PD_Feature_EEGdata['epochs_PerSubj']
cum_epochs_PerSubj = PD_Feature_EEGdata['cum_epochs_PerSubj']
ch_names           = PD_Feature_EEGdata['ch_names']
#----------------------------------------------------

# Get some parameters out for the model
n_cond           = len(np.unique(subj_labels))
n_subj, n_chans, n_features = subj_data.shape

# get the indexed of the patholfoy groups
CN_idx     = np.where(subj_labels==0)[0]
PDoff_idx  = np.where(subj_labels==1)[0]
PDon_idx   = np.where(subj_labels==2)[0]

n_subj_cond = [ np.sum(subj_labels==0), np.sum(subj_labels==1), np.sum(subj_labels==2) ]


# check for nan values find out the nan pertpetrator'
nanDict = {}
for vari in range(n_features):
    trial = np.sum(np.isnan(subj_data[:, :, vari]))
    nanDict[feature_labels[vari]] = trial  

nan_score = 100*np.sum(np.isnan(subj_data))

print(f'the zscored tensor of dims {subj_data.shape} has {nan_score} nan values')

print('data and labels loaded in tensors: Ready for deep learning procedures')


#%% Reduced tranformer without permutation after embedding and without positional encoding

# EmT Model Parameters
condExclude = 2

if condExclude == 3:
    num_classes = 3    # Number of pathology classes
else:
    num_classes = 2    # Number of pathology classes
       

n_features = subj_data.shape[-1]   # Number of features (last dimension of the data, typically the feature vector per channel)
dropout    = 0.1                  # Dropout rate for regularization (helps prevent overfitting)
num_layers = 6                    # Number of transformer layers (more layers can help with modeling complex relationships)
num_heads  = 6                    # Number of attention heads in the multi-head attention mechanism (parallel attention mechanisms)
d_model    = 60                   # Dimensionality of the embedding space (size of the vector for each token/channel)
d_ff       = 256                  # Size of the feed-forward network in each transformer layer (helps with transformation of data between attention layers)
seq_len    = n_chans              # Length of the sequence also know as the number of tokens (each token represents one unit of information, words for text, channels in my case for EEG)
input_dim  = n_features           # size of each token's feature vector: (the number of features per token, in my case per channeöls)
# This makes depth = 6 (d_model/num_heads) - This is a general reference to the relationship between `d_model` and `num_heads`

# save configuration
myRT_model_config =  {'num_layers' : num_layers, 'd_model': d_model, 'num_heads': num_heads, 'd_ff': d_ff, 'input_dim': subj_data.shape[-1],
                      'num_classes': num_classes,'dropout': dropout, 'printToggle': False}


#%% Select records for 2Class or 3Class predictions: NO Z-SCORED yet!!!!

if condExclude ==3:
    
    # make data (devoid of Nans ) torch
    CombData = torch.from_numpy( subj_data ).float()

    # # transform labels as a tensor
    labels = torch.from_numpy(subj_labels).long()
    
else:
    # initliase
    temp_subj_labels,  temp_CombData = [ [] for _ in range(2)]
    
    for subji in range(len(subj_labels)):
        if subj_labels[subji]!=condExclude: # Choose pathology group here by excluding some label
            temp_subj_labels.append(subj_labels[subji])
            temp_CombData.append(subj_data[subji, :, :]) # NOTE these are WITH nan values AND are not z scored oyet!!
    
    subj_labels = np.array(temp_subj_labels)
    
    #turn the 2's into ones id using the AD set
    subj_labels[subj_labels==2]=condExclude
    
    # # # overwrite the data
    CombData = torch.from_numpy( np.asarray(temp_CombData) ).float()

    # # transform labels as a tensor
    labels = torch.from_numpy(subj_labels).long()


print(f'data loaded are {CombData.shape} with labels = {labels.shape}')

#%% Cuda

# --- Add at the top ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move data to device (before training/testing)
CombData = CombData.to(device)
labels = labels.to(device)

#%% LOSO procedure

# initiliase outputs
loso_train_loss, loso_train_acc, loso_dev_loss, loso_dev_acc, loso_exp_loss, loso_exp_acc  = [ [ ] for _ in range(6) ]
subject_results = []

fusionProb = np.zeros( (n_subj, num_classes) ) # softmax Prob

# batc size define hewre please
BS = 6

# placeholder for weights
trainW = np.zeros( (BS, n_chans, n_subj) )
testW  = np.zeros( (n_chans, n_subj) )

# exclude the held-out recording and its paired medication-state recording
for subji in range(len(labels)):

    print(subji)

    exclude_idx = [subji]

    if condExclude == 3:

        if 16 <= subji < 31:
            exclude_idx.append(subji + 15)
        elif 31 <= subji < 46:
            exclude_idx.append(subji - 15)

    elif condExclude == 0:

        if subji < 15:
            exclude_idx.append(subji + 15)
        else:
            exclude_idx.append(subji - 15)

    train_idx = [i for i in range(len(labels)) if i not in exclude_idx]

    loso_CombData = CombData[train_idx, ...]
    dev_data      = CombData[subji,:].unsqueeze(0)
    loso_labels   = labels[train_idx]

    #----remove nans
    loso_CombData, dev_data = remove_nans(loso_CombData, dev_data)

    # -----z score data
    loso_CombData, dev_data = zscore_train_test(loso_CombData, dev_data)


    # load the data into the test dataset
    BS = BS # bacthe size
    train_dataset = TensorDataset(loso_CombData, loso_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=False, drop_last=True)


    dev_labels = torch.tensor( [ labels[subji].clone() ]  )
    dev_dataset      = TensorDataset( dev_data, dev_labels )
    dev_dataloader   = DataLoader(dev_dataset, batch_size = dev_dataset.tensors[0].shape[0], shuffle=False, drop_last=True)

      
    #--------------------------------------------------------------------------
    #% Initialise hybrid model
    
    # for CrossEntrpy MultiClass NOTe it is passing the output through a softmax not a sigmoid, change in EmT
    lossfun = nn.CrossEntropyLoss() # expects labels to be longs()
    
    # Initialize the model, and optimizer
    model = myRT.EmT(num_layers, d_model, num_heads, d_ff, input_dim, num_classes, dropout, False).to(device) # reduced without positional
    # model = myRT.EmT(num_layers, d_model, seq_len, num_heads, d_ff, input_dim, num_classes, dropout, False).to(device) # with positional
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, threshold = 0.01, factor=0.5, min_lr=5e-5)

    #--------------------------------------------------------------------------
    #% Training loop in epochs
    
    num_epochs = 50
    # numpy arrays to store loss and accuracy per epoch
    train_losses, train_accuracies, test_losses, test_accuracies, elapsed_time = [ np.zeros((num_epochs)) for _ in range(5) ]
     
    for epochi in range(num_epochs):
        
        # starttime
        start_time = time.time() 
    
        # initialise batch holding arrays
        train_batch_loss, train_batch_acc, test_batch_loss, test_batch_acc = [ [ ] for _ in range(4) ] 
        
        # training mode of the model allowed
        model.train()
        
        # load train_ batches: start loop
        for dataEEG, EEGlabels in train_dataloader:
            # print(f'this batch of EEG data has shape of {dataEEG.shape},\n the EEGlabels has shape {EEGlabels.shape}' )
            
            # cuda
            dataEEG = dataEEG.to(device)
            EEGlabels = EEGlabels.to(device)
            
            outputs, w = model(dataEEG)
            loss = lossfun(outputs, EEGlabels)
            
            #store weights (batched, channels, feat)
            trainW[:, :, subji] = torch.squeeze(w).detach().numpy()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Bring data back
            dataEEG = dataEEG.cpu()
            EEGlabels = EEGlabels.cpu()
            outputs = outputs.cpu()
    
            # Calculate accuracy: CHANGE FOR CROSSENTROPY
            pred_class = torch.argmax(outputs, dim=1) # for Multclass entropy
            pred_acc = 100*torch.mean( (pred_class == EEGlabels).float() )
            
            train_batch_loss.append( loss.item() )
            train_batch_acc.append( pred_acc )
            
            # end of batch loop
    
        # Calculate average loss and accuracy for the epoch
        train_losses[epochi] = np.mean(train_batch_loss)
        train_accuracies[epochi] = np.mean(train_batch_acc)
        elapsed_time[epochi] = time.time()  - start_time
        
        # ----- Scheduler step -----
        scheduler.step(train_losses[epochi])
        current_lr = optimizer.param_groups[0]['lr']
        
        msg = f' TRAIN: Finished epoch {epochi+1}/{num_epochs}, loss: {train_losses[epochi]:.2f}, accuracies: {train_accuracies[epochi]:.2f} %: total elapsed time {np.sum(elapsed_time):.2f} second so far '
        sys.stdout.write('\r' + msg)
  
        # After computing train_losses[epochi]
        if train_losses[epochi] <= 0.03 and train_accuracies[epochi] >= 99.70:
            print(f"\nStopping early at epoch {epochi+1} because loss reached {train_losses[epochi]:.4f}")
            break
        
       
    # save the model
    # torch.save(model.state_dict(),'loso_myRT_AD_Feature_Classifier.pt')
    # print(f' model saved to disk \n directory {os.getcwd()}')
    
    #--------------------------------------------------------------------------
    #% #REeload the model fro inspection
    
    # create a config file that stores the models initialisation paramters to meatch the keys
    # import pickle
    
   
    myRT_model_config =  {'num_layers' : num_layers, 'd_model': d_model, 'num_heads': num_heads, 'd_ff': d_ff, 'input_dim': CombData.shape[-1],
                          'num_classes': num_classes,'dropout': dropout, 'printToggle': False} # without positional
    
    # myRT_model_config =  {'num_layers' : num_layers, 'd_model': d_model, 'seq_len':seq_len, 'num_heads': num_heads, 'd_ff': d_ff, 'input_dim': CombData.shape[-1],
    #                       'num_classes': num_classes,'dropout': dropout, 'printToggle': False} # with positonal
    
    #save model n´config file
    # with open('myRT_model_config.pkl', 'wb') as f:
    #     pickle.dump(myRT_model_config, f)
        
    # # load model configuration
    # with open('myRT_model_config.pkl', 'rb') as f:
    #     myRT_model_config = pickle.load(f)
    
    # create a new models of the same class 
    trained_model = myRT.EmT(**myRT_model_config) # reduced model: no positional encoding
        
    # replace one model's parameters with those of the trained net
    trained_model.load_state_dict( model.state_dict())
    
    #--------------------------------------------------------------------------
    #% test it on all batches of the train set that has not been seen by the model
    
    trained_model.to(device)
       
    # Get the total number of batches
    num_batches = len(dev_dataloader)
    
    Num_exp = num_batches
    exp_acc, exp_loss, exp_DangerPred = [np.zeros(Num_exp) for _ in range(3)]
    elapsed_time = np.zeros(Num_exp)
    
    for experi, (batch_EEGdata, batch_EEGlabels) in enumerate(dev_dataloader):
        
        X = batch_EEGdata
        y = batch_EEGlabels
        
        trained_model.eval()
    
        start_time = time.time()
               
        # # # # New!
        X = X.to(device)
        y = y.to(device)
    
        # run the data through the loaded trained model
        # generate predictions
        yHatTrained, w = trained_model(X)
        yHatTrained = yHatTrained.detach()
    
        # get the channel weights
        testW[:, subji]  = torch.squeeze(w).detach().numpy()
        
        # generate loss
        trained_loss = lossfun( yHatTrained, y[0])
        
    
        # # # New! bring outputs back
        yHatTrained = yHatTrained.cpu()
        
        y = y.cpu()
    
        
        # labels FOR cROISSeNTROPY
        labels_trained = torch.argmax(yHatTrained).int() # boolean (minus or plus value) turned 0 to 1 label
       
        # get the softmaxed fusion probabilities
        fusionProb[subji, :] = torch.softmax(yHatTrained, dim=0).detach().numpy() # probabilities
        
     
        # comparative to real labels
        comp_trained = (labels_trained==y).float() # this is boolean True/False turned numeric 0 or 1
        
    
        # accuracy based
        trained_acc = 100*torch.mean(comp_trained).item()
       
    
        # misclassified = np.where( ( (yHatTrained> 0.5).float() )!= y)[0] # indexes of missclassified
        misclassified    = np.where( (torch.argmax(yHatTrained).float() )!= y)[0] # indexes of missclassified
        danger_missclass = torch.sum( torch.logical_and( (labels_trained==0) , (y==1) ).float() )
        percentDanger    = 100*(danger_missclass)/y.shape[0]
    
        # get every batch experiment loss and accuracy
        exp_acc[experi]  = trained_acc
        exp_loss[experi] = trained_loss.item()
        exp_DangerPred[experi] = percentDanger
    
        elapsed_time[experi] = time.time() - start_time
    
        msg = f' Unseen Batches:{experi+1}/{Num_exp}, loss: {exp_loss[experi]:.2f}, accuracies: {exp_acc[experi]:.2f}, danger predictions {exp_DangerPred[experi]:.2f} %, elapsed time {np.sum(elapsed_time):.2f} seconds'
        sys.stdout.write('\r' + msg)
    
    
    '''
    # show that the results overlap
    plt.figure(figsize=(5,5))
    plt.subplot(3,1,1)
    plt.plot(labels_trained, 'bo', mfc = 'w', markersize=10, label='Trained Pred Labels')
    plt.plot(misclassified, num_classes*np.ones( misclassified.shape[0] ),'ro', label='Missclass Labels')
    plt.plot(y,'go',mfc = 'w', label='Real Labels')
    plt.legend()
    plt.xlabel('Patient id')
    plt.ylabel('Model Prediction (0, 1 or 2)')
    plt.title(f'Model Prediction Comparison: Accuracy: {np.round(trained_acc,2)} %, Dangerous predictions {percentDanger.numpy():.2f} %')
    plt.xlim([0,1 ])
    
    
    # show me the missclassified and their
    Pre =  torch.argmax(yHatTrained).float()
    Real = y
    
    plt.subplot(3,1,2)
    plt.plot(Real, 'go-', label = 'real')
    plt.plot(Pre, 'bo-', label = 'Pred')
    plt.title('Compare missclassified idx only')
    plt.xlabel('MissClass idx')
    plt.ylabel('Outcome Class')
    plt.legend()
    
    
    plt.subplot(3,1,3)
    plt.plot(range(0,Num_exp,1), exp_acc, 'go-', label = 'Accuracy')
    plt.title(f'Classification Accuracy of Unseen Epochs \n mean acc {np.mean(exp_acc):.2f} % with Dangerous Predictions {np.mean(exp_DangerPred):.2f} %')
    plt.xlabel(f'Number of Random Experimental Batches {X.shape[0]} presented')
    plt.ylabel('% Acccuracy')
    plt.legend()
    plt.tight_layout()
    
    # plt.savefig('myRT_Results_plot.pdf', dpi=300)
    # print(os.getcwd())
    
    plt.show()
    
    '''
    
    # get the results
    loso_exp_loss.append(exp_loss)
    loso_exp_acc.append( exp_acc ) 
    
    # majoroty vote
    subject_pred = stats.mode(labels_trained)[0]
    subject_true = y[0]  # all epochs belong to same subject and label

    subject_results.append((subject_true, subject_pred))
 
# SUbjet level metrics    
true_labels, pred_labels = zip(*subject_results)   
accScore    = skm.accuracy_score (true_labels, pred_labels)
preScore    = skm.precision_score(true_labels, pred_labels,average=None) #  Precision (a.k.a. Positive Predictive Value): Measures how many predicted positives are truly positive.
recScore    = skm.recall_score   (true_labels, pred_labels,average=None) #  Recall (a.k.a. Sensitivity or True Positive Rate): Measures how many actual positives were correctly identified
f1_Score    = skm.f1_score       (true_labels, pred_labels,average=None) #  Harmonic mean of precision and recall

# compute the test confusion matrix
uC_mat = skm.confusion_matrix(true_labels, pred_labels)


#%% Manual calculation of Results

# Initialize array to store specificity per class, per batch
accuracy_i, recall_i, precision_i, f1_i, specificity_i = [ np.zeros((num_classes)) for _ in range(5) ]

# Loop through each batch

cm = uC_mat
for cls in range(num_classes):
    TP = cm[cls, cls]
    FN = np.sum(cm[cls, :]) - TP
    FP = np.sum(cm[:, cls]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Avoid division by zero
    accuracy_i[cls]  = (TP + TN) / (TP + TN + FP + FN) # Overall correctness across all classes
    recall_i[cls]    = TP / (TP + FN) if (TP + FN) > 0 else np.nan # Recall (a.k.a. Sensitivity or True Positive Rate): how well actual positives are identified
    precision_i[cls] = TP / (TP + FP) if (TP + FP) > 0 else np.nan # Precision: how many predicted positives are actually correct
    f1_i[cls]        = 2 * precision_i[cls] * recall_i[cls] / (precision_i[cls] + recall_i[cls]) if (precision_i[cls] + recall_i[cls]) > 0 else np.nan # F1-score: harmonic mean of precision and recall
    specificity_i[cls] = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # Specificity (a.k.a. True Negative Rate): how well the model identifies actual negatives

# Display result
for i, spec in enumerate(specificity_i):
    print(f"Class {i} Specificity per Class: {spec:.3f}")


top_countschanW, rank_chanW = top_interpretable_channels(testW, ch_names, top=5)

    # save results to disk
os.chdir(r'C:\Users\anton\Documents\Python Scripts\Project_Parkinon_EEG_Analysis')
save2disk = {
                        "model_config": myRT_model_config,"data_names":featCOI,
                        'testW': testW, 'rank_chanW': rank_chanW, 'top_countschanW':top_countschanW, 
                        "loso_train_loss": loso_train_loss, "loso_train_acc": loso_train_acc,                  
                        "loso_exp_loss": loso_exp_loss, "loso_exp_acc": loso_exp_acc,
                        "true_labels": true_labels, "pred_labels": pred_labels,
                        "LosoScore": accScore, 'class_accScore': accuracy_i, "class_preScore": preScore, 
                        "class_recScore": recScore, "class_f1_Score": f1_Score, "class_specificity_i": specificity_i,
                        "uC_mat": uC_mat, 'fusionProb': fusionProb
                        
                        }

print('DL run is now complete, Data stored ready for fusion')

# automatic Name saver for post processing
if ('ap_exp' in featCOI) and ('del_relP_exp' in featCOI):
    
    if condExclude== 0:
        saveFileName = f'{fileName.split("_")[0]}_cbFeatImpPart_PDoff_PDon.npz'
    elif condExclude ==1:
        saveFileName = f'{fileName.split("_")[0]}_cbFeatImpPart_CN_PDon.npz'
    elif condExclude ==2:
        saveFileName = f'{fileName.split("_")[0]}_cbFeatImpPart_CN_PDoff.npz'
    elif condExclude ==3:
        saveFileName = f'{fileName.split("_")[0]}_cbFeatImpPart_3C.npz'
        
elif 'ap_exp' in featCOI:
    
    if condExclude== 0:
        saveFileName = f'{fileName.split("_")[0]}_myFeatImpPart_PDoff_PDon.npz'
    elif condExclude ==1:
        saveFileName = f'{fileName.split("_")[0]}_myFeatImpPart_CN_PDon.npz'
    elif condExclude ==2:
        saveFileName = f'{fileName.split("_")[0]}_myFeatImpPart_CN_PDoff.npz'
    elif condExclude ==3:
        saveFileName = f'{fileName.split("_")[0]}_myFeatImpPart_3C.npz'
            
elif 'del_relP_exp' in featCOI:
    
    if condExclude== 0:
        saveFileName = f'{fileName.split("_")[0]}_stFeatImpPart_PDoff_PDon.npz'
    elif condExclude ==1:
        saveFileName = f'{fileName.split("_")[0]}_stFeatImpPart_CN_PDon.npz'
    elif condExclude ==2:
        saveFileName = f'{fileName.split("_")[0]}_stFeatImpPart_CN_PDoff.npz'
    elif condExclude ==3:
        saveFileName = f'{fileName.split("_")[0]}_stFeatImpPart_3C.npz'


print(saveFileName)

np.savez_compressed(saveFileName, **save2disk)
print(f'results saved in {os.getcwd()}')


#% plot som staff
CN_idx    =  [ (idx, t.item()) for idx, t in enumerate(true_labels) if t.item()==0]
PDoff_idx =  [ (idx, t.item()) for idx, t in enumerate(true_labels) if t.item()==1]
PDom_idx  =  [ (idx, t.item()) for idx, t in enumerate(true_labels) if t.item()==2]

# misclassified = [ (idx, 1, t, p) if t == p else (idx, 0, t, p) for idx, (t, p) in enumerate(zip(true_labels, pred_labels)) ]
misclassified = [ 1 if t == p else 0 for idx, (t, p) in enumerate(zip(true_labels, pred_labels)) ]


# show that the results overlap
fig, ax = plt.subplots(1, figsize =(12,12))

ax.plot( np.arange(0, len(pred_labels), 1), pred_labels, 'bo', mfc = 'w', markersize=10, label='Predicted Labels')
ax.plot(np.arange(0, len(pred_labels), 1), true_labels,'go',mfc = 'w', label='Real Labels')
ax.plot( np.arange(0, len(pred_labels), 1), [ 3 if misclassified[i]==0 else np.inf for i in range(len(pred_labels)) ],'ro', label='Missclass Labels')
ax.legend()

ax.set( xlabel = 'Patient id', yticks = [0, 1, 2, 3], yticklabels= ['CN', 'PDoff', 'PDon', 'Missclassified'], ylabel='Model Prediction')
# ax.set(title =f'File: {saveFileName}, Model Prediction Accuracy: {np.round(100*np.mean(misclassified),2)} % \n {feature_labels}: ')

ax.set( xlim = [-1,len(pred_labels)+1 ])
ax.set(ylim =[-0.5,3.5])

fig.savefig(fileName+'.png', dpi=300)
print(os.getcwd()+ fileName)

plt.show()

#%% print model parameters

# Total parameters
total_params = sum(p.numel() for p in model.parameters())

# Trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total_params:,}")
print(f"Trainable: {trainable_params:,}")

for name, p in model.named_parameters():
    print(name, p.shape, p.numel())
    
#%% plot channel importance

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Mean and SEM
# -----------------------

meanW = np.mean(testW,axis=1)

semW = np.std(testW,axis=1) / np.sqrt(testW.shape[1])

# -----------------------
# Plot
# -----------------------

fig,ax = plt.subplots(figsize=(12,5))

bars = ax.bar(
    range(len(ch_names)),
    meanW,
    yerr=semW,
    capsize=4
)

ax.set_xticks(range(len(ch_names)))
ax.set_xticklabels(ch_names,rotation=90)

ax.set_ylabel('Channel Importance Weight')
ax.set_xlabel('EEG Channel')

ax.set_title(
    'Average Channel Importance Across LOSO Subjects'
)

ax.grid(alpha=.3)

plt.tight_layout()
plt.show()

#%% channel importance per diagniositci contrast

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# =====================================================
# INPUTS REQUIRED
# =====================================================
# testW         : shape (n_channels, n_subjects)
# true_labels   : shape (n_subjects,)
# channel_names : list of channel names
# =====================================================

true_labels = np.array(true_labels)

nClasses = len(np.unique(true_labels))

# =====================================================
# FIGURE SETUP
# =====================================================

fig, ax = plt.subplots(figsize=(15,6))

x = np.arange(len(ch_names))

# =====================================================
# 2-CLASS PROBLEM
# =====================================================

if nClasses == 2:

    idx0 = np.where(true_labels == 0)[0]
    idx1 = np.where(true_labels == 1)[0]

    W0 = testW[:, idx0]
    W1 = testW[:, idx1]

    mean0 = np.mean(W0, axis=1)
    mean1 = np.mean(W1, axis=1)

    sem0 = sem(W0, axis=1)
    sem1 = sem(W1, axis=1)

    width = 0.40

    ax.bar(
        x - width/2,
        mean0,
        width,
        yerr=sem0,
        capsize=4,
        label='Class 0'
    )

    ax.bar(
        x + width/2,
        mean1,
        width,
        yerr=sem1,
        capsize=4,
        label='Class 1'
    )

# =====================================================
# 3-CLASS PROBLEM
# =====================================================

elif nClasses == 3:

    idx0 = np.where(true_labels == 0)[0]  # Controls
    idx1 = np.where(true_labels == 1)[0]  # PD OFF
    idx2 = np.where(true_labels == 2)[0]  # PD ON

    W0 = testW[:, idx0]
    W1 = testW[:, idx1]
    W2 = testW[:, idx2]

    mean0 = np.mean(W0, axis=1)
    mean1 = np.mean(W1, axis=1)
    mean2 = np.mean(W2, axis=1)

    sem0 = sem(W0, axis=1)
    sem1 = sem(W1, axis=1)
    sem2 = sem(W2, axis=1)

    width = 0.27

    ax.bar(
        x - width,
        mean0,
        width,
        yerr=sem0,
        capsize=4,
        label='Controls'
    )

    ax.bar(
        x,
        mean1,
        width,
        yerr=sem1,
        capsize=4,
        label='PD OFF'
    )

    ax.bar(
        x + width,
        mean2,
        width,
        yerr=sem2,
        capsize=4,
        label='PD ON'
    )

# =====================================================
# ERROR
# =====================================================

else:

    raise ValueError(
        f'Only 2-class or 3-class problems supported. Found {nClasses} classes.'
    )

# =====================================================
# COSMETICS
# =====================================================

ax.set_xticks(x)
ax.set_xticklabels(ch_names, rotation=90)

ax.set_xlabel('EEG Channel')
ax.set_ylabel('Mean Channel Importance Weight')

ax.set_title(
    'Transformer-Derived Channel Importance\n'
    'Mean ± SEM Across LOSO Test Subjects'
)

ax.legend(frameon=False)

ax.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

plt.tight_layout()
plt.show()

#%% channel importance subtraction
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# INPUTS REQUIRED
# =====================================================
# testW         : shape (n_channels, n_subjects)
# true_labels   : shape (n_subjects,)
# channel_names : list of channel names
# =====================================================

true_labels = np.array(true_labels)

# =====================================================
# CLASS MEANS
# =====================================================

classes = np.unique(true_labels)
nClasses = len(classes)

class_means = {}

for c in classes:

    idx = np.where(true_labels == c)[0]

    class_means[c] = np.mean(
        testW[:, idx],
        axis=1
    )

# =====================================================
# 2 CLASS
# =====================================================

if nClasses == 2:

    delta = class_means[1] - class_means[0]

    fig, ax = plt.subplots(figsize=(14,5))

    ax.bar(
        ch_names,
        delta
    )

    ax.axhline(
        0,
        color='k',
        linestyle='--'
    )

    ax.set_ylabel('Importance Difference')
    ax.set_xlabel('Channel')

    ax.set_title(
        'Channel Importance Difference\n'
        'Class 1 - Class 0'
    )

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# =====================================================
# 3 CLASS
# =====================================================

elif nClasses == 3:

    comparisons = [
        (1,0,'PD OFF - Controls'),
        (2,0,'PD ON - Controls'),
        (2,1,'PD ON - PD OFF')
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15,10),
        sharex=True
    )

    for ax,(c1,c0,title) in zip(axes,comparisons):

        delta = class_means[c1] - class_means[c0]

        ax.bar(
            ch_names,
            delta
        )

        ax.axhline(
            0,
            color='k',
            linestyle='--'
        )

        ax.set_ylabel('Δ Importance')

        ax.set_title(title)

        ax.grid(
            axis='y',
            linestyle='--',
            alpha=.3
        )

    axes[-1].set_xlabel('EEG Channel')

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

# =====================================================
# ERROR
# =====================================================

else:

    raise ValueError(
        f'Only 2 or 3 class problems supported. Found {nClasses}.'
    )
    
#%% Channel importance in a head montage map

import mne

def create_HeadMontage_RawInfo(ch_names, fs=None):
    """
    Parameters
    ----------
    ch_names : list of str
        Channel names.
    fs : int | None
        Sampling frequency (default = 256 Hz)

    Returns
    -------
    info : mne.Info
        Info object with BioSemi32 montage attached.
    """

    if fs is None:
        fs = 256

    # remove non-EEG channels
    scalp_chs = [ch for ch in ch_names
                 if not (ch.startswith('EXG') or ch.startswith('Status'))]

    # create info
    info = mne.create_info(ch_names=scalp_chs,
                           sfreq=fs,
                           ch_types='eeg')

    # attach BioSemi32 montage
    montage = mne.channels.make_standard_montage('biosemi32')
    info.set_montage(montage)

    return info

# for plotting and visualisations
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
import itertools

# create head montage and define adjacency
el_info = create_HeadMontage_RawInfo(ch_names, fs=None)

#% ==========================================================
# CHANNEL IMPORTANCE CONTRAST TOPOGRAPHIES
# ==========================================================
#
# INPUTS REQUIRED
#
# testW        : (n_channels, n_subjects)
# true_labels  : (n_subjects,)
# ch_names
# el_info      : output from create_HeadMontage_RawInfo()
#
# ASSUMPTIONS
#
# class 0 vs others = independent groups
#
# non-zero classes compared against each other
# are repeated measures / paired observations
#
# ==========================================================


from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

from statsmodels.stats.multitest import fdrcorrection


# ----------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------

alpha = 0.05

true_labels = np.asarray(true_labels)

classes = np.unique(true_labels)
dClass = ['Control','PD_OFF','PD_ON']
# ----------------------------------------------------------
# BUILD ALL PAIRWISE CONTRASTS
# ----------------------------------------------------------

comparisons = []

for c1, c0 in itertools.combinations(classes, 2):

    if (c1 == 0) or (c0 == 0):
        paired = False
        test_name = 'Mann-Whitney U'

    else:
        paired = True
        test_name = 'Wilcoxon'

    comparisons.append(
        {
            'c1': c1,
            'c0': c0,
            'paired': paired,
            'test': test_name,
            'label': f'{dClass[c1]} - {dClass[c0]}'
        }
    )

# ----------------------------------------------------------
# RUN ANALYSES
# ----------------------------------------------------------

contrast_results = {}

for comp in comparisons:

    c1 = comp['c1']
    c0 = comp['c0']

    idx1 = np.where(true_labels == c1)[0]
    idx0 = np.where(true_labels == c0)[0]

    W1 = testW[:, idx1]
    W0 = testW[:, idx0]

    # ------------------------------------------------------
    # EFFECT SIZE MAP
    # ------------------------------------------------------

    delta = (
        np.mean(W1, axis=1)
        -
        np.mean(W0, axis=1)
    )

    # ------------------------------------------------------
    # CHANNEL-WISE TESTING
    # ------------------------------------------------------

    pvals = np.zeros(len(ch_names))
    stats = np.zeros(len(ch_names))

    for ch in range(len(ch_names)):

        x = W1[ch, :]
        y = W0[ch, :]

        try:

            if comp['paired']:

                stat, p = wilcoxon(
                    x,
                    y,
                    alternative='two-sided',
                    zero_method='wilcox'
                )

            else:

                stat, p = mannwhitneyu(
                    x,
                    y,
                    alternative='two-sided'
                )

        except Exception:

            stat = np.nan
            p = 1.0

        stats[ch] = stat
        pvals[ch] = p

    # ------------------------------------------------------
    # FDR CORRECTION
    # ------------------------------------------------------

    reject, pvals_fdr = fdrcorrection(
        pvals,
        alpha=alpha
    )

    # ------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------

    contrast_results[comp['label']] = {

        'delta'      : delta,
        'statistic'  : stats,
        'p_raw'      : pvals,
        'p_fdr'      : pvals_fdr,
        'sig_mask'   : reject,
        'paired'     : comp['paired'],
        'test'       : comp['test']
    }

# ----------------------------------------------------------
# GLOBAL COLOUR SCALE
# ----------------------------------------------------------

all_delta = np.concatenate(
    [
        contrast_results[k]['delta']
        for k in contrast_results.keys()
    ]
)

vmax = np.max(np.abs(all_delta))

# ----------------------------------------------------------
# FIGURE
# ----------------------------------------------------------

nContrasts = len(comparisons)

fig, axes = plt.subplots(
    1,
    nContrasts,
    figsize=(4*nContrasts, 4)
)

if nContrasts == 1:
    axes = [axes]

im = None

for ax, label in zip(axes, contrast_results.keys()):

    result = contrast_results[label]

    delta = result['delta']
    sig_mask = result['sig_mask']

    im, _ = plot_topomap(
        delta,
        pos=el_info,
        axes=ax,
        cmap='RdBu_r',
        vlim=(-vmax, vmax),
        sensors=True,
        contours=12,
        res=1024,
        show=False,
        mask=sig_mask,
        mask_params=dict(
            marker='*',
            markerfacecolor='k',
            markeredgecolor='k',
            linewidth=0,
            markersize=12
        )
    )

    nSig = np.sum(sig_mask)

    ax.set_title(
        f'{label}\n'
        
    )

# ----------------------------------------------------------
# SHARED COLORBAR
# ----------------------------------------------------------

fig.subplots_adjust(right=0.88, top=0.90)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cax)


cbar.set_label(
    'Mean Importance Difference',
    fontsize=11
)

# plt.suptitle(
#     'Transformer Channel Importance Contrasts\n'
#     'Colours = Mean Difference | * = FDR Significant'
# )

# plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# OPTIONAL SUMMARY TABLE
# ----------------------------------------------------------

for contrast_name, result in contrast_results.items():

    print('\n')
    print('='*80)
    print(contrast_name)
    print('='*80)

    sig_idx = np.where(result['sig_mask'])[0]

    if len(sig_idx) == 0:

        print('No FDR-significant channels.')

    else:

        for idx in sig_idx:

            print(
                f'{ch_names[idx]:<8s}'
                f' p_raw={result["p_raw"][idx]:.5f}'
                f' p_fdr={result["p_fdr"][idx]:.5f}'
                f' delta={result["delta"][idx]:.4f}'
            )

#%% Channel FREQUENCY importance in a head montage map

import mne

def create_HeadMontage_RawInfo(ch_names, fs=None):
    """
    Parameters
    ----------
    ch_names : list of str
        Channel names.
    fs : int | None
        Sampling frequency (default = 256 Hz)

    Returns
    -------
    info : mne.Info
        Info object with BioSemi32 montage attached.
    """

    if fs is None:
        fs = 256

    # remove non-EEG channels
    scalp_chs = [ch for ch in ch_names
                 if not (ch.startswith('EXG') or ch.startswith('Status'))]

    # create info
    info = mne.create_info(ch_names=scalp_chs,
                           sfreq=fs,
                           ch_types='eeg')

    # attach BioSemi32 montage
    montage = mne.channels.make_standard_montage('biosemi32')
    info.set_montage(montage)

    return info

# for plotting and visualisations
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
import itertools

# create head montage and define adjacency
el_info = create_HeadMontage_RawInfo(ch_names, fs=None)

#% ==========================================================
# CHANNEL IMPORTANCE CONTRAST TOPOGRAPHIES
# ==========================================================
#
# INPUTS REQUIRED
#
# testW        : (n_channels, n_subjects)
# true_labels  : (n_subjects,)
# ch_names
# el_info      : output from create_HeadMontage_RawInfo()
#
# ASSUMPTIONS
#
# class 0 vs others = independent groups
#
# non-zero classes compared against each other
# are repeated measures / paired observations
#
# ==========================================================


from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

from statsmodels.stats.multitest import fdrcorrection


# ----------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------

alpha = 0.05

true_labels = np.asarray(true_labels)

classes = np.unique(true_labels)
dClass = ['Control','PD_OFF','PD_ON']
# ----------------------------------------------------------
# BUILD ALL PAIRWISE CONTRASTS
# ----------------------------------------------------------

comparisons = []

for c1, c0 in itertools.combinations(classes, 2):

    if (c1 == 0) or (c0 == 0):
        paired = False
        test_name = 'Mann-Whitney U'

    else:
        paired = True
        test_name = 'Wilcoxon'

    comparisons.append(
        {
            'c1': c1,
            'c0': c0,
            'paired': paired,
            'test': test_name,
            'label': f'{dClass[c1]} - {dClass[c0]}'
        }
    )

# ----------------------------------------------------------
# RUN ANALYSES
# ----------------------------------------------------------

contrast_results = {}

for comp in comparisons:

    c1 = comp['c1']
    c0 = comp['c0']

    idx1 = np.where(true_labels == c1)[0]
    idx0 = np.where(true_labels == c0)[0]

    W1 = testW[:, idx1]
    W0 = testW[:, idx0]

    # ------------------------------------------------------
    # EFFECT SIZE MAP
    # ------------------------------------------------------

    top1, _ = top_interpretable_channels(W1, ch_names, top=5)
    top0, _ = top_interpretable_channels(W0, ch_names, top=5)
    
    
    delta = top1 - top0

    # ------------------------------------------------------
    # CHANNEL-WISE TESTING
    # ------------------------------------------------------

    pvals = np.zeros(len(ch_names))
    stats = np.zeros(len(ch_names))

    for ch in range(len(ch_names)):

        x = W1[ch, :]
        y = W0[ch, :]

        try:

            if comp['paired']:

                stat, p = wilcoxon(
                    x,
                    y,
                    alternative='two-sided',
                    zero_method='wilcox'
                )

            else:

                stat, p = mannwhitneyu(
                    x,
                    y,
                    alternative='two-sided'
                )

        except Exception:

            stat = np.nan
            p = 1.0

        stats[ch] = stat
        pvals[ch] = p

    # ------------------------------------------------------
    # FDR CORRECTION
    # ------------------------------------------------------

    reject, pvals_fdr = fdrcorrection(
        pvals,
        alpha=alpha
    )

    # ------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------

    contrast_results[comp['label']] = {

        'delta'      : delta,
        'statistic'  : stats,
        'p_raw'      : pvals,
        'p_fdr'      : pvals_fdr,
        'sig_mask'   : reject,
        'paired'     : comp['paired'],
        'test'       : comp['test']
    }

# ----------------------------------------------------------
# GLOBAL COLOUR SCALE
# ----------------------------------------------------------

all_delta = np.concatenate(
    [
        contrast_results[k]['delta']
        for k in contrast_results.keys()
    ]
)

vmax = np.max(np.abs(all_delta))

# ----------------------------------------------------------
# FIGURE
# ----------------------------------------------------------

nContrasts = len(comparisons)

fig, axes = plt.subplots(
    1,
    nContrasts,
    figsize=(4*nContrasts, 4)
)

if nContrasts == 1:
    axes = [axes]

im = None

for ax, label in zip(axes, contrast_results.keys()):

    result = contrast_results[label]

    delta = result['delta']
    sig_mask = result['sig_mask']

    im, _ = plot_topomap(
        delta,
        pos=el_info,
        axes=ax,
        cmap='RdBu_r',
        vlim=(-vmax, vmax),
        sensors=True,
        contours=12,
        res=1024,
        show=False,
        mask=sig_mask,
        mask_params=dict(
            marker='*',
            markerfacecolor='k',
            markeredgecolor='k',
            linewidth=0,
            markersize=12
        )
    )

    nSig = np.sum(sig_mask)

    ax.set_title(
        f'{label}\n'
        
    )

# ----------------------------------------------------------
# SHARED COLORBAR
# ----------------------------------------------------------

fig.subplots_adjust(right=0.88, top=0.90)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cax)


cbar.set_label(
    'Difference in P(channel ∈ Top-5)',
    fontsize=11
)

# plt.suptitle(
#     'Transformer Channel Importance Contrasts\n'
#     'Colours = Mean Difference | * = FDR Significant'
# )

# plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# OPTIONAL SUMMARY TABLE
# ----------------------------------------------------------

for contrast_name, result in contrast_results.items():

    print('\n')
    print('='*80)
    print(contrast_name)
    print('='*80)

    sig_idx = np.where(result['sig_mask'])[0]

    if len(sig_idx) == 0:

        print('No FDR-significant channels.')

    else:

        for idx in sig_idx:

            print(
                f'{ch_names[idx]:<8s}'
                f' p_raw={result["p_raw"][idx]:.5f}'
                f' p_fdr={result["p_fdr"][idx]:.5f}'
                f' freq_delta={result["delta"][idx]:.4f}'
            )

