### Imports ###

# Data manipulation and other stuff : 
import numpy as np
import pandas as pd
import re
#pd.set_option('display.max_rows', 10)
from matplotlib import pyplot as plt
import seaborn as sns

# Utils for NLP : 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Utils for encoding : 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# Utils for regression : 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

from sklearn.utils.class_weight import compute_class_weight

# Utils for Metrics calculation : 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, hamming_loss, accuracy_score, jaccard_score, classification_report, roc_auc_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve

from metrics_utils import *

# Custom preprocessing : 
from preprocess_NLP import *

# Tensorflow/keras
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense

import functools

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

from tqdm.notebook import tqdm

from datasets.dataset_dict import DatasetDict
from datasets import Dataset

import pickle


model_dir = '../../models/'
pred_dir = '../../predictions/'

# Choose pretrained model to use (e.g. 'camembert-base' or 'camembert/camembert-large')
pretrained_model = 'camembert-base'


### Load data ###
target_feature = 'atcd.endo'

# Loading X : 
df_nlp = pd.read_csv('./../../Data/Generate/donnees_entree_nlp_sans_endo.csv', usecols=['Anonymisation', 'Date', 'Nature', 'Résumé'])
df_nlp_orig = df_nlp.copy()
print('X shape is :', df_nlp.shape)

# Loading Y : 
recueil_orig  = pd.read_excel('./../../Data/Raw/Recueil (1).xlsx').drop('Unnamed: 90', axis=1)
recueil_orig = recueil_orig[['Numéro anonymat', 'atcd.endo', 'irm.lusg', 'tv.douloureux', 'irm.externe', 'sf.dig.diarrhee', 'echo.lusg', 'echo.lusd', 'ef.hormone.dpc', 'effet.hormone']]
recueil = recueil_orig.copy()
recueil.replace(['Na', 'NA'], np.nan, inplace=True)
recueil = recueil.rename(columns={'Numéro anonymat': 'Anonymisation'})
recueil = recueil[['Anonymisation'] + [target_feature]]

print('Y shape is :', recueil.shape)
print(f'Nombre de patientes dans le df_nlp : {len(df_nlp.Anonymisation.unique())}')
num_labels = len(pd.unique(recueil_orig[target_feature]))
print(f'Nombre de classes pour{target_feature}: {num_labels}')

if 'DJ-055' in list(df_nlp['Anonymisation']):
    df_nlp.loc[df_nlp['Anonymisation']=='DJ-055', 'Anonymisation'] ='NJ-055'
'NJ-055' in list(df_nlp['Anonymisation'])
'DJ-055' in list(df_nlp['Anonymisation'])


### Preprocessing

# IMPORTANT: Lowercase and removal of special characters has to be applied before 'correction_series', else words will not be found in the correction dictionnary
df_nlp.Résumé = df_nlp.Résumé.apply(remove_special_characters)
df_nlp.Résumé = df_nlp.Résumé.apply(lowercase_text)
df_nlp.Résumé = df_nlp.Résumé.apply(correction_series)

### Data preparation for training

# Merge receuil and nlp dataframes and rename columns
#df_nlp = df_nlp.groupby('Anonymisation')['Résumé'].agg(' '.join).reset_index()  # not joining all texts for patients anymore
data = pd.merge(df_nlp, recueil, on='Anonymisation', how='inner')
data = data.rename(columns={'Anonymisation': 'patient', 'Résumé': 'text', target_feature: 'outcome'})
# Format labels
def label_outcome(row):
    if row['outcome'] == 0:
        return 'absent'
    elif row['outcome'] == 1:
        return 'present'
    else:
        return 'missing'
data['str_outcome'] = data.apply(label_outcome, axis=1)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast = False)

def tokenize_batch(samples, tokenizer, max_length):
    text = [sample["text"] for sample in samples]
    labels = torch.tensor([sample["outcome"] for sample in samples])
    str_labels = [sample["str_outcome"] for sample in samples]
    # The tokenizer handles
    # - Tokenization (amazing right?)
    # - Padding (adding empty tokens so that each example has the same length)
    # - Truncation (cutting samples that are too long)
    # - Special tokens (in CamemBERT, each sentence ends with a special token </s>)
    # - Attention mask (a binary vector which tells the model which tokens to look at. For instance it will not compute anything if the token is a padding token)
    #tokens = tokenizer(text, padding="longest", return_tensors="pt")
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}


## Train/val/test split

prop_train = 0.8
prop_val = 0.1
prop_test = 0.1

# Choose how to split: 
# 'patientwise': On patient-level (such that no partient is in the train and val/test set)
# 'patientmix': Not seperating patients in train vs. val/test set
method_split = 'patientwise'

### Split based on patients
if method_split == 'patientwise':
    patient_outcomes = data.groupby('patient')['outcome'].max().reset_index()
    patient_outcomes['nrows'] = list(data['patient'].value_counts())

    train_patients, test_patients = train_test_split(
        patient_outcomes,
        test_size=prop_test,
        stratify=patient_outcomes['outcome'],
        random_state=42
    )

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=0.125,
        stratify=train_patients['outcome'],
        random_state=42
    )

    train = data.loc[data.patient.isin(list(train_patients.patient))]
    val = data.loc[data.patient.isin(list(val_patients.patient))]
    test = data.loc[data.patient.isin(list(test_patients.patient))]

    npatients_train = len(pd.unique(train.patient))
    npatients_val = len(pd.unique(val.patient))
    npatients_test = len(pd.unique(test.patient))

    print(f'Number of patients in train set: {npatients_train}')
    print(f'Number of patients in validation set: {npatients_val}')
    print(f'Number of patients in test set: {npatients_test}')

### Simple approach, patients mixed in train and val/test
else:
    train, test = train_test_split(data, random_state=42, test_size=prop_test, stratify=data['outcome'])
    train, val = train_test_split(train, random_state=42, test_size=0.125, stratify=train['outcome'])

    print(f'Number of samples in train set: {train.shape[0]}, = {round((train.shape[0]/data.shape[0])*100, 2)} %')
    print(f'Number of samples in validation set: {val.shape[0]}, = {round((val.shape[0]/data.shape[0])*100, 2)} %')
    print(f'Number of samples in test set: {test.shape[0]}, = {round((test.shape[0]/data.shape[0])*100, 2)} %')

## Optional: Upsampling to correct class imbalance

use_upsampling = False

if use_upsampling:

    ### Add more samples to the minority class by dubplicating the last N samples for each patient

    train_original = train.copy()
    train_negative = train.loc[train.outcome==0]
    train_positive = train.loc[train.outcome==1]
    train_missing = train.loc[train.outcome==2]

    if train_negative.shape[0] > train_positive.shape[0]:
        data_train_majority = train_negative
        data_train_minority = train_positive
    else:
        data_train_majority = train_positive
        data_train_minority = train_negative

    nsamples_minority = data_train_minority.shape[0]
    nsamples_majority = data_train_majority.shape[0]
    nsamples_total_before_upsampling = nsamples_majority+nsamples_minority
    nsamples_to_add = data_train_majority.shape[0] - data_train_minority.shape[0]
    npatients = len(pd.unique(train.patient))
    nsamples_to_add_per_patient = int(nsamples_to_add/npatients)
    bias = data_train_minority.shape[0]/data_train_majority.shape[0]
    nsamples_in_total_after_upsampling = nsamples_majority+nsamples_minority+nsamples_to_add
    print(f'Number of samples in the minority class: {nsamples_minority}')
    print(f'Number of samples in the majority class: {nsamples_majority}')
    print(f'Number of samples in total now, before upsampling: {nsamples_total_before_upsampling}')
    print(f'Number of samples it add to minority class: {nsamples_to_add}')
    print(f'Number of samples to be obtained in total after upsampling: {nsamples_in_total_after_upsampling}')

    train_upsampled = train.copy()
    patients = pd.unique(train.patient)
    for patient in patients:
        train_patient = data_train_minority.loc[data_train_minority.patient==patient]
        train_upsampled = pd.concat([train_upsampled, train_patient.tail(nsamples_to_add_per_patient)])
    print(f'Number of samples obtained so far by augmentation: {train_upsampled.shape[0]}')

    # The division by number of patients doesn't result in all samples to add being assigned.
    # Iterate through patients and add more samples one-by-one, until we have enough
    while train_upsampled.shape[0] < nsamples_in_total_after_upsampling:
        ipatient = 0
        while ipatient < len(patients):
            patient = patients[ipatient]
            train_patient = data_train_minority.loc[data_train_minority.patient==patient]
            train_upsampled = pd.concat([train_upsampled, train_patient.tail(1)])
            if train_upsampled.shape[0] == nsamples_in_total_after_upsampling:
                break
            ipatient += 1
    print(f'Number of samples obtained by augmentation finally: {train_upsampled.shape[0]}')
    train = train_upsampled.copy()

## Transformation to DataLoader format

# Transformer les données en format qui peut être lu dans les librairies 'HuggingFace'

max_length = 512

train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

train_dataloader = DataLoader(
    dataset["train"], 
    batch_size=16,
    shuffle=True, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer, max_length=max_length)
)
val_dataloader = DataLoader(
    dataset["validation"], 
    batch_size=16, 
    shuffle=False, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer, max_length=max_length)
)
test_dataloader = DataLoader(
    dataset["test"], 
    batch_size=16, 
    shuffle=False, 
    collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer, max_length=max_length)
)

### Training

learning_rate = 3e-5 #3e-5
# Choose loss function: Either 'crossentropyloss', 'weightedcrossentropyloss' or 'focalloss'
loss_function = 'crossentropyloss'
if use_upsampling and (loss_function != 'crossentropyloss'):
    raise ValueError('If you use upsampling, you need to use crossentropyloss')

# Calculate class weights
if loss_function == 'weightedcrossentropyloss':
    class_weights = compute_class_weight(class_weight="balanced", classes=pd.unique(train_original['outcome']), y=train_original['outcome'].values)  # Using train_original because in case upsampling is used as well, compute weights without taking duplicate samples into account
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    print('Class weights:')
    print(class_weights)
else:
    class_weights = None

class FocalLoss(nn.Module):
    def __init__(self, gamma=6, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class LightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, loss_function, class_weights = None, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            # Si `from_scratch` est vrai, on charge uniquement la config (nombre de couches, hidden size, etc.) et pas les poids du modèle 
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            # Cette méthode permet de télécharger le bon modèle pré-entraîné directement depuis le Hub de HuggingFace sur lequel sont stockés de nombreux modèles
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def training_step(self, batch):
        out = self.forward(batch)

        logits = out.logits
        # -------- MASKED --------
        if loss_function == 'crossentropyloss':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_function == 'weightedcrossentropyloss':
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_function == 'focalloss':
            loss_fn = FocalLoss()
        else:
            raise ValueError(f'Invalid loss function specified: {loss_function}')
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        # ------ END MASKED ------

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)

        preds = torch.max(out.logits, -1).indices
        # -------- MASKED --------
        acc = (batch["labels"] == preds).float().mean()
        # ------ END MASKED ------
        self.log("valid/acc", acc)

        f1 = f1_score(batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        """La fonction predict step facilite la prédiction de données. Elle est 
        similaire à `validation_step`, sans le calcul des métriques.
        """
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


target_feature_txt = target_feature.replace('.', '_')
txt_upsampling = '_upsampling' if use_upsampling else ''
model_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="valid/acc", 
    mode="max",
    dirpath=model_dir,
    filename=f'{pretrained_model}_{target_feature_txt}_split_{method_split}_learning_rate_{learning_rate}_{loss_function}{txt_upsampling}',
    save_top_k=1,
    save_weights_only=True
)

# Set up CSV logger
csv_logger = CSVLogger("logs", name="camembert_training")
camembert_trainer = pl.Trainer(
    max_epochs=20,
    #gpus=1,
    logger=csv_logger,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]
)

lightning_model = LightningModel(pretrained_model, num_labels, lr=learning_rate, weight_decay=0., loss_function=loss_function, class_weights=class_weights)

camembert_trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

pickle.dump(camembert_trainer, open(model_dir + f'{pretrained_model}_{target_feature_txt}_split_{method_split}_learning_rate_{learning_rate}_{loss_function}_{txt_upsampling}', 'wb'))

### Evaluate performance

# Load the best model checkpoint
best_model_path = model_checkpoint.best_model_path
best_model = LightningModel.load_from_checkpoint(best_model_path, model_name=pretrained_model, num_labels=num_labels, lr=3e-5, weight_decay=0.)

# Predict on the test dataset
predictions = []
true_labels = []

best_model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculations for inference
    for batch in test_dataloader:
        preds = best_model.predict_step(batch, batch_idx=0).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        
        predictions.extend(preds)
        true_labels.extend(labels)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')
conf_matrix = confusion_matrix(true_labels, predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

## Show performance by patient
# Merge predictions patient-wise (since we have several texts per patient), choosing the most frequent prediction

test.loc[:, 'predictions'] = predictions
predictions_per_patient_text = test.sort_values('patient', ascending=True).groupby('patient')['predictions'].value_counts()
predictions_by_patient_majority = []
for patient in pd.unique(test.patient):
    # In case the number of predictions for the classes are the same, assign most frequent outcome
    if (predictions_per_patient_text[patient].shape[0]==2) and (predictions_per_patient_text[patient].loc[0] == predictions_per_patient_text[patient].loc[1]):
        predictions_by_patient_majority = predictions_by_patient_majority + [test['outcome'].value_counts().nlargest(1).index[0]]
    else:
        predictions_by_patient_majority = predictions_by_patient_majority + [predictions_per_patient_text[patient].idxmax()]

true_classes_by_patient = list(test.drop_duplicates(subset='patient').sort_values('patient', ascending=True)['outcome'])

if len(predictions_by_patient_majority) != len(true_classes_by_patient):
    raise ValueError('Mismatch in the number of samples')
    
# Calculate evaluation metrics if the lengths match
accuracy_by_patient_majority = accuracy_score(true_classes_by_patient, predictions_by_patient_majority)
precision_by_patient_majority = precision_score(true_classes_by_patient, predictions_by_patient_majority, average='macro')
recall_by_patient_majority = recall_score(true_classes_by_patient, predictions_by_patient_majority, average='macro')
f1_by_patient_majority = f1_score(true_classes_by_patient, predictions_by_patient_majority, average='macro')
conf_matrix_by_patient_majority = confusion_matrix(true_classes_by_patient, predictions_by_patient_majority)

print('##### Performance patient-wise:')
print(f'Accuracy: {accuracy_by_patient_majority:.4f}')
print(f'Precision: {precision_by_patient_majority:.4f}')
print(f'Recall: {recall_by_patient_majority:.4f}')
print(f'F1 Score: {f1_by_patient_majority:.4f}')
print('Confusion Matrix:')
print(conf_matrix_by_patient_majority)