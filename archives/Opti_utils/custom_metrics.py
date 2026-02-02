
### Custom metric functions for ML model performance
# Author: Maxime Mock

from sklearn.metrics import make_scorer, confusion_matrix, recall_score

def custom_metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TP = CM[1][1]
    FP = CM[0][1]
    TN = CM[0][0]
    FN = CM[1][0]
    specificite = TN/(FP+TN)
    return specificite

def sensi(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)

def speci(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def mix_sensi_speci(y_true, y_pred):
    return (recall_score(y_true, y_pred, pos_label=0) + recall_score(y_true, y_pred, pos_label=1))/2

# scorer = make_scorer(custom_metric, greater_is_better=True)
# mix_recall = make_scorer(mix_sensi_speci)
# sensibilite = make_scorer(sensi)
# specificite = make_scorer(speci)