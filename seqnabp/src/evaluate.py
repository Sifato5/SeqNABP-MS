import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); oh=np.eye(p.shape[1])[y]
    try: auc=roc_auc_score(oh,p,average="macro",multi_class="ovr")
    except ValueError: auc=float("nan")
    try: ap=average_precision_score(oh,p,average="macro")
    except ValueError: ap=float("nan")
    return {"ACC":accuracy_score(y,p.argmax(1)),
      "REC":recall_score(y,p.argmax(1),average="macro",zero_division=0),
      "PRE":precision_score(y,p.argmax(1),average="macro",zero_division=0),
      "F1":f1_score(y,p.argmax(1),average="macro",zero_division=0),"AUC":auc,"AUPRC":ap}
