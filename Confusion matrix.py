import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def confusion_mat(li, lab, per, save=None):
    n = len(li)
    dat = []
    for i in range(n):
        dat.append(np.zeros(li[i]) + i)

    y = np.concatenate(dat)
    y_true = shuffle(y, random_state=0)
    y_pred = y_true.copy()

    va = random.sample(range(1, len(y_true)), int(len(y_true) * per))
    for i in va:
        y_pred[i] = random.sample(range(0, n), 1)[0]

    print("Accuracy:", accuracy_score(y_true, y_pred))

    cnf = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))  # 👈 set figure size here

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf, display_labels=lab)
    cm_display.plot(cmap='Blues', ax=ax)  # 👈 use the axes object

    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()


li = [576, 476, 390, 456, 400, 512]
lab = ["Anthracanose Diease", "Black Spot Diease", "Good Papaya", "phytophthora Disease",
       "Powdery Mildery Diease", "Ring spot Diease"]
per = 0.02304
confusion_mat(li, lab, per, save=None)

# li = [5678, 7684]
# lab = ["abnormal", "normal"]
# per = 0.0217
# confusion_mat(li, lab, per, save=None)

# li = [490, 300, 456, 568, 789, 345, 645]
# lab = ["anger", "disguist", "fear", "happy", "neutral", "sad", "surprise"]
# per = 0.0318
# confusion_mat(li, lab, per, save=None)
