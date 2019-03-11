from imblearn.metrics import specificity_score
from sklearn.metrics import make_scorer

def specificity(y, y_pred):
    return specificity_score(y, y_pred, average='macro')

specificity_scorer = make_scorer(specificity, greater_is_better=True)