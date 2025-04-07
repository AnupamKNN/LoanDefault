from loandefault.entity.artifact_entity import ClassificationMetricArtifact
from loandefault.exception.exception import LoanDefaultException
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sys

def get_classification_score(y_true, y_pred):
    try:
        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precission_score = precision_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(accuracy_score = model_accuracy_score, f1_score= model_f1_score,
                                                             precision_score= model_precission_score,
                                                             recall_score= model_recall_score)
        return classification_metric
    except Exception as e:
        raise LoanDefaultException(e, sys)