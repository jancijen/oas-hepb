from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

sns.set_style('white')

def curves_eval(y_true, y_pred, sample_weights=None):
    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    
    # PRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred, sample_weight=sample_weights)
    ax = axs[0]
    
    au_prc = average_precision_score(y_true, y_pred, sample_weight=sample_weights)
    
    ax.plot(recall, precision, label=f'PRC curve (area = {au_prc:.3f})')
    ax.legend()
    ax.set(title='PRC', xlabel='Recall', ylabel='Precision')
    
    # PRC - baseline
    if sample_weights is not None:
        pos_proportion = sample_weights[(y_true == True)].sum() / sample_weights.sum()
    else:
        pos_proportion = (y_true == True).sum() / len(y_true)

    ax.axhline(y=pos_proportion, color='.3', linestyle='--')
    
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, sample_weight=sample_weights)
    ax = axs[1]
    
    auc_val = roc_auc_score(y_true, y_pred, sample_weight=sample_weights)
    
    ax.plot(fpr, tpr, label=f'ROC curve (area = {auc_val:.3f})')
    ax.legend()
    ax.set(title='ROC', xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    # ROC - baseline
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.3')
    plt.show()

def evaluate_classifier(y_true, y_pred, y_pred_prob, metrics, sample_weights=None):
    # Metrics
    if metrics:
        print('Metric values:\n')
        
        for metric_name, metric_fn in metrics:
            print('{}: {:.3f}'.format(metric_name, metric_fn(y_true, y_pred, sample_weight=sample_weights)))
        
        print('\n')
        
    # Confusion matrix
    cm_labels = [True, False]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels,sample_weight=sample_weights)
    cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
    
    # Plot confusion matrix
    ax = sns.heatmap(cm_df, annot=True, fmt=',')
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix');
    plt.show()
    
    # Curves
    curves_eval(y_true, y_pred_prob, sample_weights)

def get_threshold(true_values, predictions, target_fpr=None, target_tpr=None, sample_weights=None):
    """
    Source: https://github.com/Merck/bgc-pipeline/blob/29300da912fd1836eea8e285e2e50f5326f021f3/bgc_detection/evaluation/confusion_matrix.py#L64
    
    Calculate threshold that should be used a given FPR or TPR value, based on given true values and predictions.
    Can be seen as a horizontal or vertical cut of a ROC curve
    :param true_values: Series of true values
    :param predictions: Series of predictions
    :param target_fpr: Target TPR to be achieved (or None to ignore)
    :param target_tpr: Target FPR to be achieved (or None to ignore)
    :return: threshold that should be used a given FPR or TPR value
    """
    if target_fpr is None and target_tpr is None:
        raise AttributeError('Specify one of TPR and FPR')
    if target_fpr and target_tpr:
        raise AttributeError('Specify only one of TPR and FPR')
    prev_threshold = None
    fprs, tprs, thresholds = roc_curve(true_values, predictions, sample_weight=sample_weights)
    for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
        if target_fpr is not None and fpr > target_fpr:
            break
        if target_tpr is not None and tpr > target_tpr:
            break
        prev_threshold = threshold
    if not prev_threshold:
        raise AttributeError('Target FPR or TPR not achievable')

    return prev_threshold
