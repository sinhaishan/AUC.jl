using AUC
using Base.Test

y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
y_score = [0.3, 0.2, 0.3, 0.23, 0.5, 0.34, 0.45, 0.54, 0.6, 0.7, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.6, 0.7, 0.5, 0.2, 0.1, 0.7, 0.2, 0.7, 0.4]


"""
    roc_auc_score(y_true, y_score)

This function returns the area under the curve (AUC) for the receiver operating characteristic 
curve (ROC). This function takes two vectors, `y_true` and `y_score`. The vector `y_true` is the 
observed `y` in a binary classification problem. And the vector `y_score` is the real-valued 
prediction for each observation.
"""
function roc_auc_score(y_true, y_score)
    if length(Set(y_true)) == 1
        warn("Only one class present in y_true.\n
              The AUC is not defined in that case; returning -Inf.")
        res = -Inf
    else
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        res = auc(fpr, tpr, true)
    end
    res
end

res = roc_auc_score(y, y_score)

@test res == 0.0
