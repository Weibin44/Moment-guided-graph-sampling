import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, dtype=bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def label_classification(embeddings, y, split):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    train_idx = split['train'].cpu().numpy()
    valid_idx = split['valid'].cpu().numpy()
    test_idx = split['test'].cpu().numpy()
    best_score, best_clf, best_c = -1.0, None, None
    for c in 2.0 ** np.arange(-10, 10):
        clf = OneVsRestClassifier(
            LogisticRegression(solver='liblinear', C=c)
        )
        clf.fit(X[train_idx], Y[train_idx])
        score = accuracy_score(
            Y[valid_idx], prob_to_one_hot(clf.predict_proba(X[valid_idx]))
        )
        if score > best_score:
            best_score, best_clf, best_c = score, clf, c

    y_test = Y[test_idx]
    y_pred = best_clf.predict_proba(X[test_idx])
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'TestAcc': accuracy,
        'F1Mi': micro,
        'F1Ma': macro,
        'ValAcc': best_score,
        'C': best_c,
    }
