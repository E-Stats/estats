# dca_demo_highdim.py
import matplotlib; matplotlib.use("Agg")          # <-- fix the Qt issue
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets       import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.linear_model    import LogisticRegression

from energystats.decomposition import dca as dca_mod  # your implementation

# ---------------------- 1. load the 4 096-dim data --------------------------
faces     = fetch_olivetti_faces()          # only ~1 MB download, once
X, y      = faces.data, faces.target        # X 400×4096, y in {0,…,39}

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25,
                                      stratify=y, random_state=1)
std = StandardScaler().fit(Xtr)
Xtr, Xte = std.transform(Xtr), std.transform(Xte)

# ---------------------- 2. helper for the ridge/LogReg score ---------------
def logreg_score(Ztr, Zte):
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial',
                             solver='lbfgs', C=10).fit(Ztr, ytr)
    return clf.score(Zte, yte), clf.predict_proba(Zte).argmax(1)

raw_acc, _ = logreg_score(Xtr, Xte)

# ---------------------- 3. PCA baseline ------------------------------------
k = 5
pca      = PCA(k, random_state=0).fit(Xtr)
pca_acc, _ = logreg_score(pca.transform(Xtr), pca.transform(Xte))

# ---------------------- 4. DCA (your code) ---------------------------------
Ztr_dca = dca_mod.dca(Xtr, ytr, k=k)['projections']
Zte_dca = dca_mod.dca(Xte, yte, k=k)['projections']  # project held-out set
dca_acc, yhat = logreg_score(Ztr_dca, Zte_dca)

# ---------------------- 5. plots (saved, not shown) -------------------------
plt.figure(figsize=(5,3.5))
labels, scores = ['Raw 4096', f'PCA {k}', f'DCA {k}'], [raw_acc, pca_acc, dca_acc]
bars = plt.bar(labels, scores, color="#FDBE02")
for rect, s in zip(bars, scores):
    plt.text(rect.get_x()+rect.get_width()/2, s+0.01, f"{s:.3f}",
             ha="center", va="bottom")
plt.ylim(0,1); plt.ylabel("Accuracy (test)")
plt.title("Olivetti faces – multinomial LogReg after reduction")
plt.tight_layout(); plt.savefig("bar.png", dpi=140)

plt.figure(figsize=(4,4))
plt.scatter(yte, yhat, s=14, alpha=.6)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'k--', lw=1)
plt.xlabel("True label"); plt.ylabel("Predicted label")
plt.title(f"DCA({k}) + LogReg  acc={dca_acc:.3f}")
plt.tight_layout(); plt.savefig("scatter.png", dpi=140)
