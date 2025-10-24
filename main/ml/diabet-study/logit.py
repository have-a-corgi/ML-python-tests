import prepareset as ps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ppn = LogisticRegression()
ppn.fit(ps.X_train_std, ps.y_train)
y_pred = ppn.predict(ps.X_test_std)


print('Misclassified examples: %d' % (ps.y_test != y_pred).sum())
print('Accuracy: %.3f' % accuracy_score(ps.y_test, y_pred))
print('Accuracy: %.3f' % ppn.score(ps.X_test_std, ps.y_test))