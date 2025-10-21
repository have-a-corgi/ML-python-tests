import loadiris as irs
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Расшепляем 30-70
X_train, X_test, y_train, y_test = train_test_split(irs.X, irs.y, test_size=0.3, random_state=1, stratify=irs.y)

#Нормализуем по тренинговым данным
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = LogisticRegression()
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

print('Misclassified examples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))