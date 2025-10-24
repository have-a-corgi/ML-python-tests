import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/dmitry/Projects/python/MLData/diabetes_dataset.csv")
df = df.sample(frac=1)

df_x = df[['age', 'gender']]
df_y = df[['diagnosed_diabetes']]

#Кодирование категориальных признаков
#gender - категоральный признак и должен быть переведен в числовой аналог
map_dict = {'Male' : 0,
            'Female' : 1,
            'Other': 2}
nx = df_x['gender'].map(map_dict)
df_x.loc[:, 'gender'] = nx.loc[:]

df_x = df_x.to_numpy()
df_y = df_y.to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=1, stratify=df_y)

#Нормализуем по тренинговым данным
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(y_train)


