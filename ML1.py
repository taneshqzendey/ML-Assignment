import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.drop(columns=['ID', 'Constituency'], inplace=True)
test_data.drop(columns=['ID', 'Constituency'], inplace=True)

def convert_to_numeric(x):
    try:
        if isinstance(x, str):
            x = x.replace(' Crore+', 'e7').replace(' Lac+', 'e5').replace(' Thou+', 'e3').replace(' Nil', '0')
            return eval(x)
        else:
            return x
    except:
        return np.nan
def nameinterpret(x):
    if x.find("Dr.")!=-1:
        return 1
    else:
        return 0
    
train_data['Total Assets'] = train_data['Total Assets'].apply(convert_to_numeric)
train_data['Liabilities'] = train_data['Liabilities'].apply(convert_to_numeric)
train_data['Candidate'] = train_data['Candidate'].apply(nameinterpret)
test_data['Total Assets'] = test_data['Total Assets'].apply(convert_to_numeric)
test_data['Liabilities'] = test_data['Liabilities'].apply(convert_to_numeric)
test_data['Candidate'] = test_data['Candidate'].apply(nameinterpret)

X = train_data.drop(columns=['Education'])
y = train_data['Education']

encoder = LabelEncoder()
X['Party'] = encoder.fit_transform(X['Party'])
test_data['Party'] = encoder.transform(test_data['Party'])
X['state'] = encoder.fit_transform(X['state'])
test_data['state'] = encoder.transform(test_data['state'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=(42))
numeric_features = ['Criminal Case','Candidate']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

param_grid = {
  'n_estimators': [50,100,150],
  'min_samples_split': [10,15]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', grid_search)])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

test_predictions = model.predict(test_data)

predict_df = pd.DataFrame({'ID': range(0, len(test_predictions)), 'Education': test_predictions})
predict_df.to_csv('predict.csv', index=False)
