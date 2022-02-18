from random import random
from unittest import result
import pandas as pd
import plotly.express as px

df = pd.read_csv("Admission_Predict.csv")
TOEFL_Score = df["TOEFL Score"].tolist()
GRE_Score = df["GRE Score"].tolist()

fig = px.scatter(x=TOEFL_Score, y=GRE_Score)
fig.show()

import plotly.graph_objects as go

TOEFL_Score = df["TOEFL Score"].tolist()
GRE_Score = df["GRE Score"].tolist()

Result = df["Chance of admit"]
colors=[]
for data in Result:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")

fig = go.Figure(data=go.Scatter(
    x=TOEFL_Score,
    y=GRE_Score,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()

score = df["GRE Score","TOEFL Score"]

#results
results = df["Chance of admit"]


from sklearn.model_selection import train_test_split 

score_test, score_train, results_test, results_train = train_test_split(score, results, test_size = 0.25, random_state = 0)
print(score_train)

classifier = LogisticRegression(random_state = 0)
classifier.fit(score_train, results_train)

results_pred = classifier.predict(score_test)

from sklearn.metric import accuracy_score
print ("Accuracy :", accuracy_score(results_test,results_pred))