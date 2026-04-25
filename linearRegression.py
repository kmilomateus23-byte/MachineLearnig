import pandas as pd
from sklearn.linear_model import LinearRegression


def trainModel():
    data = {
        "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
        "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
    }

    df = pd.DataFrame(data)

    X = df[["Study Hours"]]
    y = df[["Final Grade"]]

    model = LinearRegression()
    model.fit(X, y)

    return model



model = trainModel()



def calculateGrade(hours):
    
    df = pd.DataFrame([[hours]], columns=["Study Hours"])

  
    result = model.predict(df)[0][0]

    return round(float(result), 2)