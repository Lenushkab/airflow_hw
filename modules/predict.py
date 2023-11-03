import os
from datetime import datetime

import dill
import pandas as pd
import json

path = os.environ.get('PROJECT_PATH', '..')

def predict():
    #сортировка моделей по дате
    sorted_models = sorted(os.listdir(f'{path}/data/models'))

    with open(f'{path}/data/models/{sorted_models[-1]}', 'rb') as file:   #самая новая по времени модель
        model = dill.load(file)

    preds = pd.DataFrame(columns=['car_id', 'pred'])

    test_cars = os.listdir(f'{path}/data/test')     #автомобили для предсказания
    for item in test_cars:
        with open(f'{path}/data/test/{item}', 'rb') as file:
            car = json.load(file)

        df = pd.DataFrame(car, index = [0])
        y = model.predict(df)
        x = {'car_id':df.id, 'pred':y}
        df1 = pd.DataFrame(x)
        preds = pd.concat([preds, df1], axis = 0)
    print(preds)

    preds.to_csv(path_or_buf=f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)



if __name__ == '__main__':
    predict()
