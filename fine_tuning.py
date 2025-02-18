import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os

# TODO:
#  - загрузка датасетов,
#  - задание каждому из них своего item_id
#  - объединение в один TimeSeriesDataFrame
#  - fine-tuning модели Chronos
# 1. Загрузка и подготовка данных для обучения
df_train = pd.read_csv("./large_training_data.csv", skipinitialspace=True)  # Большой корпус данных
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], unit='ms')
df_train['ds'] = df_train['timestamp']
df_train = df_train.set_index('ds')
df_train['item_id'] = 0
train_data = TimeSeriesDataFrame(df_train)

# 2. Обучение и сохранение FineTuned модели
prediction_length = 3  # Длина горизонта прогнозирования
model_path = "./fine_tuned_chronos"  # Путь для сохранения модели
os.makedirs(model_path, exist_ok=True) # Создаем папку, если ее нет

predictor = TimeSeriesPredictor(
    target='close',
    prediction_length=prediction_length,
    freq='D'
).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,
    enable_ensemble=False,
    save_path=model_path,  # Указываем путь для сохранения
)

# 3. Загрузка обученной модели и ее использование для предсказаний
loaded_predictor = TimeSeriesPredictor.load(model_path)  # Загружаем модель

# Загрузка новых данных для предсказаний
df_new = pd.read_csv("./new_data_for_prediction.csv", skipinitialspace=True)
df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
df_new['ds'] = df_new['timestamp']
df_new = df_new.set_index('ds')
df_new['item_id'] = 0
new_data = TimeSeriesDataFrame(df_new)

# Получение предсказаний с помощью загруженной модели
predictions = loaded_predictor.predict(new_data)

# Вывод или сохранение предсказаний
print(predictions)
# predictions.to_csv("./predictions.csv")  # Сохранение в CSV файл