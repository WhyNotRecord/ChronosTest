# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import pandas as pd


# todo Py4J, JPype, run py-script by subprocess, Docker
# todo https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
# Press the green button in the gutter to run the script.

def create_and_train_predictor(train_data, prediction_length, model_path=None, time_limit=60):
    """Создает и обучает TimeSeriesPredictor."""
    return TimeSeriesPredictor(
        target='close',
        prediction_length=prediction_length,
        path=model_path,
        freq='D'
    ).fit(
        train_data=train_data,
        hyperparameters={
            "Chronos": [
                {"model_path": "bolt_base", "ag_args": {"name_suffix": "ZeroShot"}},
                # {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
            ]
        },
        time_limit=time_limit,
        enable_ensemble=False,
    )


def plot_predictions(ax, predictions, train_data, title, max_history_length=50):
    """Отрисовывает предсказания и обучающие данные на графике."""
    index = predictions.index.get_level_values('timestamp')
    ax.plot(index, predictions['mean'], color='darkblue', label='Предсказания')
    ax.plot(index, predictions['0.3'], color='gray')
    ax.plot(index, predictions['0.7'], color='gray')
    plottable_data = train_data[-max_history_length:]
    ax.plot(plottable_data.index.get_level_values('timestamp'), plottable_data['close'], color='blue', label='Обучающие данные')
    ax.set_title(title)
    ax.legend()


if __name__ == '__main__':
    df = pd.read_csv("./ADAUSDT_ccompare_17.12.01-25.01.01_1d_export.txt", skipinitialspace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['ds'] = df['timestamp']
    df.drop(columns=['close_timestamp'], inplace=True)
    df = df.set_index('ds')
    df['item_id'] = 0

    prediction_length = 10
    short_data_length = 250
    med_data_length = 360
    max_history_length = 30
    initial_shift = 15  # Начальный сдвиг
    num_shifts = 5  # Количество сдвигов
    shifts = list(range(initial_shift, initial_shift - num_shifts, -1))  # Формируем список сдвигов

    fig, axes = plt.subplots(len(shifts), 3, figsize=(10, 4 * len(shifts)))  # Создаем сетку графиков
    # Задаем путь к каталогу, где будут храниться модели
    models_dir = "./my_models"  # Или любой другой путь
    os.makedirs(models_dir, exist_ok=True)  # Создаем каталог, если он не существует

    for i, shift in enumerate(shifts):
        # Обработка данных для текущего сдвига
        df_shifted = df.copy()  # Важно создать копию, чтобы не изменить исходный DataFrame
        if shift > 0:
            df_shifted.drop(df_shifted.tail(shift).index, inplace=True)
        data_full = TimeSeriesDataFrame(df_shifted)
        train_data_full, test_data_full = data_full.train_test_split(prediction_length)
        data_short = data_full[-short_data_length - prediction_length:]
        train_data_short, test_data_short = data_short.train_test_split(prediction_length)
        data_medium = data_full[-med_data_length - prediction_length:]
        train_data_medium, test_data_medium = data_medium.train_test_split(prediction_length)

        # Обучение моделей с указанием save_path
        model_path_full = os.path.join(models_dir, f"full_data_shift_{shift}") # Уникальное имя для каждой модели
        predictor_full = create_and_train_predictor(train_data_full, prediction_length, model_path_full)
        model_path_med = os.path.join(models_dir, f"med_data_shift_{shift}") # Уникальное имя для каждой модели
        predictor_med = create_and_train_predictor(train_data_medium, prediction_length, model_path_med)
        model_path_short = os.path.join(models_dir, f"short_data_shift_{shift}") # Уникальное имя для каждой модели
        predictor_short = create_and_train_predictor(train_data_short, prediction_length, model_path_short)

        print(f"Leaderboard for shift {shift}:")
        print(predictor_full.leaderboard(test_data_full))
        print(predictor_med.leaderboard(test_data_medium))
        print(predictor_short.leaderboard(test_data_short))

        # Получение предсказаний
        predictions_full = predictor_full.predict(train_data_full)
        predictions_med = predictor_med.predict(train_data_medium)
        predictions_short = predictor_short.predict(train_data_short)

        # Построение графиков
        plot_predictions(axes[i, 0], predictions_full, test_data_full, f"Предсказания (полные данные, сдвиг {shift})")
        plot_predictions(axes[i, 1], predictions_med, test_data_medium, f"Предсказания (последние {med_data_length}, сдвиг {shift})")
        plot_predictions(axes[i, 2], predictions_short, test_data_short, f"Предсказания (последние {short_data_length}, сдвиг {shift})")

    plt.tight_layout()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
