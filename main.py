# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import pandas as pd


# todo Py4J, JPype, run py-script by subprocess, Docker
# todo https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO test with additional parameters (Incorporating the covariates), then on my dataset
    # Загружаем ваши данные из CSV
    df = pd.read_csv("./ADAUSDT_ccompare_17.12.01-25.01.01_1d_export.txt", skipinitialspace=True)

    # Преобразуем столбец 'timestamp' в datetime, если он еще не в этом формате
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # <--- Важно!
    # Создаем копию столбца 'timestamp' (это важно!)
    df['ds'] = df['timestamp']  # ds - стандартное название для AutoGluon
    df.drop(columns=['close_timestamp'], inplace=True)  # Удаляем столбец 'close_timestamp'
    # Устанавливаем 'timestamp' в качестве индекса
    df = df.set_index('ds')  # <--- Ключевой шаг!

    # Добавляем столбец item_id со значением 0 для всех строк
    df['item_id'] = 0
    df.drop(df.tail(40).index, inplace=True)

    # Создаем TimeSeriesDataFrame
    # data = TimeSeriesDataFrame.from_path("./test.csv")
    # "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
    data = TimeSeriesDataFrame(df)

    print(data.head())
    prediction_length = 3
    short_data_length = 250
    max_history_length = 50

    train_data, test_data = data.train_test_split(prediction_length)
    data_short = data[-short_data_length - prediction_length:]
    train_data_short, test_data_short = data_short.train_test_split(prediction_length)

    predictor = TimeSeriesPredictor(target='close', prediction_length=prediction_length, freq='D').fit(
        train_data=train_data,
        hyperparameters={
            "Chronos": [
                {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
                # {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
            ]
        },
        # features=['open', 'volume'],
        # hyperparameters={"Chronos": {"fine_tune": True, "fine_tune_lr": 1e-4, "fine_tune_steps": 2000}}
        time_limit=60,  # time limit in seconds (for fine-tuning?)
        enable_ensemble=False,
    )
    predictor_short = TimeSeriesPredictor(target='close', prediction_length=prediction_length, freq='D').fit(
        train_data=train_data_short,
        hyperparameters={
            "Chronos": [
                {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
                # {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
            ]
        },
        # features=['open', 'volume'],
        # hyperparameters={"Chronos": {"fine_tune": True, "fine_tune_lr": 1e-4, "fine_tune_steps": 2000}}
        time_limit=60,  # time limit in seconds (for fine-tuning?)
        enable_ensemble=False,
    )

    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 300)

    print(predictor.leaderboard(test_data))
    print(predictor_short.leaderboard(test_data))

    print("Starting inference")

    # predictions = predictor.predict(train_data, model="ChronosFineTuned[bolt_small]")
    predictions = predictor.predict(train_data)
    predictions_short = predictor_short.predict(train_data_short)
    feature_importance = predictor.feature_importance()
    print(feature_importance)

    # Создаем figure и axes для графиков
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 2 графика в одном столбце

    # Строим графики с помощью matplotlib.pyplot
    index_full = predictions.index.get_level_values('timestamp')
    axes[0].plot(index_full, predictions['mean'], color='darkblue', label='Предсказания (по всем отсчётам)')
    axes[0].plot(index_full, predictions['0.3'], color='gray')
    axes[0].plot(index_full, predictions['0.7'], color='gray')
    plottable_data = test_data[-max_history_length:]
    axes[0].plot(plottable_data.index.get_level_values('timestamp'), plottable_data['close'],
                 color='blue', label='Обучающие данные')  # Пример: добавляем обучающие данные
    axes[0].set_title("Предсказания на полной последовательности")
    axes[0].legend()  # Добавляем легенду

    index_short = predictions_short.index.get_level_values('timestamp')
    axes[1].plot(index_short, predictions_short['mean'], color='darkblue', label='Предсказания (по последним 250)')
    axes[1].plot(index_short, predictions_short['0.3'], color='gray')
    axes[1].plot(index_short, predictions_short['0.7'], color='gray')
    plottable_data_short = test_data_short[-max_history_length:]
    axes[1].plot(plottable_data_short.index.get_level_values('timestamp'), plottable_data_short['close'],
                 color='blue', label='Обучающие данные')  # Пример: добавляем обучающие данные
    axes[1].set_title("Предсказания на последних 250 отсчетах")
    axes[1].legend()  # Добавляем легенду

    plt.tight_layout()  # Автоматическое размещение графиков
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
