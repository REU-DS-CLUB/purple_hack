def feature_drop(data):
    
    return data.copy().loc[:, data.nunique() != 1].drop(columns=["feature642", "feature756"], axis = 1) 


def get_categorical_columns():

    pass


def drop_ununique_features(data):
    # создает датафрейм только с колонками, содержащими больше одного уникального значения
    return data.copy().loc[:, data.nunique() != 1]


def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()

    # инициализируем множество для хранения индексов признаков, которые нужно удалить
    features_to_remove = set()

    # проходимся по всем элементам матрицы корреляции
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            # если корреляция между двумя столбцами выше заданного порога
            if corr_matrix.iloc[i, j] > threshold:
                # определяем, какой из двух признаков удалить
                colname = corr_matrix.columns[j]
                features_to_remove.add(colname)

    # удаляем признаки
    df_reduced = df.drop(columns=features_to_remove)

    return df_reduced


def download_raw_data_from_drive_and_open_in_pandas(file_id="1cS6pE2ZD127iSVEiLRDzGd_b65J7GTd9",
                                                    file_path="Data/train_ai_comp_final_dp.parquet"):
    import pandas as pd
    import gdown
    # скачивает файл
    gdown.download(id=file_id, output=file_path)

    return pd.read_parquet(file_path, index='id')
