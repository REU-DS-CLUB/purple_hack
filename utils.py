

def feature_drop(data):
    return data.copy().loc[:, data.nunique() != 1].drop(columns="feature756")



def get_categorical_columns():
    pass



"""
def remove_highly_correlated_features(df, threshold=0.9):

    data = df.copy()
    data = data.sample(n = 100000)
    corr_matrix = data.corr().abs()

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
    #df_reduced = df.drop(columns=features_to_remove)

    return features_to_remove
"""


import pandas as pd

def remove_highly_correlated_features(df, threshold=0.9, max_samples=100000):
    # Создание копии DataFrame для предотвращения изменений в оригинале
    df_copy = df.copy()

    # Если размер DataFrame больше максимального количества образцов, используем случайную выборку
    if len(df_copy) > max_samples:
        df_copy = df_copy.sample(n=max_samples, random_state=1)

    # Вычисление матрицы корреляции
    corr_matrix = df_copy.corr().abs()

    # Идентификация колонок, которые не должны удаляться
    protected_columns = ["target", 'id']

    # Список колонок для удаления
    columns_to_remove = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col_name_i = corr_matrix.columns[i]
            col_name_j = corr_matrix.columns[j]

            # Проверка, не являются ли обе колонки защищенными
            if col_name_i in protected_columns or col_name_j in protected_columns:
                continue

            # Если корреляция между колонками выше порога
            if corr_matrix.iloc[i, j] >= threshold:
                # Сравнение корреляции колонок с целевой переменной
                correlation_with_target_i = corr_matrix.loc[col_name_i, "target"]
                correlation_with_target_j = corr_matrix.loc[col_name_j, "target"]

                # Удаление колонки с меньшей корреляцией с целевой переменной
                if correlation_with_target_i < correlation_with_target_j:
                    if col_name_i not in columns_to_remove and col_name_i not in protected_columns:
                        columns_to_remove.append(col_name_i)
                else:
                    if col_name_j not in columns_to_remove and col_name_j not in protected_columns:
                        columns_to_remove.append(col_name_j)

    return columns_to_remove

    
    # Удаляем высококоррелированные признаки, выбранные к удалению
    
    return features_to_remove





def download_raw_data_from_drive_and_open_in_pandas(file_id="1cS6pE2ZD127iSVEiLRDzGd_b65J7GTd9",
                                                    file_path="raw_data.parquet"):
    import pandas as pd
    import gdown
    # скачивает файл
    gdown.download(id=file_id, output=file_path)

    return pd.read_parquet(file_path)