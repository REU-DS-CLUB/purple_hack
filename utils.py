def faeture_drop(data):
    return data.copy().loc[:, data.nunique() != 1]

def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()

    # Инициализируем множество для хранения индексов признаков, которые нужно удалить
    features_to_remove = set()

    # Проходимся по всем элементам матрицы корреляции
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            # Если корреляция между двумя столбцами выше заданного порога
            if corr_matrix.iloc[i, j] > threshold:
                # Определяем, какой из двух признаков удалить
                colname = corr_matrix.columns[j]
                features_to_remove.add(colname)

    # Удаляем признаки
    df_reduced = df.drop(columns=features_to_remove)

    return df_reduced
