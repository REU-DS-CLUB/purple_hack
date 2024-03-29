# purple_hack - Сбер

Прогнозирование оттока зарплатного клиента ФЛ

## Описание кейса

🟣 Проблематика

Банк тратит определенную сумму на привлечение зарплатного клиента (клиента, который официально получает заработную плату на карту банка.).
Зарплатный клиент приносит банку больше прибыли, чем обычный клиент. Когда зарплатный клиент уводит свою зарплату из банка в другой банк, то банк теряет прибыль, такие события называются оттоком. 
Проблема заключается в том, что банк узнает об оттоке по факту отсутствия заработной платы.


🟣 Желаемый результат

Модель машинного обучения, которая предсказывает на ежедневной основе отток зарплатных клиентов из банка до возникновения самого события оттока, используя данные поведения клиента: транзакции, продукты, мобильное приложение, терминалы, прочее.

## Структура проекта

- файл EDA.ipynb содержит исследовательский анализ данных нашей команды, предшествующий обучению предиктивных моделей


## Данные

Открытый датасет представлен в виде файла с расширением .parquet, содержащим более 500 тыс. строк - уникальных клиентов банка и 1070 колонок - их замаскированных признаков

## Команда REU DS CLUB

| Участник                       | bio       | Контакты                        |
|--------------------------------|-----------|---------------------------------|
| Пашинская Пелагея              | Team lead | https://t.me/polyanka003        |
| Морозова Мария                 | DA        | https://t.me/kheydelberg        |
| Иванов Александр               | MLE       | https://t.me/lild1tz            |
| Мичурин Артем                  | DE        | https://t.me/amichurin_rubbles  |
