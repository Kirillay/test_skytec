import dask.dataframe as dd                 #Импортирую нужные библиотеки                
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Определяю количество записей
n_records = 100_000_000

# Создаю данные
numeric_data = np.random.rand(n_records)
start_date = datetime(2010, 1, 1)
date_range = [start_date + timedelta(days=i)
    for i in np.random.randint(0, 365*5, n_records)]
string_data = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_records)

# Создаю DataFrame с использованием Dask
df = dd.from_pandas(pd.DataFrame({
    'numeric': numeric_data,
    'date': date_range,
    'string': string_data
}), npartitions=10) # Указал количество партиций

# Сохранаю DataFrame в файл в формате CSV
df.to_csv('test_dataset_skytec.csv', single_file=True, index=False)

# Считывание и процессинг

# Загружаю CSV файл
df = dd.read_csv('test_dataset_skytec.csv')

# Удаляю пустые строки
df = df.dropna()

# Удаляю дубликаты
df = df.drop_duplicates()

# Использую функцию для преобразования строк, которые не содержат цифр
def replace_non_numeric(s):
    return s if re.search(r'\d', s) else np.nan

# Применяю функцию к колонке 'string'
df['string'] = df['string'].map(replace_non_numeric, meta=('x', 'object'))

# Удаляю записи в промежутке от 1 до 3 часов ночи
df['date'] = dd.to_datetime(df['date'])

# Фильтрую записи
df = df[~((df['date'].dt.hour >= 1) & (df['date'].dt.hour < 3))]

# Чтобы увидеть результат, нужно запустить вычисления
result = df.compute()

# Чтобы сохранить очищенный файл DataFrame в новый CSV, используем эту функцию
result.to_csv('cleaned_test_dataset_skytec.csv', index=False)

# !!!Расчёт метриков!!!

# Агрегация по времени с использованием функции groupby
aggregated_df = df.groupby(df['date'].dt.floor('H')).agg(
    unique_strings = ('string', 'nunique'),  # Количество уникальных значений в колонке 'string'
    mean_numeric = ('numeric', 'mean'),   # Среднее значение в колонке 'numeric'
    median_numeric = ('numeric', 'median')  # Медиана в колонке 'numeric'
). reset_index()        # Добавим функцию, чтобы привести DataFrame к стандартному формату в виде колонки, а не индекса
# Merge с метриками. Использование функции
merged_df = pd.merge_asof(
    df,
    aggregated_df,
    on = 'date',    # Указываем объединение по колонке 'data'
    direction = 'backward'  # Используем backward, чтобы брать ближающую метрику не позже текущего времени
)

# Сохраним результат 
merged_df.to_csv('merged_test_dataset_skytec.csv', index=False)

# SQL-запрос для агрегации по времени
"""
SELECT 
    DATE_TRUNC('hour', date) AS hour,           # Округляет временную метку до ближайшего часа
    COUNT(DISTINCT string) AS unique_strings,   # Подсчитывает количество уникальных значений
    AVG(numeric) AS mean_numeric,               # Вычисляет среднее значение
    MEDIAN(numeric) AS median_numeric           # Вычисляет медиану 
FROM
    название_таблицы
WHERE
    date NOT BETWEEN 'YYYY-MM-DD 01:00:00' AND 'YYYY-MM-DD 03:00:00'  # Фильтрует записи, чтобы исключить те, которые находятся в промежутке от 1 до 3 часов ночи
GROUP BY
    hour
ORDER BY
    hour;  
"""

# Извлекаем колонку 'numeric'
numeric_data = merged_df['numeric']

# Выбрал метод BootStrap для расчёта. Считаю его самым оптимальным
n_iterations = 10000 # Количество итерации для выборок
bootstrapped_means = []

for _ in range(n_iterations):
    # Рэсэмплирование с возвращением
    sample = np.random.choice(numeric_data, size=len(numeric_data), replace=True)
    bootstrapped_means.append(np.mean(sample))

# Вычисляем 95% интервал
lower_bound = np.percentile(bootstrapped_means, 2.5)
upper_bound = np.percentile(bootstrapped_means, 97.5)

# Построение гистограммы
plt.figure(figsize=(12, 6))
sns.histplot(numeric_data, bins=30, kde=True, stat='плотность', color='skyblue', label='Числовое значение')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower 95% CI')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper 95% CI')
plt.title('Гистограмма Для Числовых Данных с 95% доверительным интервалом')
plt.xlabel('Числовое значение')
plt.ylabel('Плотность')
plt.legend()
plt.show()