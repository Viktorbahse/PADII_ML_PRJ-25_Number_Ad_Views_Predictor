import streamlit as st
from api_client import set_active_model, predict_one_item, predict_csv, get_models
import logging
from logging.handlers import RotatingFileHandler
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from schemas import *
import re

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "app.log")
handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

st.title("Выберите модель")

models = get_models()
model_names = list(models['models'].keys())

selected_model = st.selectbox("Модель", model_names)

if st.button("Установить модель"):
    response = set_active_model(selected_model)
    st.success(f"Модель установлена: {response['active_model']}")
    logger.info(f"Модель установлена: {response['active_model']}")

st.title("Прогноз доли пользователей, увидевших рекламное объявление 1, 2, 3 раза")

def validate_input(input_string):
    pattern = r'^\d+(,\d+)*$'
    return re.match(pattern, input_string) is not None

cpm = st.number_input("Введите CPM", min_value=0.0, value=50.0, step=1.0)
hour_start, hour_end = st.slider("Выберите диапазон времени", 0, 2500, (500, 1200), 1)
publishers = st.text_input("publishers", "1,2,3")
if not validate_input(publishers):
    st.error("Ошибка: Введите числа, разделённые запятыми.")
user_ids = st.text_input("user_ids", "1,2,3")
if not validate_input(user_ids):
    st.error("Ошибка: Введите числа, разделённые запятыми.")
audience_size = len([uid for uid in user_ids.split(',') if uid.strip()])

if st.button("Предсказать"):
    data = {
        "cpm": cpm,
        "hour_start": hour_start,
        "hour_end": hour_end,
        "publishers": publishers,
        "audience_size": audience_size,
        "user_ids": user_ids
    }
    if not validate_input(user_ids) or not validate_input(publishers):
        st.error("Ошибка: Введите числа, разделённые запятыми.")
    else:
        prediction = predict_one_item(data)
        st.success(f"Предсказания: {prediction}")
        logger.info(f"Предсказания: {prediction}")

st.header("Предсказание из CSV")
uploaded_file = st.file_uploader("Выберите CSV файл")

if st.button("Предсказать для датасета"):
    if uploaded_file is not None:
        response = predict_csv(uploaded_file)
        if response.status_code == 200:
            st.success("Предсказания готовы!")
            logger.info(f"Предсказания готовы!")
            st.download_button("Скачать предсказания", response.content, "predictions.csv")
        else:
            st.error("Ошибка при получении предсказаний.")
            logger.error(f"Ошибка при получении предсказаний")


def load_and_validate_users(file) -> pd.DataFrame:
    logger.info(f"Загрузка данных из файла в датафрейм")
    df = pd.read_csv(file, sep='\t')
    errors = []
    valid_rows = []

    for i, row in df.iterrows():
        try:
            User(**row.to_dict())
            valid_rows.append(row)
        except Exception as e:
            errors.append(f"Строка {i}: {str(e)}")
            logger.error(f"Ошибка при загрузке строки в датафрейм")

    if errors:
        st.warning(f"Найдены ошибки в данных users: {errors[:5]}... (всего {len(errors)})")
        logger.warning(f"Найдены ошибки в данных при загрузке")

    return pd.DataFrame(valid_rows)


def load_and_validate_history(file) -> pd.DataFrame:
    logger.info(f"Загрузка данных из файла в датафрейм")
    df = pd.read_csv(file, sep='\t')
    errors = []
    valid_rows = []

    for i, row in df.iterrows():
        try:
            HistoryItem(**row.to_dict())
            valid_rows.append(row)
        except Exception as e:
            errors.append(f"Строка {i}: {str(e)}")
            logger.error(f"Ошибка при загрузке строки в датафрейм")

    if errors:
        st.warning(f"Найдены ошибки в данных history: {errors[:5]}... (всего {len(errors)})")
        logger.warning(f"Найдены ошибки в данных при загрузке")

    return pd.DataFrame(valid_rows)


def load_and_validate_validate(file) -> pd.DataFrame:
    logger.info(f"Загрузка данных из файла в датафрейм")
    df = pd.read_csv(file, sep='\t')
    errors = []
    valid_rows = []

    for i, row in df.iterrows():
        try:
            AddInput(**row.to_dict())
            valid_rows.append(row)
        except Exception as e:
            errors.append(f"Строка {i}: {str(e)}")
            logger.error(f"Ошибка при загрузке строки в датафрейм")

    if errors:
        st.warning(f"Найдены ошибки в данных validate: {errors[:5]}... (всего {len(errors)})")
        logger.warning(f"Найдены ошибки в данных при загрузке")

    return pd.DataFrame(valid_rows)


def analyze_user(user_id, users_df, history_df):
    user_data = users_df[users_df['user_id'] == user_id]
    if user_data.empty:
        return None

    history = history_df[history_df['user_id'] == user_id]

    return {
        'user_info': user_data.iloc[0].to_dict(),
        'ads_count': len(history),
        'publishers_distribution': history['publisher'].value_counts().to_dict(),
        'hourly_distribution': history['hour'].value_counts().sort_index().to_dict(),
        'cpm_distribution': {
            'mean': history['cpm'].mean(),
            'median': history['cpm'].median(),
            'min': history['cpm'].min(),
            'max': history['cpm'].max()
        }
    }


st.header("Загрузка файлов для EDA")
users_file = st.file_uploader("Загрузить файл users.tsv", type=["tsv"])
history_file = st.file_uploader("Загрузить файл history.tsv", type=["tsv"])
validate_file = st.file_uploader("Загрузить файл validate.tsv", type=["tsv"])

if 'users_df' not in st.session_state:
    st.session_state.users_df = pd.DataFrame()
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame()
if 'validate_df' not in st.session_state:
    st.session_state.validate_df = pd.DataFrame()

if st.button("Загрузить и проанализировать данные"):
    logger.info(f"Загрузка EDA")
    if users_file is not None:
        st.session_state.users_df = load_and_validate_users(users_file)
        st.success(f"Загружено {len(st.session_state.users_df)} пользователей")
        logger.info(f"Загружено {len(st.session_state.users_df)} пользователей")

    if history_file is not None:
        st.session_state.history_df = load_and_validate_history(history_file)
        st.success(f"Загружено {len(st.session_state.history_df)} записей истории")
        logger.info(f"Загружено {len(st.session_state.users_df)} записей истории")

    if validate_file is not None:
        st.session_state.validate_df = load_and_validate_validate(validate_file)
        st.success(f"Загружено {len(st.session_state.validate_df)} записей валидации")
        logger.info(f"Загружено {len(st.session_state.users_df)} записей валидации")

    if not st.session_state.users_df.empty:
        st.subheader("Анализ данных пользователей")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Распределение по полу")
            logger.info(f"Выведено распределение по полу")
            sex_counts = st.session_state.users_df['sex'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)
        with col2:
            st.write("Распределение по возрасту")
            logger.info(f"Выведено распределение по возрасту")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.users_df['age'], bins=20, kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("Анализ зависимостей (users)")
        numeric_cols = st.session_state.users_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            st.write("Матрица корреляций:")
            logger.info(f"Выведена матрица корреляций")
            corr_matrix = st.session_state.users_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.write("Парные зависимости:")
            logger.info(f"Выведены парные зависимости")
            fig = sns.pairplot(st.session_state.users_df[numeric_cols])
            st.pyplot(fig)

        st.write("Статистики по возрасту:")
        st.write(st.session_state.users_df['age'].describe())
        st.write("Топ 10 городов:")
        st.write(st.session_state.users_df['city_id'].value_counts().head(10))

    if not st.session_state.history_df.empty:
        st.subheader("Анализ истории показов")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Распределение по часам")
            logger.info(f"Выведено распределение по часам")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.history_df['hour'], bins=24, ax=ax)
            st.pyplot(fig)
        with col2:
            st.write("Распределение CPM")
            logger.info(f"Выведено распределение по CPM")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.history_df['cpm'], ax=ax, bins=30)
            st.pyplot(fig)

        st.subheader("Анализ зависимостей (history)")
        numeric_cols = st.session_state.history_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            st.write("Матрица корреляций:")
            logger.info(f"Выведена матрица корреляций")
            corr_matrix = st.session_state.history_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.write("Парные зависимости:")
            logger.info(f"Выведены парные зависимости")
            fig = sns.pairplot(st.session_state.history_df[numeric_cols])
            st.pyplot(fig)

        st.write("Топ 10 платформ:")
        st.write(st.session_state.history_df['publisher'].value_counts().head(10))

    if not st.session_state.validate_df.empty:
        st.subheader("Анализ валидационных данных")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Распределение hour_start")
            logger.info(f"Выведено распределение по hour_start")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.validate_df['hour_start'], bins=24, ax=ax)
            st.pyplot(fig)
        with col2:
            st.write("Распределение hour_end")
            logger.info(f"Выведено распределение по hour_end")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.validate_df['hour_end'], bins=24, ax=ax)
            st.pyplot(fig)

        st.subheader("Анализ зависимостей (validate)")
        numeric_cols = st.session_state.validate_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            st.write("Матрица корреляций:")
            logger.info(f"Выведена матрица корреляций")
            corr_matrix = st.session_state.validate_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.write("Парные зависимости:")
            logger.info(f"Выведены парные зависимости")
            fig = sns.pairplot(st.session_state.validate_df[numeric_cols])
            st.pyplot(fig)

        st.write("Распределение CPM:")
        logger.info(f"Выведено распределение по CPM")
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.validate_df['cpm'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)
        st.write("Распределение audience_size:")
        logger.info(f"Выведено распределение по audience_size")
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.validate_df['audience_size'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

if not st.session_state.users_df.empty and not st.session_state.history_df.empty:
    st.subheader("Анализ конкретного пользователя")
    user_id_to_analyze = st.number_input("Введите ID пользователя для анализа", min_value=0)
    if st.button("Проанализировать пользователя"):
        analysis = analyze_user(user_id_to_analyze, st.session_state.users_df, st.session_state.history_df)
        if analysis:
            st.write("Информация о пользователе:")
            logger.info(f"Выведена информация о пользователе {user_id_to_analyze}")
            st.write(analysis['user_info'])
            st.write(f"Всего показов: {analysis['ads_count']}")
            st.write("Распределение по платформам:")
            fig, ax = plt.subplots()
            sns.barplot(x=list(analysis['publishers_distribution'].keys()),
                        y=list(analysis['publishers_distribution'].values()), ax=ax)
            st.pyplot(fig)
            st.write("Распределение по часам:")
            fig, ax = plt.subplots()
            x_values = [hour % 24 for hour in analysis['hourly_distribution'].keys()]
            y_values = list(analysis['hourly_distribution'].values())
            sns.lineplot(x=x_values, y=y_values, ax=ax)
            st.pyplot(fig)
            st.write("Статистики по CPM:")
            st.write(analysis['cpm_distribution'])
        else:
            st.warning("Пользователь не найден")

def binary_search(hour_start, hour_end, publishers, audience_size, user_ids, target):
    left = 0
    right = 10000000
    ans = -1
    while left < right:
        mid = (left + right) // 2  
        data = {
            "cpm": mid,
            "hour_start": hour_start,
            "hour_end": hour_end,
            "publishers": publishers,
            "audience_size": audience_size,
            "user_ids": user_ids
        }

        prediction = predict_one_item(data)
        st.success(f"Предсказания: {prediction}")
        logger.info(f"Предсказания: {prediction}")

        auditorium = audience_size*(prediction['at_least_one']+prediction['at_least_two']+prediction['at_least_three'])
        if target > auditorium:
            left = mid + 1
        else:
            right = mid
    return right    

def budget_calculator():
    st.title("Калькулятор бюджета")
    
    st.header("Желаемое количество просмотров")
    target_views = st.number_input(
        "Введите количество просмотров:", 
        min_value=0, 
        max_value=10000000, 
        value=10_000,
        step=10
    )

    st.header("Период показа объявления")
    target_duration = st.number_input("Продолжительность", min_value=0, max_value=300, value = 50, step=1)
    
    st.header("Выбор площадок")
    select_all = st.checkbox("Выбрать все площадки", key="select_all")
    cols = st.columns(3)
    selected_platforms = []
    for i in range(1, 22):
        with cols[(i-1)//7]:
            if select_all:
                checked = st.checkbox(f"Площадка {i}", value=True, key=f"platform_{i}")
            else:
                checked = st.checkbox(f"Площадка {i}", key=f"platform_{i}")
            
            if checked:
                selected_platforms.append(i)
    
    st.header("Аудитория")
    uploaded_file = st.file_uploader(
        "Загрузите CSV/TSV файл с user_id (обязательно)", 
        type=["csv", "tsv", "txt"],  
        help="Файл должен содержать столбец 'user_id'. Разделитель: запятая (CSV) или табуляция (TSV)"
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_users = pd.read_csv(uploaded_file, sep='\t')
            else:  
                df_users = pd.read_csv(uploaded_file, sep='\t')
            
            if "user_id" not in df_users.columns:
                print(df_users.columns)
                st.error("❌ В файле нет столбца 'user_id'!")
            else:
                st.success(f"✅ Файл загружен. Найдено {len(df_users)} пользователей")
                st.dataframe(df_users.head(3))  
            
        except Exception as e:
            st.error(f"❌ Ошибка чтения файла: {str(e)}")
    
    if st.button("Рассчитать бюджет", type="primary"):
        if not selected_platforms:
            st.warning("Выберите хотя бы одну площадку!")
        if not uploaded_file:
            st.warning("ℹ️ Загрузите CSV файл, чтобы начать расчет")  
        else:
            user_ids_str = ",".join(df_users["user_id"].astype(str))
            publishers_str = ",".join(map(str, selected_platforms))

            cost_per_view = binary_search(1, target_duration, publishers_str, df_users.shape[0], publishers_str, target_views)
            total_cost = (target_views / 1000) * cost_per_view
            if cost_per_view == 10000000:
                st.success(f"""
                ### Результаты расчета:
                - **Желаемое количество просмотров:** {target_views:,}
                - **Выбранные площадки:** {sorted(selected_platforms)}
                - **Расчетный бюджет:** более ${total_cost:,.2f} USD
                - **CPM:** более ${cost_per_view} USD
                """)
            else:
                st.success(f"""
                ### Результаты расчета:
                - **Желаемое количество просмотров:** {target_views:,}
                - **Выбранные площадки:** {sorted(selected_platforms)}
                - **Расчетный бюджет:** ${total_cost:,.2f} USD
                - **CPM:** ${cost_per_view} USD
                """)

budget_calculator()
