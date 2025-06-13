import streamlit as st
from api_client import set_active_model, predict_one_item, predict_csv, get_models

st.title("Прогноз доли пользователей, увидевших рекламное объявление 1,2,3 раза")

models = get_models()
model_names = list(models['models'].keys())

selected_model = st.selectbox("Выберите модель", model_names)

if st.button("Установить модель"):
    response = set_active_model(selected_model)
    st.success(f"Модель установлена: {response['active_model']}")

st.header("Предсказание для рекламного объявления")
cpm = st.number_input("CPM", min_value=0.0, value=250.0)
hour_start = st.number_input("hour_start", min_value=0, value=100)
hour_end = st.number_input("hour_end", min_value=0, value=345)
publishers = st.text_input("publishers", "1,2,3")
audience_size = st.number_input("audience_size", min_value=0, value=3)
user_ids = st.text_input("user_ids", "1,2,3")

if st.button("Предсказать"):
    data = {
        "cpm": cpm,
        "hour_start": hour_start,
        "hour_end": hour_end,
        "publishers": publishers,
        "audience_size": audience_size,
        "user_ids": user_ids
    }
    prediction = predict_one_item(data)
    st.success(f"Предсказания: {prediction}")

st.header("Предсказание из CSV")
uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if st.button("Предсказать для датасета"):
    if uploaded_file is not None:
        response = predict_csv(uploaded_file)
        if response.status_code == 200:
            st.success("Предсказания готовы!")
            st.download_button("Скачать предсказания", response.content, "predictions.csv")
        else:
            st.error("Ошибка при получении предсказаний.")