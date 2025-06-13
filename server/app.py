from classes import *
from schemas import *
from fastapi import FastAPI, HTTPException, UploadFile, Path, Body, File
from fastapi.responses import StreamingResponse
import pandas as pd
from contextlib import asynccontextmanager
from joblib import load
from joblib import dump
import multiprocessing
import logging
from logging.handlers import RotatingFileHandler
import os
from io import StringIO

MODELS_DIR =  "models"
AVAILABLE_MODELS = {}

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "app.log")

    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

@asynccontextmanager
async def lifespan(app: FastAPI):

    import sys
    sys.modules['__main__'].DummyModelFirst = DummyModelFirst
    sys.modules['__main__'].DummyModelSecond = DummyModelSecond

    app.state.models = {}
    app.state.active_model = None

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    for dir_name, sub_dirs, file_names in os.walk(MODELS_DIR):
        for file_name in file_names:
            model_path = os.path.join(dir_name, file_name)
            if os.path.exists(model_path):
                model_name = file_name.replace(".joblib", "")
                app.state.models[model_name] = load(model_path)
                
    yield
    app.state.models = None
    app.state.active_model = None

setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI(lifespan=lifespan)

@app.post(
    "/set/{model_name}",
    response_model=SetHandlerResponse,
    description="Выбираем модель для предсказания."
)
def set_active_model(
    model_name: str = Path(
        ...,
        title="Имя модели",
        description="Название модели, которую нужно использовать.",
        example="DummyModelFirst"
    )
) -> SetHandlerResponse:

    logger.info(f"Запрос на установку активной модели: {model_name}")

    available_models = getattr(app.state, "models", {})
    if model_name not in available_models:
        error_msg = (
            f"Модель '{model_name}' не найдена. "
            f"Доступные модели: {list(available_models.keys())}."
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    app.state.active_model = model_name
    logger.info(f"Активная модель успешно установлена: {model_name}")

    return SetHandlerResponse(status="success", active_model=model_name)


@app.post(
    "/predict_csv",
    description="Предсказывает просмотры и возвращает CSV с результатами."
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV-файл с данными рекламных объявлений.")
) -> StreamingResponse:

    logger.info("Запрос на предсказание просмотров из CSV-файла.")

    if app.state.active_model is None:
        logger.error("Попытание предсказания без установленной активной модели.")
        raise HTTPException(
            status_code=400,
            detail="Активная модель не установлена. Пожалуйста, установите модель через /set."
        )

    try:
        df = pd.read_csv(file.file, sep='\t')
    except Exception as e:
        logger.error(f"Ошибка при чтении CSV: {e}")
        raise HTTPException(status_code=400, detail="Невозможно прочитать CSV-файл.")

    model_name = app.state.active_model
    model = app.state.models.get(model_name)
    if model is None:
        logger.error(f"Модель '{model_name}' отсутствует в app.state.models")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка: модель не найдена.")

    try:
        prediction = model.predict(df)
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при выполнении предсказания.")

    logger.info(
        f"Сформировано {len(prediction)} предсказаний "
        f"для модели '{model_name}'."
    )

    buf = StringIO()
    prediction.to_csv(buf, index=False, sep='\t')
    buf.seek(0)

    filename = f"predictions_{model_name}.csv"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return StreamingResponse(buf, media_type="text/csv", headers=headers)


@app.post(
    "/predict_one_item",
    response_model=PredictOneItemHandlerResponse,
    description="Предсказание количества просмотров для одного рекламного объявления."
)
def predict_one_item(
    payload: AddInput = Body(..., description="Данные объявления.")
) -> PredictOneItemHandlerResponse:

    logger.info("Запрос на предсказание просмотров для одного объявления.")

    active_model = getattr(app.state, "active_model", None)
    if not active_model:
        logger.error("Попытка предсказания без установленной активной модели.")
        raise HTTPException(
            status_code=400,
            detail="Активная модель не установлена. Установите модель через POST /set/{model_name}."
        )

    df = pd.DataFrame([{
        "cpm": payload.cpm,
        "hour_start": payload.hour_start,
        "hour_end": payload.hour_end,
        "publishers": payload.publishers,
        "audience_size": payload.audience_size,
        "user_ids": payload.user_ids,
    }])

    model = app.state.models.get(active_model)
    if model is None:
        logger.error(f"Модель '{active_model}' не найдена в app.state.models.")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка: модель загружена некорректно."
        )

    try:
        pred = model.predict(df)       
        at_least_one = pred["at_least_one"].sum()
        at_least_two = pred["at_least_two"].sum()
        at_least_three = pred["at_least_three"].sum()
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ошибка при выполнении предсказания."
        )

    logger.info(
        f"Модель '{active_model}' вернула "
        f"at_least_one={at_least_one}, "
        f"at_least_two={at_least_two}, "
        f"at_least_three={at_least_three}."
    )

    return PredictOneItemHandlerResponse(
        model=active_model,
        at_least_one=at_least_one,
        at_least_two=at_least_two,
        at_least_three=at_least_three
    )

@app.get("/models", response_model=ModelsHendlerResponse)
def models() -> Annotated[Dict[str, Dict[str, Any]], ModelsHendlerResponse]:
    logger.info("Запрос на получение списка доступных моделей.")
    response = {}
    for model_name in app.state.models:
        response[model_name] = 'Available'
    logger.info(f"Возвращаем список моделей: {list(response.keys())}")
    return ModelsHendlerResponse(models= response)

