from pydantic import BaseModel, conint, confloat, validator, Field
from typing import Annotated, Dict, Any, Optional


class SetHandlerResponse(BaseModel):
    status: str
    active_model: str

COMMA_INTS_PATTERN = r'^[0-9]+(?:,[0-9]+)*$'

class AddInput(BaseModel):
    cpm: float = Field(..., example=225.0, ge=0.0, description="Цена объявления для которой надо сделать прогноз.")
    hour_start: int = Field(..., example=15, ge=-1, description="Предположительное время запуска рекламного объявления.")
    hour_end: int = Field(..., example=345, gt=-1, description="Предположительное время остановки рекламного объявления.")
    publishers: str = Field(..., example="1,2,3", description="Площадки на которых объявление может быть показано.")
    audience_size: int = Field(..., example=5, ge=0, description="Размер аудитории объявления.")
    user_ids: str = Field(..., example="12,44,46,50,58", description="Аудитория объявления.")

class PredictOneItemHandlerResponse(BaseModel):
    model: str
    at_least_one: float
    at_least_two: float
    at_least_three: float

class ModelsHendlerResponse(BaseModel):
    models: Dict[str, str]

class User(BaseModel):
    user_id: int
    sex: int
    age: conint(ge=0, le=120)
    city_id: int

class HistoryItem(BaseModel):
    hour: int
    cpm: confloat(ge=0)
    publisher: int
    user_id: int

