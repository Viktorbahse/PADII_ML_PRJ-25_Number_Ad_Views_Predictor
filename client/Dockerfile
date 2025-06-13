FROM python:slim
WORKDIR /app

COPY . .
COPY ../common/ common/

RUN pip3 install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
