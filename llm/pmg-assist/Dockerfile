FROM python:3.12-slim

WORKDIR /app

RUN pip install pipenv

COPY data/pmg_data.parquet.brotli data/pmg_data.parquet.brotli
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --ignore-pipfile --system

COPY pmg-assist .

EXPOSE 5001

CMD gunicorn --bind 0.0.0.0:5001 app:app