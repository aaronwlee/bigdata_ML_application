FROM openjdk:slim

COPY --from=python:3.9-bullseye / /

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "2", "app:app", "-b", "0.0.0.0:80"]