FROM openjdk:slim

COPY --from=python:3.9-bullseye / /

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]