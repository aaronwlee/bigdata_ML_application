# Abstract
Server-side rendering-based applications are built using Flask and Jinja engines, and bigdata is managed using MongoDB and Spark. Perform machine learning through Spark MLlib using managed data.

# Perform
- Save csv data to MongoDB
- Load data stored in MongoDB into Spark
- Run data analysis with Spark MLlib

# Preparations
1. Install docker
2. Install python 3.9
3. Install openjdk > 11 - It may be different depending on your OS.
4. pip install -r ./requirements.txt
5. spark application may needed

# Run
1. docker run -p 27017:27017 -d mongo:latest
2. run in mode
    - sudo MONGO_URI=mongodb://localhost:27017/bigdata python3 app.py
    - sudo MONGO_URI=mongodb://localhost:27017/bigdata nohup python3 app.py > output.log &

# Node
Since pyspark does not work multi-threading in the docker compose environment right now, it runs directly in the local environment.
