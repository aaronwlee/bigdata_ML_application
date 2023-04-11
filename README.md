# Abstract
Server-side rendering-based applications are built using Flask and Jinja engines, and bigdata is managed using MongoDB and Spark. Perform machine learning through Spark MLlib using managed data.

# Perform
- Save csv data to MongoDB
- Load data stored in MongoDB into Spark
- Run data analysis with Spark MLlib

# Preparations
1. download .env file - Because it has sensitive content, please download it from slack.
2. install openjdk > 11 - It may be different depending on your OS.
3. pip install -r ./requirements.txt --user
4. spark application may needed

# Run
docker run -p 27017:27017 -d mongo:latest
MONGO_URI=mongodb://localhost:27017/bigdata python app.py
