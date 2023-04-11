#!/bin/bash

echo "Building Stage"
docker build -t aaronwoolee/bigdata-ml-app:v1.1 .

echo "Pushing Stage"
docker push aaronwoolee/bigdata-ml-app:v1.1