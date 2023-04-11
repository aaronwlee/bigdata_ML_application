import matplotlib as mpl
mpl.use('Agg')

from flask import current_app, request
from flask_socketio import emit
from database import get_mongo_spark_for_thread
from main import socketio

# Import PySpark Pandas
import pyspark.pandas as ps
# Import pandas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans

from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px

from io import BytesIO
import os

connections = []


def emit_img(buffer):
    id = request.sid
    print("emit_img", id, connections)
    if id not in connections:
        raise Exception("disconnected")
    buffer.seek(0)
    for line in buffer:
        emit("img-stream", { "status": "ongoing", "data": line })
    
    emit("img-stream", { "status": "done" })

def emit_dataframe(df):
    id = request.sid
    print("emit_img", id, connections)
    if id not in connections:
        raise Exception("disconnected")
    emit("html", df.to_html())

def emit_message(msg):
    id = request.sid
    print("emit_img", id, connections)
    if id not in connections:
        raise Exception("disconnected")
    emit("message", msg)


@socketio.on('connect')
def test_connect():
    id = request.sid
    if id not in connections:
        connections.append(id)
    print("connected")

@socketio.on('disconnect')
def test_disconnect():
    id = request.sid
    if id in connections:
        connections.remove(id)
    print('Client disconnected')


@socketio.on('start')
def handle_message(collection):
    emit_message("[Stage: 1] Spark Init")
    sc = get_mongo_spark_for_thread()

    try:
        emit_message("[Stage: 1] Spark Ready")
        emit_message("[Stage: 2] Load data from MongoDB")
        df = sc.read.format("mongodb").option("database", "bigdata").option("collection", collection).load()
        df = df.drop("_id")
        emit_dataframe(df.toPandas().head(5))
        emit_message(f"[Stage: 2] DataFrame Dimensions : {(df.count(),len(df.columns))}")

        ## 3. data cleaning stage
        emit_message("[Stage: 3] Data cleaning stage")

        df_clean = df.select(sorted(df.columns))

        ## 3.1. Remove white space of the columns
        emit_message("[Stage: 3.1] Remove white space of the columns")
        for colname in df_clean.columns:
            df_clean = df_clean.withColumn(colname, F.trim(F.col(colname)))
        emit_message(f"DataFrame Dimensions : {(df_clean.count(),len(df_clean.columns))}")

        ## 3.2. Combine duplicate columns using "coalesce"
        emit_message("[Stage: 3.2] Combine duplicate columns using 'coalesce'")
        ### Combine and fill out duplicate columns
        single = ["SAMPLEID","HOLEID","PROJECTCODE","SAMPTO","SAMPLETYPE","SAMPFROM"]
        tuple = [item for item in df_clean.columns if item not in single]
        key=[]
        tuple_checked=[]
        to_drop=[]
        for t1 in tuple:
            if(t1 not in tuple_checked):
                tuple_dict1 = t1.split("_")
                k1 = tuple_dict1[0]
                key.append(t1)
                tuple_checked.append(t1)
                to_drop.append(t1)
            
                for t2 in tuple:
                    if(t1 != t2):
                        tuple_dict2 = t2.split("_")
                        k2=tuple_dict2[0]
                        if(k1==k2):
                            key.append(t2) 
                            tuple_checked.append(t2)
                            to_drop.append(t2)

                b=[]
                [b.append(df_clean[col]) for col in key]
                #coalesce columns to get one column
                add_col=k1+"_ppm"
                tuple_checked.append(add_col)
                df_clean=df_clean.withColumn(add_col,F.coalesce(*b)).drop(*to_drop)
                key.clear()
                to_drop.clear()
        
        emit_message(f"DataFrame Dimensions : {(df_clean.count(),len(df_clean.columns))}")

        ## 3.3. Handle null and convert special characters
        emit_message("[Stage 3.3]. Handle null and convert special characters")
        pdf_clean=df_clean.toPandas()
        emit_dataframe(pdf_clean.head())
        
        pdf_clean.set_index(['HOLEID','SAMPLEID','PROJECTCODE','SAMPFROM','SAMPTO','SAMPLETYPE'],inplace=True)
        null_counts_df=pdf_clean.isna().sum().to_frame("null_count")
        null_counts_df.index.set_names(['Element'], inplace=True)
        null_counts_df.reset_index(inplace=True)
        null_counts_df["percent_of_null"] = (null_counts_df["null_count"]/pdf_clean.shape[0] ) * 100
        emit_dataframe(null_counts_df.head(2))

        ### 3.3.1. Drop Columns with more than 10 % nulls
        emit_message("[Stage 3.3.1] Drop Columns with more than 10 % nulls")
        threshold= round(0.9* pdf_clean.shape[0]) + 1 
        #print(threshold)
        pdf_clean.dropna(thresh=threshold,axis=1, inplace=True)
        emit_message(pdf_clean.shape)

        ### 3.3.2. Drop rows with more Any Null Column
        emit_message("[3.3.2] Drop rows with more Any Null Column")
        pdf_clean.dropna(inplace=True)
        emit_message(pdf_clean.shape)
        emit_dataframe(pdf_clean.head(3))

        ### 3.3.3. Replace values that include "<" with half the value to the right of ">"
        emit_message('[3.3.3] Replace values that include "<" with half the value to the right of ">"')
        def handle_less_than(a_value):
            if a_value.find("<")!=-1:
                return float(a_value.split("<",1)[1])/2
            elif a_value.find(">")!=-1 :
                return a_value.split(">",1)[1]
            else:
                return a_value 
            
        #Convert all column to strings before replacing values. 

        pdf_clean = pdf_clean.astype(str)
        pdf_clean[pdf_clean.columns]=pdf_clean[pdf_clean.columns].applymap(handle_less_than)
        pdf_clean = pdf_clean.astype(float)

        emit_dataframe(pdf_clean.head(3))
        # emit_dataframe(pdf_clean.info())

        ## 4. Exploratory Data Analysis
        emit_message("[Stage: 4] Exploratory Data Analysis")
        ### 4.1. Numerical and Categorical Variables
        emit_message("[Stage 4.1] Numerical and Categorical Variables")

        sdf_clean = sc.createDataFrame(pdf_clean.reset_index())
        sdf_EDA = sdf_clean
        pdf_EDA = pdf_clean

        categorical_var = [ t[0] for t in sdf_EDA.dtypes if t[1]=='string']
        numerical_var = list(set(sdf_EDA.columns)-set(categorical_var))
        emit_message(f"Numerical vars   -> {numerical_var}",)
        emit_message(f"Categorical vars -> {categorical_var}")

        ## 4.2. Pairplot of Numerical Variables
        emit_message("[Stage 4.2] Pairplot of Numerical Variables")
        # sns.pairplot(pdf_EDA, diag_kind="hist", corner=True)
        # buf = BytesIO()
        # plt.savefig(buf, format='png')
        # emit_img(buf)

        ## 4.3. Log transformation
        emit_message("[Stage 4.3] Log transformation")
        # pdf_log_transformed = pdf_EDA.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
        # pdf_log_transformed.plot.hist(subplots=True, legend=True, layout=(6, 4), figsize=(15, 15))
        # buf = BytesIO()
        # plt.savefig(buf, format='png')
        # emit_img(buf)


        ## 4.4. Correlation and Heatmap
        emit_message("[Stage 4.4] Correlation and Heatmap")
        # plt.figure(figsize=(10, 10))
        all_numeric_var = numerical_var
        correlation = pdf_EDA[all_numeric_var].iloc[:, 0:].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        # sns.heatmap(correlation, mask=mask, fmt='.1f',annot=True,square=True, cmap="coolwarm")
        emit_dataframe(pdf_EDA.head(3))

        ## 5. Preprocessing and Transformations
        emit_message("[Stage 5] Preprocessing and Transformations")

        ## set up prep dataframes
        pdf_prep = pdf_EDA
        sdf_prep = sdf_EDA


        categorical_var = [ t[0]  for t in sdf_prep.dtypes if t[1]=='string']
        numerical_var = list(set(sdf_EDA.columns)-set(categorical_var))
        columns_to_scale = numerical_var
        emit_message(f"Numerical vars   -> {numerical_var}")
        emit_message(f"Categorical vars -> {categorical_var}")
        emit_message(f"columns_to_scale -> {columns_to_scale}")

        ## 5.1. MinMaxScaler Using UDF
        emit_message("[Stage 5.1] MinMaxScaler Using UDF")

        sdf_udf_scale=sdf_prep
        # UDF for converting column type from vector to double type
        unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

        # Iterating over columns to be scaled
        for i in columns_to_scale:
            # VectorAssembler Transformation - Converting column to vector type
            assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

            # MinMaxScaler Transformation
            scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

            # Pipeline of VectorAssembler and MinMaxScaler
            pipeline = Pipeline(stages=[assembler, scaler])

            # Fitting pipeline on dataframe
            sdf_udf_scale = pipeline.fit(sdf_udf_scale).transform(sdf_udf_scale).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")
            
            
        ## Select scaled columns
        names = {x + "_scaled": x for x in columns_to_scale}
        cols_expanded_lcd = [F.col(c).alias(names[c]) for c in names.keys()]
        sdf_udf_scale = sdf_udf_scale.select(*categorical_var,*cols_expanded_lcd)

        #Assembler to get features
        ##for umap
        df_udf_scale = sdf_udf_scale.toPandas()
        ## for pca
        sdf_udf_scale = VectorAssembler(inputCols=columns_to_scale, outputCol='features').transform(sdf_udf_scale)


        emit_message("After Scaling :")
        emit_dataframe(sdf_udf_scale.toPandas().head(3))

        ## 5.2. MinMaxScaler and VectorAssembler Pipeline
        emit_message("[Stage 5.2] MinMaxScaler and VectorAssembler Pipeline")

        # set up each processing step with the correct input columns and output
        assemble          = VectorAssembler(inputCols=columns_to_scale, outputCol='assemble_features')
        MinMax_scale      = MinMaxScaler(inputCol='assemble_features',outputCol='features')
        MM_pipeline       = Pipeline(stages=[assemble, MinMax_scale])
        MM_model          = MM_pipeline.fit(sdf_prep)
        sdf_scale            = MM_model.transform(sdf_prep)
        emit_dataframe(sdf_scale.toPandas().head(3))

        ## 6. Unsupervised MachinLearning
        emit_message("[Stage 6] Unsupervised MachinLearning")
        
        ### 6.1. PCA Feature Reduction
        emit_message("[Stage 6.1] PCA Feature Reduction")
        pca = PCA(k=9,inputCol="features",outputCol="pcaFeatures")
        pca_model          = pca.fit(sdf_scale)
        sdf_pca            = pca_model.transform(sdf_scale)
        emit_dataframe(sdf_pca.toPandas().head(3))

        pcs = np.round(pca_model.pc.toArray(),9)

        columns_to_scale[:-1]

        pcs = np.round(pca_model.pc.toArray(),9)
        df_pc = pd.DataFrame(pcs, columns = ['PC'+str(i) for i in range(1, 10)], index = columns_to_scale)
        ##The variance explained
        variance_arr = np.cumsum(pca_model.explainedVariance.toArray())
        d = np.argmax(variance_arr >=0.95) + 1
        emit_message('The number of dimensions required to preserve 95% of the data set variance is {}.'.format(d))

        ### 6.1.1. Optimal Number of Clusters Using Sillhoute Scores
        emit_message("[Stage 6.1.1] Optimal Number of Clusters Using Sillhoute Scores")
        km_pca = KMeans(featuresCol = 'pcaFeatures')
        evaluator = ClusteringEvaluator()
        cluster_list = []
        sillhoute_scores = []

        # test between k=2 and 10 
        for cluster_num in range(2,10):
            # set the KMeans stage of the pipe to hold each value of K and the random seed = 1 and fit that pipe to data  
            kmeans = km_pca.setK(cluster_num).setSeed(1)  
            km_model  = kmeans.fit(sdf_pca)
            
            # build a preds dataset of each k value
            preds = km_model.transform(sdf_pca)

            # silhouette score each prediction set and print formatted output 
            silhouette = evaluator.evaluate(preds)
            cluster_list.append(cluster_num)
            sillhoute_scores.append(silhouette)
            emit_message(f'Tested: {cluster_num} clusters: {silhouette}')

        # Display the graph 
        fig  = plt.subplots(figsize=(10,5))
        plt.plot(cluster_list, sillhoute_scores )
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sillhoute Scores')
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        emit_img(buf)

        ## 6.2. TSNE Feature Reduction
        emit_message("[Stage 6.2] TSNE Feature Reduction")
        pdf_tsne=df_udf_scale.copy(deep=True)
        pdf_tsne.set_index(['HOLEID','SAMPLEID','PROJECTCODE','SAMPFROM','SAMPTO','SAMPLETYPE'],inplace=True)
        pdf_tsne = pdf_tsne.reindex(sorted(pdf_tsne.columns), axis=1)
        emit_dataframe(pdf_tsne.head(3))

        ### 6.2.1. 2D TSNE Feature Reduction
        emit_message("[Stage 6.2.1] 2D TSNE Feature Reduction")
        # We want to get TSNE embedding with 2 dimensions
        n_components = 2
        tsne = TSNE(n_components,learning_rate=50,init='random')
        tsne_features = tsne.fit_transform(pdf_tsne)
        emit_message(tsne_features.shape)

        tsne_features[1:4,:]
        pdf_tsne['Tsne_X'] = tsne_features[:,0]
        pdf_tsne['Tsne_Y']= tsne_features[:,1]

        emit_dataframe(pdf_tsne.head(2))

        pdf_tsne_red = pdf_tsne[["Tsne_X","Tsne_Y"]]

        stsne_reduced_df = sc.createDataFrame(pdf_tsne_red)
        colsAssmb = stsne_reduced_df.columns
        tsne_assemble = VectorAssembler(inputCols=colsAssmb, outputCol='features')
        stsne_reduced_df = tsne_assemble.transform(stsne_reduced_df)
        emit_dataframe(stsne_reduced_df.toPandas().head(3))

        ### 6.2.2. Optimal Number of Clusters Using Sillhoute Scores
        emit_message("[Stage 6.2.2] Optimal Number of Clusters Using Sillhoute Scores")
        km = KMeans(featuresCol = 'features')

        # set up evaluator 
        evaluator = ClusteringEvaluator()
        cluster_list = []
        sillhoute_scores = []

        # test between k=2 and 10 
        for cluster_num in range(2,15):
            # set the KMeans stage of the pipe to hold each value of K and the random seed = 1 and fit that pipe to data  
            kmeans = km.setK(cluster_num).setSeed(1)  
            km_model  = kmeans.fit(stsne_reduced_df)
            
            # build a preds dataset of each k value
            preds = km_model.transform(stsne_reduced_df)

            # silhouette score each prediction set and print formatted output 
            silhouette = evaluator.evaluate(preds)
            cluster_list.append(cluster_num)
            sillhoute_scores.append(silhouette)
            emit_message(f'Tested: {cluster_num} clusters: {silhouette}')

        # Display the graph 
        fig  = plt.subplots(figsize=(10,5))
        plt.plot(cluster_list, sillhoute_scores )
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sillhoute Scores')
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        emit_img(buf)

        ## 6.2.3. KMeans Clustering on TSNE Results
        emit_message("[Stage 6.2.3] KMeans Clustering on TSNE Results")

        num_clusters=8
        # set the random seed for the algorithm and the value for k 
        kmeans = km.setK(num_clusters).setSeed(1)  
        # fit model and transform the data showing a cut of the data to check output
        km_model  = kmeans.fit(stsne_reduced_df)

        km_centers = km_model.clusterCenters()
        km_centers = np.asarray(km_centers)

        prediction = km_model.transform(stsne_reduced_df).select('prediction').collect()
        labels = [p.prediction for p in prediction ]

        clusters = km_model.transform(stsne_reduced_df)
        clusters =clusters.drop("features")
        emit_dataframe(clusters.toPandas().head(5))

        # vis_umap = clusters.toPandas()

        # kmeans_umap_reduced_df=vis_umap.copy(deep=True)
        kmeans_umap_reduced_df =clusters.toPandas()

        grouped = kmeans_umap_reduced_df.groupby('prediction')

        colors = { 0:'blue', 1:'orange', 2:'green', 3:'skyblue', 4:'black', 5:'yellow',6:'magenta', 7:'red'}
        fig, ax = plt.subplots(figsize=(10,10))
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='Tsne_X', y='Tsne_Y', label=key, color=colors[key])
            plt.xlabel('X')
            plt.ylabel('Y')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        emit_img(buf)

        ## 6.3. UMAP Feature Reduction
        emit_message("[Stage 6.3] UMAP Feature Reduction")

        pdf_umap=df_udf_scale
        pdf_umap.set_index(['HOLEID','SAMPLEID','PROJECTCODE','SAMPFROM','SAMPTO','SAMPLETYPE'],inplace=True)
        pdf_umap = pdf_umap.reindex(sorted(pdf_umap.columns), axis=1)
        emit_dataframe(pdf_umap.head())



        umap_3d = UMAP(n_components=3, init='random', random_state=0)
        umap_reduced_ar = umap_3d.fit_transform(pdf_umap)
        umap_fig_2d = px.scatter(
            umap_reduced_ar, x=0, y=1
        )
        umap_fig_3d = px.scatter_3d(
            umap_reduced_ar, x=0, y=1, z=2
            
        )
        umap_fig_3d.update_traces(marker_size=5)

        buf = BytesIO()
        umap_fig_2d.write_image(buf, format='png')
        emit_img(buf)
        buf = BytesIO()
        umap_fig_3d.write_image(buf, format='png')
        emit_img(buf)


        umap_reduced_df = pd.DataFrame(data = umap_reduced_ar,columns = ["Comp1","Comp2","Comp3"])

        umap_reduced_df = umap_reduced_df.astype(float)
        sumap_reduced_df = sc.createDataFrame(umap_reduced_df)
        colsAssmb=sumap_reduced_df.columns
        Umap_assemble = VectorAssembler(inputCols=colsAssmb, outputCol='features')
        sumap_reduced_df = Umap_assemble.transform(sumap_reduced_df)
        emit_dataframe(sumap_reduced_df.toPandas().head(3))

        ### 6.3.1. Optimal Number of Clusters Using Sillhoute Scores
        emit_message("[Stage 6.3.1] Optimal Number of Clusters Using Sillhoute Scores")
        km = KMeans(featuresCol = 'features')

        # set up evaluator 
        evaluator = ClusteringEvaluator()
        cluster_list_umap = []
        sillhoute_scores_umap = []

        # test between k=2 and 10 
        for cluster_num in range(2,15):
            # set the KMeans stage of the pipe to hold each value of K and the random seed = 1 and fit that pipe to data  
            kmeans = km.setK(cluster_num).setSeed(1)  
            km_model  = kmeans.fit(sumap_reduced_df)
            
            # build a preds dataset of each k value
            preds_umap = km_model.transform(sumap_reduced_df)

            # silhouette score each prediction set and print formatted output 
            silhouette = evaluator.evaluate(preds_umap)
            cluster_list_umap.append(cluster_num)
            sillhoute_scores_umap.append(silhouette)
            emit_message(f'Tested: {cluster_num} clusters: {silhouette}')

        
        #Display the graph 
        fig  = plt.subplots(figsize=(10,5))
        plt.plot(cluster_list_umap, sillhoute_scores_umap)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sillhoute Scores')
        plt.grid(True)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        emit_img(buf)

        ### 6.3.2. KMeans Clustering on UMAP Results
        emit_message("[Stage 6.3.2] KMeans Clustering on UMAP Results")
        num_clusters=7
        # set the random seed for the algorithm and the value for k 
        kmeans = km.setK(num_clusters).setSeed(1)  
        # fit model and transform the data showing a cut of the data to check output
        km_model  = kmeans.fit(sumap_reduced_df)

        km_centers = km_model.clusterCenters()
        km_centers = np.asarray(km_centers)

        prediction = km_model.transform(sumap_reduced_df).select('prediction').collect()
        labels = [p.prediction for p in prediction ]

        clusters = km_model.transform(sumap_reduced_df)
        clusters = clusters.drop("features")
        emit_dataframe(clusters.toPandas().head(3))

        kmeans_umap_reduced_df = clusters.toPandas()

        grouped = kmeans_umap_reduced_df.groupby('prediction')

        colors = { 0:'blue', 1:'orange', 2:'green', 3:'skyblue', 4:'red', 5:'yellow',6:'magenta',7:'orange'}
        fig, ax = plt.subplots(figsize=(10,10))
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='Comp1', y='Comp2', label=key, color=colors[key])
            plt.xlabel('X')
            plt.ylabel('Y')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        emit_img(buf)

        emit_message("[Final] All Stages are done")
    except Exception as err:
        print(err)
        emit_message(f"[Error] {err}")
    finally:
        sc.stop()



    