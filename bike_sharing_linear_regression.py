from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as f
import pyspark


def run(spark_session: pyspark.sql.session.SparkSession):
    bike_sharing = spark_session.read.csv("day.csv", header=True)

    bike_sharing01 = bike_sharing.select(
        bike_sharing.season.astype("int"),
        bike_sharing.yr.astype("int"),
        bike_sharing.mnth.astype("int"),
        bike_sharing.holiday.astype("int"),
        bike_sharing.weekday.astype("int"),
        bike_sharing.workingday.astype("int"),
        bike_sharing.weathersit.astype("int"),
        bike_sharing.temp.astype("double"),
        bike_sharing.atemp.astype("double"),
        bike_sharing.hum.astype("double"),
        bike_sharing.windspeed.astype("double"),
        bike_sharing.cnt.astype("int").alias("label")
    )

    assembler = VectorAssembler()
    assembler.setInputCols(bike_sharing01.columns[:-1])
    assembler.setOutputCol("features")
    train, test = bike_sharing01.randomSplit((0.7, 0.3))

    train01 = assembler.transform(train)

    train02 = train01.select("features", "label")

    lr = LinearRegression()

    model = lr.fit(train02)

    test2 = assembler.transform(test)
    test02 = test2.select("features", "label")
    out = model.transform(test02)

    e = RegressionEvaluator()
    e.evaluate(out, {e.metricName: "r2"})
    e.evaluate(out, {e.metricName: "rmse"})

    res = out.select(f.abs(f.col("label")-f.col("prediction")).alias("diff"))
    accs = res.select(f.when(f.col("diff") < 300, 1).otherwise(0).alias("is_accurate"))
    accs.limit(3).toPandas()
    accs.agg(f.mean("is_accurate").alias("accuracy")).toPandas()

    # using MinMaxScaler to scale features:
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(train02)
    scaler_model.transform(train02)


def construct_pipeline():
    feature_columns = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed"
    ]

    sql_transformer = SQLTransformer(statement="""
        SELECT
            cast(season as int),
            cast(yr as int),
            cast(mnth as int),
            cast(holiday as int),
            cast(weekday as int),
            cast(workingday as int),
            cast(weathersit as int),
            cast(temp as double),
            cast(atemp as double),
            cast(hum as double),
            cast(windspeed as double),
            cast(cnt as int) as label
        FROM __THIS__
    """)

    assembler = VectorAssembler().setInputCols(feature_columns[:-1]).setOutputCol("features")

    feature_label_tf = SQLTransformer(statement="""
        SELECT features, label
        FROM __THIS__
    """)

    pipeline = Pipeline(stages=[
        sql_transformer,
        assembler,
        feature_label_tf,
        LinearRegression()
    ])

    return pipeline


def save_pipeline_model(pipeline_model: PipelineModel, foldername: str):
    pipeline_model.save(foldername)


def load_pipeline_model(foldername: str):
    return PipelineModel.load(foldername)
