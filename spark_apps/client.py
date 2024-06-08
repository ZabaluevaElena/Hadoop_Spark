import logging
from os.path import join
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_timestamp
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, QuantileDiscretizer

ml1mPath = "/opt/spark/data"
TEST_SIZE = 0.1
SEED = 42

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('/opt/spark/apps/client.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_dataset(path, spark):
    movies = (spark.read.option("delimiter","::")
              .csv(join(path, "movies.dat"))
              .toDF("movie_id","title","genres"))
    users = (spark.read.option("delimiter","::")
             .csv(join(path, "users.dat"))
             .toDF("user_id","gender","age","occupation","zipcode")
             .withColumn("age", col("age").cast(IntegerType())))
    ratings = (spark.read.option("delimiter","::")
               .csv(join(path, "ratings.dat"))
               .toDF("user_id","movie_id","rating","time")
               .withColumn("rating", col("rating").cast(DoubleType()))
               .withColumn("time", to_timestamp(from_unixtime(col("time"))))
            )
    return ratings.join(users,"user_id").join(movies,"movie_id")
    

def get_transformator(features, label):
    stages = []
    for column in features:
        if column == 'age':
            str_indexer = QuantileDiscretizer(inputCol='age', numBuckets=4,
                                              outputCol=column + "Index")
        else:
            str_indexer = StringIndexer(inputCol=column,
                                    outputCol=column + "Index",
                                    handleInvalid='keep')
        encoder = OneHotEncoder(inputCols=[str_indexer.getOutputCol()],
                                outputCols=[column + "Vec"],
                                handleInvalid='keep')
        stages += [str_indexer, encoder]
    
    
    assembler_inputs = [c + "Vec" for c in features]
    stages += [VectorAssembler(inputCols=assembler_inputs, outputCol=label)]
    return Pipeline().setStages(stages)

logger.info('Start Application')
console_handler.flush()
file_handler.flush()
spark = (SparkSession.builder
        .appName("SparkFMApp")
        .getOrCreate())

spark.conf.set("spark.sql.shuffle.partitions", SEED)

logger.info('Loading data...')
data = load_dataset(ml1mPath, spark)
logger.info('Loading data... done!')
logger.info('Processing data...')
features = ['user_id', 'gender', 'age', 'occupation', 'title']
target = ['rating']

filter_data = data.select(features+target)

prep_pipe = get_transformator(features, 'features')

(train, test) = data.randomSplit([1-TEST_SIZE, TEST_SIZE])
logger.info('Processing data... done!')

logger.info('Training model...')
ml = (FMRegressor()
      .setLabelCol("rating")
      .setFeaturesCol("features")
      .setFactorSize(32)
      .setSeed(SEED)
      .setMaxIter(100)
      .setRegParam(0.01)
      .setStepSize(0.02)
     )
learn_pipe = Pipeline().setStages([prep_pipe, ml])

fm = learn_pipe.fit(train)

logger.info('Training model... done!')

pred_train = fm.transform(train)
pred_test = fm.transform(test)

evaluator = (RegressionEvaluator()
  .setLabelCol("rating")
  .setPredictionCol("prediction")
  .setMetricName("rmse"))

rmse_train = evaluator.evaluate(pred_train)
rmse_test = evaluator.evaluate(pred_test)

logger.info(f"RMSE train = {rmse_train:.3f}, test = {rmse_test:.3f}")

fm.save(join(ml1mPath, 'model'))
logger.info("Application running... done!")
spark.stop()
