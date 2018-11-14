from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


class DistRS(object):
    def __init__(self, n_factors=10, epochs=10, alpha=0.01):
        self.spark = SparkSession \
            .builder \
            .appName('budget-utilization') \
            .enableHiveSupport() \
            .getOrCreate()
        self.sc = self.spark.sparkContext()

        self.n_factors = n_factors
        self.epochs = epochs
        self.alpha = alpha

        self.model = None

    def load_data(self, data_path, aggregate=True):
        data = self.sc.textFile(data_path)
        data = data\
            .map(lambda tokens: tokens.split(',')) \
            .map(lambda tokens: ((int(tokens[0]), int(tokens[4])), int(tokens[5])))

        if aggregate:
            data = data\
                .reduceByKey(lambda x, y: x + y)

        ratings = data\
            .map(lambda t: Rating(t[0][0], t[0][1], t[2]))

        return ratings

    def train(self, ratings):
        self.model = ALS\
            .trainImplicit(ratings, self.n_factors, self.epochs, self.alpha)

    def predict(self, data):
        predictions = self.model\
            .predictAll(data)\
            .map(lambda r: ((r[0], r[1]), r[2]))

        return predictions

    def save_model(self, to_path):
        self.model\
            .save(self.sc, to_path)


def evaluate(ratings, predictions):
    comparable = ratings\
        .map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

    mse = comparable\
        .map(lambda r: (r[1][0] - r[1][1])**2).mean()

    print("Mean Squared Error is: {}".format(mse))


def load_model(sc, from_path):
    model = MatrixFactorizationModel\
        .load(sc, from_path)

    return model
