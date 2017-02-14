import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.log4j.{Level, Logger}
/**
  * Created by Rashmi on 2/8/2017.
  */
object LinearRegressionChimpanzeeProblem {

  def main(args: Array[String]): Unit ={

    System.setProperty("hadoop.home.dir","E:\\big data analytics\\hadoop");
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.INFO);

    // Load and parse the data. This filw contains hour information and the position of chimpanzee
    val data = sc.textFile("data\\inputdataset.data")

    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    parsedData.take(1).foreach(f=>println(f))

    // Split data into training (85%) and test (15%).
    val Array(training, test) = parsedData.randomSplit(Array(0.85, 0.15))

    // Building the model
    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()


    // Evaluate model on test examples and compute training error
    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPreds2.map{ case(v, p) => math.pow((v - p), 2) }.mean()

    println("training Mean Squared Error = " + MSE)
    println("test Mean Squared Error = " + MSE2)

    // Save and load model
    model.save(sc, "data\\LinearRegressionWithSGDModel")
    val sameModel = LinearRegressionModel.load(sc, "data\\LinearRegressionWithSGDModel")
  }

}
