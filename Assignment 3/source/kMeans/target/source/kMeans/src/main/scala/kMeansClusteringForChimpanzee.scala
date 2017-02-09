import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

 /*
  * Created by Rashmi on 2/8/2017.
  */
object kMeansClusteringForChimpanzee {

  def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir","E:\\big data analytics\\hadoop");
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)
    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF)
    // Load and parse the data
    val data = sc.textFile("data/input_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    //Look at how training data is!
   // parsedData.foreach(f=>println(f))
    // Cluster the data into two classes using KMeans
    val numClusters = 3
    val numIterations = 25
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data: ")
    clusters.predict(parsedData).zip(parsedData).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters.save(sc, "data/KMeansOuput")
    val sameModel = KMeansModel.load(sc, "data/KMeansOutput")


  }


}
