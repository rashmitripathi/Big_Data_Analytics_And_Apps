import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Puchu on 2/1/2017.
  */
object SparkSortWordsByLength {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir","D:\\winutils")

    val sparkConf = new SparkConf().setAppName("SortWordsByLength").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)
    val input=sc.textFile("input.txt")
    val fil_input = input.flatMap((line=>{line.split(" ")}))
    val output =fil_input.flatMap(line=>{line.replaceAll("[^a-zA-Z\\s]", "").toLowerCase.split(" ")})
      .map(word=>(word,word.size)).distinct().cache()
    val o=output.collect().sortBy(-_._2)

    //output .saveAsTextFile("output1")
    val file = "output.txt"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    o.foreach{case(word,count)=>{
      writer.write(word+":"+count+" \r\n")
    }}
    writer.close()
  }

}
