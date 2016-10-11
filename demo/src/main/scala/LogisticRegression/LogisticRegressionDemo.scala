package scala.LogisticRegression

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegressionDemo {

  val conf : SparkConf = new SparkConf
  val sc : SparkContext = new SparkContext(conf)

  def main(args : Array[String]) : Unit ={
    val trainData_path : String = args(0)
    val features_path : String = args(1)
    val weights_path : String = args(2)
    val trainRDD :  RDD[LabeledPoint]= FeatureTransform.dataToLabeledPointRDD(sc, features_path, trainData_path)
    val weights : Vector = LRWithLBFGS(trainRDD)
    val saveWeight = sc.parallelize(weights.toArray,1)
    saveWeight.saveAsTextFile(weights_path)
  }

  def LRWithLBFGS(data : RDD[LabeledPoint]): Vector ={
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data)
    model.weights
  }
}
