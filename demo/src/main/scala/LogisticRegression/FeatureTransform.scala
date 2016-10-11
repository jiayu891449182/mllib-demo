package scala.LogisticRegression

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.{ArrayBuffer, HashMap, ListBuffer}

object FeatureTransform {

  /**
    * @param sc
    * @param path_features
    * @param path_data
    * @return
    */
  def dataToLabeledPointRDD(sc : SparkContext, path_features : String, path_data : String): RDD[LabeledPoint] ={
    val data : RDD[String] = sc.textFile(path_data)
    val features : RDD[String] = sc.textFile(path_features).coalesce(1)
    val featureMap : HashMap[String, Int] = loadFeatures(features)
    val rddLabeledPoint = stringToLabeledPoint(data,featureMap)
    rddLabeledPoint
  }

  // 将feature存放到hashmap，feature作为key，每个feature对应一个序号，该序号作为value
  private def loadFeatures(features :  RDD[String]) : HashMap[String, Int] = {
    val featureMap = new HashMap[String, Int]()
    features.collect().foreach{ feature =>
      val arr = feature.split("\t")
      featureMap.put(arr(0),arr(1).toInt)
    }
    featureMap
  }

  /**
    * @param dataRDD
    * @param featureMap
    * @return
    */
  private def stringToLabeledPoint(dataRDD : RDD[String], featureMap : HashMap[String, Int]) : RDD[LabeledPoint] = {
    dataRDD.map(line => line.split("\t")).map { features =>
      val indexValueList = new ListBuffer[Tuple2[Int,Double]]()
      val indexArrary = new ArrayBuffer[Int]()
      val valueArrary = new ArrayBuffer[Double]()
      val label = features(0).toDouble
      for(i <- 1 until features.size){
        val featureString = features(i)
        val array : Array[String] = featureString.split(":")
        if (array.length>1 && featureMap.contains(featureString)) {
          indexValueList += new Tuple2(featureMap.get(featureString).get,1.0)
        } else if (array.length>1 && featureMap.contains(array(0))) {
          indexValueList += new Tuple2(featureMap.get(array(0)).get,array(1).toDouble)
        }
      }
      //稀疏向量要求按照index从小到大排序
      val sortList = indexValueList.toList.sortWith( _._1 < _._1)
      for (j <- sortList) {
        indexArrary += j._1
        valueArrary += j._2
      }
      val vectorSize = featureMap.size
      //稀疏向量，形式为（向量大小,index数组，value数组）
      val featureVector : SparseVector = new SparseVector(vectorSize, indexArrary.toArray, valueArrary.toArray)
      LabeledPoint(label,featureVector)
    }
  }

  private def stringToSVMLabeledPoint(dataRDD : RDD[String], featureMap : HashMap[String, Int]) : RDD[String] = {
    dataRDD.map(line => line.split("\t")).map { features =>
      val indexValueList = new ListBuffer[Tuple2[Int,Double]]()
      val indexArrary = new ArrayBuffer[Int]()
      val valueArrary = new ArrayBuffer[Double]()
      val label = features(0).toDouble
      for(i <- 1 until features.size){
        val featureString = features(i)
        val array : Array[String] = featureString.split(":")
        if (array.length>1 && featureMap.contains(featureString)) {
          indexValueList += new Tuple2(featureMap.get(featureString).get,1.0)
        } else if (array.length>1 && featureMap.contains(array(0))) {
          indexValueList += new Tuple2(featureMap.get(array(0)).get,array(1).toDouble)
        }
      }
      val sortList = indexValueList.toList.sortWith( _._1 < _._1)
      val str = new StringBuilder()
      str.append(label).append(" ")
      for (j <- sortList) {
        str.append(j._1).append(":").append(j._2).append(" ")
      }
      str.substring(0,str.length-1).toString
    }
  }
}
