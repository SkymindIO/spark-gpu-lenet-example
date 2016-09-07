/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License")
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.sparkgpuexample

import java.util
import java.util.{Collections, Random}

import org.apache.commons.lang3.time.StopWatch

import scala.collection.JavaConversions._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4s.layers.convolutional.Convolution2D
import org.deeplearning4s.layers.pooling.MaxPooling2D
import org.deeplearning4s.layers.{Dense, DenseOutput}
import org.deeplearning4s.models.Sequential
import org.deeplearning4s.optimizers.SGD
import org.deeplearning4s.regularizers.l2
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ListBuffer

/**
  * Two-layer MLP for MNIST.
  *
  * @author David Kale
  */
object LeNetMnistExampleSpark extends App {
  private val log: Logger = LoggerFactory.getLogger(LeNetMnistExample.getClass)

  private val nbRowsval = 28
  private val nbColumnsval = 28
  private val nbChannelsval = 1
  private val nbOutputval = 10
  private val batchSizeval = 64
  private val nbEpochsval = 100
  private val rngSeedval = 123
  private val weightDecay = 0.005
  private val learningRate = 0.1
  private val batchSize = 32
  private val rngSeed = 12345
  //Set up network configuration (as per standard DL4J networks)
  val nChannels = 1
  val outputNum = 10
  val iterations = 1
  val seed = 123



  System.out.println("Build model....")

  val builder = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .learningRate(0.1)//.biasLearningRate(0.02)
    //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
    .weightInit(WeightInit.XAVIER)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(Updater.ADAGRAD)
    .list()
    .layer(0, new ConvolutionLayer.Builder(5, 5)
      .stride(1, 1)
      .nOut(20)
      .activation("identity")
      .build())
    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2,2)
      .stride(2,2)
      .build())
    .layer(2, new ConvolutionLayer.Builder(5, 5)
      //Note that nIn needed be specified in later layers
      .stride(1, 1)
      .nOut(50)
      .activation("identity")
      .build())
    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2,2)
      .stride(2,2)
      .build())
    .layer(4, new DenseLayer.Builder().activation("relu")
      .nOut(500).build())
    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .activation("softmax")
      .build()).setInputType(InputType.convolutionalFlat(28,28,1))
    .backprop(true).pretrain(false)

  val conf = builder.build()
  val net = new MultiLayerNetwork(conf)
  net.init()


  //Create spark context, and load data values memory
  val sparkConf = new SparkConf()
  sparkConf.setMaster("local[*]")
  sparkConf.setAppName("MNIST")
  val sc = new SparkContext(sparkConf)

  val examplesPerDataSetObject = 100
  val mnistTrain = new MnistDataSetIterator(examplesPerDataSetObject, true, 12345)
  val mnistTest = new MnistDataSetIterator(examplesPerDataSetObject, false, 12345)
  val trainData = new ListBuffer[DataSet]
  val testData = new ListBuffer[DataSet]
  while(mnistTrain.hasNext()) trainData.add(mnistTrain.next())
  Collections.shuffle(trainData,new Random(12345))
  while(mnistTest.hasNext()) testData.add(mnistTest.next())

  //Get training data. Note that using parallelize isn't recommended for real problems
  val train: RDD[DataSet] = sc.parallelize(trainData).cache()
  //val test : RDD[DataSet] = sc.parallelize(testData)



  //Create Spark multi layer network from configuration
  val tm  = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
    .workerPrefetchNumBatches(0)
    .saveUpdater(true)
    .averagingFrequency(10)                            //Do 5 minibatch fit operations per worker, then average and redistribute parameters
    .batchSizePerWorker(examplesPerDataSetObject)     //Number of examples that each worker uses per fit operation
    .build()


  val sparkExample = new SparkDl4jMultiLayer(sc,net,tm)
  println(net.getLayerWiseConfigurations.toYaml)
  val stopWatchForSpark = new StopWatch
  stopWatchForSpark.start()
  var finalModel = sparkExample.fit(train.toJavaRDD())
  for(i <- 0 to nbEpochsval - 1) {
    finalModel = sparkExample.fit(train.toJavaRDD())
  }

  stopWatchForSpark.stop()
  System.out.println("Spark model trained in " + stopWatchForSpark.getTime)
  System.out.println("Evaluate spark model....")
  mnistTest.reset
  val evaluator = new Evaluation(nbOutputval)

  while(mnistTest.hasNext){
    val next: DataSet = mnistTest.next()
    val output: INDArray = finalModel.output(next.getFeatureMatrix)
    evaluator.eval(next.getLabels, output)
  }


  System.out.println(evaluator.stats())
  System.out.println("****************Spark Example finished********************")



}
