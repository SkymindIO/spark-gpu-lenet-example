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
object LeNetMnistExample extends App {
  private val log: Logger = LoggerFactory.getLogger(LeNetMnistExample.getClass)

  private val nbRowsval = 28
  private val nbColumnsval = 28
  private val nbChannelsval = 1
  private val nbOutputval = 10
  private val batchSizeval = 64
  private val nbEpochsval = 10
  private val rngSeedval = 123
  private val weightDecay = 0.005
  private val learningRate = 0.01
  private val batchSize = 32
  private val rngSeed = 12345
  //Set up network configuration (as per standard DL4J networks)
  val nChannels = 1
  val outputNum = 10
  val iterations = 1
  val seed = 123



  println("Build model....")

  val builder = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .regularization(true).l2(0.0005)
    .learningRate(0.01)//.biasLearningRate(0.02)
    //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
    .weightInit(WeightInit.XAVIER)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(Updater.NESTEROVS).momentum(0.9)
    .list()
    .layer(0, new ConvolutionLayer.Builder(5, 5)
      //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
      .nIn(nChannels)
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
  // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
  // new ConvolutionLayerSetup(builder,28,28,1)

  val conf = builder.build()
  val net = new MultiLayerNetwork(conf)
  net.setListeners(new ScoreIterationListener(1))
  net.init()
  val examplesPerDataSetObject = 1000
  val mnistTrain = new MnistDataSetIterator(examplesPerDataSetObject, true, 12345)
  val mnistTest = new MnistDataSetIterator(examplesPerDataSetObject, false, 12345)

  val singleModelStopWatch = new StopWatch
  singleModelStopWatch.start()
  for(i <- 0 to nbEpochsval) {
    net.fit(mnistTrain)
    mnistTrain.reset
  }

  singleModelStopWatch.stop()
  println("Single threaded trained in " + singleModelStopWatch.getTime)
  mnistTest.reset

  val singleEval = new Evaluation(nbOutputval)
  while(mnistTest.hasNext){
    val next = mnistTest.next
    val output = net.output(next.getFeatureMatrix)
    singleEval.eval(next.getLabels, output)
  }


  println(singleEval.stats())
  println("****************Single Example finished********************")

}
