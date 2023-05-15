package org.example;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;

public class Discriminator {
    private MultiLayerNetwork model;

    public Discriminator() {
        int numInputs = 784;
        int numOutputs = 1;
        double dropoutProb = 0.3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.0002, 0.5, 0.999, 1e-8))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(1024)
                        .activation(Activation.RELU).dropOut(dropoutProb).build())
                .layer(new DenseLayer.Builder().nIn(1024).nOut(512)
                        .activation(Activation.RELU).dropOut(dropoutProb).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(256)
                        .activation(Activation.RELU).dropOut(dropoutProb).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(256).nOut(numOutputs).activation(Activation.SIGMOID).build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    public double train(INDArray inputs, INDArray labels) {
        model.setInput(inputs);
        model.setLabels(labels);
        model.fit();
        return model.score();
    }

    public void setParams(Adam optimizer, LossFunction lossFunction) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(optimizer)
                .list()
                .layer(new OutputLayer.Builder((ILossFunction) lossFunction)
                        .activation(Activation.SIGMOID)
                        .nIn(256)
                        .nOut(1)
                        .build())
                .build();

        model.setLayerWiseConfigurations(conf);
    }


    public MultiLayerNetwork getModel() {
        return model;
    }
}