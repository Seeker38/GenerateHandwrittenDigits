package org.example;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Generator {
    private static MultiLayerNetwork model;

    public Generator() {
        int numInputs = 100;
        int numOutputs = 784;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(org.nd4j.linalg.learning.config.Adam.builder().learningRate(0.01).beta1(0.5).build())
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(256)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(512)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(1024)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(1024).nOut(numOutputs).activation(Activation.TANH).build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
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

    public static INDArray forward(INDArray latentSpaceSamples) {
        INDArray generated = model.output(latentSpaceSamples);
        generated = generated.reshape(latentSpaceSamples.size(0), 1, 28, 28);
        return generated;
    }

    public double train(INDArray latentSpaceSamples, INDArray labels) {
        model.setInput(latentSpaceSamples);
        model.setLabels(labels);
        model.computeGradientAndScore();
        model.fit();
        return model.score();
    }

    public INDArray generateSamples(int batchSize) {
        INDArray noise = Nd4j.randn(batchSize, 100);
        INDArray generated = model.output(noise);
        generated = generated.reshape(batchSize, 1, 28, 28);
        return generated;
    }
}