package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class Discriminator {
    private MultiLayerNetwork model;

    public Discriminator() {
        // Setting the neural network configuration
        int numInputs = 784;
        int numOutputs = 1;
        int numHiddenNodes1 = 1024;
        int numHiddenNodes2 = 512;
        int numHiddenNodes3 = 256;
        double dropoutProb = 0.3;
        double learningRate = 0.0002;
        int numIterations = 1;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(111)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes1)
                        .activation(new ActivationReLU())
                        .dropOut(dropoutProb)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(numHiddenNodes1)
                        .nOut(numHiddenNodes2)
                        .activation(new ActivationReLU())
                        .dropOut(dropoutProb)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(numHiddenNodes2)
                        .nOut(numHiddenNodes3)
                        .activation(new ActivationReLU())
                        .dropOut(dropoutProb)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(numHiddenNodes3)
                        .nOut(numOutputs)
                        .activation(new ActivationSigmoid())
                        .lossFunction(new LossBinaryXENT())
                        .build());

        // Building the neural network model
        this.model = new MultiLayerNetwork(builder.build());
        this.model.init();
        this.model.setListeners(new ScoreIterationListener(numIterations));
    }

    public INDArray predict(INDArray input) {
        return this.model.output(input);
    }

    public void fit(INDArray input, INDArray label) {
        this.model.fit(input, label);
    }
}