package org.example;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.nativeblas.Nd4jCuda;


import javax.imageio.ImageIO;

public class HandwrittenDigitsGenerator {


    private static final int EPOCHS = 10;
    private static final int BATCH_SIZE = 128;

    public static void main(String[] args) throws IOException {

        // Load the MNIST dataset
        DataSetIterator trainData = new MnistDataSetIterator(BATCH_SIZE, 100);

        // Create the discriminator network
        MultiLayerConfiguration discriminatorConf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.0001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(784)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();

        MultiLayerNetwork discriminator = new MultiLayerNetwork(discriminatorConf);
        discriminator.init();
        discriminator.addListeners(new ScoreIterationListener(1));

        // Create the generator network
        MultiLayerConfiguration generatorConf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.0001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(784)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();

        MultiLayerNetwork generator = new MultiLayerNetwork(generatorConf);
        generator.init();

        // Train the networks
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Train the discriminator
            for (int i = 0; i < trainData.next().numExamples(); i++) {
                DataSet ds = trainData.next();
                double realLabel = 1.0;
                double fakeLabel = 0.0;
//                double[] epsilonArray = {1e-6};


                Nd4jCuda.Workspace workspace = new Nd4jCuda.Workspace();
                Set<ArrayType> arrayTypes = Collections.singleton(ArrayType.ACTIVATIONS);

                // Create WorkspaceConfiguration
                WorkspaceConfiguration config = WorkspaceConfiguration.builder()
                        .initialSize(10000000)  // Adjust the size as per your requirements
                        .policyAllocation(AllocationPolicy.STRICT)
                        .policyMirroring(MirroringPolicy.HOST_ONLY)
                        .policyLearning(LearningPolicy.FIRST_LOOP)
                        .build();

                // Create configMap and workspaceNames
                Map<ArrayType, WorkspaceConfiguration> configMap = Collections.singletonMap(ArrayType.ACTIVATIONS, config);
                Map<ArrayType, String> workspaceNames = Collections.singletonMap(ArrayType.ACTIVATIONS, "MyWorkspace");

                // Create LayerWorkspaceMgr
                LayerWorkspaceMgr workspaceMgr = new LayerWorkspaceMgr(arrayTypes, configMap, workspaceNames);



                INDArray features = ds.getFeatures();
                INDArray labels = Nd4j.create(new double[]{realLabel});
                discriminator.setInput(features);
                discriminator.setLabels(labels);
                discriminator.calculateGradients(ds.getFeatures(), ds.getLabels(), null, null);
//                discriminator.backpropGradient(Nd4j.create(epsilonArray), workspaceMgr);
                discriminator.backpropGradient(Nd4j.scalar(1e-6), workspaceMgr);

                INDArray noise = Nd4j.rand(new int[]{1, 10});
                int[] generatedImage = generator.predict(noise);
                INDArray generatedImageINDArray = Nd4j.create(generatedImage);

                labels = Nd4j.create(new double[]{fakeLabel});
                generator.setInput(noise);
                generator.setLabels(labels);
                generator.calculateGradients(ds.getFeatures(), ds.getLabels(), null, null);

                discriminator.backpropGradient(generatedImageINDArray, workspaceMgr);
            }

            // Check for Nash equilibrium
            double discriminatorAccuracy = discriminator.evaluate(trainData).accuracy();
            if (discriminatorAccuracy == 1.0) {
                System.out.println("Nash equilibrium reached at epoch " + epoch);
                break;
            }
        }

        // Print the result of training
        System.out.println("Training complete");
        System.out.println("Discriminator accuracy: " + discriminator.evaluate(trainData).accuracy());
        System.out.println("Generator accuracy: " + generator.evaluate(trainData).accuracy());

        // Generate some handwritten digits
        List<Image> generatedDigits = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            double[] input = new double[]{i};
            INDArray inputINDArray = Nd4j.create(input);
            generator.setInput(inputINDArray);
//            generator.calculateGradients();
//            generator.backprop();
//            Image image = generator.outputToImage();
            INDArray output = generator.output(inputINDArray, false);
            Image image = outputToImage(output);
            generatedDigits.add(image);
        }

        // Save the generated digits
        File outputDir = new File("generated_digits");
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        for (int i = 0; i < generatedDigits.size(); i++) {
            ImageIO.write((RenderedImage) generatedDigits.get(i), "png", new File(outputDir, "generated_digit_" + i + ".png"));
//            ImageIO.write(generatedDigits.get(i), "png", new File(outputDir, "generated_digit_" + i + ".png"));
        }
    }
    private static Image outputToImage(INDArray output) {
        long[] shape = output.shape();
        int numRows = (int) shape[0];
        int numColumns = (int) shape[1];
        BufferedImage image = new BufferedImage(numColumns, numRows, BufferedImage.TYPE_BYTE_GRAY);
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numColumns; col++) {
                float pixelValue = output.getFloat(row, col);
                int rgbValue = (int) (pixelValue * 255);
                int pixel = (rgbValue << 16) | (rgbValue << 8) | rgbValue;
                image.setRGB(col, row, pixel);
            }
        }
        return image;
    }
}