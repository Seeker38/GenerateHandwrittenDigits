package org.example;

import java.awt.image.WritableRaster;
import java.io.File;
import java.util.*;
import java.util.Random;

import ai.djl.pytorch.engine.PtEngine;
import ai.djl.util.cuda.CudaUtils;
import org.nd4j.jita.workspace.CudaWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;


import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;



import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.IOException;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;


public class GANS {
    Discriminator discriminator;
    Generator generator;
    DataSetIterator trainIter;



    public static void main(String[] args) throws IOException {

        // Set seed for reproducibility
        long seed = 1234;
        Random random = new Random(seed);

        // Check if GPU is available
        int gpuCount = Engine.getInstance().getGpuCount();
        NDManager manager = NDManager.newBaseManager();
        if (gpuCount > 0) {
            // Use GPU for computations
            System.out.println("GPU available");
            manager = manager.newSubManager(Device.gpu());
        } else {
            // Use CPU for computations
            System.out.println("GPU not available");
            manager = manager.newSubManager(Device.cpu());
        }



        // Creating the data transformation pipeline
        DataNormalization scaler = new ImagePreProcessingScaler(-0.5, 0.5);
        int batchSize = 32;
        int numExamples = 60000; // number of examples in the MNIST training set
        boolean binarize = true;
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, numExamples, binarize);
        trainIter.setPreProcessor(scaler);


        // Loading the MNIST dataset
        DataSet ds = trainIter.next();
        INDArray features = ds.getFeatures();
        INDArray labels = ds.getLabels();

        // Displaying the first 16 samples
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("MNIST Samples");
            frame.setLayout(new GridLayout(4, 4));
            for (int i = 0; i < 16; i++) {
                BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
                for (int j = 0; j < 28 * 28; j++) {
                    int x = j % 28;
                    int y = j / 28;
                    int value = (int) (255 * (features.getDouble(i, j) + 0.5));
                    img.setRGB(x, y, value << 16 | value << 8 | value);
                }
                JLabel label = new JLabel(new ImageIcon(img));
                JPanel panel = new JPanel();
                panel.add(label);
                frame.add(panel);
            }
            frame.pack();
            frame.setVisible(true);
        });



        // Creating the Generator and Discriminator instances
        Discriminator discriminator = new Discriminator();
        Generator generator = new Generator();



    }

//    public void train(int num_epochs, float lr, int batch_size) {
//        // Set up loss function and optimizers
//        LossFunction loss_function = LossFunctions.LossFunctionBuilder.buildLossFunctionByString("BinaryCrossentropy");
//        Adam optimizer_discriminator = new Adam(lr);
//        Adam optimizer_generator = new Adam(lr);
//
//        List<BufferedImage> images = new ArrayList<>();
//
//        for (int epoch = 0; epoch < num_epochs; epoch++) {
//            System.out.println("Epoch " + epoch);
//
//            for (int n = 0; n < trainIter.numExamples() / batch_size; n++) {
//                // Data for training the discriminator
//                INDArray real_samples = trainIter.next().getFeatures();
//                INDArray real_samples_labels = Nd4j.ones(batch_size, 1);
//                INDArray latent_space_samples = Nd4j.randn(batch_size, 100);
//                INDArray generated_samples = generator.forward(latent_space_samples);
//                INDArray generated_samples_labels = Nd4j.zeros(batch_size, 1);
//                INDArray all_samples = Nd4j.vstack(real_samples, generated_samples);
//                INDArray all_samples_labels = Nd4j.vstack(real_samples_labels, generated_samples_labels);
//
//                // Training the discriminator
//                discriminator.setParams(optimizer_discriminator, loss_function);
//                double loss_discriminator = discriminator.train(all_samples, all_samples_labels);
//
//                // Data for training the generator
//                latent_space_samples = Nd4j.randn(batch_size, 100);
//
//                // Training the generator
//                generator.setParams(optimizer_generator, loss_function);
//                double loss_generator = generator.train(latent_space_samples, Nd4j.ones(batch_size, 1));
//
//                // Print losses
//                if (n % 100 == 0) {
//                    System.out.println("Discriminator loss: " + loss_discriminator + ", Generator loss: " + loss_generator);
//                }
//
//                // Generate sample images from the generator
//                if (n % 1000 == 0) {
//                    INDArray latent_space_samples_sample = Nd4j.randn(16, 100);
//                    INDArray generated_samples_sample = generator.forward(latent_space_samples_sample);
//                    generated_samples_sample = generated_samples_sample.mul(0.5).add(0.5); // scale to [0, 1]
//                    for (int i = 0; i < 16; i++) {
//                        BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
//                        WritableRaster raster = img.getRaster();
//                        for (int j = 0; j < 28 * 28; j++) {
//                            int x = j % 28;
//                            int y = j / 28;
//                            int value = (int) (255 * generated_samples_sample.getDouble(i, j));
//                            raster.setSample(x, y, 0, value);
//                        }
//                        images.add(img);
//                    }
//                }
//            }
//        }
//
//        // Display sample images
//        JFrame frame = new JFrame("GAN Samples");
//        frame.setLayout(new GridLayout(4, 4));
//        for (BufferedImage img : images) {
//            JLabel label = new JLabel(new ImageIcon(img));
//            JPanel panel = new JPanel();
//            panel.add(label);
//            frame.add(panel);
//        }
//        frame.pack();
//        frame.setVisible(true);
//    }
}
