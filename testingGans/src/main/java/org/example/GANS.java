package org.example;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;


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


import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.ImageIcon;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class GANS {
    public static void main(String[] args) throws IOException {

        // Setting the random seed
        Random rand = new Random(111);
        Nd4j.getRandom().setSeed(111);

        // Checking if GPU is available
        String device = "";
        if (Nd4j.getAffinityManager().getNumberOfDevices() > 0) {
            device = "cuda";
            System.out.println("GPU available");
        } else {
            device = "cpu";
            System.out.println("GPU not available");
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



        Discriminator discriminator = new Discriminator();

        
    }
}
