//package org.example;
//import ai.djl.ModelException;
//import ai.djl.training.dataset.Batch;
//import ai.djl.training.dataset.RandomAccessDataset;
//import ai.djl.training.loss.Loss;
//import ai.djl.translate.TranslateException;
//import ai.djl.util.ZipUtils.EntryFilter;
//import ai.djl.util.ZipUtils.ZipEntryProcessor;
//import ai.djl.util.ZipUtils.ZipOutputStreamBuilder;
//
//import java.io.IOException;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.List;
//
//public class GANS {
//
//    private static final int LATENT_DIM = 100;
//    private static final int BATCH_SIZE = 32;
//    private static final int EPOCHS = 50;
//    private static final float LEARNING_RATE = 0.0001f;
//    private static final int NUM_SAMPLES = 16;
//
//    public static void main(String[] args) throws IOException, ModelException, TranslateException {
//        // Set random seed for reproducibility
//        System.setProperty("ai.djl.pytorch.randomSeed", "123");
//
//        // Select device based on availability of a GPU
//        String device = "cpu";
//        if (Boolean.parseBoolean(System.getProperty("ai.djl.pytorch.useGpu", "false"))) {
//            device = "cuda";
//        }
//
//        // Define image transformations
//        ImageTransform transform = new ImageTransform.Builder()
//                .add(new ToTensor())
//                .add(new Normalize(0.5f, 0.5f))
//                .build();
//
//        // Load MNIST training dataset
//        RandomAccessDataset trainSet = HandwrittenDigitsGenerator.randomTrainSet(60000, transform);
//
//        // Create DataLoader for training dataset
//        DataLoader trainLoader = trainSet.getData(trainSet.getBatchSize(), true);
//
//        // Display a 4x4 grid of real samples
//        displaySamples(trainLoader, 4, 4);
//
//        // Create Discriminator and move it to the selected device
//        Discriminator discriminator = new Discriminator();
//        discriminator.setDevice(device);
//
//        // Create Generator and move it to the selected device
//        Generator generator = new Generator();
//        generator.setDevice(device);
//        // Define loss function
//        Loss lossFunction = new SigmoidBinaryCrossEntropyLoss();
//
//        // Create optimizers
//        Adam optimizerDiscriminator = Adam.builder()
//                .optLearningRate(LEARNING_RATE)
//                .build();
//        Adam optimizerGenerator = Adam.builder()
//                .optLearningRate(LEARNING_RATE)
//                .build();
//
//        // Create empty list to store generated images
//        List<Image> images = new ArrayList<>();
//
//        // Main training loop
//        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
//            for (Batch batch : trainLoader) {
//                NDList realSamples = batch.getData().singletonOrThrow();
//                NDList mnistLabels = batch.getLabels().singletonOrThrow();
//
//                // Move data to the selected device
//                realSamples.attach(device);
//                mnistLabels.attach(device);
//
//                // Generate random samples from latent space
//                NDList latentSpaceSamples = generator.generateLatentSamples(realSamples.size());
//
//                try (GradientCollector collector = trainer.newGradientCollector()) {
//                    // Train discriminator
//                    NDList discriminatorInput = new NDList(realSamples, latentSpaceSamples);
//                    NDList discriminatorOutput = discriminator.forward(discriminatorInput);
//
//                    NDList realSamplesLabels = new NDList(mnistLabels);
//                    NDList generatedSamplesLabels = new NDList(generator.getGeneratedLabels(realSamples.size()));
//
//                    NDList discriminatorLoss = lossFunction.evaluate(discriminatorOutput, realSamplesLabels);
//                    discriminatorLoss.backward();
//                    optimizerDiscriminator.step();
//                    collector.backward(discriminatorLoss.get(0));
//
//                    // Train generator
//                    NDList generatedSamples = generator.generateSamples(realSamples.size());
//                    NDList generatorOutput = discriminator.forward(new NDList(generatedSamples));
//                    NDList generatorLoss = lossFunction.evaluate(generatorOutput, realSamplesLabels);
//                    generatorLoss.backward();
//                    optimizerGenerator.step();
//                    collector.backward(generatorLoss.get(0));
//                }
//            }
//
//            // Print loss for last batch of the epoch
//            System.out.printf("Epoch [%d/%d], Discriminator Loss: %.4f, Generator Loss: %.4f%n",
//                    epoch, EPOCHS, discriminatorLoss.getFloat(0), generatorLoss.getFloat(0));
//
//            // Generate and store batch of new images
//            NDList generatedImages = generator.generateSamples(NUM_SAMPLES);
//            for (int i = 0; i < NUM_SAMPLES; i++) {
//                Image generatedImage = ImageFactory.getInstance().fromNDArray(generatedImages.singletonOrThrow().get(i));
//                images.add(generatedImage.toImage());
//            }
//        }
//
//        // Save generated images as animated GIF
//        Path outputPath = Paths.get("generated_images.gif");
//        ImageVisualization.saveAsGif(images, outputPath, 5);
//    }
//}
//

