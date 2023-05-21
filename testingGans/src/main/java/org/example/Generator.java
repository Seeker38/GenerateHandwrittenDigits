package org.example;


import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;

class Generator extends SequentialBlock {
    private static final int LATENT_DIM = 100;
    private Device device;
    Generator() {
        super();
        Block block = Linear.builder().setUnits(128).build();
        addChildBlock("linear1", block);
        block = Activation.reluBlock();
        addChildBlock("relu1", block);

        block = Linear.builder().setUnits(256).build();
        addChildBlock("linear2", block);
        block = Activation.reluBlock();
        addChildBlock("relu2", block);

        block = Linear.builder().setUnits(784).build();
        addChildBlock("linear3", block);
        block = Activation.tanhBlock();
        addChildBlock("tanh", block);
    }
    Generator(Device device) {
        super();
        this.device = device;

        Block block = Linear.builder().setUnits(128).build();
        addChildBlock("linear1", block);
        block = Activation.reluBlock();
        addChildBlock("relu1", block);

        block = Linear.builder().setUnits(256).build();
        addChildBlock("linear2", block);
        block = Activation.reluBlock();
        addChildBlock("relu2", block);

        block = Linear.builder().setUnits(784).build();
        addChildBlock("linear3", block);
        block = Activation.tanhBlock();
        addChildBlock("tanh", block);
    }

    NDList generateSamples(int batchSize) {
        NDList latentSamples = generateLatentSamples(batchSize);
        return forward(null, latentSamples, false, null);
    }

    NDList generateLatentSamples(int batchSize) {
        NDManager manager = NDManager.newBaseManager();
        Shape shape = new Shape(batchSize, LATENT_DIM);
        NDArray randomSamples = manager.randomUniform(0, 1, shape, DataType.FLOAT32);
        return new NDList(randomSamples);
    }

    NDList getGeneratedLabels(int batchSize) {
        NDManager manager = NDManager.newBaseManager();
        NDArray ones = manager.ones(new Shape(batchSize, 1), DataType.FLOAT32);
        return new NDList(ones);
    }
}