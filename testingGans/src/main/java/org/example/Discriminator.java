package org.example;

import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;

import ai.djl.nn.Activation;

class Discriminator extends SequentialBlock {
    Discriminator() {
        super();
        Block block = Linear.builder().setUnits(256).build();
        addChildBlock("linear1", block);
        block = Activation.reluBlock();
        addChildBlock("relu1", block);
        block = Dropout.builder().optRate(0.5f).build();
        addChildBlock("dropout1", block);

        block = Linear.builder().setUnits(128).build();
        addChildBlock("linear2", block);
        block = Activation.reluBlock();
        addChildBlock("relu2", block);
        block = Dropout.builder().optRate(0.5f).build();
        addChildBlock("dropout2", block);

        block = Linear.builder().setUnits(1).build();
        addChildBlock("linear3", block);
        block = Activation.sigmoidBlock();
        addChildBlock("sigmoid", block);
    }
}