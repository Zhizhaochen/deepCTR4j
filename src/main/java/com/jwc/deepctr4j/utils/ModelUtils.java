package com.jwc.deepctr4j.utils;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

/**
 * 
 * @author craig
 *
 */
public class ModelUtils {

	public static ComputationGraphConfiguration.GraphBuilder initBasicNNConfiguration() {

		return new NeuralNetConfiguration.Builder()
				.trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.activation(Activation.RELU)
				.updater(new Adam(0.01))
				.weightInit(WeightInit.XAVIER)
				.graphBuilder();
	}

	public ModelUtils() {
	}

}
