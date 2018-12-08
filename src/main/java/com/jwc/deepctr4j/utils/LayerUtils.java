package com.jwc.deepctr4j.utils;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

/**
 * 
 * 对输入层的封装
 * 
 * @author craig
 *
 */
public class LayerUtils {

	
	public static void inputLayer() {
		
	}
	
	/**
	 * 
	 * @return
	 */
	public static EmbeddingLayer embedding(int in, int out) {
		return new EmbeddingLayer.Builder().nIn(in).nOut(out).build();
	}
	
	/**
	 * 
	 * @return
	 */
	public static DenseLayer dense(int in, int out) {
		return new DenseLayer.Builder().activation(Activation.TANH).nIn(in).nOut(out).build();
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 */
	public static EmbeddingSequenceLayer embeddingSequence(int in, int inputLength, int out) {
		return new EmbeddingSequenceLayer.Builder().nIn(in).inputLength(inputLength).nOut(out)
	        .activation(Activation.IDENTITY)
	        .weightInit(WeightInit.XAVIER)
	        .build();
	}
	
	

}
