package com.jwc.deepctr4j.utils;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.jwc.deepctr4j.layer.DCNLayer;
import com.jwc.deepctr4j.layer.FMLayer;
import com.jwc.deepctr4j.layer.MeanPoolLayer;

/**
 * 各算法的实现，方便调参，不对模型搭积木的过程做过度封装
 * 
 * @author craig
 *
 */
public class ModelDesigner {

	public static ComputationGraph DCN(int in, int out, int nLayer, int crossLayerEmbedingSize) {
		
		ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
				.trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.activation(Activation.RELU)
				.updater(new Adam(0.01))
				.weightInit(WeightInit.XAVIER)
				.graphBuilder()
				.addInputs("input_f_1", "input_f_2", "input_f_3", "input_f_4")
				
				.addLayer("emb_f_1", LayerUtils.embedding(10, 6), "input_f_1")
				.addLayer("emb_f_2", LayerUtils.embedding(100, 6), "input_f_2")
				.addLayer("emb_f_3", LayerUtils.embedding(200, 6), "input_f_3")
				.addLayer("emb_f_4", LayerUtils.embedding(300, 6), "input_f_4")
				.addVertex("emb_merge", new MergeVertex(), "emb_f_1", "emb_f_2", "emb_f_3", "emb_f_4") // Perform depth concatenation
				// Cross part.
				.addLayer("cross", new DCNLayer(true, 24, Activation.TANH, WeightInit.XAVIER), "emb_merge")
				// Deep part.
				.addLayer("dense_1st", LayerUtils.dense(24, 100), "emb_merge")
				.addLayer("dense_2nd", LayerUtils.dense(100, 100), "dense_1st")
				.addLayer("dense_3th", LayerUtils.dense(100, 100), "dense_2nd")
				
				// Perform depth concatenation.
				.addVertex("deep_cross_merge", new MergeVertex(), "cross", "dense_3th") 
				// 标准OutputLayer.  
				.addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.activation(Activation.SOFTMAX)
						.nIn(124).nOut(out)
						.build(), "deep_cross_merge") 
				.setOutputs("out")
//				.setInputTypes(InputType.feedForward(10), InputType.feedForward(10))
				.build();
		
		ComputationGraph net = new ComputationGraph(config);
		net.init();
		
		return net;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param nLayer
	 * @param crossLayerEmbedingSize
	 * @return
	 */
	public static ComputationGraph DeepFM(int in, int out, int nLayer, int fieldSize, int fieldEmbedingSize, int deepEmbeddingSize) {
		
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.activation(Activation.RELU)
				.updater(new Adam(0.01))
				.weightInit(WeightInit.XAVIER)
				.graphBuilder()
				.addInputs("input_1", "input_2", "input_3", "input_4", "input_5", "input_6")
				
				/*-----------------------------------------一次项部分----------------------------------------*/
				//数值
				.addLayer("emb_linear_1", LayerUtils.dense(1, 1), "input_1")
				//单值离散
				.addLayer("emb_linear_2", LayerUtils.embedding(13, 1), "input_2")
				.addLayer("emb_linear_3", LayerUtils.embedding(5, 1), "input_3")
				.addLayer("emb_linear_4", LayerUtils.embedding(3, 1), "input_4")
				//多值离散
				.addLayer("emb_linear_5", LayerUtils.embeddingSequence(15, 4, 1), "input_5")
				.addLayer("mean_pool_linear_5", new MeanPoolLayer(), "emb_5")
				//单值离散
				.addLayer("emb_linear_6", LayerUtils.embedding(1988, 1), "input_6")
				.addVertex("fm_linear", new MergeVertex(), "emb_linear_1", "emb_linear_2", "emb_linear_3", "emb_linear_4", "mean_pool_5", "emb_linear_6")   
				
				/*-----------------------------------------二次项部分----------------------------------------*/
				//数值
				.addLayer("emb_poly_1", LayerUtils.dense(1, fieldEmbedingSize), "input_1")
				//单值离散
				.addLayer("emb_poly_2", LayerUtils.embedding(13, fieldEmbedingSize), "input_2")
				.addLayer("emb_poly_3", LayerUtils.embedding(5, fieldEmbedingSize), "input_3")
				.addLayer("emb_poly_4", LayerUtils.embedding(3, fieldEmbedingSize), "input_4")
				//多值离散
				.addLayer("emb_poly_5", LayerUtils.embeddingSequence(15, 4, 1), "input_5")
				.addLayer("mean_pool_poly_5", new MeanPoolLayer(), "emb_5")
				//单值离散
				.addLayer("emb_poly_6", LayerUtils.embedding(1988, fieldEmbedingSize), "input_6")
				.addVertex("fm_poly", new FMLayer(fieldSize, fieldEmbedingSize), "emb_poly_1", "emb_poly_2", "emb_poly_3", "emb_poly_4", "mean_pool_poly_5", "emb_poly_6")
				
				/*-----------------------------------------Deep部分----------------------------------------*/
				.addVertex("deep_merge", new MergeVertex(), "emb_poly_1", "emb_poly_2", "emb_poly_3", "emb_poly_4", "mean_pool_poly_5", "emb_poly_6")   
				.addLayer("deep_1_out", LayerUtils.dense(fieldSize * fieldEmbedingSize, deepEmbeddingSize), "deep_merge")
				.addLayer("deep_2_out", LayerUtils.dense(deepEmbeddingSize, deepEmbeddingSize), "deep_1_out")
				
				/*----------------------------------------Sigmod部分-----------------------------------------*/
				.addVertex("merge_sigmod", new MergeVertex(), "fm_linear", "fm_poly", "deep_2_out")   
				// 标准OutputLayer.  
				.addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
						.activation(Activation.SIGMOID)
						.nIn(fieldSize + fieldEmbedingSize + deepEmbeddingSize)
						.nOut(1)
						.build(), "merge_sigmod") 
				.setOutputs("out")
                .build();
		
		ComputationGraph net = new ComputationGraph(conf);
        net.init();
        return net;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param nLayer
	 * @param crossLayerEmbedingSize
	 * @return
	 */
	public static ComputationGraph PNN(int in, int out, int nLayer, int crossLayerEmbedingSize) {
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.activation(Activation.RELU)
				.updater(new Adam(0.01))
				.weightInit(WeightInit.XAVIER)
				.graphBuilder()
				//TODO ....
				.build();
		ComputationGraph net = new ComputationGraph(conf);
        net.init();
        return net;
	}
		
	
}


