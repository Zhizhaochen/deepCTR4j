package com.jwc.deepctr4j.core;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.google.common.base.Charsets;
import com.jwc.deepctr4j.layer.MeanPoolLayer;
import com.jwc.deepctr4j.utils.Field;
import com.jwc.deepctr4j.utils.Fields;
import com.jwc.deepctr4j.utils.ModelUtils;

import lombok.NoArgsConstructor;

/**
 * 
 * @author craig
 *
 */
public class DeepCTRProblem {
	
	private List<Field> fieldList;
	private int labelIndex = -1;

	public DeepCTRProblem schema (List<Field> fieldList) throws IOException {
		if(fieldList.size() < 2) {
			throw new IOException("size of fieldList should be large than 2.");
		}
		for (int i = 0; i < fieldList.size(); i++) {
			
		}
		return this;
	} 
	
	public static void main(String[] args) throws IOException {
		
		DeepCTRProblem problem = new DeepCTRProblem();
		problem.schema(Fields.newCategorical(""), Fields.newCategorical(""), Fields.newLabel("y") );
	}
	
	
	public DeepCTRProblem() throws IOException {
		
		List<String> corpus = FileUtils.readLines(new File("/Users/craig/fineFile"), Charsets.UTF_8);
		CorpusIterator iter = new CorpusIterator(corpus, 32);
		
        ComputationGraphConfiguration conf = ModelUtils.initBasicNNConfiguration()
				.addInputs("input_1", "input_2", "input_3", "input_4", "input_5", "input_6")
				//数值
				.addLayer("emb_1", new DenseLayer.Builder().nIn(1).nOut(1).build(), "input_1")
				//单值离散
				.addLayer("emb_2", new EmbeddingLayer.Builder().nIn(13).nOut(1).build(), "input_2")
				.addLayer("emb_3", new EmbeddingLayer.Builder().nIn(5).nOut(1).build(), "input_3")
				.addLayer("emb_4", new EmbeddingLayer.Builder().nIn(3).nOut(1).build(), "input_4")
				//多值离散
				.addLayer("emb_5", new EmbeddingSequenceLayer.Builder().nIn(15).inputLength(4).nOut(1)
		                .activation(Activation.IDENTITY) 
		                .weightInit(WeightInit.XAVIER) 
		                .build(),
		                "input_5") 
				.addLayer("mean_pool_5", new MeanPoolLayer(), "emb_5")
				//单值离散
				.addLayer("emb_6", new EmbeddingLayer.Builder().nIn(1988).nOut(1).build(), "input_6")
				.addVertex("emb_merge", new MergeVertex(), "emb_1", "emb_2", "emb_3", "emb_4", "mean_pool_5", "emb_6") 
				
				// 标准OutputLayer.  
				.addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.XENT) 
						.activation(Activation.SIGMOID) 
						.nIn(6) 
						.nOut(1) 
						.build(), "emb_merge") 
				.setOutputs("out") 
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        System.out.println(net.summary());
        net.fit(iter, 10);
	}

}
