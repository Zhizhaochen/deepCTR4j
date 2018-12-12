package com.jwc.deepctr4j.layer;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 
 * @author craig
 *
 */
@SuppressWarnings("serial")
@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class PNNLayer extends SameDiffLambdaVertex {

	private int fieldSize;
	private int fieldEmbeddingSize;
	
	@Override
	public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
		// TODO Auto-generated method stub
		return null;
	}

}






