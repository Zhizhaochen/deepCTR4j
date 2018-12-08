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
 * 实现DeepFM二次项部分的操作, multiple inputs, single output layers usable only in ComputationGraph, without any parameters.
 * 
 * @author craig
 *
 */
@SuppressWarnings("serial")
@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class FMLayer extends SameDiffLambdaVertex {
	
	private int fieldSize;
	private int fieldEmbeddingSize;
	
	/**
	 * TODO
	 * May be more faster...
	 */
	@Override
	public SDVariable defineVertex(SameDiff sd, VertexInputs inputs) {
		
		SDVariable sumAndSquare = sd.zero("sumAndSquare", new int[] {fieldEmbeddingSize});
		SDVariable squareAndSum = sd.zero("squareAndSum", new int[] {fieldEmbeddingSize});
		for (int j = 0; j < fieldSize; j++) {
			SDVariable vi = inputs.getInput(j);
			sumAndSquare = sumAndSquare.add(vi);
			SDVariable square = sd.square(vi);
			squareAndSum = squareAndSum.add(square);
		}
		sumAndSquare = sd.square(sumAndSquare);
		return sumAndSquare.sub("sub", squareAndSum).div(2);
	}

	
	
}


