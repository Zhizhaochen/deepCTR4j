package com.jwc.deepctr4j.layer;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import lombok.NoArgsConstructor;

/**
 * 
 * @author craig
 *
 */
@NoArgsConstructor
@SuppressWarnings("serial")
public class MeanPoolLayer extends SameDiffLambdaLayer {

	@Override
	public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
		return sameDiff.mean(layerInput, 2);
	}
	
}
