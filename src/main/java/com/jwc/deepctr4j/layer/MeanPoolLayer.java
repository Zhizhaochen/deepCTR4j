package com.jwc.deepctr4j.layer;

import java.util.Arrays;

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
		
		System.out.println(layerInput.getArr());
		SDVariable mean = sameDiff.mean(layerInput, 2);
		System.out.println("after: " + Arrays.toString(mean.getShape()));
		return mean;
	}
	
}
