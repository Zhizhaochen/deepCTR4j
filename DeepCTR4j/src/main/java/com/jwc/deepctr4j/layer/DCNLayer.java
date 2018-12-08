package com.jwc.deepctr4j.layer;

import java.util.Map;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * DCN层的实现
 * 
 * @author craig
 *
 */
@NoArgsConstructor
@Getter
@Setter
public class DCNLayer extends SameDiffLayer {

	private int embeddingSize;
	private Activation activation;
	private boolean isBottom = false;
	
	public static INDArray x0;
	
	/**
    *
    * @param nIn        Number of inputs to the layer
    * @param nOut       Number of outputs - i.e., layer size
    * @param activation Activation function
    * @param weightInit Weight initialization for the weights
    */
   public DCNLayer(boolean isBottom, int embeddingSize, Activation activation, WeightInit weightInit){
	   this.embeddingSize = embeddingSize;
       this.activation = activation;
       this.weightInit = weightInit;
       this.isBottom = isBottom;
   }
   
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public InputType getOutputType(int layerIndex, InputType inputType) {
		return InputType.feedForward(embeddingSize);
	}
	
	/**
	 * 定义参数
	 */
	@Override
	public void defineParameters(SDLayerParams params) {
		
		params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, embeddingSize, 1);
		params.addBiasParam(DefaultParamInitializer.BIAS_KEY, embeddingSize, 1);
	}

	/**
	 * 初始化参数
	 */
	@Override
	public void initializeParameters(Map<String, INDArray> params) {
		params.get(DefaultParamInitializer.BIAS_KEY).assign(0);
		initWeights(embeddingSize, 1, weightInit, params.get(DefaultParamInitializer.WEIGHT_KEY));
	}

	/**
	 * Call method
	 */
	@Override
	public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
		
		SDVariable layerWeights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable layerBias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

		SDVariable cross = null;
        if(isBottom) {
        	SDVariable inputTrans = sd.transpose(layerInput);
        	DCNLayer.x0 = inputTrans.getArr();
        	cross = inputTrans.mmul(layerInput.mmul(layerWeights)).add(layerBias).add(inputTrans);
        }else {
        	SDVariable mm_1 = layerInput.mmul("mm_1", layerWeights);
        	SDVariable mm_2 = sd.var(DCNLayer.x0).mmul("mm_2", mm_1);
        	cross = mm_2.add(layerBias).add(sd.transpose(layerInput));
        }
        return activation.asSameDiff("cross_out", sd, sd.transpose(cross));
	}

	public static void main(String[] args) {
	}

}
