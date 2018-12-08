package com.jwc.deepctr4j.core;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.google.common.collect.Lists;

import jodd.util.StringUtil;
import lombok.Getter;
import lombok.Setter;

/**
 * 
 * @author craig
 *
 */
@SuppressWarnings("serial")
@Getter
@Setter
public class CorpusIterator implements MultiDataSetIterator {

	private ConfigurationContext configurationContext;
	
	private List<String> corpus;
	private MultiDataSetPreProcessor preProcessor;
	private final int batchSize;
	private final int totalBatches;
	private int currentBatch = 0;

	/*---------------------------------------------------------------------------------*/
	public CorpusIterator(List<String> corpus, int batchSize) {
		this.corpus = corpus;
		this.batchSize = batchSize;
		this.totalBatches = (int) Math.ceil((double) corpus.size() / batchSize);
	}

	@Override
	public boolean hasNext() {
		return currentBatch < totalBatches;
	}

	@Override
	public MultiDataSet next() {
		return next(batchSize);
	}

	@Override
	public MultiDataSet next(int sampleSize) {

		int i = currentBatch * batchSize;
		int currentBatchSize = Math.min(batchSize, corpus.size() - i - 1);
		
		INDArray input_age = Nd4j.zeros(currentBatchSize, 1);
		INDArray input_zodiac = Nd4j.zeros(currentBatchSize, 1);
		INDArray input_bloodType = Nd4j.zeros(currentBatchSize, 1);
		INDArray input_gender = Nd4j.zeros(currentBatchSize, 1);
		INDArray input_hoby = Nd4j.zeros(currentBatchSize, 4);  //多值离散, padding = 4
		INDArray input_item = Nd4j.zeros(currentBatchSize, 1);
		INDArray prediction = Nd4j.zeros(currentBatchSize, 1);
		INDArray input_hoby_mask = Nd4j.zeros(currentBatchSize, 4);
		
		for (int j = 0; j < currentBatchSize; j++) {
			
			String rowIn = corpus.get(j);
			String [] featureArr = StringUtil.split(rowIn, " ");
			
			input_age.putScalar(j, 0, Double.parseDouble(featureArr[0]));
			input_zodiac.putScalar(j, 0, Double.parseDouble(featureArr[1]));
			input_bloodType.putScalar(j, 0, Double.parseDouble(featureArr[2]));
			input_gender.putScalar(j, 0, Double.parseDouble(featureArr[3]));
			input_item.putScalar(j, 0, Double.parseDouble(featureArr[5]));
			prediction.putScalar(j, 0, Double.parseDouble(featureArr[6]));
			//处理多值离散
			Double [] hobyArr = Lists.newArrayList(StringUtil.split(featureArr[4], ",")).stream()
					.map(s -> Double.valueOf(s)).toArray(Double[]::new);
			//一次性替换
			input_hoby.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, hobyArr.length) }, 
					Nd4j.create(ArrayUtils.toPrimitive(hobyArr)));
			//一次性替换
			input_hoby_mask.put(new INDArrayIndex[] { NDArrayIndex.point(j), NDArrayIndex.interval(0, hobyArr.length) }, 
					Nd4j.ones(hobyArr.length));
		}
		
		++currentBatch;
		return new org.nd4j.linalg.dataset.MultiDataSet(
				new INDArray[] { input_age, input_zodiac, input_bloodType, input_gender, input_hoby, input_item }, 
				new INDArray[] { prediction },
				new INDArray[] { null, null, null, null, input_hoby_mask, null }, 
				null);
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
		this.preProcessor = preProcessor;
	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		return null;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public void reset() {
		// but we still can do it manually before the epoch starts
		currentBatch = 0;
	}

	public int batch() {
		return currentBatch;
	}

	public int totalBatches() {
		return totalBatches;
	}

}
