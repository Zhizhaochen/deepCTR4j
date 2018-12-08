package com.jwc.deepctr4j.utils;

/**
 * 
 * @author craig
 *
 */
public enum FeatureType {
	
	/**
	 * 单值离散
	 */
	CATEGORICAL, 
	
	/**
	 * 连续数值型.
	 */
	NUMERIC, 
	
	/**
	 * 多值离散
	 */
	MULTI_CATEGORICAL,

	/**
	 * y列, 预测目标
	 */
	LABEL;
	
}
