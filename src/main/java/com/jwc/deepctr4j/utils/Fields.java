package com.jwc.deepctr4j.utils;

/**
 * 
 * @author craig
 *
 */
public class Fields {

	/**
	 * 
	 * @param fieldName
	 * @return
	 */
	public static Field newNumeric(String fieldName) {
		return new Field(FeatureType.NUMERIC, fieldName);
	}
	
	/**
	 * 
	 * @param fieldName
	 * @return
	 */
	public static Field newCategorical(String fieldName) {
		return new Field(FeatureType.CATEGORICAL, fieldName);
	}
	
	/**
	 * 
	 * @param fieldName
	 * @return
	 */
	public static Field newMultiCategorical(String fieldName) {
		return new Field(FeatureType.MULTI_CATEGORICAL, fieldName);
	}
	
	/**
	 * 
	 * @param fieldName
	 * @return
	 */
	public static Field newLabel(String fieldName) {
		return new Field(FeatureType.LABEL, fieldName);
	}
	

}
