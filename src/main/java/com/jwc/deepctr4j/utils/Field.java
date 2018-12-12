package com.jwc.deepctr4j.utils;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Field {
	
	private FeatureType featureType;
	private String fieldName;
	
	public static Field create(FeatureType featureType, String fieldName) {
		return new Field(featureType, fieldName);
	}
	
}
