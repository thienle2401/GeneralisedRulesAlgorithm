/*
 * Copyright (C) 2017 Thien Le, Frederic Stahl - University of Reading, UK
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package uk.ac.rdg.bdata;

import java.util.*;
import java.util.Map.Entry;


public class Utilities {
	
	/**
	 * Parse attribute-value in form [attributeIndex, attributeValue]
	 * [type, attributeIndex, attriubteLux] or [type, ]
	 * 
	 * 
	 * @param termList	List of all possible terms/ conditions for the dataset
	 * @param attributeValueDobule Term/ condition in form of [attributeIndex, attributeValue]
	 * @return
	 */
	public static GRules.Term parseTermFromDoubleArray(List<GRules.Term> termList, double[] attributeValueDobule){
		
		Iterator<GRules.Term> termsIterator = termList.iterator();
		
		while(termsIterator.hasNext()){
			
                    GRules.Term term = termsIterator.next();

                    if(term.isMatching(attributeValueDobule)){
                        return term;
                    }	
		}
		return null;
	}
        
        public static GRules.Term parseTermFromString(List<GRules.Term> termList, String termString){
            
            String[] splittedString= termString.split(";");
            double[] doubleVector = new double[splittedString.length];
 
            for (int i = 0; i < splittedString.length; i++) {
                doubleVector[i] = Double.parseDouble(splittedString[i]);
            }
            
            Iterator<GRules.Term> termsIterator = termList.iterator();
 
            while(termsIterator.hasNext()){
                GRules.Term term = termsIterator.next();
                if(term.isMatching(doubleVector)){
                    return term;
                }	
            }
            
            return null;
        }
	
	/**
	 * Compare two attributeValue double[] in form of [attributeIndex, attributeValue]
	 * @param doubleVector1
	 * @param doubleVector2
	 * @return true if two attributeValue are identical
	 */
	public static boolean compareTwoDoubleAttributeValue(double[] doubleVector1, double[] doubleVector2){
		
		if(doubleVector1[0] == doubleVector2[0] && doubleVector1[1] == doubleVector2[1]){
			return true;
		}
		
		return false;
	}
	
	public static double[] parseStringToDoubleArray(String stringIn){
		
		String doubleString = stringIn;
		String[] splittedString= doubleString.split(";");
		
		double[] doubleVector = new double[splittedString.length];
		
		for (int i = 0; i < splittedString.length; i++) {
			doubleVector[i] = Double.parseDouble(splittedString[i]);
		}
		
		return doubleVector;
	}
	
	public static String keyWithBestValueFromMap(Map<String,Double> mapIn){
		
		Set<Entry<String, Double>> mapSet = mapIn.entrySet();
		Iterator<Entry<String, Double>> entryIterator = mapSet.iterator();
		
		String currentHighestKey = null;
		Double currentHighestValue = null;
		
		while(entryIterator.hasNext()){
			Entry<String, Double> anEntry = entryIterator.next();
			
			if(currentHighestKey == null){
				currentHighestKey = anEntry.getKey();
				currentHighestValue = anEntry.getValue();
			}else{
				if(anEntry.getValue() >= currentHighestValue){
					currentHighestKey = anEntry.getKey();
					currentHighestValue = anEntry.getValue();
				}
			}
			
		}
		
		return currentHighestKey;
		
	}
        
        private String printRule(){
            return "";
        }
}
