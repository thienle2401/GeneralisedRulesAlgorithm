/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uk.ac.rdg.bdata;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.Frequency;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Loggers;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author tle
 */
public class GRules {
    
    private static final long serialVersionUID = 1L;
    private static final Logger myLogger = LogManager.getLogger(GRules.class);
    
    final static int wraPrunning = 1;
    final static int jmeasurePrunning = 2;
    
    private final int prunningOption = GRules.jmeasurePrunning;
    
    private final int headMaxTerms =  1;
    private final double headCoverageThreshold = 0.2;
    
    public GRules(){
    
    }
    public List<Rule> learnStreamingRules(Instances streamingExamples) throws Exception{
        
         // train the rules learner
        List<List<Term>> rightSideRules = learnRightSide(streamingExamples);
        List<Rule> completedRules = induceCompleteRules(rightSideRules, streamingExamples);       
        
        return completedRules;
    }
    
    public double learRulesEvaluation(Instances streamingExamples) throws Exception{
        
        List<List<Term>> rightSideRules = learnRightSide(streamingExamples);
        List<Rule> completedRules = induceCompleteRules(rightSideRules, streamingExamples);  
        
        Iterator<Rule> rulesIterator;
        List<Rule> kBestRules = null;
                
        // get k best rules
        for(int kRules = 10; kRules <= 50; kRules = kRules + 5){
            
            TreeMap<Double, Rule> heuristicSortedTreeMap = new TreeMap<>(Collections.reverseOrder());

            rulesIterator = completedRules.iterator();

            while(rulesIterator.hasNext()){

                Rule aRule = rulesIterator.next();

                double selectingHeuristic = calcualteWRA(streamingExamples, aRule.getLeftSide(), aRule.getRightSdie());
//                double selectingHeuristic = calculateSupport(dataset, aRule.getLeftSide());
//                double selectingHeuristic = calcualteRuleJMeasure(dataset, aRule.getLeftSide(), aRule.getRightSdie());
//                double selectingHeuristic = calcualteConfidence(dataset, aRule.getLeftSide(), aRule.getRightSdie());
                
//                myLogger.trace(aRule.nicePrint());
                heuristicSortedTreeMap.put(selectingHeuristic, aRule);            
            }        

            kBestRules = new ArrayList<>();

            // get k best rules
            Set<Map.Entry<Double,Rule>> wraSortedTreeMapSet = heuristicSortedTreeMap.entrySet();
            Iterator<Map.Entry<Double,Rule>> heuristicSortedTreeMapSetIterator = wraSortedTreeMapSet.iterator();

            int ruleCount = 0;
            while(heuristicSortedTreeMapSetIterator.hasNext() && ruleCount < kRules){
                Map.Entry<Double,Rule> anEntry = heuristicSortedTreeMapSetIterator.next();
                Rule aRule = anEntry.getValue();

                kBestRules.add(aRule);
                ruleCount++;
            }

            double kRulesTruePositiveRate = 100 * calcualteTruePostiveRate(streamingExamples, kBestRules);
            double kRulesFalsePositiveRate = 100 * calcuateFalsePositiveRate(streamingExamples, kBestRules);


            myLogger.trace("Select " + kRules + " best rules");
            myLogger.trace("True positive rate: " + kRulesTruePositiveRate);
            myLogger.trace("False postive rate: " + kRulesFalsePositiveRate);    
            
            kBestRules.stream().forEach((Rule aRule) -> {
                myLogger.trace(aRule.nicePrint());
            });
        }
        
        return calcualteTruePostiveRate(streamingExamples, kBestRules);
    }
    
    
    public List<List<Term>> learnRightSide(Instances dataset) throws Exception{
                 
        Instances originalDataset = new Instances(dataset);
        
//        myLogger.trace("No. instances: " + originalDataset.size());
        
        // array of possible combination of terms for right hand-side
        List<List<Term>> rightSideSet = new ArrayList<>();
        
        // list of all possible term (original, shouldn't be changed)
        List<Term> allTerms_original = new ArrayList<>();
        List<Term> allTerms = new ArrayList<>();
        
        // list of all used terms
        List<Term> usedTerms = new ArrayList<>();
        
        
        // generate all possible rule term, this list shouldn't be changed throughout the learning process
        allTerms_original.addAll(generatePossibleRuleTerms(originalDataset));
        allTerms.addAll(allTerms_original);
        
        // create D dataset from the orginal dataset
        Instances datasetD = new Instances(originalDataset);
        
        // total number of terms from the dataset
        int noTerms = allTerms_original.size();
        
        // metadata object for learning process, this to provide more inforamtion on the learning procoess
        Map<List<Term>, String> setMetaData = new HashMap<>();
 
        
        while(usedTerms.size() < noTerms - 1){
            
            // add all the terms to used terms and they should not be used to induce a new set
            List<Term> termsD = new ArrayList<>(allTerms);

            // a set should represent a complete hand-side
            List<Term> aSetOfTerms = new ArrayList<>();
            String setMetaDataString = "";
                    
            // first remove all used terms from terms set
            removeUsedTerms(termsD, usedTerms);
            
            while(aSetOfTerms.size() < headMaxTerms){

                Map<String, Double> termProbability = new HashMap<>();
    
                for (int termI = 0; termI < termsD.size(); termI++) {
                    
                    @SuppressWarnings("UnusedAssignment")
                    NormalDistribution attributeDistribution = null;
                                            
                    // check if the attribute is numeric, then create Gaussian for the value from the dataset
                    if(termsD.get(termI).isNumeric()){
                        
                        // statistic for the values
                        SummaryStatistics valuesSummaryStatistics = new SummaryStatistics();
                        
                        // go through all the instances and get values of the attribute
                        for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {
                            
                            double attribteValue = datasetD.get(instanceI).value((int) termsD.get(termI).getAttributeIndex());
                            valuesSummaryStatistics.addValue(attribteValue);
                        }
                        
//                        myLogger.trace(valuesSummaryStatistics);
                        myLogger.trace(termsD.get(termI).getAttribute());
                        
                        double meanValue = valuesSummaryStatistics.getMean() == 0.0d ? 0.01 : valuesSummaryStatistics.getStandardDeviation();
                        double sdValue = valuesSummaryStatistics.getStandardDeviation() == 0.0d ? valuesSummaryStatistics.getMean() + (valuesSummaryStatistics.getMean() * 0.01) : valuesSummaryStatistics.getStandardDeviation();
                        
                        if(sdValue == 0.0){
                            continue;
                        }
                        
                        attributeDistribution = new NormalDistribution(meanValue, sdValue);               
                        Map<Double, Double> attribteDistributionProbabilityValues = new HashMap<>();
                        
                        for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {
                            double instanceAttributeValue = datasetD.get(instanceI).value((int) termsD.get(termI).getAttributeIndex());
                            @SuppressWarnings("null")
                            double probabilityValue = attributeDistribution.density(instanceAttributeValue); 
                            attribteDistributionProbabilityValues.put(instanceAttributeValue, probabilityValue);
                        }
                        
                        double[] bounds = createNumericTerm(attribteDistributionProbabilityValues);
                        termsD.get(termI).setLowerBoundNumeric(bounds[0]);              
                        termsD.get(termI).setUpperBoundNumeric(bounds[1]);                       
                        
                        myLogger.trace(termsD.get(termI));
                        
                        String termString = termsD.get(termI).getAttributeValueDobuleString();
                        
                        int termCoveredCount = 0;
                        
                        for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {
                            if(termsD.get(termI).coveredInstance(datasetD.get(instanceI))){
                                termCoveredCount++;
                            }
                        }
                        
                        double numericTermRangeProbability = (double) termCoveredCount / (double) datasetD.size();
                        
                        termProbability.put(termString, numericTermRangeProbability);
                        
                    }else if(termsD.get(termI).isCategorial()){
                        
                        String termString = termsD.get(termI).getAttributeValueDobuleString();
                        int termCoveredCount = 0;
                        
                        for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {
                             if(termsD.get(termI).coveredInstance(datasetD.get(instanceI))){
                                 termCoveredCount++;
                             }
                        }
                        
//                        myLogger.trace(termsD.get(termI));
//                        myLogger.trace(termCoveredCount + ":" + datasetD.size());
                        
                        double termProbabilityValue = (double) termCoveredCount / datasetD.size();
                        
                        // put term String and its corresponding probability into the map
                        termProbability.put(termString, termProbabilityValue);
                    }
                } 
                
                // if there are not enough term, then stop inducing

                
                myLogger.trace("Terms probability:");
                myLogger.trace(termProbability);

                // get the term with largest count from frequency distribution
                String curentTermWithLargestCount = Utilities.keyWithBestValueFromMap(termProbability);
                double highestProbabilityValue = termProbability.get(curentTermWithLargestCount);
                
                myLogger.trace("Term String with highest probability: " + curentTermWithLargestCount);
                myLogger.trace("Coverage value: " + highestProbabilityValue);
                // no more terms then stop looking
                if(curentTermWithLargestCount == null){
                   break;
                }               
                
                // If additional term leads to zero coverage then the inducing process should stop
                if(highestProbabilityValue == 0 || highestProbabilityValue < headCoverageThreshold){
                    break;
                }

                // parse current term with largest value into dobule[], [attributeIndex, attributre]
                double[] currentTermWithLargestCountDouble = Utilities.parseStringToDoubleArray(curentTermWithLargestCount);                
                
                // parse the term in form on double[] [attributeIndex, attributeValue] into an actual term object
                Term currentBestTerm = Utilities.parseTermFromDoubleArray(allTerms, currentTermWithLargestCountDouble);
                myLogger.trace("Term String with highest probability: " + currentBestTerm);
                
                // need to remove instances covered by the term from datasetD
                Iterator<Instance> datasetDIterator = datasetD.iterator();
                
                setMetaDataString += "[";
                myLogger.trace("Dataset size before removing instances not covered by best term: " + datasetD.size());
                setMetaDataString += datasetD.size();
                
                while(datasetDIterator.hasNext()){

                    Instance anInstance = datasetDIterator.next();
                    
                    if(currentBestTerm.coveredInstance(anInstance) == false){
                        datasetDIterator.remove();
                    }
                }

                myLogger.trace("Dataset size after removing instances not covered by best term: " + datasetD.size());
                setMetaDataString += ":" + datasetD.size();
                
                setMetaDataString += "]";
                
                //  remove term related to the selected term
                Iterator<Term> termIterator = termsD.iterator();
                
                setMetaDataString += "[";
                myLogger.trace("size before removing best term: " + termsD.size());
                setMetaDataString += termsD.size();
                
                while(termIterator.hasNext()){

                    Term aTerm = termIterator.next();

                    if(aTerm.getAttributeIndex() == currentTermWithLargestCountDouble[1]){
                        termIterator.remove();
                    }
                }

                myLogger.trace("size after removing best term: " + termsD.size());
                setMetaDataString += ":" + termsD.size();
                setMetaDataString += "]";

                if(!aSetOfTerms.isEmpty() ){
                    setMetaDataString += "";
                }
                
                aSetOfTerms.add(currentBestTerm);
                
            }
           
            // fixing bogus term error (when a term does not cover any instances at all)
            if(aSetOfTerms.isEmpty()){
                break;
            }
            
            // set metadata for the set
            setMetaData.put(aSetOfTerms, setMetaDataString);
            
            // add all terms from the set to used terms list
            for (int termI = 0; termI < aSetOfTerms.size(); termI++) {
                usedTerms.add(aSetOfTerms.get(termI));
            }
            
            // reset D to original dataset
            datasetD.clear();
            datasetD.addAll(originalDataset);
            
            // add set of sets
            rightSideSet.add(aSetOfTerms);
            
            

       }
                
        Set<Map.Entry<List<Term>, String>> setMetaDatSet = setMetaData.entrySet();
        Iterator<Map.Entry<List<Term>, String>> setMetaDatSetIterator = setMetaDatSet.iterator();
        
        while(setMetaDatSetIterator.hasNext()){
            
            Map.Entry<List<Term>, String> anEntry = setMetaDatSetIterator.next();
            
            List<Term> termList = anEntry.getKey();
            
            Iterator<Term> termListIterator = termList.iterator();
            
            while(termListIterator.hasNext()){
                
                Term aTerm = termListIterator.next();
                
                myLogger.info(aTerm.toString());
                
                if(termListIterator.hasNext() == true){
                    myLogger.info( " AND " );
                }
            }
            
            myLogger.info( " Stats: " + anEntry.getValue());
        }
        
        return new ArrayList<>(rightSideSet);
    }    
    
    /**
     * Generate all possible terms (attribute-value) from all attributes from the training dataset.
     * 
     * @param originalDataset training dataset, Instances structure from Weka
     * @return list of all possible terms from all attributes, expect the class
     */
    public List<Term> generatePossibleRuleTerms(Instances originalDataset){
        
        // list of all attribute-value (term)
        List<Term> allTerms = new ArrayList<>();
        
        // number of total attributes from the dataset
        int noAttribute = originalDataset.numAttributes();
        
        // loop through each attribute in turn
        for (int attributeNo = 0; attributeNo < noAttribute; attributeNo++) {
            
            // get the actual attribute object for the attribute
            Attribute anAttribute = originalDataset.attribute(attributeNo);
            
            // get the index of the attribute in the dataset
            double anAttributeIndex = anAttribute.index();
            
            // loop through value of the attribute (Categorical)
            if(anAttribute.isNominal()){
                
                // get total number of values for the attribute
                int noValue = anAttribute.numValues();
                
                // go through each value of the attribute, create a term and addd to the list
                
                for (int valueNo = 0; valueNo < noValue; valueNo++) {
                    
                    // create a Term object for an attribute-value
                    Term aTerm = new Term(anAttribute, anAttributeIndex, (double) valueNo);
                    
                    // add the Term to the list
                    allTerms.add(aTerm);
                }
                
                
            }else if(anAttribute.isNumeric()){
                
               // create new Term for the Attribute
               Term aTerm = new Term(anAttribute, anAttributeIndex);
               
                // add the Term to the list
                allTerms.add(aTerm);
               
            }
        }
        
        return allTerms;
    }
    
    /**
     * Given a list of terms, remove all terms from second argument.
     * 
     * @param allTerms list of terms
     * @param usedTerms these terms to be removed
     */
    public void removeUsedTerms(List<Term> allTerms, List<Term> usedTerms){
        
        // a new list of terms, with all terms from used terms list removed from all terms
        
        Iterator<Term> allTermsIterator = allTerms.iterator();
       
        
        // go thorugh each term from all term and remove it, if it matches any term from used terms list
        while(allTermsIterator.hasNext()){
            
            Iterator<Term> usedTermsIeterator = usedTerms.iterator();
            Term aTerm = allTermsIterator.next();
            
            while(usedTermsIeterator.hasNext()){
                
                Term aTerm_j = usedTermsIeterator.next();
                
                if(aTerm.getAttributeValueDobuleString().equals(aTerm_j.getAttributeValueDobuleString())){
                    
                    allTermsIterator.remove();
                    
                }
                
            }
        }

    }
    
    /**
     * Remove all terms related to the input terms list (at attribute level)
     * all terms in termList_i that relates to termList_j, will be removed
     * 
     * @param termList_i
     * @param termList_j
     */
    
    public void removeRelativedTermsAttributeLevel(List<Term> termList_i, List<Term> termList_j){
        
        Iterator<Term> termList_i_Iterator = termList_i.iterator();
        
        while(termList_i_Iterator.hasNext()){
            
            Term anTerm = termList_i_Iterator.next();
            double anTermAttributeIndex= anTerm.getAttributeIndex();
            
            boolean toBeRemoved = false;
            
            for (int termI = 0; termI < termList_j.size(); termI++) {
                
                if(termList_j.get(termI).getAttributeIndex() == anTermAttributeIndex){
                    toBeRemoved = true;
                }
            }
            if(toBeRemoved){
                termList_i_Iterator.remove();
            }
        } 
    }

    /**
     * Remove all data instances from the dataset for a given rule
     * 
     * @param dataset the input dataset
     * @param aRule the rule is used to classify instances and these instances will be removed from the dataset
     */
    public void removeInstancesCoveredByRule(Instances dataset, Rule aRule){
        
        List<Term> aLeftSide = aRule.getLeftSide();
        List<Term> aRightSide = aRule.getRightSdie();
        
        Iterator<Instance> datasetIterator = dataset.iterator();
        
        while(datasetIterator.hasNext()){
            
            Instance anInstance = datasetIterator.next();
            
            if(instanceCoveredByTermsList(anInstance, aLeftSide) == true &&
                    instanceCoveredByTermsList(anInstance, aRightSide) == true){
                datasetIterator.remove();
                
            }
        }
    }
    
    /**
     * Create a a numeric term in form of lower &lt; value &gt; upper
     * 
     * @param attribteDistributionProbabilityValues Map of the values from a numeric attribute and their corresponding probability
     * @return  lower and upper bound from best and 2nd probabilities
     */
    public double[] createNumericTerm(Map<Double, Double> attribteDistributionProbabilityValues){
        
        //  lower and upper bound values, bound[0] = lowerBoundValue, bounds[1] = upperBoundValue
        double[] bounds = new double[2];
        
        // select value with highest probability
        double highestValue = 0.0;
        double highestValueProbability = 0.0;
        
        Set<Map.Entry<Double, Double>> valueProbabilitySet = attribteDistributionProbabilityValues.entrySet();
        Iterator<Map.Entry<Double, Double>> valueProbabilitySetIterator = valueProbabilitySet.iterator();
        
        while(valueProbabilitySetIterator.hasNext()){
            Map.Entry<Double, Double> anEntry = valueProbabilitySetIterator.next();
            
            // check if the probability of current entry is greater than currrent selected one
            if(anEntry.getValue() >= highestValueProbability){
                highestValue = anEntry.getKey();
                highestValueProbability = anEntry.getValue();
            } 
        }
        
        // now looking for lower bound and upper bound
        valueProbabilitySetIterator = valueProbabilitySet.iterator();
        
        double lowerBoundHighestValue = 0.0d;
        double lowerBoundHighestProbability = 0.0d;
        boolean lowerBoundSelected = false;
        
        double upperBoundHighestValue = 0.0d;
        double upperBoundHighestProbability = 0.0d;       
        boolean upperBoundSelected = false;
        
        while(valueProbabilitySetIterator.hasNext()){
            
            Map.Entry<Double, Double> anEntry = valueProbabilitySetIterator.next();
            
            // looking for a lower bound with best probability
            if(anEntry.getKey() < highestValue && anEntry.getValue() >= lowerBoundHighestProbability){
                lowerBoundHighestValue = anEntry.getKey() ;
                lowerBoundHighestProbability = anEntry.getValue();
                lowerBoundSelected = true;
            }
            
            // looking for an upper bound with best probability
            if(anEntry.getKey() > highestValue && anEntry.getValue() >= upperBoundHighestProbability){
                upperBoundHighestValue = anEntry.getKey() ;
                upperBoundHighestProbability = anEntry.getValue();    
                upperBoundSelected = true;
            }
            
        }
        
        bounds[0] = lowerBoundSelected ? lowerBoundHighestValue : highestValue;
        bounds[1] = upperBoundSelected ? upperBoundHighestValue : highestValue;
        
        return bounds;
    }
    
    /**
     * check if all instances from a given dataset is covered/ satisfied by both BODY and HEAD
     * 
     * @param dataset instances to be checked
     * @param leftHandSide these rule term(s) represent the BODY
     * @param rightHandside these rule term(s) represent the HEAD
     * @return true if all instances covered by the rule, false otherwise
     */
    
    public boolean allInstancesCoveredByTermsBothSides(Instances dataset, List<Term> leftHandSide, List<Term> rightHandside){
        
        if(rightHandside.isEmpty() || leftHandSide.isEmpty()){
            return false;
        }
        
        Iterator<Instance> datasetIterator = dataset.iterator();
        while(datasetIterator.hasNext()){
            Instance anInstance = datasetIterator.next();
            
            // 1st check if the instance is totally covered by the right hand-side
           for (int termI = 0; termI < rightHandside.size(); termI++) {
               if(rightHandside.get(termI).coveredInstance(anInstance) == false){
                   return false;
               }
           }
           
           // 2nd check if the instance is totally covered by the right hand-side
            for (int termI = 0; termI < leftHandSide.size(); termI++) {
               if(leftHandSide.get(termI).coveredInstance(anInstance) == false){
                   return false;
               }
           }
           
        }
        
        return true;
    }
    
    /**
     * Test whether an instance is covered by given term(s)
     * 
     * @param instanceIn
     * @param termList
     * @return TRUE or FALSE
     */
    public boolean instanceCoveredByTermsList(Instance instanceIn, List<Term> termList){
        
        for (int termI = 0; termI < termList.size(); termI++) {
            
            // if not all terms covered the instance, then return false
            if(termList.get(termI).coveredInstance(instanceIn) == false){
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Calculate J-measure value  
     * 
     * @param xyConditionalProbabilityIn 
     * @param headPorbability
     * @param bodyProbability
     * @return J-measure value
     */
    public double calculateJMeasure(double xyConditionalProbabilityIn, double headPorbability, double bodyProbability){
        
        double jMesuare = xyConditionalProbabilityIn * Utils.log2(xyConditionalProbabilityIn / headPorbability) +
                            (1.0d -  xyConditionalProbabilityIn) * Utils.log2((1.0d - xyConditionalProbabilityIn) / (1.0d - headPorbability));
        
        double Jmeasure = jMesuare * bodyProbability;
        
        return (Double.isNaN(Jmeasure) == false) ? Jmeasure : 0.0d;
    }    

    /**
     * Calculate J-measure of the dataset for given body/ left and head/ right
     * 
     * @param datasetIn
     * @param bodyPart
     * @param headPart
     * @return 
     */
    public double calcualteRuleJMeasure(Instances datasetIn, List<Term> bodyPart, List<Term> headPart){
        
        int bodyCoveredCount = 0;
        int headCoveredCount = 0;
        int bothBodyHeadCovered = 0;
        
        for (int i = 0; i < datasetIn.size(); i++) {
            
            // check if the instance is covered by body/ left side
            if(instanceCoveredByTermsList(datasetIn.get(i), bodyPart) == true){
                bodyCoveredCount++;
                
                // check whether the instance is also covered by head/ right
                if(instanceCoveredByTermsList(datasetIn.get(i), headPart) == true){
                    bothBodyHeadCovered++;
                }
            }
            
            // if the instance is not covered by body/ left, then check if
            // the instance is covered only head/ right
            if(instanceCoveredByTermsList(datasetIn.get(i), headPart) == true){
                headCoveredCount++;
            }
        }
        
        // calcate required probabilities 
        double xyConditionalProbability = (double) bothBodyHeadCovered / bodyCoveredCount;
        double headProbability = (double) headCoveredCount / datasetIn.size();
        double bodyProbability = (double) bodyCoveredCount / datasetIn.size();
        
        double JMeasureValue = calculateJMeasure(xyConditionalProbability, headProbability, bodyProbability);
        
        return JMeasureValue;
    }
    
    /**
     * Calculate Weighted Relative Accuracy (WRA) for from the input dataset for a given rule.
     * 
     * @param dataset the dataset
     * @param aLeftHandSide left-side (BODY) of the rule
     * @param aRightHandSide right-side (HEAD) of the rule
     * @return WRA value of the rule on the dataset
     */
    public double calcualteWRA(Instances dataset, List<Term> aLeftHandSide, List<Term> aRightHandSide){
        
        double totalPositive;
        double totalNegative;
        
        double ruleTruePositive = 0;
        double ruleFalsePositive = 0;
        
        double ruleTrueNegative = 0;
        double ruleFalseNegative = 0;
        
        
        // calcualte truePositive, trueNegative, falsePostive, falseNegative for the Rules
        for (int instanceI = 0; instanceI < dataset.size(); instanceI++) {
            
            Instance anInstance = dataset.get(instanceI);
            
            // calculate true postive and true negative instances for the rules
            if(instanceCoveredByTermsList(anInstance, aLeftHandSide)){
                
                if(instanceCoveredByTermsList(anInstance, aRightHandSide)){
                    ruleTruePositive++;
                }else{
                    ruleFalsePositive++;
                }
            }else if(instanceCoveredByTermsList(anInstance, aRightHandSide)){
                ruleFalsePositive++;
            }else{
                ruleTrueNegative++;
            }
        }
        
        // calculate totalPositive and totalNegative
        totalPositive = ruleTruePositive + ruleFalseNegative;
        totalNegative = ruleFalsePositive + ruleTrueNegative;
        
        // start calcualting WRA      
        double ruleWRA = (ruleTruePositive + ruleFalsePositive) / (totalPositive + totalNegative) * 
                (  (ruleTruePositive / (ruleTruePositive + ruleFalsePositive) - ( totalPositive / (totalPositive + totalNegative) )));
        
        return ruleWRA; 
    }
    
    /**
     * Calculate support value of a given rule on the dataset
     * 
     * @param dataset the dataset
     * @param bodySide left-side or BODY part of the rule
     * @return support value for the rule on the given dataset
     */
    public double calculateSupport(Instances dataset, List<Term> bodySide){
        
        Iterator<Instance> datasetIterator = dataset.iterator();
        int supportCount = 0;
        
        while(datasetIterator.hasNext()){
            
            Instance anInstance = datasetIterator.next();
            
            if(instanceCoveredByTermsList(anInstance,bodySide)){
                supportCount++;
            }
            
        }
        return !dataset.isEmpty() ? (double) supportCount / (double) dataset.size() : 0.0d;
    }
    
    /**
     * Calculate confidence value of a given rule on the dataset
     * 
     * @param dataset the dataset
     * @param bodySide left-side or BODY part of the rule
     * @param HeadSide right-side or HEAD part of the rule
     * @return confidence value for the rule on the given dataset
     */
    public double calcualteConfidence(Instances dataset, List<Term> bodySide, List<Term> HeadSide){
        Iterator<Instance> datasetIterator = dataset.iterator();
        int confidenceCount = 0;
        int supportCount = 0;
        while(datasetIterator.hasNext()){
            
            Instance anInstance = datasetIterator.next();
            
            if(instanceCoveredByTermsList(anInstance,bodySide)){
                supportCount++;
                if(instanceCoveredByTermsList(anInstance,HeadSide)){
                    confidenceCount++;
                }
            }
            
        }
        return !dataset.isEmpty() ? (double) confidenceCount / (double) supportCount : 0.0d;
    }

        /**
     * Calculate true positive rate, 
     * The rate of true positive is calculated by no. true positive / total positive
     * 
     * @param dataset
     * @param selectedRules
     * @return 
     */
    
    public double calcualteTruePostiveRate(Instances dataset, List<Rule> selectedRules){
        
        // calculate total postive example
        int truePositive = 0;
        int falseNegative = 0;
        
        Iterator<Instance> instancesIterator;
        
        // go through each selected rule
        for (Rule aRule : selectedRules) {
            
            // got HEAD from the rules
            List<Term> HeadTerms = aRule.getRightSdie();
            List<Term> bodyTerm = aRule.getLeftSide();
            
            // check whether this rule is positive (covered by the HEAD only)
            instancesIterator = dataset.iterator();
            
            while(instancesIterator.hasNext()){
                
                Instance anInstance = instancesIterator.next();
                
                if(instanceCoveredByTermsList(anInstance, HeadTerms) == true && instanceCoveredByTermsList(anInstance, bodyTerm) == true){
                    truePositive++;
                }
                
                if(instanceCoveredByTermsList(anInstance, HeadTerms) == false && instanceCoveredByTermsList(anInstance, bodyTerm) == true){
                    falseNegative++;
                }
            }
        }
               
        double truePostitveRate = (double) truePositive / (double) (truePositive + falseNegative);
        
        return truePostitveRate;
        
    }
    
    /**
     * Calculate false positive rate,
     * the false positive rate is calculated by no. false positive / total negative
     * 
     * @param dataset the dataset that will be used for the calculation
     * @param selectedRules the rule that is used for the calculation
     * @return false positive for the given rules and the input dataset
     */
    public double calcuateFalsePositiveRate(Instances dataset, List<Rule> selectedRules){
        
         // calculate total negative example
        int falsePostive = 0;    
        int trueNegative = 0;
        
        Iterator<Instance> instancesIterator;   
        
         // go through each selected rule
        for (Rule aRule : selectedRules) {
            
            // got HEAD from the rules
            List<Term> HeadTerms = aRule.getRightSdie();
            List<Term> bodyTerm = aRule.getLeftSide();
            
            // check whether this rule is positive (covered by the HEAD only)
            instancesIterator = dataset.iterator();
            
            while(instancesIterator.hasNext()){
                
                Instance anInstance = instancesIterator.next();
                
                if(instanceCoveredByTermsList(anInstance, bodyTerm) == true && instanceCoveredByTermsList(anInstance, HeadTerms) == false){
                    falsePostive++;
                }
                
                if(instanceCoveredByTermsList(anInstance, bodyTerm) == false && instanceCoveredByTermsList(anInstance, HeadTerms) == false){
                    trueNegative++;
                }                
                
            }
        }       
        
        double falsePositiveRate = (double) falsePostive / (double) (falsePostive + trueNegative);
        
        return falsePositiveRate;
    }
    
    /**
     * For each generated right-hand side (HEAD), all possible left-hand side (BODY) will be generated.
     * 
     * @param rightSideSetIn list of generated possible right-hand side (HEAD)
     * @param dataset dataset used to generate the rules
     * @return 
     * @throws Exception
     */
    public List<Rule> induceCompleteRules(List<List<Term>> rightSideSetIn, Instances dataset) throws Exception{
        
        //init rule collection 
        List<Rule> ruleCollection = new ArrayList<>();
        
       
        for (int rightSideI = 0; rightSideI < rightSideSetIn.size(); rightSideI++) {
            boolean noMoreTermForLeft = false;
            
            // output message indicate to go to next part

            myLogger.trace("=============================================");            
            myLogger.trace("Introducing rules for: " + rightSideSetIn.get(rightSideI));
            myLogger.trace("  BODY:");

            
            Instances originalDataset = new Instances(dataset);
            
            // this block of code should be optimised, but not now
            List<Term> allTerms_original = new ArrayList<>();
            allTerms_original.addAll(generatePossibleRuleTerms(originalDataset));
            
            List<Term> allTerms = new ArrayList<>();
            allTerms.addAll(allTerms_original);
            
            // get 1st right hand-side
            List<Term> aRightSide = rightSideSetIn.get(rightSideI);

            // remove all relative terms 
            removeRelativedTermsAttributeLevel(allTerms, aRightSide);

    //        RemoveInstancesCoveredByTerms(datasetD, aRightSide);

            myLogger.trace("current right hand-side: " + aRightSide);

            List<Term> usedTermsForLeft = new ArrayList<>();

            while(originalDataset.size() > 0 && allTerms.size() > 0){
                Instances datasetD = new Instances(originalDataset);
                
                // create an empty left-hand (BODY) side, terms will be evaluated and incrementally added to the rule
                List<Term> aLeftSide = new ArrayList<>();
                
                // current max J-Measure value for the rule
                double ruleCurrentMaxJmeasure = 0.0d;
                
                double previousWraChange = 0.0d;
                int inducedTermsCount = 0;
                
//                while(allInstancesCoveredByTermsBothSides(datasetD, aLeftSide, aRightSide) == false && allTerms.isEmpty() == false && aLeftSide.size() <= 2){
                while(allInstancesCoveredByTermsBothSides(datasetD, aLeftSide, aRightSide) == false && allTerms.isEmpty() == false){
                    
                    boolean stopInducing = false;
                    
                    // frequency for term for left hand-side
                    Frequency termFrequency = new Frequency();
                    
                    //frequency for both left and rights sides
                    Frequency conditionalTermFrequency = new Frequency();
                    
                    
                    // Map of conditonal probability for each term
                    Map<String, Double> conditionalTermProbability = new HashMap<>();

                    // Map of J-measure value for each term
                    Map<String, Double> jMeasureTerms = new HashMap<>();
                    
                    // conditional probability for each term for the right hand-side

//                    myLogger.trace(datasetD);
                    // calcualte probablity of each term for a set
                    for (int termI = 0; termI < allTerms.size(); termI++) {
                        
                        // check whether the term is categorical or numeric
                        if(allTerms.get(termI).isCategorial()){
                           int termCoveredCount = 0;
                           int termAndRightSideCoveredCount = 0;
                           int rightSideCoveredCount = 0;

                           String termString = allTerms.get(termI).getAttributeValueDobuleString();

                           for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {
                               
                               // check whether the right/ head side is covered
                               if(instanceCoveredByTermsList(datasetD.get(instanceI), aRightSide) == true){
                                   rightSideCoveredCount++;
                               }
                               
                               if(allTerms.get(termI).coveredInstance(datasetD.get(instanceI))){

                                   termCoveredCount++;
                                   
                                   termFrequency.addValue(termString);

                                   // check if covered (Right|Left)
                                   if(instanceCoveredByTermsList(datasetD.get(instanceI), aRightSide)){
                                       termAndRightSideCoveredCount++;
                                       conditionalTermFrequency.addValue(termString);
                                   }
                               }
                               
                               
                           }
                           
                           double conditionalRightLeftProbability;
                           if(termCoveredCount != 0){
                               conditionalRightLeftProbability = (double) termAndRightSideCoveredCount / (double) termCoveredCount;
                           }else{
                               conditionalRightLeftProbability = 0.0;
                           }
                            
                           double leftProbability = (double) termCoveredCount / datasetD.size();
                           double rightProbability = (double) rightSideCoveredCount / datasetD.size();
                           
                           
                           // put term String and its corresponding probability into the map
                           conditionalTermProbability.put(termString, conditionalRightLeftProbability);

                           double JMeasureValue = calculateJMeasure(conditionalRightLeftProbability, rightProbability, leftProbability);
                           jMeasureTerms.put(termString, JMeasureValue);
                           
                        }else if(allTerms.get(termI).isNumeric()){
                            
                            // statistic for the values
                            SummaryStatistics valuesSummaryStatistics = new SummaryStatistics();
                            NormalDistribution termDistribution;
                            // go through all the instances and get values of the numeric
                            for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {

                                // create distribution for given target class
                                if(instanceCoveredByTermsList(datasetD.get(instanceI), aRightSide)){
                                    double attribteValue = datasetD.get(instanceI).value((int) allTerms.get(termI).getAttributeIndex());
                                    valuesSummaryStatistics.addValue(attribteValue);  
                                }
                                
                                
                            }
                            
//                            myLogger.trace(valuesSummaryStatistics);
                            
                            
                            double distributionMean = valuesSummaryStatistics.getMean();
                            double distributionSD = valuesSummaryStatistics.getStandardDeviation() == 0.0d ? distributionMean + (distributionMean * 0.01) : valuesSummaryStatistics.getStandardDeviation();
                            
                            
                            // all values are the same, and standard deviation is 0.0
                            if(distributionSD == 0.0){
//                                usedTermsForLeft.add(allTerms.get(termI));
                                continue;
                            }
                            
                            termDistribution = new NormalDistribution(distributionMean, distributionSD);  
                            // Map for numeric attribute-value
                            Map<Double, Double> attribteDistributionProbabilityValues = new HashMap<>();
                            
                            for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {
                                double instanceAttributeValue = datasetD.get(instanceI).value((int) allTerms.get(termI).getAttributeIndex());
                                double probabilityValue = termDistribution.density(instanceAttributeValue);
                                
                                attribteDistributionProbabilityValues.put(instanceAttributeValue, probabilityValue);
                            }
                            
                            double[] bounds = createNumericTerm(attribteDistributionProbabilityValues);
                            
                            allTerms.get(termI).setLowerBoundNumeric(bounds[0]);              
                            allTerms.get(termI).setUpperBoundNumeric(bounds[1]);
                            myLogger.trace(Arrays.toString(bounds));
                            
                            String termString = allTerms.get(termI).getAttributeValueDobuleString();
                            
                           int termCoveredCount = 0;
                           int termAndRightSideCoveredCount = 0;
                           int rightSideCoveredCount = 0;

                            for (int instanceI = 0; instanceI < datasetD.size(); instanceI++) {

                               // check whether the right/ head side is covered
                               if(instanceCoveredByTermsList(datasetD.get(instanceI), aRightSide) == true){
                                   rightSideCoveredCount++;
                               }
                               
                                if(allTerms.get(termI).coveredInstance(datasetD.get(instanceI))){

                                    termCoveredCount++;

                                    termFrequency.addValue(termString);

                                    // check if covered (rightSide|term)
                                    if(instanceCoveredByTermsList(datasetD.get(instanceI), aRightSide)){
                                        termAndRightSideCoveredCount++;
                                        conditionalTermFrequency.addValue(termString);
                                    }
                                }
                            }
                           
                            double conditionalRightLeftProbability = (double) termAndRightSideCoveredCount / (double) termCoveredCount  ;
                            double leftProbability = (double) termCoveredCount / datasetD.size();
                            double rightProbability = (double) rightSideCoveredCount / datasetD.size();
                            
                            double JMeasureValue = calculateJMeasure(conditionalRightLeftProbability, rightProbability, leftProbability);
                            
                            conditionalTermProbability.put(termString, conditionalRightLeftProbability);
                            jMeasureTerms.put(termString, JMeasureValue);
                        }
                    }

                    myLogger.trace("Frequency for current iteration:");
                    myLogger.trace(termFrequency);
                    
                    myLogger.trace("Conditional frequency for current iteration:");
                    myLogger.trace(conditionalTermFrequency);
                                        
                    myLogger.trace("Conditional probability:");
                    myLogger.trace(conditionalTermProbability);

                    myLogger.trace("Term with best conditional probability: ");
                    myLogger.trace(Utilities.parseTermFromString(allTerms, Utilities.keyWithBestValueFromMap(conditionalTermProbability)));
                    myLogger.trace(Utilities.keyWithBestValueFromMap(conditionalTermProbability) + " - " + conditionalTermProbability.get(Utilities.keyWithBestValueFromMap(conditionalTermProbability)));

                    myLogger.trace("Term with best J-measure value: ");
                    myLogger.trace(Utilities.keyWithBestValueFromMap(jMeasureTerms) + " - " + jMeasureTerms.get(Utilities.keyWithBestValueFromMap(jMeasureTerms)));
                    myLogger.trace(Utilities.parseTermFromString(allTerms, Utilities.keyWithBestValueFromMap(jMeasureTerms)));
                    
                    String curentTermWithLargestCount = Utilities.keyWithBestValueFromMap(conditionalTermProbability);
                                        
                    if(curentTermWithLargestCount == null){
                        break;
                    }
                    
                    // parse current term with largest value into dobule[], [attributeIndex, attributre]
                    double[] currentTermWithLargestCountDouble = Utilities.parseStringToDoubleArray(curentTermWithLargestCount);

                    // need to remove instances not covered by the term from datasetD
                    Iterator<Instance> datasetDIterator = datasetD.iterator();
                    
                    // parse the term in form on double[] [attributeIndex, attributeValue] into an actual term object
                    Term currentBestTerm = Utilities.parseTermFromDoubleArray(allTerms, currentTermWithLargestCountDouble);

                    myLogger.trace("Term with highest count current iteration: " + currentBestTerm);

                    if(currentBestTerm == null){
                        break;
                    }

                    List<Term> aLeftSideUndedicded = new ArrayList<>(aLeftSide);
                    aLeftSideUndedicded.add(currentBestTerm);

                    myLogger.trace("Dataset size before removing instances not covered by best term: " + datasetD.size());

                    double currentWraChange = calcualteWRA(originalDataset, aLeftSideUndedicded, aRightSide) - calcualteWRA(originalDataset, aLeftSide, aRightSide);

                    while(datasetDIterator.hasNext()){
                        Instance anInstance = datasetDIterator.next();
                        if(currentBestTerm.coveredInstance(anInstance) == false){
                            datasetDIterator.remove();
                        }
                    }


                    myLogger.trace("Dataset size after removing instances not covered by best term: " + datasetD.size());

                    //  remove term related to the selected term
                    Iterator<Term> termIterator = allTerms.iterator();

                    myLogger.trace("size before removing best term: " + allTerms.size());
                    myLogger.trace("size before removing best term: " + allTerms);

                    while(termIterator.hasNext()){

                        Term aTerm = termIterator.next();

                        if(aTerm.getAttributeIndex() == currentTermWithLargestCountDouble[1]){
                            termIterator.remove();
                        }
                    }

                    myLogger.trace("size after removing best term: " + allTerms.size());
                    myLogger.trace("size after removing best term: " + allTerms);
                                        
                    // check 0 coverage 
                    if((conditionalTermProbability.get(curentTermWithLargestCount) != 0.0d)){

                        switch(prunningOption){
                            case GRules.wraPrunning:
                                if(inducedTermsCount == 0){
                                    aLeftSide.add(currentBestTerm);
                                    inducedTermsCount++;      
                                    previousWraChange  = currentWraChange;
                                }else{
                                    if(currentWraChange > previousWraChange){
                                        aLeftSide.add(currentBestTerm);
                                        inducedTermsCount++;                                          
                                    }else{
                                        break;
                                    }
                                }                            
                                break;
                            case GRules.jmeasurePrunning:
                                // calculate current J-measure value
                                double currentJmeasureValue = calcualteRuleJMeasure(originalDataset, 
                                        aLeftSide, 
                                        aRightSide);

                                // calculate prospective J-measure for the rule
                                List<Term> tempLeft = (ArrayList) ((ArrayList) aLeftSide).clone();
                                tempLeft.add(currentBestTerm);
                                double prospectiveJmeasureValue = calcualteRuleJMeasure(originalDataset, 
                                        tempLeft, 
                                        aRightSide);

                                myLogger.trace(prospectiveJmeasureValue);
                                if(aLeftSide.isEmpty() == true){
                                     aLeftSide.add(currentBestTerm);
                                    inducedTermsCount++;                                       
                                }else if(prospectiveJmeasureValue != Double.NaN && prospectiveJmeasureValue > currentJmeasureValue){
                                    aLeftSide.add(currentBestTerm);
                                    inducedTermsCount++;                               
                                }else{
                                    stopInducing = true;
                                }
                                break;
                            default:   
                        }
                    }else{
                        usedTermsForLeft.add(currentBestTerm);
                        
                        if(inducedTermsCount == 0){
                            noMoreTermForLeft = true;
                        }
                        // stop looking for new rule-term
                        break;
                    }
                    
                    // no more suitable term to be added, so stop inducing
                    if(stopInducing){
                        //stop looking for new rule-term
                        break;
                    }
                }
                
                if(noMoreTermForLeft){
                    break;
                }
                
                Rule aRule = new Rule(aLeftSide, aRightSide);
                ruleCollection.add(aRule);

                // so these terms should not be used anymore
                usedTermsForLeft.addAll(aLeftSide);

                myLogger.trace(aLeftSide);
                myLogger.trace(" THEN ");
                myLogger.trace(aRightSide);
                                
                myLogger.trace("Dataset size before removing: " + originalDataset.size());
                
                removeInstancesCoveredByRule(originalDataset, aRule);

                double ruleConfidence = calcualteConfidence(dataset, aRule.getLeftSide(), aRule.getRightSdie());
                double ruleSupport = calculateSupport(dataset, aRule.getLeftSide());
                
                // remove 0 coverage rules
                if(ruleConfidence != 0){
                    if(aLeftSide.size() >0){
                       // append rule to output window
                       myLogger.trace("  " + aLeftSide + "(Conf: " + ruleConfidence + ", Supp: " + ruleSupport + ")");
                                  
                   }                   
                }

                // init all possible terms list
                allTerms.addAll(allTerms_original);

                // remove all relative terms 
                removeRelativedTermsAttributeLevel(allTerms, aRightSide);

                //also remove used terms
                removeUsedTerms(allTerms, usedTermsForLeft);
            }
        }
        // end of learning left hand-side loop
        
        // output rules and their correspoding metrics for analysis
        StringBuilder ruleString = new StringBuilder();
        Iterator<Rule> rulesIterator = ruleCollection.iterator();
        
        
        while(rulesIterator.hasNext()){
            Rule aRule = rulesIterator.next();
            
            // print out the rule content
            ruleString.append(aRule.nicePrint());
            
            // add support
            ruleString.append(";");
            ruleString.append(calculateSupport(dataset, aRule.getLeftSide()));
            
            // add confidence
            ruleString.append(";");
            ruleString.append(calcualteConfidence(dataset, aRule.getLeftSide(), aRule.getRightSdie()));
            
            // add WRA
            ruleString.append(";");
            ruleString.append(calcualteWRA(dataset, aRule.getLeftSide(), aRule.getRightSdie()));
            
            ruleString.append("\n");
        }
        
        myLogger.trace(ruleString.toString());
        
        // get k best rules
        for(int kRules = 5; kRules <= 50; kRules = kRules + 5){
            
            TreeMap<Double, Rule> heuristicSortedTreeMap = new TreeMap<>(Collections.reverseOrder());

            rulesIterator = ruleCollection.iterator();

            while(rulesIterator.hasNext()){

                Rule aRule = rulesIterator.next();

                double selectingHeuristic = calcualteWRA(dataset, aRule.getLeftSide(), aRule.getRightSdie());
//                double selectingHeuristic = calculateSupport(dataset, aRule.getLeftSide());
//                double selectingHeuristic = calcualteRuleJMeasure(dataset, aRule.getLeftSide(), aRule.getRightSdie());
//                double selectingHeuristic = calcualteConfidence(dataset, aRule.getLeftSide(), aRule.getRightSdie());
                
//                myLogger.trace(aRule.nicePrint());
                heuristicSortedTreeMap.put(selectingHeuristic, aRule);            
            }        

            List<Rule> kBestRules = new ArrayList<>();

            // get k best rules
            Set<Map.Entry<Double,Rule>> wraSortedTreeMapSet = heuristicSortedTreeMap.entrySet();
            Iterator<Map.Entry<Double,Rule>> heuristicSortedTreeMapSetIterator = wraSortedTreeMapSet.iterator();

            int ruleCount = 0;
            while(heuristicSortedTreeMapSetIterator.hasNext() && ruleCount < kRules){
                Map.Entry<Double,Rule> anEntry = heuristicSortedTreeMapSetIterator.next();
                Rule aRule = anEntry.getValue();

                kBestRules.add(aRule);
                ruleCount++;
            }

            double kRulesTruePositiveRate = 100 * calcualteTruePostiveRate(dataset, kBestRules);
            double kRulesFalsePositiveRate = 100 * calcuateFalsePositiveRate(dataset, kBestRules);


            myLogger.trace("Select " + kRules + " best rules");
            myLogger.trace("True positive rate: " + kRulesTruePositiveRate);
            myLogger.trace("False postive rate: " + kRulesFalsePositiveRate);    
            
            kBestRules.stream().forEach((aRule) -> {
                myLogger.trace(aRule.nicePrint());
            });
        }
        
        return ruleCollection;
    }
    
    private Rule predict(Instance anInstance, List<Rule> rulesCollection){
        
        
        for (Rule aRule : rulesCollection) {
            List<Term> bodyPart = aRule.getLeftSide();  
            
            if(instanceCoveredByTermsList(anInstance, bodyPart) == true){
                return aRule;
            }
        }
        
        // check if the Instance is covered by body
        return null;
    }
    
    public class Term implements Comparable<Object>{
        
        private final static double TypeNomial = 1;
        private final static double TypeNumeric = -1;
        
    
	private final Attribute attribute;
	private double attributeIndex;
	private double attributeValue;
        private final double attributeType;
        
        private  double lowerBoundNumeric;
        private  double upperBoundNumeric;
        
	public Term(Attribute attributeIn, double attributeIndexIn, double attributeValue){
		
		attribute = attributeIn;
		
		setAttributeIndex(attributeIndexIn);
		setAttributeValue(attributeValue);
                
                // set type of the attribute
                if(attributeIn.isNominal()){
                    attributeType = TypeNomial;
                }else{
                    attributeType = 0;
                }
                
                lowerBoundNumeric = Double.NaN;
                upperBoundNumeric =  Double.NaN;
	}
        
        /**
         * Constructor for numeric attribute, each numeric attribute is
         * represented by a term in form of x <= value < y
         * 
         * @param attributeIn
         * @param attributeIndexIn
         */
        public Term(Attribute attributeIn, double attributeIndexIn){
            
            attribute = attributeIn;
            setAttributeIndex(attributeIndexIn);
            
            // set type of the attribute
            if(attributeIn.isNumeric()){
                attributeType = TypeNumeric;
            }else{
                attributeType = 0;
            }
        }
        
	@Override
	public int compareTo(Object anotherTerm) {
		// TODO Auto-generated method stub
		
		if(!(anotherTerm instanceof GRules.Term)){
			throw new ClassCastException("A rule term object is expected.");
		}
		
		// check if two rule terms are the same
		if(getAttributeIndex() == ((GRules.Term) anotherTerm).getAttributeIndex() &&
				getAttributeValue() == ((GRules.Term) anotherTerm).getAttributeValue()){
			
			return 0;
		}else{
			return 1;
		}
	}
	
	public boolean coveredInstance(Instance instanceIn){
            
            if (isCategorial()) {
                return instanceIn.value((int) attributeIndex) == attributeValue;
            }else if(isNumeric()){
                return (upperBoundNumeric >= instanceIn.value((int) attributeIndex) && lowerBoundNumeric < instanceIn.value((int) attributeIndex));
            }
            
            return false;
	}
	
	public boolean parseAttributeValueDouble(Double[] attributeValueDouble){
		return false;
	}
	
        @Override
	public String toString(){
		
		String termString = "";
		
                if(isCategorial()){
                    termString += attribute.name();
                    termString += "=";
                    termString += attribute.value((int) attributeValue);
                }else if(isNumeric()){
                    termString += Double.toString(lowerBoundNumeric);
                    termString += " < ";
                    termString += attribute.name();
                    termString += " =< ";
                    termString += Double.toString(upperBoundNumeric);
                }
                


		return termString;
	}


	/**
	 * Returns a term/ condition of the term in form of [attributeIndex, attributeValue]
	 * 
	 * @return	the term in form double vector, [attributeIndex, attributeValue]
	 */
	public double[] getAttributeValueDobule(){
		double[] attributeValueDouble = new double[4];
		
                if(isCategorial()){
                    
                 attributeValueDouble[0] = TypeNomial;
                 attributeValueDouble[1] = attributeIndex;
                 attributeValueDouble[2] = attributeValue;
                    
                }else if(isNumeric()){
                    
                 attributeValueDouble[0] = TypeNumeric;
                 attributeValueDouble[1] = attributeIndex;
                 attributeValueDouble[2] = lowerBoundNumeric;    
                 attributeValueDouble[3] = upperBoundNumeric;   
                }
		
		return attributeValueDouble;
	}
	
	public String getAttributeValueDobuleString(){
		String attributeValueDouble = "";
		
                if(isCategorial()){
                    attributeValueDouble = Double.toString(TypeNomial);
                    attributeValueDouble += ";" + Double.toString(attributeIndex);
                    attributeValueDouble += ";" + Double.toString(attributeValue);                   
                }else if(isNumeric()){
                    attributeValueDouble = Double.toString(TypeNumeric);
                    attributeValueDouble += ";" + Double.toString(attributeIndex); 
                    attributeValueDouble += ";" + Double.toString(lowerBoundNumeric);
                    attributeValueDouble += ";" + Double.toString(upperBoundNumeric);
                }
                

		
		return attributeValueDouble;
	}
	
	/**
	 * Check where an attributeValue [attributeIndex, attributeValue] is equivalent to the term
	 * 
	 * @param attributeValueDobule attributeValue [attributeIndex, attributeValue] 
	 * @return true if the term is represented by @param [attributeIndex, attributeValue], otherwise, false;
	 */
	public boolean isMatching(double[] attributeValueDobule){
		
            if(attributeValueDobule[0] == Term.TypeNomial){
		if(attributeValueDobule[1] == attributeIndex && attributeValueDobule[2] == attributeValue){
			return true;
		}              
            }else if(attributeValueDobule[0] == Term.TypeNumeric){
 		if(attributeValueDobule[1] == attributeIndex){
			return true;
		}                     
            }
            
            
            return false;  
	}

	public double getAttributeIndex() {
		return attributeIndex;
	}



	private void setAttributeIndex(double attributeIndex) {
		this.attributeIndex = attributeIndex;
	}

	public double getAttributeValue() {
		return attributeValue;
	}

	private void setAttributeValue(double attributeValue) {
		this.attributeValue = attributeValue;
	}

        public void setLowerBoundNumeric(double lowerBoundNumeric) {
            this.lowerBoundNumeric = lowerBoundNumeric;
        }

        public void setUpperBoundNumeric(double upperBoundNumeric) {
            this.upperBoundNumeric = upperBoundNumeric;
        }

	public Attribute getAttribute() {
		return attribute;
	}
        
        public boolean isCategorial(){
            return attributeType == Term.TypeNomial;
        }
        
        public boolean isNumeric(){
            return attributeType == Term.TypeNumeric;
        }

        public double getLowerBoundNumeric() {
            return lowerBoundNumeric;
        }

        public double getUpperBoundNumeric() {
            return upperBoundNumeric;
        }
        
        public double[] termCode(){
            
            double[] termCode = new double[4];
            
            if(attributeType == TypeNomial){
                termCode[0] = TypeNomial;
                termCode[1] = attributeIndex;
                termCode[2] = attributeValue;
                termCode[3] = Double.NaN;
            }else{
                termCode[0] = TypeNumeric;
                termCode[1] = attributeIndex;
                termCode[2] = lowerBoundNumeric;
                termCode[3] = upperBoundNumeric;
            }
            
            return termCode;
        }
    }
    
    public class Rule{
        private final List<Term> leftSide;
        private final List<Term> rightSdie;
        
        int truePositive;
        int falsePositive;
        int trueNegative;
        int falseNegative;
        
        final int delayThreshold = 5;
        int warningCount;

        int createdTime;
        int lastModified;

        public int triggerNo;
        public Rule(List<Term> leftSideIn, List<Term> rightSdieIn){

            leftSide = new ArrayList<>(leftSideIn);
            rightSdie = new ArrayList<>(rightSdieIn);
            
            truePositive = 0;
            falsePositive = 0;
            trueNegative = 0;
            falseNegative = 0;
            warningCount = 0;
            createdTime = 0;
            lastModified = 0;
            triggerNo = 0;
        }

        public List<Term> getLeftSide() {
            return leftSide;
        }

        public List<Term> getRightSdie() {
            return rightSdie;
        }

        public String nicePrint(){
            StringBuilder ruleString = new StringBuilder();

            Iterator<Term> ruleBodyIterator = leftSide.iterator();
            while(ruleBodyIterator.hasNext()){

                Term aTerm = ruleBodyIterator.next();
                
                ruleString.append(aTerm);

                if(ruleBodyIterator.hasNext()){
                    ruleString.append(" & ");
                }
            }

            ruleString.append(" THEN ");

            // append the HEAD
            Iterator<Term>  ruleHeadIterator = rightSdie.iterator();
            while(ruleHeadIterator.hasNext()){

                Term aTerm = ruleHeadIterator.next();

                ruleString.append(aTerm);

                if(ruleHeadIterator.hasNext()){
                    ruleString.append(" & ");
                }
            }

            return ruleString.toString();
        }     

        public int getTruePositive() {
            return truePositive;
        }

        public int getFalsePositive() {
            return falsePositive;
        }

        public int getTrueNegative() {
            return trueNegative;
        }

        public int getFalseNegative() {
            return falseNegative;
        }
        
        public void wasTruePositive(){
            truePositive++;
            triggerNo++;
        }
       
        public void setCreatedTime(int createdTime) {
            this.createdTime = createdTime;
        }
        
        public double getTrueFalsePositiveRate(){
            
            if(falsePositive == 0){
                return (double) truePositive / 1.0d + Math.log(createdTime);
            }else{
                return (double) truePositive / (double) falsePositive + Math.log(createdTime);
            }
        }
        
        
        public void wasFalsePositive(){
            falsePositive++;
            triggerNo++;
        }
        
        public void wasTrueNegative(){
            trueNegative++;
            triggerNo++;
        }
        
        public void wasFalseNegative(){
            falseNegative++;
            triggerNo++;
        }
        
        public double tpr(){
           double value = (double) truePositive / (truePositive + falseNegative);
           return Double.isNaN(value) ? 0.0d : value;
        }
        
        public double fpr(){
            double value = (double) falsePositive / (falsePositive + trueNegative);
            return Double.isNaN(value) ? 0.0d : value;
        }
        
        public double pDistance(){
            return tpr() - fpr();
        }
        
        public int getTriggerCount(){
            return triggerNo;
        }
        
        public boolean isRemoveable(){     
            return falsePositive > 5;
        }
    }
    
    public static void main(String[] args){
        GRules GRulesLearner = new GRules();
        
        try {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/UciDataSets/nominal/Manchester.arff");
        Instances originalDataset = source.getDataSet();
        
        // split data 80/20
        int trainSize = (int) Math.round(originalDataset.numInstances() * 0.8
            / 100);
        int testSize = originalDataset.numInstances() - trainSize;
        Instances train = new Instances(originalDataset, 0, trainSize);
        Instances test = new Instances(originalDataset, trainSize, testSize);
        
        train.randomize(new java.util.Random(0));
        
        // train the rules learner
        List<List<Term>> rightSideRules = GRulesLearner.learnRightSide(train);
        List<Rule> completedRules = GRulesLearner.induceCompleteRules(rightSideRules, originalDataset);
        
        // try to predict an instance
        myLogger.info(test.get(10));
        myLogger.info(GRulesLearner.predict(test.get(10), completedRules).nicePrint());
        
        } catch (Exception ex) {
            myLogger.error(ex);
        }
    }
}
