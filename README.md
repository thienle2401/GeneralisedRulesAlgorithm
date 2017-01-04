# Generalised Rules Algorithm

The expressiveness of descriptive models is of paramount importance in applications that examine the causality of relationships between variables. The implementation was used to carry out the experiment as presented in:
> _"Le, T., Stahl, F., Gaber, M. M., & Wrench, C. A Statistical Learning Method to Fast Generalised Rule Induction Directly from Raw Measurements"._

## Getting Started

The implementation is a Maven project and the algorithm required specified dependencies to work. The algorithm can be used alone to learn rules from training data and evaluate unseen data instances from learnt result. 


The algorithm uses instance and Instances classes as an input of a dataset and a data instance. Therefore, the code can work along with an existing development that has made use of Instance, Instances, Attribute classes from Weka.

### Running the code

Generate a ruleset from a training dataset. 

```java
// define che learner
GRules GRulesLearner = new GRules();

// load data from arff (Weka formmatted) data file
DataSource source = new DataSource("data/UciDataSets/nominal/Manchester.arff");

// split data 80/20
int trainSize = (int) Math.round(originalDataset.numInstances() * 0.8
/ 100);
int testSize = originalDataset.numInstances() - trainSize;
Instances train = new Instances(originalDataset, 0, trainSize);
Instances test = new Instances(originalDataset, trainSize, testSize);
        
// train the rules learner
List<List<Term>> rightSideRules = GRulesLearner.learnRightSide(train);
List<Rule> completedRules = GRulesLearner.induceCompleteRules(rightSideRules, originalDataset);

// get an unseen data instance from test set
Instance unseenIsntance = test.get(10);

// predict the HEAD part for a given data instance if these is any rule with BODY matches the data instance
Rule matchedRule = GRulesLearner.predict(test.get(10), completedRules).nicePrint();
System.out.println(matchedRule.nicePrint());
```

### Use the algorithm in development

Depend on the desired output when classifying a data instance, the user can modify the predict() in GRules class to return a required output such as index and value of the attribute-value pair(s) in HEAD part.

## Dependencies Used

* Weka-package (04/2016)
* Commons Math 3.6

## Authors

* **Thien Duyen Le**, University of Reading (t.d.le@reading.ac.uk)

Other contributors who participated in this project:

* Frederic Stahl, University of Reading
* Mohamed Medhat Gaber, Birmingham City University
* Chris Wrench, University of Reading

## Citing 

If you want to refer to this implementation in a publication, please cite the following paper:

> Le, T., Stahl, F., Gaber, M. M., & Wrench, C. A Statistical Learning Method to Fast Generalised Rule Induction Directly from Raw Measurements


## Acknowledgments

This development and the research has been supported by the UK Engineering, and Physical Sciences Research Council (EPSRC) grant EP/M016870/1.