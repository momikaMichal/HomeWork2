package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

//todo: make sure we didnt read the class value when scanning the attributes

class BasicRule {
    int attributeIndex;
    int attributeValue;
}

class Rule {
    List<BasicRule> listOfBasicRules;
    double returnValue;

    public Rule() {
        listOfBasicRules = new LinkedList<BasicRule>();
        // TODO: what is the default returnValue if constructing a new rule
    }

    public List<BasicRule> getListOfBasicRules() {
        return listOfBasicRules;
    }
}

class Node {
    Node[] children;
    Node parent;
    BasicRule basicRule = new BasicRule();
    Rule nodeRule;
    Instances instances;

    public Node(Rule fathersRule) {
        this.nodeRule = fathersRule;
        //buildTree should add a new BasicRule to current list of BasicRules
    }

    public boolean isPerfectlyClassified() {
        double firstClassValue = instances.instance(0).classValue();

        for (Instance instance : instances) {
            double currentClassValue = instance.classValue();
            if (currentClassValue != firstClassValue)
                return false;
        }

        return true;
    }

    public boolean isLeaf() {
        if (this.children.length == 0)
            return true;
        return false;
    }

    // TODO: Input: Instance object (a subset of the training data), attribute index (int). - make sure it's fine to use the instance of the node object
    public double calcInfoGain(int attributeIndex) {
        double numOfInstances = this.instances.numInstances();
        //class values: 'recurrence-events', 'no-recurrence-events'
        int countRecurrenceEvents = 0;

        //count number of class-index=0 in order to calculate the probabilities for entropy
        for (int i = 0; i < numOfInstances; i++) {
            if (instances.instance(i).classValue() == 0) {
                countRecurrenceEvents++;
            }
        }

        double rootEntropy = calcEntropy(countRecurrenceEvents / numOfInstances);

        //we have attributeIndex (=color)
        //now need to divide to yellow/blue/red and calc entropy for each one
        double weightedAverageOfEntropyAccordingToAttributeIndex = 0;
        Attribute attribute = this.instances.attribute(attributeIndex);
        Enumeration<Object> attributeValues = attribute.enumerateValues();

        while (attributeValues.hasMoreElements()) {
            Object attributeValue = attributeValues.nextElement();
            //count total number instances having this attribute value and count the
            //number of instances having this attribute and class-index=0
            //these 2 params will give us the weighted entropy
            int totalNumberOfInstancesWithCurrentAttributeValue = 0;
            int numberOfInstancesWithCurrentAttributeValueAndClassIndexZero = 0;
            int numberOfInstancesWithCurrentAttributeValueAndClassIndexOne = 0;

            for (int i = 0; i < numOfInstances; i++) {
                Instance currentInstance = instances.instance(i);
                String currentInstanceAttributeValue = currentInstance.stringValue(attributeIndex);

                if (currentInstanceAttributeValue.equals(attributeValue)) {
                    totalNumberOfInstancesWithCurrentAttributeValue++;
                    if (currentInstance.classValue() == 0) {
                        numberOfInstancesWithCurrentAttributeValueAndClassIndexZero++;
                    } else {
                        numberOfInstancesWithCurrentAttributeValueAndClassIndexOne++;
                    }
                }
            }

            if (totalNumberOfInstancesWithCurrentAttributeValue != 0) {
                weightedAverageOfEntropyAccordingToAttributeIndex +=
                        (totalNumberOfInstancesWithCurrentAttributeValue / numOfInstances) *
                                calcEntropy(numberOfInstancesWithCurrentAttributeValueAndClassIndexZero /
                                        totalNumberOfInstancesWithCurrentAttributeValue);
            }
        }
        return rootEntropy - weightedAverageOfEntropyAccordingToAttributeIndex;
    }

    private double calcEntropy(double probability1) {
        return -((probability1 * Math.log(probability1)) + ((1 - probability1) * Math.log(1 - probability1)));
    }
}

public class DecisionTree implements Classifier {

    // Our members
    private List<Node> leavesNodes = new LinkedList<Node>();
    private Queue<Node> m_nodesQueue;
    private int m_numOfAttributes;

    // Given members
    public enum PruningMode {
        None, Chi, Rule
    }

    private Node rootNode;
    private PruningMode m_pruningMode;
    Instances validationSet;
    private List<Rule> rules = new ArrayList<Rule>();

    //tree rules includes all leaves rules. use this method only after building the tree
    private void setTreesRules() {
        for (Node node : leavesNodes) {
            rules.add(node.nodeRule);
        }
    }

    @Override
    public void buildClassifier(Instances arg0) throws Exception {

        // We subtract 1 since classValue is included
        m_numOfAttributes = arg0.numAttributes() - 1;

        // Adding root
        rootNode = new Node(new Rule());
        m_nodesQueue = new LinkedList<>();
        m_nodesQueue.add(rootNode);

        // Build the tree
        buildTree(arg0);
    }

    /**
     * Builds a tree based on a given instances set.
     *
     * @param data
     */
    private void buildTree(Instances data) {

        rootNode.instances = data;

        // TODO: after removing the root the condition doesn't hold and therefore we don't enter the "While"
        while (!m_nodesQueue.isEmpty()) {

            Node currentNode = m_nodesQueue.remove();

            if (currentNode.isPerfectlyClassified()) {
                // currentNode is a leaf. set its returning value and add it to leaves list
                currentNode.nodeRule.returnValue = currentNode.instances.instance(0).classValue();
                leavesNodes.add(currentNode);
            } else {
                // currentNode is not a leaf, thus need to be forked
                Attribute bestAttribute = findBestAttribute(currentNode); // color
                Enumeration<Object> attributeValues = bestAttribute.enumerateValues(); // green yellow red
                currentNode.children = new Node[bestAttribute.numValues()];

                //create each child with its fathers list of rules(nodeRules) so far
                for (int i = 0; i < currentNode.children.length; i++) {
                    currentNode.children[i] = new Node(currentNode.nodeRule);
                }


                int i = 0;// i indicates the index of the value of the attribute
                while (attributeValues.hasMoreElements()) {
                    Object attributeValue = attributeValues.nextElement();
                    Node currentChild = currentNode.children[i];
                    for (Instance instance : currentNode.instances) {
                        String bestAttributeValueOfCurrentInstance = instance.stringValue(bestAttribute);
                        if (bestAttributeValueOfCurrentInstance.equals(attributeValue)) {
                            currentChild.instances.add(instance);
                        }
                    }
                    //set attribute index&value for building the rules
                    currentChild.basicRule.attributeValue = i;
                    currentChild.basicRule.attributeIndex = bestAttribute.index();
                    //add the created basicRule to the list of the nodes rules
                    currentChild.nodeRule.listOfBasicRules.add(currentChild.basicRule);
                    currentChild.parent = currentNode;

                    i++;
                }

                // Add the children to the queue
                for (Node child : currentNode.children) {
                    m_nodesQueue.add(child);
                }
            }
        }
    }

    /**
     * Finds the attribute that brings us closer to perfect classification
     *
     * @param node
     * @return the best attribute
     */
    private Attribute findBestAttribute(Node node) {
        Attribute bestAttribute = null;
        double maxImpurity = Integer.MIN_VALUE;
        double currentNodeImpurity;

        // Iterate all attributes and find the one which gives the highest informationGain
        for (int i = 0; i < m_numOfAttributes; i++) {
            currentNodeImpurity = node.calcInfoGain(i);

            if (currentNodeImpurity > maxImpurity) {
                bestAttribute = node.instances.attribute(i);
                maxImpurity = currentNodeImpurity;
            }
        }

        return bestAttribute;
    }

    /**
     * Calculate the average on a given instances set (could be the training, test or validation set).
     * The average error is the total number of classification mistakes on the input instances set
     * and divides that by the number of instances in the input set.
     *
     * @param data
     * @return the average error
     */
    private double calcAvgError(Instances data) {

        int numOfInstances = data.numInstances();
        int numOfErrors = 0;

        for (int i = 0; i < numOfInstances; i++) {
            double predictedClassValue = classifyInstance(data.instance(i));
            double actualClassValue = data.instance(i).classValue();

            // Compare predicted class value to actual class value
            if (predictedClassValue != actualClassValue)
                numOfErrors++;
        }

        double averageError = numOfErrors / numOfInstances;
        return averageError;
    }

    /**
     * Calculates the chi square statistic of splitting the data according to this attribute as learned in class.
     *
     * @param data
     * @param attributeIndex
     * @return chi square statistic
     */
    private double calcChiSquare(Instances data, int attributeIndex) {

        int numOfValuesOfAttribute = data.attribute(attributeIndex).numValues();
        int numOfInstances = data.numInstances();
        Enumeration<Object> attributeValues = data.attribute(attributeIndex).enumerateValues();

        int[] Df = new int[numOfValuesOfAttribute]; // each cell represents number of instances with the same value of the attribute
        int[] pf = new int[numOfValuesOfAttribute]; // each cell represents number of instances with the same value of the attribute where classValue=0
        int[] nf = new int[numOfValuesOfAttribute]; // each cell represents number of instances with the same value of the attribute where classValue=1
        int[] p0 = new int[numOfValuesOfAttribute]; // each cell represents number of instances with classValue=0 divided by the total number of instances
        int[] p1 = new int[numOfValuesOfAttribute]; // each cell represents number of instances with classValue=1 divided by the total number of instances

        double chiSquareStatistic = 0;

        for (Instance instance : data) {

            while (attributeValues.hasMoreElements()) {
                Object attributeValue = attributeValues.nextElement();
                String currentInstanceAttributeValueAsString = instance.stringValue(attributeIndex);

                if (attributeValue.equals(currentInstanceAttributeValueAsString)) {
                    int indexOfValue = instance.attribute(attributeIndex).indexOfValue(currentInstanceAttributeValueAsString);
                    Df[indexOfValue]++; //Increment the cell of this value according ot its index

                    if (instance.classValue() == 0) {
                        pf[indexOfValue]++;
                        p0[indexOfValue]++;
                    } else {
                        nf[indexOfValue]++;
                        p1[indexOfValue]++;
                    }
                    continue;
                }
            }
        }

        for (int i = 0; i < numOfValuesOfAttribute; i++) {

            p0[i] = p0[i] / numOfInstances;
            p1[i] = p1[i] / numOfInstances;

            // For each attribute value, calculate:
            int E0 = Df[i] * p0[i];
            int E1 = Df[i] * p1[i];

            chiSquareStatistic += Math.pow(pf[i] - E0, 2) / E0 + Math.pow(nf[i] - E1, 2);
        }

        return chiSquareStatistic;
    }

    /**
     * Calculates for a given instance the number of consecutive conditions that hold in a certain node
     *
     * @param currentNode
     * @param instance
     * @return number of consecutive conditions
     */
    private int calculateNumberOfConsecutiveConditions(Node currentNode, Instance instance) {
        int numberOfConsecutiveConditions = 0;
        List<BasicRule> listOfBasicRules = currentNode.nodeRule.getListOfBasicRules();

        for (BasicRule basicRule : listOfBasicRules) {
            if (instance.value(basicRule.attributeIndex) == basicRule.attributeValue) {
                numberOfConsecutiveConditions++;
            } else {
                // TODO: make sure the logic is right: make sure that ConsecutiveConditions refer to ConsecutiveConditions starting form the first consition
                return numberOfConsecutiveConditions;
            }
        }
        return numberOfConsecutiveConditions;
    }

    /**
     * Sets the pruning mode according to the given pruning mode.
     *
     * @param pruningMode
     */
    public void setPruningMode(PruningMode pruningMode) {
        m_pruningMode = pruningMode;
    }

    /**
     * Sets the validation according to the given validation.
     *
     * @param validation
     */
    public void setValidation(Instances validation) {
        validationSet = validation;
    }

    /**
     * The classification of an instance is done by searching for the most suitable rule
     * The definition for the most suitable rule is follow by this steps:
     * (1) If an instance meets all conditions for a given rule then its classification will be the rule returning value.
     * (2) If there is no such a rule then you need to find the most suitable one, the one that meets the largest number
     * of consecutive conditions (from the most left condition – meaning the largest path from the root).
     * (3) If there are more than one rule with the largest number from the previous step then classify with the majority
     * of the returning values of those rules.
     *
     * @param instance the instance to classify
     * @return the predicted class value
     */
    @Override
    public double classifyInstance(Instance instance) {

        Node nodeWithMostSuitableRule = null;
        int maxNumberOfConsecutiveConditions = Integer.MIN_VALUE;

        for (Node leaf : leavesNodes) {
            // Iterating all leaves in order to find the node with the most suitable rule.
            int currentNumberOfConsecutiveConditions = calculateNumberOfConsecutiveConditions(leaf, instance);
            if (currentNumberOfConsecutiveConditions > maxNumberOfConsecutiveConditions) {
                // if the number of consecutive condition of the current node is higher than maximum,
                // then update maximum and set nodeWithMostSuitableRule with the current node
                maxNumberOfConsecutiveConditions = currentNumberOfConsecutiveConditions;
                nodeWithMostSuitableRule = leaf;
            } else if (currentNumberOfConsecutiveConditions == maxNumberOfConsecutiveConditions) {
                // if the number of consecutive condition of the current node is equal to the maximum,
                // then compare the return values of these 2 nodes and update maximum to the node with the higher return value.
                if (leaf.nodeRule.returnValue > nodeWithMostSuitableRule.nodeRule.returnValue)
                    nodeWithMostSuitableRule = leaf;
            }
        }

        return nodeWithMostSuitableRule.nodeRule.returnValue;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // Don't change
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // Don't change
        return null;
    }
}