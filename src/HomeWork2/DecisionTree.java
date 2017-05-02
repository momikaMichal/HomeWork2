package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

class BasicRule {
    int attributeIndex;
    int attributeValue;

    public BasicRule (int attributeIndex, int attributeValue) {
        this.attributeIndex = attributeIndex;
        this.attributeValue = attributeValue;
    }
}

class Rule {
    List<BasicRule> listOfBasicRules; // TODO: check if allowed to change name from basicRule to listOfBasicRules
    double returnValue;

    public Rule() {
        listOfBasicRules = new LinkedList<BasicRule>();
        // TODO: what is the default returnValue if constructing a new rule
    }
}

class Node {

    // DO NOT delete or change name - these are given members - we are not allowed to change them
    Node[] children;
    Node parent;
    int attributeIndex;
    double returnValue;
    Rule nodeRule = new Rule();

    // Our members
    BasicRule basicRule;
    Instances instances;

    public Node(Instances data, Rule fathersRule) {
        this.nodeRule = fathersRule;
        this.instances = new Instances(data, -1);
    }

    public boolean sameClassValueForAllInstances() {
        double classValueOfFirstInstance = instances.instance(0).classValue();

        for (Instance instance : instances) {
            double currentClassValue = instance.classValue();
            if (currentClassValue != classValueOfFirstInstance)
                return false;
        }

        return true;
    }

    public boolean sameAttributesValuesForAllInstances() {
        for (int i = 1; i < instances.size(); i++) {
            for (int j = 0; j < instances.numAttributes() - 1; j++) {
                if (instances.instance(0).value(j) != instances.instance(i).value(j)) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Finds the attribute that brings us closer to perfect classification
     *
     * @return the best attribute
     */
    public Attribute findBestAttribute() {
        Attribute bestAttribute = null;
        double maxImpurityReduction = 0;
        double currentNodeImpurityReduction;

        // Iterate all attributes and find the one which gives the highest informationGain
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            currentNodeImpurityReduction = this.calcInfoGain(i);

            if (currentNodeImpurityReduction > maxImpurityReduction) {
                bestAttribute = this.instances.attribute(i);
                maxImpurityReduction = currentNodeImpurityReduction;
            }
        }

        return bestAttribute;
    }

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

        double rootEntropy = calcEntropy((double) countRecurrenceEvents / numOfInstances);

        //we have attributeIndex (=color)
        //now need to divide to yellow/blue/red and calc entropy for each one
        double weightedAverageOfEntropyAccordingToAttributeIndex = 0;
        Attribute attribute = this.instances.attribute(attributeIndex);
        Enumeration<Object> attributeValues = attribute.enumerateValues();

        while (attributeValues.hasMoreElements()) {
            Object attributeValue = attributeValues.nextElement();
            /*
            count total number instances having this attribute value and count the
            number of instances having this attribute and class-index=0
            these 2 params will give us the weighted entropy
            */
            int totalNumberOfInstancesWithCurrentAttributeValue = 0;
            int numberOfInstancesWithCurrentAttributeValueAndClassIndexZero = 0;

            for (int i = 0; i < numOfInstances; i++) {
                Instance currentInstance = instances.instance(i);
                String currentInstanceAttributeValue = currentInstance.stringValue(attributeIndex);

                if (currentInstanceAttributeValue.equals(attributeValue)) {
                    totalNumberOfInstancesWithCurrentAttributeValue++;
                    if (currentInstance.classValue() == 0) {
                        numberOfInstancesWithCurrentAttributeValueAndClassIndexZero++;
                    }
                }
            }

            if (totalNumberOfInstancesWithCurrentAttributeValue != 0 && numberOfInstancesWithCurrentAttributeValueAndClassIndexZero != 0) {

                double entropyResult = calcEntropy((double) numberOfInstancesWithCurrentAttributeValueAndClassIndexZero /
                        totalNumberOfInstancesWithCurrentAttributeValue);

                double product = ((double) totalNumberOfInstancesWithCurrentAttributeValue / numOfInstances) * entropyResult;
                weightedAverageOfEntropyAccordingToAttributeIndex += product;
            }
        }

        return rootEntropy - weightedAverageOfEntropyAccordingToAttributeIndex;
    }

    private double calcEntropy(double probability1) {
        double probability2 = 1 - probability1;

        if (probability1 == 0) {
            probability1 = 1;
        }
        if (probability2 == 0) {
            probability2 = 1;
        }

        double entropy = - ((probability1 * Math.log(probability1)) + (probability2 * Math.log(probability2)));
        return entropy;
    }
}

public class DecisionTree implements Classifier {

    // Our members
    private List<Node> leavesNodes = new LinkedList<Node>();
    private Queue<Node> m_nodesQueue;
    private int m_numOfAttributes;
    private static final double THRESHOLD = 15.51;

    // Given members
    // DO NOT delete or change name - these are given members - we are not allowed to change them
    private Node rootNode;

    public enum PruningMode {
        None, Chi, Rule
    }

    private PruningMode m_pruningMode;
    Instances validationSet;
    private List<Rule> leavesRules = new ArrayList<Rule>(); // TODO: check if allowed to change name from rules to leavesRules and what's the meaning of it, sent ben a message

    @Override
    public void buildClassifier(Instances arg0) throws Exception {

        // We subtract 1 since classValue is included
        m_numOfAttributes = arg0.numAttributes() - 1;

        // Adding root
        rootNode = new Node(arg0, new Rule());
        m_nodesQueue = new LinkedList<>();
        m_nodesQueue.add(rootNode);

        // Set pruning mode to "chi"
        m_pruningMode = PruningMode.Chi;

        // Build the tree
        buildTree(arg0);

        // For debugging
        rootNode = rootNode;
    }

    /**
     * Builds a tree based on a given instances set.
     *
     * @param data
     */
    private void buildTree(Instances data) {

        rootNode.instances = data;

        while (!m_nodesQueue.isEmpty()) {

            Node currentNode = m_nodesQueue.remove();

            // If there are instances in the current node there's no need to handle it
            if (currentNode.instances.size() != 0) {
                double classValue = currentNode.instances.instance(0).classValue();

                // If the instances of currentNode have the same class value - the current node is a leaf
                // Or if the instances of currentNode have the same attributes values - the current node is a leaf
                if (currentNode.sameClassValueForAllInstances() || currentNode.sameAttributesValuesForAllInstances()) {
                    currentNode.nodeRule.returnValue = classValue;
                    currentNode.returnValue = classValue;
                    leavesNodes.add(currentNode);
                    continue;
                }

                // Otherwise, if currentNode is not a leaf, it has to be forked - find best attribute for splitting the data
                Attribute bestAttribute = currentNode.findBestAttribute();
                int indexOfBestAttribute = bestAttribute.index(); // TODO: use it after ben answers to set value for                            attributeIndex member

                // TODO: decide where to locate the chi square test and the logic
                // If your chi squared statistic is less than the threshold you prune.
                //double chiSquareStatistic = calcChiSquare(currentNode.instances, bestAttribute.index());
                //if (chiSquareStatistic >= THRESHOLD) {

                //create each child with its fathers list of rules(nodeRules) so far
                currentNode.children = new Node[bestAttribute.numValues()];
                for (int i = 0; i < currentNode.children.length; i++) {
                    currentNode.children[i] = new Node(data, currentNode.nodeRule);
                }

                // k indicates the index of the value of the attribute
                int k = 0;
                Enumeration<Object> attributeValues = bestAttribute.enumerateValues();
                while (attributeValues.hasMoreElements()) {

                    Object attributeValue = attributeValues.nextElement();
                    Node currentChild = currentNode.children[k];

                    // Update the child members values
                    currentChild.basicRule = new BasicRule(k, bestAttribute.index());
                    currentChild.nodeRule.listOfBasicRules.add(currentChild.basicRule);
                    currentChild.parent = currentNode;

                    // Update the instances member for that child
                    for (Instance instance : currentNode.instances) {
                        String bestAttributeValueOfCurrentInstance = instance.stringValue(bestAttribute);
                        if (bestAttributeValueOfCurrentInstance.equals(attributeValue)) {
                            currentChild.instances.add(instance);
                        }
                    }

                    k++;
                }

                // Add the children to the queue
                for (Node child : currentNode.children) {
                    m_nodesQueue.add(child);
                }
            }
        }
    }

    /**
     * Calculate the average on a given instances set (could be the training, test or validation set).
     * The average error is the total number of classification mistakes on the input instances set
     * and divides that by the number of instances in the input set.
     *
     * @param data
     * @return the average error
     */
    public double calcAvgError(Instances data) {

        int numOfInstances = data.numInstances();
        int numOfErrors = 0;

        for (int i = 0; i < numOfInstances; i++) {
            double predictedClassValue = classifyInstance(data.instance(i));
            double actualClassValue = data.instance(i).classValue();

            // Compare predicted class value to actual class value
            if (predictedClassValue != actualClassValue)
                numOfErrors++;
        }

        double averageError = (double) numOfErrors / numOfInstances;
        return averageError;
    }

    private void rulePruning() {
        setTreesRules();

        for (int i = 0; i < leavesNodes.size(); i++) {

            Node currentNode = leavesNodes.get(i);

            // Calculate average error before pruning
            double averageErrorBeforePruning = calcAvgError(currentNode.instances);

            // Remove the last basic rule in the list
            List<BasicRule> basicRulesOfCurrentRule = currentNode.nodeRule.listOfBasicRules;
            BasicRule RemovedBasicRule = basicRulesOfCurrentRule.remove(basicRulesOfCurrentRule.size() - 1);

            // Calculate average error after pruning
            calcAvgError(currentNode.parent.instances);

            // TODO: to be continued
        }
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

        double p0 = 0;
        double p1 = 0;

        int[] Df = new int[numOfValuesOfAttribute]; // each cell represents number of instances with the same value of the attribute
        int[] pf = new int[numOfValuesOfAttribute]; // each cell represents number of instances with the same value of the attribute where classValue=0
        int[] nf = new int[numOfValuesOfAttribute]; // each cell represents number of instances with the same value of the attribute where classValue=1

        double chiSquareStatistic = 0;

        for (Instance instance : data) {

            if (instance.classValue() == 0) {
                p0++;
            } else {
                p1++;
            }

            Enumeration<Object> attributeValues = data.attribute(attributeIndex).enumerateValues();
            while (attributeValues.hasMoreElements()) {
                Object attributeValue = attributeValues.nextElement();
                String currentInstanceAttributeValueAsString = instance.stringValue(attributeIndex);

                if (attributeValue.equals(currentInstanceAttributeValueAsString)) {
                    int indexOfValue = instance.attribute(attributeIndex).indexOfValue(currentInstanceAttributeValueAsString);
                    Df[indexOfValue]++; //Increment the cell of this value according ot its index

                    if (instance.classValue() == 0) {
                        pf[indexOfValue]++;
                    } else {
                        nf[indexOfValue]++;
                    }
                    continue;
                }
            }
        }

        p0 = p0 / numOfInstances;
        p1 = p1 / numOfInstances;

        for (int i = 0; i < numOfValuesOfAttribute; i++) {

            if (Df[i] != 0) {
                // For each attribute value, calculate:
                double E0 = Df[i] * p0;
                double E1 = Df[i] * p1;

                double a = Math.pow(pf[i] - E0, 2) / E0;
                double b = Math.pow(nf[i] - E1, 2) / E1;
                chiSquareStatistic += a + b;
            }
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
        List<BasicRule> listOfBasicRules = currentNode.nodeRule.listOfBasicRules;

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
     * Set tree rules according to leaves rules.
     **/
    private void setTreesRules() {
        for (Node node : leavesNodes) {
            leavesRules.add(node.nodeRule);
        }
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
     * Returns the amount of rules generated from the tree
     *
     * @return amount of rules
     */
    public int getAmountOfRules() {
        setTreesRules(); // TODO: remove after pruning is implemented
        return leavesRules.size();
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