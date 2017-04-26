package HomeWork2;

import java.util.*;
import java.util.Queue;

import weka.classifiers.Classifier;
import weka.core.*;

class BasicRule {
    int attributeIndex;
    int attributeValue;
}

class Rule {
    List<BasicRule> basicRule;
    double returnValue;
}

class Node {
    Node[] children;
    Node parent;
    int attributeIndex;
    double returnValue;
    Rule nodeRule = new Rule();
    Instances instances;

    public boolean isPerfectlyClassified() {
        double firstClassValue = instances.instance(0).classValue();

        for (Instance instance : instances) {
            double currentClassValue = instance.classValue();
            if (currentClassValue != firstClassValue)
                return false;
        }

        return true;
    }

    public double calcNodeImpurity(){
    }
}

public class DecisionTree implements Classifier {
    private Node rootNode;

    public enum PruningMode {None, Chi, Rule}

    private PruningMode m_pruningMode;
    Instances validationSet;
    private List<Rule> rules = new ArrayList<Rule>();
    private Queue<Node> m_nodesQueue;

    @Override
    public void buildClassifier(Instances arg0) throws Exception {

        m_nodesQueue = new LinkedList<>();
        m_nodesQueue.add(rootNode);

        // Build the tree
        buildTree(arg0);
    }

    private void buildTree(Instances data) {
        Node currentNode = m_nodesQueue.remove();

        while (!m_nodesQueue.isEmpty()) {
            if (!currentNode.isPerfectlyClassified()) {
                Attribute bestAttribute = findBestAttribute(currentNode); // color
                Enumeration<Object> attributeValues = bestAttribute.enumerateValues(); // green yellow red
                currentNode.children = new Node[bestAttribute.numValues()];

                int i = 0;
                while (attributeValues.hasMoreElements()) {
                    Object attributeValue = attributeValues.nextElement();
                    for (Instance instance : currentNode.instances) {
                        String bestAttributeValueOfCurrentInstance = instance.stringValue(bestAttribute);
                        Node currentChild;
                        if (bestAttributeValueOfCurrentInstance.equals(attributeValue)) {
                            currentChild = currentNode.children[i];
                            currentChild.instances.add(instance);
                        }
                    }
                    i++;
                }

                // Add the children to the queue
                for (Node child : currentNode.children) {
                    m_nodesQueue.add(child);
                    //todo: update rest of nodes fields
                }
            }

            // Move to the next node
            currentNode = m_nodesQueue.remove();
        }
    }

    private Attribute findBestAttribute(Node node) {
        double currentNodeImpurity = node.calcNodeImpurity();
        return null;
    }

    public void setPruningMode(PruningMode pruningMode) {
        m_pruningMode = pruningMode;
    }

    public void setValidation(Instances validation) {
        validationSet = validation;
    }

    @Override
    public double classifyInstance(Instance instance) {
        //TODO: implement classifyInstance
        return 0;
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
