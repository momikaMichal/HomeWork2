package HomeWork2;

// TODO: check that java.io.* is allowed
import weka.core.Instances;

import java.io.*;
import java.text.MessageFormat;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	/**
	 * no pruning- trainingCancer, testingCancer
	 * chi pruning- trainingCancer with chaiPruning mode, testingCancer
	 * postPruning - trainingCancer, validationCancer
	 */
    //todo: make sure that the order of the sets is correct
	private static void calssifyAndTest(DecisionTree decisionTree, Writer writer, DecisionTree.PruningMode pruningMode, Instances trainingCancer, Instances testingCancer) throws Exception{
		double averageTrainError,averageTestError;
		int amountOfRulesGeneratedFromTheTree;

		decisionTree.setPruningMode(pruningMode);
		decisionTree.buildClassifier(trainingCancer);
		averageTrainError = decisionTree.calcAvgError(trainingCancer);
        if (pruningMode==DecisionTree.PruningMode.Rule){
            decisionTree.setValidationSet(testingCancer);
            decisionTree.postPruning();
        }
		averageTestError = decisionTree.calcAvgError(testingCancer);
		amountOfRulesGeneratedFromTheTree = decisionTree.getAmountOfRules();

		writer.write(
                            "Decision Tree with " + pruningMode.toString() + " pruning\n"
                                    + "The average train error of the decision tree with " + pruningMode + " pruning is " + averageTrainError + "\n"
                                    + "The average test error of the decision tree with " + pruningMode + " pruning is " + averageTestError + "\n"
									+ "The amount of rules generated from the tree " + amountOfRulesGeneratedFromTheTree + "\n\n\n"
		);
	}

	public static void main(String[] args) throws Exception {
        Instances trainingCancer = loadData("cancer_train.txt");
        Instances testingCancer = loadData("cancer_test.txt");
        Instances validationCancer = loadData("cancer_validation.txt");

        Writer writer = null;
        try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("h2.txt"), "utf-8"));

			//Decision Tree with No pruning
			calssifyAndTest(new DecisionTree(),writer, DecisionTree.PruningMode.None,trainingCancer,testingCancer);

			//Decision Tree with Chi pruning
			calssifyAndTest(new DecisionTree(),writer, DecisionTree.PruningMode.Chi,trainingCancer,testingCancer);

			//Decision Tree with Rule pruning
			calssifyAndTest(new DecisionTree(),writer, DecisionTree.PruningMode.Rule,trainingCancer,validationCancer);

		} catch (IOException ex) {
            System.out.println(MessageFormat.format("An error occurred, ex:{0}", ex));
        } finally {
            try { //todo: there is a way to avoid this ugly finally ...
                writer.close();
            } catch (Exception ex) {
                System.out.println(MessageFormat.format("An error occurred while trying to finalize, ex:{0}", ex));
            }
        }
    }
}
