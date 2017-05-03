package HomeWork2;

// TODO: check that java.io.* is allowed
import java.io.*;
import java.text.MessageFormat;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork2.DecisionTree.PruningMode;
import weka.core.Instances;

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

	public static void main(String[] args) throws Exception {
        Instances trainingCancer = loadData("cancer_train.txt");
        Instances testingCancer = loadData("cancer_test.txt");
        Instances validationCancer = loadData("cancer_validation.txt");
        // TODO: we need to use validationCancer somehow

        // TEST
        Instances test = loadData("test.txt");
        DecisionTree dtt = new DecisionTree();
        dtt.setPruningMode(PruningMode.None);
        dtt.buildClassifier(test);
        dtt.calcAvgError(test);

        //Write results to hw2.txt
        Writer writer = null;
//        try {
//            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("h2.txt"), "utf-8"));
//
//            for (PruningMode mode : PruningMode.values()) {
//
//                //Train classifier
//                DecisionTree dt = new DecisionTree();
//                dt.setPruningMode(mode);
//                dt.buildClassifier(trainingCancer);
//
//                double averageTrainError = dt.calcAvgError(trainingCancer);
//                double averageTestError = dt.calcAvgError(testingCancer);
//                double amountOfRulesGeneratedFromTheTree = dt.getAmountOfRules();
//
//                if (mode.equals(PruningMode.None)) {
//                    writer.write(
//                            "Decision Tree with No pruning\n"
//                                    + "The average train error of the decision tree is " + averageTrainError + "\n"
//                                    + "The average test error of the decision tree is " + averageTestError + "\n"
//                    );
//                } else {
//                    writer.write(
//                            "Decision Tree with " + mode.toString() + " pruning\n"
//                                    + "The average train error of the decision tree with " + mode + " pruning is " + averageTrainError + "\n"
//                                    + "The average test error of the decision tree with " + mode + " pruning is " + averageTestError + "\n"
//                    );
//                }
//
//                writer.write("The amount of rules generated from the tree " + amountOfRulesGeneratedFromTheTree + "\n");
//            }
//        } catch (IOException ex) {
//            System.out.println(MessageFormat.format("An error occurred, ex:{0}", ex));
//        } finally {
//            try {
//                writer.close();
//            } catch (Exception ex) {
//                System.out.println(MessageFormat.format("An error occurred while trying to finalize, ex:{0}", ex));
//            }
//        }
    }
}
