package com.randomoptimize;

import opt.SimulatedAnnealing;
import shared.Instance;

public class SAOptimizer extends NNOptimzeBase {

    public double[] m_cr = {0.02, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7};

    @Override
    public void run() {

        createNeuralNet();

        for (int trainingIterations : m_iterations) {
            runAlgo(trainings, trainingIterations);
        }
    }

    public void runAlgo(Instance[] data, int trainingIterations) {
        finals = "";
        for (double crates : m_cr) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            optimizations = new SimulatedAnnealing(1E12, crates, neuralNetProblems);
            train(optimizations, networks, trainingIterations); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = optimizations.getOptimal();
            networks.setWeights(optimalInstance.getData());

            // Calculate Training Set Statistics //
            double predicted, actual;
            start = System.nanoTime();
            for (int j = 0; j < data.length; j++) {
                networks.setInputValues(data[j].getData());
                networks.run();

                actual = Double.parseDouble(data[j].getLabel().toString());
                predicted = Double.parseDouble(networks.getOutputValues().toString());
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }

            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);

            finals += "\nTrain Results for SA:" + "," + crates + "," + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect  + "\nTraining time: " + myFormatting.format(trainingTime)
                    + " seconds\nTesting time: " + myFormatting.format(testingTime) + " seconds\n";
            runTestAlgo(testings, crates);
        }
    }

    public void runTestAlgo(Instance[] data, double crates) {
        double correct = 0, incorrect = 0;
        for (int j = 0; j < data.length; j++) {
            networks.setInputValues(data[j].getData());
            networks.run();

            double actual = Double.parseDouble(data[j].getLabel().toString());
            double predicted = Double.parseDouble(networks.getOutputValues().toString());
            if (Math.abs(predicted - actual) < 0.5) {
                correct++;
            } else {
                incorrect++;
            }

        }

        finals += "\nTest Results for GA: " + "," + crates + ","  + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect  + "\nTraining time: "
                + " seconds\nTesting time: "  + " seconds\n";
        System.out.println(finals);
    }

}

