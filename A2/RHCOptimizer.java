package com.randomoptimize;

import opt.RandomizedHillClimbing;
import shared.Instance;

public class RHCOptimizer extends NNOptimzeBase {

    @Override
    public  void run() {

        createNeuralNet();

        for (int trainingIterations : m_iterations) {
            runAlgo(trainings, trainingIterations);
        }
    }


    public  void runAlgo(Instance[] data, int trainingIterations) {
        finals = "";
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            optimizations = new RandomizedHillClimbing(neuralNetProblems);
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
                if (Math.abs(predicted - actual) < 0.5) {
                    correct++;
                } else {
                    incorrect++;
                }
            }

            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);

            finals += "\nTrain Results for RHC:" + "," + "," + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect  + "\nTraining time: " + myFormatting.format(trainingTime)
                    + " seconds\nTesting time: " + myFormatting.format(testingTime) + " seconds\n";
            runTestAlgo(testings, 0);
    }

    public  void runTestAlgo(Instance[] data, int q) {
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

        finals += "\nTest Results for RHC: " + "," + ","  + ": \nCorrectly classified " + correct + " instances." +
                "\nIncorrectly classified " + incorrect;

        System.out.println(finals);
    }


}

