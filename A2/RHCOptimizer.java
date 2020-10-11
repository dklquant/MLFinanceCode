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
            double correct = 0, incorrect = 0;
            optimizations = new RandomizedHillClimbing(neuralNetProblems);
            train(optimizations, networks, trainingIterations); //trainer.train();

            Instance optimalInstance = optimizations.getOptimal();
            networks.setWeights(optimalInstance.getData());

            double predicted, actual;
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

            finals += "\nRHC:" + "," + "," + ": \nCorrect " + correct +
                    "\nIncorrect " + incorrect;
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

        finals += "\nRHC: "  + ": \nCorrect " + correct + "\nIncorrect " + incorrect;

        System.out.println(finals);
    }


}

