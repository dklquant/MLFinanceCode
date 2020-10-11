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
            double correct = 0, incorrect = 0;
            optimizations = new SimulatedAnnealing(1E12, crates, neuralNetProblems);
            train(optimizations, networks, trainingIterations);

            Instance optimalInstance = optimizations.getOptimal();
            networks.setWeights(optimalInstance.getData());

            double notActual, real;
            for (int j = 0; j < data.length; j++) {
                networks.setInputValues(data[j].getData());
                networks.run();

                real = Double.parseDouble(data[j].getLabel().toString());
                notActual = Double.parseDouble(networks.getOutputValues().toString());
                if (Math.abs(notActual - real) < 0.5) {
                    correct++;
                } else {
                    incorrect++;
                }
            }

            finals += "\nSA:" + "," + crates + "," + "\nCorrect " + correct +
                    "\nIncorrect " + incorrect;
            runTestAlgo(testings, crates);
        }
    }

    public void runTestAlgo(Instance[] data, double crates) {
        double correct = 0;
        double incorrect = 0;
        for (int j = 0; j < data.length; j++) {
            networks.setInputValues(data[j].getData());
            networks.run();

            double real = Double.parseDouble(data[j].getLabel().toString());
            double notActual = Double.parseDouble(networks.getOutputValues().toString());
            if (Math.abs(notActual - real) < 0.5) {
                correct++;
            } else {
                incorrect++;
            }

        }

        finals += "\nSA: " + "," + crates + ","  + ": \nCorrect " + correct +
                "\nIncorrect " + incorrect;
        System.out.println(finals);
    }

}

