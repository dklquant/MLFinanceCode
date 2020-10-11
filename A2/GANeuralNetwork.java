package com.randomoptimize;

import opt.OptimizationAlgorithm;
import opt.ga.StandardGeneticAlgorithm;
import shared.Instance;

public class GANeuralNetwork extends NNOptimzeBase {

    public  int[] m_population = {100,250,500,750,1000,1250};
    public  int[] m_mate = {1,2,14,30,50,60};
    public  int[] m_mutes = {2,5,10,15,20,25};

    private  OptimizationAlgorithm createOptimzer(int population, int mate, int mute) {
        return new StandardGeneticAlgorithm(population, mate, mute, neuralNetProblems);
    }

    @Override
    public void run() {

        createNeuralNet();

        for (int trainingIterations : m_iterations) {
            runAlgo(trainings, trainingIterations);
        }
    }

    public  void runAlgo(Instance[] data, int trainingIterations) {
        finals = "";
        for (int q = 0; q < m_population.length; q++) {
            double correct = 0, incorrect = 0;
            optimizations = new StandardGeneticAlgorithm(m_population[q], m_mate[q], m_mutes[q], neuralNetProblems);
            train(optimizations, networks, trainingIterations); //trainer.train();

            Instance optimalInstance = optimizations.getOptimal();
            networks.setWeights(optimalInstance.getData());

            // Calculate Training Set Statistics //
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

            finals += "\nGA Results:" + "," + m_population[q] + "," + m_mate[q] + "," + m_mutes[q] + "," + ": \nCorrect " + correct  +
                    "\nIncorrect " + incorrect;
            runTestAlgo(testings, q);
        }
    }

    public void runTestAlgo(Instance[] data, int q) {
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

        finals += "\nGA Results: " + "," + m_population[q] + "," + m_mate[q] + "," + m_mutes[q] + ","  + ": \nCorrect " + correct +
                "\nIncorrect " + incorrect;
        System.out.println(finals);
    }
}

