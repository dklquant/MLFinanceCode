package com.randomoptimize;

/*
 * Created by Kyle West on 9/29/2018.
 * Adapted from PhishingWebsites.java by Daniel Cai (in turn adapted from AbaloneTest.java by Hannah Lau)
 */
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.*;

import java.text.DecimalFormat;
import java.util.Arrays;

public class NNOptimzeBase {
    private final  Instance[] instances = OptimizeUtil.initializeInstances();
    protected final  Instance[] trainings = Arrays.copyOfRange(instances, 0, 600);
    protected final  Instance[] testings = Arrays.copyOfRange(instances, 600, 950);

    private final  DataSet commandSets = new DataSet(trainings);

    private final  int inputting = 24;
    private final  int hiddens = 3;
    private final  int putputting = 1;

    private  BackPropagationNetworkFactory NNFactory = new BackPropagationNetworkFactory();

    private  ErrorMeasure errors = new SumOfSquaresError();

    protected  BackPropagationNetwork networks;
    protected  NeuralNetworkOptimizationProblem neuralNetProblems;

    protected  OptimizationAlgorithm optimizations;
    protected  String finals = "";

    protected  DecimalFormat myFormatting = new DecimalFormat("0.0");

    protected  final int[] m_iterations = {100,200,1000,2000,2500,3000,4000,5000, 10000, 20000, 100000};

    public  void createNeuralNet() {
        networks = NNFactory.createClassificationNetwork(
                new int[] {inputting, hiddens, putputting});
        neuralNetProblems = new NeuralNetworkOptimizationProblem(commandSets, networks, errors);
    }


    public  void run() {
        return;
    }

    protected  void train(
            OptimizationAlgorithm myOptimizer,
            BackPropagationNetwork network,
            int iteration) {

        for(int i = 0; i < iteration; i++) {
            myOptimizer.train();
            for(int j = 0; j < trainings.length; j++) {
                network.setInputValues(trainings[j].getData());
                network.run();

                Instance output = trainings[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
            }

        }
    }

}
