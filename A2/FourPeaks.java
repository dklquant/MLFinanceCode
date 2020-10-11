package com.randomoptimize;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import java.io.PrintStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import opt.DiscreteChangeOneNeighbor;
import opt.GenericHillClimbingProblem;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import shared.FixedIterationTrainer;

public class FourPeaks {
    private static final int N = 300;
    private static final int T = N / 5;
    private static final int[] m_iterations = new int[] {
            100, 200, 500, 1000, 2000, 3000, 5000, 10000, 50000, 100000
    };

    public FourPeaks() {
    }

    public static void main(String[] var0) {
        int[] var1 = new int[N];
        Arrays.fill(var1, 2);
        FourPeaksEvaluationFunction var2 = new FourPeaksEvaluationFunction(T);
        DiscreteUniformDistribution var3 = new DiscreteUniformDistribution(var1);
        DiscreteChangeOneNeighbor var4 = new DiscreteChangeOneNeighbor(var1);
        DiscreteChangeOneMutation var5 = new DiscreteChangeOneMutation(var1);
        SingleCrossOver var6 = new SingleCrossOver();
        DiscreteDependencyTree var7 = new DiscreteDependencyTree(0.1D, var1);
        GenericHillClimbingProblem var8 = new GenericHillClimbingProblem(var2, var3, var4);
        GenericGeneticAlgorithmProblem var9 = new GenericGeneticAlgorithmProblem(var2, var3, var5, var6);
        GenericProbabilisticOptimizationProblem var10 = new GenericProbabilisticOptimizationProblem(var2, var3, var7);

        for (int iter : m_iterations) {
            System.out.println("Iterations " + iter);
            RandomizedHillClimbing var11 = new RandomizedHillClimbing(var8);
            FixedIterationTrainer var12 = new FixedIterationTrainer(var11, iter);
            var12.train();
            PrintStream var10000 = System.out;
            double var10001 = var2.value(var11.getOptimal());
            var10000.println("RHC: " + var10001);

            SimulatedAnnealing var13 = new SimulatedAnnealing(1.0E11D, 0.95D, var8);
            var12 = new FixedIterationTrainer(var13, iter);
            var12.train();
            var10000 = System.out;
            var10001 = var2.value(var13.getOptimal());
            var10000.println("SA: " + var10001);

            StandardGeneticAlgorithm var14 = new StandardGeneticAlgorithm(7000, 90, 7, var9);
            var12 = new FixedIterationTrainer(var14, iter);
            var12.train();
            var10000 = System.out;
            var10001 = var2.value(var14.getOptimal());
            var10000.println("GA: " + var10001);
            MIMIC var15 = new MIMIC(200, 20, var10);
            var12 = new FixedIterationTrainer(var15, iter);
            var12.train();
            var10000 = System.out;
            var10001 = var2.value(var15.getOptimal());
            var10000.println("MIMIC: " + var10001);
        }
    }
}
