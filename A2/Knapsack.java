package com.randomoptimize;

//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import java.util.Arrays;
import java.util.Random;
import opt.DiscreteChangeOneNeighbor;
import opt.GenericHillClimbingProblem;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.KnapsackEvaluationFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import shared.FixedIterationTrainer;

public class Knapsack {
    private static final Random random = new Random();
    private static final int NUM_ITEMS = 40;
    private static final int COPIES_EACH = 4;
    private static final double MAX_VALUE = 50.0D;
    private static final double MAX_WEIGHT = 50.0D;
    private static final double MAX_KNAPSACK_WEIGHT = 3200.0D;
    private static final int[] m_iterations = new int[] {
            100, 200, 500, 1000, 2000, 3000, 5000, 10000, 50000, 100000
    };

    public Knapsack() {
    }

    public static void main(String[] var0) {
        for (int iter : m_iterations) {
            int[] var1 = new int[40];
            Arrays.fill(var1, 4);
            double[] var2 = new double[40];
            double[] var3 = new double[40];

            for (int var4 = 0; var4 < 40; ++var4) {
                var2[var4] = random.nextDouble() * 50.0D;
                var3[var4] = random.nextDouble() * 50.0D;
            }

            int[] var19 = new int[40];
            Arrays.fill(var19, 5);
            KnapsackEvaluationFunction var5 = new KnapsackEvaluationFunction(var2, var3, 3200.0D, var1);
            DiscreteUniformDistribution var6 = new DiscreteUniformDistribution(var19);
            DiscreteChangeOneNeighbor var7 = new DiscreteChangeOneNeighbor(var19);
            DiscreteChangeOneMutation var8 = new DiscreteChangeOneMutation(var19);
            UniformCrossOver var9 = new UniformCrossOver();
            DiscreteDependencyTree var10 = new DiscreteDependencyTree(0.1D, var19);
            GenericHillClimbingProblem var11 = new GenericHillClimbingProblem(var5, var6, var7);
            GenericGeneticAlgorithmProblem var12 = new GenericGeneticAlgorithmProblem(var5, var6, var8, var9);
            GenericProbabilisticOptimizationProblem var13 = new GenericProbabilisticOptimizationProblem(var5, var6, var10);
            RandomizedHillClimbing var14 = new RandomizedHillClimbing(var11);
            FixedIterationTrainer var15 = new FixedIterationTrainer(var14, iter);
            var15.train();
            System.out.println(var5.value(var14.getOptimal()));
            SimulatedAnnealing var16 = new SimulatedAnnealing(100.0D, 0.95D, var11);
            var15 = new FixedIterationTrainer(var16, iter);
            var15.train();
            System.out.println(var5.value(var16.getOptimal()));
            StandardGeneticAlgorithm var17 = new StandardGeneticAlgorithm(200, 150, 25, var12);
            var15 = new FixedIterationTrainer(var17, iter);
            var15.train();
            System.out.println(var5.value(var17.getOptimal()));
            MIMIC var18 = new MIMIC(200, 100, var13);
            var15 = new FixedIterationTrainer(var18, iter);
            var15.train();
            System.out.println(var5.value(var18.getOptimal()));
        }
    }
}
