package com.randomoptimize;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import java.io.PrintStream;
import java.util.Random;
import opt.GenericHillClimbingProblem;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.NQueensFitnessFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import shared.FixedIterationTrainer;

public class NQueens {
    private static final int N = 10;
    private static final int[] m_iterations = new int[] {
            100, 200, 500, 1000, 2000, 3000, 5000, 10000, 50000, 100000
    };

    public NQueens() {
    }

    public static void main(String[] var0) {
        for (int iter : m_iterations) {
            int[] var1 = new int[N];
            Random var2 = new Random(N);

            for (int var3 = 0; var3 < N; ++var3) {
                var1[var3] = var2.nextInt();
            }

            NQueensFitnessFunction var19 = new NQueensFitnessFunction();
            DiscretePermutationDistribution var4 = new DiscretePermutationDistribution(N);
            SwapNeighbor var5 = new SwapNeighbor();
            SwapMutation var6 = new SwapMutation();
            SingleCrossOver var7 = new SingleCrossOver();
            DiscreteDependencyTree var8 = new DiscreteDependencyTree(0.1D);

            GenericHillClimbingProblem var9 = new GenericHillClimbingProblem(var19, var4, var5);
            GenericGeneticAlgorithmProblem var10 = new GenericGeneticAlgorithmProblem(var19, var4, var6, var7);
            GenericProbabilisticOptimizationProblem var11 = new GenericProbabilisticOptimizationProblem(var19, var4, var8);

            RandomizedHillClimbing var12 = new RandomizedHillClimbing(var9);
            FixedIterationTrainer var13 = new FixedIterationTrainer(var12, iter);
            var13.train();
            long var14 = System.currentTimeMillis();
            PrintStream var10000 = System.out;
            double var10001 = var19.value(var12.getOptimal());
            var10000.println("RHC: " + var10001);
            System.out.println("RHC: Board Position: ");
            System.out.println("Time : " + (System.currentTimeMillis() - var14));
            System.out.println("============================");

            SimulatedAnnealing var16 = new SimulatedAnnealing(10.0D, 0.1D, var9);
            var13 = new FixedIterationTrainer(var16, iter);
            var13.train();
            var14 = System.currentTimeMillis();
            var10000 = System.out;
            var10001 = var19.value(var16.getOptimal());
            var10000.println("SA: " + var10001);
            System.out.println("SA: Board Position: ");
            System.out.println("Time : " + (System.currentTimeMillis() - var14));
            System.out.println("============================");

            var14 = System.currentTimeMillis();
            StandardGeneticAlgorithm var17 = new StandardGeneticAlgorithm(200, 0, 10, var10);
            var13 = new FixedIterationTrainer(var17, iter);
            var13.train();
            var10000 = System.out;
            var10001 = var19.value(var17.getOptimal());
            var10000.println("GA: " + var10001);
            System.out.println("GA: Board Position: ");
            System.out.println("Time : " + (System.currentTimeMillis() - var14));
            System.out.println("============================");

            var14 = System.currentTimeMillis();
            MIMIC var18 = new MIMIC(200, 10, var11);
            var13 = new FixedIterationTrainer(var18, iter);
            var13.train();
            var10000 = System.out;
            var10001 = var19.value(var18.getOptimal());
            var10000.println("MIMIC: " + var10001);
            System.out.println("MIMIC: Board Position: ");
            System.out.println("Time : " + (System.currentTimeMillis() - var14));
        }
    }
}
