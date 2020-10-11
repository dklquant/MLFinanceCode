package com.randomoptimize;

import java.util.HashMap;

public class NNOptimizerFactory {
    public static HashMap<String, NNOptimzeBase> m_algos = new HashMap<>();

    public static void create() {
        m_algos.put("RHC", new RHCOptimizer());
        m_algos.put("GA", new GANeuralNetwork());
        m_algos.put("SA", new SAOptimizer());
    }

    public static void main (String[] args) {
        create();
        for (String key : m_algos.keySet()) {
            NNOptimzeBase algo = m_algos.get(key);
            algo.run();
        }
    }
}
