package com.randomoptimize;

import shared.Instance;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Scanner;

public class OptimizeUtil {

    public static Instance[] initializeInstances() {

        double[][][] attributes = new double[950][][];

        BufferedReader br = new BufferedReader(new FileReader(new File("")));

        for(int i = 0; i < attributes.length; i++) {
            Scanner scan = new Scanner(br.readLine());
            scan.useDelimiter(",");

            attributes[i] = new double[2][];
            attributes[i][0] = new double[24];
            attributes[i][1] = new double[1];

            for(int j = 0; j < 24; j++)
                attributes[i][0][j] = Double.parseDouble(scan.next());

            attributes[i][1][0] = Double.parseDouble(scan.next());
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }

}

