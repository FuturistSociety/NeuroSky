/**
 * @author Steven L. Moxley
 * @version 1.2
 */
package org.futurist.neuralnet.evolve;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Playground {
	
	public static void main(String[] args) {
		String outputFile = "C:\\Users\\smoxley\\Downloads\\network-stats.txt";
		int numObservations = 5;
		
		Random yRNG = new Random();
		Double[] y = new Double[numObservations];
		for(int i = 0; i < numObservations; i++) {
			y[i] = yRNG.nextDouble();
		}
		
		Evolver evolver = new Evolver(y);
		File output = new File(outputFile);
		try {
			FileWriter writer = new FileWriter(output);
			writer.write(evolver.getFittestNetwork().getStats());
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}