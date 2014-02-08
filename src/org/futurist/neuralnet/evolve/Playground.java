package org.futurist.neuralnet.evolve;

import java.util.Random;

public class Playground {
	
	public static void main(String[] args) {
		int numObservations = 5;
		
		Random yRNG = new Random();
		Double[] y = new Double[numObservations];
		for(int i = 0; i < numObservations; i++) {
			y[i] = yRNG.nextDouble();
		}
		
		Evolver evolver = new Evolver(y);
		evolver.getFittestNetwork().printStats();
	}

}