/**
 * @author Steven L. Moxley
 * @version 1.2
 */
package org.futurist.neuralnet.evolve;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.genetics.Chromosome;
import org.apache.commons.math3.genetics.FixedGenerationCount;
import org.apache.commons.math3.genetics.GeneticAlgorithm;
import org.apache.commons.math3.genetics.OnePointCrossover;
import org.apache.commons.math3.genetics.ElitisticListPopulation;
import org.apache.commons.math3.genetics.Population;
import org.apache.commons.math3.genetics.StoppingCondition;
import org.apache.commons.math3.genetics.TournamentSelection;
import org.futurist.neuralnet.Network;

public class Evolver extends Thread {

	public static final int DEFAULT_POPULATION_SIZE = 5;
	public static final double DEFAULT_ELITE_RATE = 0.25;
	public static final int DEFAULT_CROSSOVER_RATE = 1;
	public static final double DEFAULT_MUTATION_RATE = 0.1;
	public static final int DEFAULT_NUM_GENERATIONS = 10;
	public static final int DEFAULT_TIMEOUT_SECONDS = 60;

	private Double[] yData;
	private ElitisticListPopulation initialPop;
	private StoppingCondition stopCond;
	private GeneticAlgorithm ga;
	private Population finalPop;
	private Chromosome bestInitial;
	private Chromosome bestFinal;

	/**
	 * Default constructor to create an <code>Evolver<code>.
	 * @param y the target/ideal/correct values that the Evolver should consider the fittest possible values.
	 */
	public Evolver(Double[] y) {
		yData = y;

		// initialize a random population
		initialPop = getRandomPopulation(DEFAULT_POPULATION_SIZE, DEFAULT_ELITE_RATE);

		// set the stopping condition
		stopCond = new FixedGenerationCount(DEFAULT_NUM_GENERATIONS);

		// initialize a new genetic algorithm
		ga = new GeneticAlgorithm(new OnePointCrossover<Integer>(), DEFAULT_CROSSOVER_RATE, new NeuralNetworkMutation(), DEFAULT_MUTATION_RATE, new TournamentSelection(initialPop.getPopulationSize() / 2));

		run();
	}

	/**
	 * Constructor to create an <code>Evolver<code> with the given parameters.
	 * @param y the target/ideal/correct values that the Evolver should consider the fittest possible values.
	 * @param popSize the size of the initial population.
	 * @param eliteRate the percentage of most fit individuals to survive into the next generation.
	 * @param numGen the number of generations to run before the evolution is considered finished.
	 * @param crossoverRate the probability of the <code>CrossoverPolicy</code> being applied to parent chromosomes.
	 * @param mutationRate the probability of the <code>MutationPolicy</code> being applied to offspring chromosomes.
	 */
	public Evolver(Double[] y, int popSize, int eliteRate, int numGen, double crossoverRate, double mutationRate) {
		yData = y;

		// initialize a random population
		initialPop = getRandomPopulation(popSize, eliteRate);

		// set the stopping condition
		stopCond = new FixedGenerationCount(numGen);

		// initialize a new genetic algorithm
		ga = new GeneticAlgorithm(new OnePointCrossover<Integer>(), crossoverRate, new NeuralNetworkMutation(), mutationRate, new TournamentSelection(initialPop.getPopulationSize() / 2));

		run();
	}

	/**
	 * Generate a random <code>ElitisticListPopulation</code> of <code>NeuralNetworkChromosomes</code> of the given size with the given elitism rate.
	 * @param size the size of the population.
	 * @param eliteRate the percentage of most fit individuals to survive into the next generation.
	 * @return the random random <code>ElitisticListPopulation</code> of <code>NeuralNetworkChromosomes</code>.
	 */
	public ElitisticListPopulation getRandomPopulation(int size, double eliteRate) {
		ElitisticListPopulation pop = new ElitisticListPopulation(size, eliteRate);
		Random rng = new Random();
		for(int i = 0; i < size; i++) {
			int startNodes = 2 + rng.nextInt(99);
			ArrayList<Object> chromosome = new ArrayList<Object>();

			chromosome.add(0, i);
			chromosome.add(1, new ChiSquaredDistribution(startNodes*yData.length));
			chromosome.add(2, startNodes);
			chromosome.add(3, 2 + rng.nextInt(99));
			chromosome.add(4, 2 + rng.nextInt(99));
			chromosome.add(5, 3 + rng.nextInt(98));
			Double rate = 0.0;
			while(rate <= 0 || rate > 5) {
				rate = 0.01 + rng.nextInt(5) - Math.abs((1/rng.nextDouble()));
			}
			Double scale = 0.0;
			while(scale <= 0 || scale > 5) {
				scale = 0.01 + rng.nextInt(5) - Math.abs((1/rng.nextDouble()));
			}

			chromosome.add(6, rate);	
			chromosome.add(7, scale);
			pop.addChromosome(new NeuralNetworkChromosome(chromosome, yData));
			System.out.println("Added chromosome " + chromosome.get(0) + " to population with " +  chromosome.get(1).toString() + ", " + chromosome.get(2) + " input neurons, " + chromosome.get(3) + " layer neurons, " + chromosome.get(4) + " synapses, " + chromosome.get(5) + " layers, " + chromosome.get(6) + " learning rate, and " + chromosome.get(7) + " learning scale.");
		}
		return pop;
	}

	/**
	 * Run the <code>GeneticAlgorithm</code>.
	 */
	public void run() {

		long startTime = System.nanoTime();

		// save the most fit chromosome from the initial random population for later comparison
		bestInitial = initialPop.getFittestChromosome();

		// run the algorithm
		finalPop = ga.evolve(initialPop, stopCond);

		// best chromosome from the final population
		bestFinal = finalPop.getFittestChromosome();

		Double initialFitness = bestInitial.getFitness();
		Double finalFitness = bestFinal.getFitness();
		System.out.println("The most fit chromosome in the final population has a fitness of " + finalFitness + " compared to " + initialFitness + " in the initial random population's, a " + 100 * (finalFitness-initialFitness) / initialFitness + "% improvement.");
		System.out.println("The genetic algorithm finished running in " + (System.nanoTime()-startTime) / 1000000000.0 + " seconds.");
	}

	/**
	 * Get the fittest <code>Chromosomes</code> after the population has finished evolving.
	 * @return the fittest <code>Chromosomes</code>.
	 */
	public Chromosome getBestFinalChromosome() {
		return bestFinal;
	}
	
	/**
	 * Get the fittest <code>Network</code> after the population has finished evolving.
	 * @return the fittest <code>Network</code>.
	 */
	public Network getFittestNetwork() {
		NeuralNetworkChromosome fittest = (NeuralNetworkChromosome) bestFinal;
		return fittest.getNeuralnet();
	}

}