/**
 * @author Steven L. Moxley
 * @version 0.1
 */
package org.futurist.neuralnet.evolve;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.distribution.AbstractRealDistribution;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.CauchyDistribution;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.distribution.WeibullDistribution;
import org.apache.commons.math3.genetics.Chromosome;
import org.apache.commons.math3.genetics.MutationPolicy;

public class NeuralNetworkMutation implements MutationPolicy {

	public static final int numParams = 8;

	protected Random rng;
	protected ArrayList<AbstractRealDistribution> dists;

	/**
	 * Default constructor to create a <code>NeuralNetworkMutation</code>.
	 */
	public NeuralNetworkMutation() {
		rng = new Random();
		dists = new ArrayList<AbstractRealDistribution>();

	}

	/**
	 * Mutate the given original <code>Chromosome</code> by randomly deciding how many genes to change and setting them to values randomly drawn from a range of valid values or return an error if the given <code>Chromosome</code> is not an instance of <code>NeuralNetworkChromosome</code>.
	 * @param c the original <code>Chromosome</code>.
	 * @return the mutated <code>Chromosome</code>.
	 */
	public Chromosome mutate(Chromosome c) {
		if(c instanceof NeuralNetworkChromosome) {
			NeuralNetworkChromosome original = (NeuralNetworkChromosome) c;

			// define the valid AbstractRealDistributions
			dists.add(new BetaDistribution(original.getStartNodes()*original.getNumOutputs(), original.getStartEdges()));
			dists.add(new CauchyDistribution());
			dists.add(new ChiSquaredDistribution(original.getStartNodes()*original.getNumOutputs()));
			dists.add(new ExponentialDistribution(original.getStartNodes()*original.getNumOutputs()));
			dists.add(new FDistribution(original.getStartNodes()*original.getNumOutputs(), original.getStartEdges()));
			dists.add(new NormalDistribution());
			dists.add(new TDistribution(original.getStartNodes()*original.getNumOutputs()));
			dists.add(new UniformRealDistribution());
			dists.add(new WeibullDistribution(original.getStartNodes()*original.getNumOutputs(), original.getStartEdges()));

			// create a mutant representation with placeholder parameters
			NeuralNetworkChromosome mutant = original;
			
			// set the mutant's ID to the original's ID concatenated with a random integer
			mutant.setID(rng.nextInt());

			// randomly decide which gene to mutate
			Boolean[] paramsToChange = new Boolean[numParams];
			for(int i = 1; i < numParams; i++) {
				paramsToChange[i] = rng.nextBoolean();

				// mutate a gene if it was selected
				if(paramsToChange[i]) {
					switch(i) {
					case 1: mutant.setDistribution(selectDistribution());
					case 2: mutant.setStartNodes(selectStartNodes());
					case 3: mutant.setLayerNodes(selectLayerNodes());
					case 4: mutant.setStartEdges(selectStartEdges());
					case 5: mutant.setLayers(selectLayers());
					case 6: mutant.setLearnRate(selectLearnRate());
					case 7: mutant.setLearnScale(selectLearnScale());
					}
				}
			}

			return mutant;
		} else {
			//throw new MathIllegalArgumentException(new DummyLocalizable("NeuralNetworkMutation works only with NeuralNetworkChromosome, not "), original);
			System.out.println("NeuralNetworkMutation works only with NeuralNetworkChromosome");
			return null;
		}
	}

	/**
	 * Select a randomly chosen <code>AbstractRealDistribution</code>.
	 * @return the randomly chosen <code>AbstractRealDistribution</code>.
	 */
	public AbstractRealDistribution selectDistribution() {
		return dists.get(rng.nextInt(dists.size()));
	}

	/**
	 * Select a randomly chosen number of input neurons from 2 to 100.
	 * @return the randomly chosen number of input neurons.
	 */
	public Integer selectStartNodes() {
		return 2 + rng.nextInt(99);
	}

	/**
	 * Select a randomly chosen number of neurons per hidden layer from 2 to 100.
	 * @return the randomly chosen number of neurons per hidden layer.
	 */
	public Integer selectLayerNodes() {
		return 2 + rng.nextInt(99);
	}

	/**
	 * Select a randomly chosen number of input synapses from 2 to 100.
	 * @return the randomly chosen number of input synapses.
	 */
	public Integer selectStartEdges() {
		return 2 + rng.nextInt(99);
	}

	/**
	 * Select a randomly chosen number of layers from 3 to 100.
	 * @return the randomly chosen number of layers.
	 */
	public Integer selectLayers() {
		return 3 + rng.nextInt(98);
	}

	/**
	 * Select a randomly chosen learning rate from 0.01 to 5.
	 * @return the randomly chosen learning rate.
	 */
	public Double selectLearnRate() {
		Double rate = 0.0;
		while(rate <= 0 || rate > 5) {
			rate = 0.01 + rng.nextInt(5) - Math.abs((1/rng.nextDouble()));
		}
		return rate;
	}

	/**
	 * Select a randomly chosen backpropagation learning scale factor from 0.01 to 5.
	 * @return the randomly chosen backpropagation learning scale factor.
	 */
	public Double selectLearnScale() {
		Double scale = 0.0;
		while(scale <= 0 || scale > 5) {
			scale = 0.01 + rng.nextInt(5) - Math.abs((1/rng.nextDouble()));
		}
		return scale;
	}

}
