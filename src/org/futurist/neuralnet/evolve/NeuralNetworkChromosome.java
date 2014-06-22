/**
 * @author Steven L. Moxley
 * @version 1.2
 */
package org.futurist.neuralnet.evolve;

import java.util.List;

import org.apache.commons.math3.distribution.AbstractRealDistribution;
import org.apache.commons.math3.exception.util.DummyLocalizable;
import org.apache.commons.math3.genetics.AbstractListChromosome;
import org.apache.commons.math3.genetics.InvalidRepresentationException;
import org.futurist.neuralnet.Network;

public class NeuralNetworkChromosome extends AbstractListChromosome<Object> {

	protected Integer id;
	protected AbstractRealDistribution dist;	// probability distribution that determines which nodes to connect with an edge
	protected Integer startNodes;				// number of nodes in the input layer/functional unit
	protected Integer layerNodes;				// number of nodes per hidden layer
	protected Integer startEdges;				// number of connected synapses in the initial network
	protected Integer layers;					// number of hidden layers/hidden networks
	protected Double learnRate;					// learning weight; used for re-weighting edges
	protected Double learnScale;				// factor by which learning rate decays linearly w.r.t. distance from output layer
	protected Number[] yData;					// the target/ideal/correct values used to assess this NeuralNetworkChromosome's fitness
	protected Network neuralnet;
	protected Double fitness;

	/**
	 * Default constructor to create a NeuralNetworkChromosome with the given representation of a <code>Network</code>.
	 * @param representation[0] the unique ID of this Network.
	 * @param representation[1] the statistical distribution drawn from to determine neurons' initial values and firing thresholds.
	 * @param representation[2] the number of input neurons.
	 * @param representation[3] the number of neurons in each hidden layer.
	 * @param representation[4] the number of synapses.
	 * @param representation[5] the the number of layers.
	 * @param representation[6] the the learning rate used for reinforcement in backpropagation.
	 * @param representation[7] the scaling (or dampening) factor used to modulate the learning rate as backpropagation takes place.
	 * @param y the correct/ideal/target values that are used to train the network.
	 */
	public NeuralNetworkChromosome(List<Object> representation, Number[] y) {
		super(representation);

		yData = y;
		
		if(representation.get(0) instanceof Integer && (Integer) representation.get(0) >= 0) {
			id = (Integer) representation.get(0);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The network's ID must be an Integer greater than or equal to 0, but was set to " + representation.get(0) + "."), (Object) representation.get(0));
		}
		
		if(representation.get(1) instanceof AbstractRealDistribution) {
			dist = (AbstractRealDistribution) representation.get(1);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The network's statistical distribution must be an AbstractRealDistribution, but was set to " + representation.get(1) + "."), (Object) representation.get(1));
		}
		
		if(representation.get(2) instanceof Integer && (Integer) representation.get(2) >= 2 && (Integer) representation.get(2) <= 100) {
			startNodes = (Integer) representation.get(2);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The number of nodes in the first layer must be an Integer between 2 and 100, but was set to " + representation.get(2) + "."), (Object) representation.get(2));
		}
		
		if(representation.get(3) instanceof Integer && (Integer) representation.get(3) >= 2  && (Integer) representation.get(3) <= 100) {
			layerNodes = (Integer) representation.get(3);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The number of nodes in the hidden layers must be an Integer between 2 and 100, but was set to " + representation.get(3) + "."), (Object) representation.get(3));
		}
		
		if(representation.get(4) instanceof Integer && (Integer) representation.get(4) >= 2 && (Integer) representation.get(4) <= 100) {
			startEdges = (Integer) representation.get(4);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The number of synaptic edges in the first layer must be an Integer between 2 and 100, but was set to " + representation.get(4) + "."), (Object) representation.get(4));
		}
		
		if(representation.get(5) instanceof Integer && (Integer) representation.get(5) >= 3 && (Integer) representation.get(5) <= 100) {
			layers = (Integer) representation.get(5);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The number of layers must be an Integer between 3 and 100, but was set to " + representation.get(5) + "."), (Object) representation.get(5));
		}
		
		if(representation.get(6) instanceof Double && (Double) representation.get(6) >= 0.01 && (Double) representation.get(6) <= 5.0) {
			learnRate = (Double) representation.get(6);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The learning rate must be a Double between 0.01 and 5.0, but was set to " + representation.get(6) + "."), (Double) representation.get(6));
		}
		
		if(representation.get(7) instanceof Double && (Double) representation.get(7) >= 0.01 && (Double) representation.get(7) <= 5.0) {
			learnScale = (Double) representation.get(7);
		} else {
			throw new InvalidRepresentationException(new DummyLocalizable("The learning scale must be a Double between 0.01 and 5.0, but was set to " + representation.get(7) + "."), (Double) representation.get(7));
		}
		
		neuralnet = new Network(id, dist, startNodes, layerNodes, yData.length, startEdges, layers, learnRate, learnScale, yData);
	}

	/**
	 * Get the ID number of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the ID.
	 */
	public Integer getID() {
		return id;
	}
	
	/**
	 * Set the ID number of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param i the ID.
	 */
	public void setID(Integer i) {
		id = i;
	}
	
	/**
	 * Get the <code>AbstractRealDistribution</code> of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the <code>AbstractRealDistribution</code>.
	 */
	public AbstractRealDistribution getDistribution() {
		return dist;
	}

	/**
	 * Set the <code>AbstractRealDistribution</code> of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param d the <code>AbstractRealDistribution</code>,
	 */
	public void setDistribution(AbstractRealDistribution d) {
		dist = d;
	}

	/**
	 * Get the number of input neurons of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the number of input neurons.
	 */
	public Integer getStartNodes() {
		return startNodes;
	}

	/**
	 * Set the number of input neurons of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param s the number of input neurons.
	 */
	public void setStartNodes(Integer s) {
		startNodes = s;
	}

	/**
	 * Get the number of neurons per layer of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the number of neurons per layer.
	 */
	public Integer getLayerNodes() {
		return layerNodes;
	}

	/**
	 * Set the number of neurons per layer of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param l the number of neurons per layer.
	 */
	public void setLayerNodes(Integer l) {
		layerNodes = l;
	}

	/**
	 * Get the number of input synapses of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the number of input synapses.
	 */
	public Integer getStartEdges() {
		return startEdges;
	}

	/**
	 * Set the number of input synapses of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param s the number of input synapses.
	 */
	public void setStartEdges(Integer s) {
		startEdges = s;
	}
	
	/**
	 * Get the number of layers of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the number of layers.
	 */
	public Integer getLayers() {
		return layers;
	}

	/**
	 * Set the number of layers of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param l the number of layers.
	 */
	public void setLayers(Integer l) {
		layers = l;
	}

	/**
	 * Get the learning rate of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the learning rate.
	 */
	public Double getLearnRate() {
		return learnRate;
	}

	/**
	 * Set the learning rate of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param l the learning rate.
	 */
	public void setLearnRate(Double l) {
		learnRate = l;
	}

	/**
	 * Get the backpropagation learning scale factor of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the backpropagation learning scale factor.
	 */
	public Double getLearnScale() {
		return learnScale;
	}

	/**
	 * Set the backpropagation learning scale factor of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @param l the backpropagation learning scale factor.
	 */
	public void setLearnScale(Double l) {
		learnScale = l;
	}

	/**
	 * Get <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the <code>Network</code>.
	 */
	public Network getNeuralnet() {
		neuralnet = new Network(id, dist, startNodes, layerNodes, yData.length, startEdges, layers, learnRate, learnScale, yData);
		return neuralnet;
	}

	/**
	 * Get the number of output neurons of the <code>Network</code> represented by this <code>NeuralNetworkChromosome</code>.
	 * @return the number of output neurons.
	 */
	public Integer getNumOutputs() {
		return yData.length;
	}

	/**
	 * Get the correct/ideal/target values that are used to train the network.
	 * @return the values.
	 */
	public Number[] getData() {
		return yData;
	}

	/**
	 * Get this <code>NeuralNetworkChromosome's</code> fitness.
	 * @return the fitness.
	 */
	@Override
	public double fitness() {
		if(fitness == null || fitness <= 0) {
			neuralnet.run();
			fitness = 1 - neuralnet.getAverageError();
		}
		
		return fitness;
	}

	/**
	 * Dummy method to satisfy the compiler.  Actual validity checking is done in the default constructor.
	 * UPGRADE: Implement the same checks as the default constructor since this class allows "set" methods.
	 */
	@Override
	protected void checkValidity(List<Object> representation) throws InvalidRepresentationException {
		
	}

	/**
	 * Get a copy of this <code>NeuralNetworkChromosome</code>.
	 * @return the copy.
	 */
	@Override
	public AbstractListChromosome<Object> newFixedLengthChromosome(List<Object> representation) {
		return new NeuralNetworkChromosome(representation, yData);
	}

}
