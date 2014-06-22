/**
 * @author Steven L. Moxley
 * @version 1.2
 * To do for next release: implement getClosenessCentrality() and getBetweennessCentrality()
 */
package org.futurist.neuralnet.node;

import java.util.ArrayList;

import org.futurist.neuralnet.Edge;

public abstract class Node implements Runnable {

	final ArrayList<Integer> coordinateID;
	Double value;
	Double threshold;
	Integer numFires;
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 */
	public Node(ArrayList<Integer> i, Double v) { 
		coordinateID = new ArrayList<Integer>(i.size());
		for(int c = 0; c < i.size(); c++) {
			coordinateID.add(c, i.get(c));
		}
		value = v;
		threshold = new Double(1.0);
		numFires = 0;
	}
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 * @param t the firing threshold of this Node.
	 */
	public Node(ArrayList<Integer> i, Double v, Double t) { 
		coordinateID = new ArrayList<Integer>(i.size());
		for(int c = 0; c < i.size(); c++) {
			coordinateID.add(c, i.get(c));
		}
		value = v;
		threshold = t;
		numFires = 0;
	}

	public void run() {
		System.out.println("Neuron " + getID() + " is firing the value " + value + "!");
	}
	
	/** 
	 * Returns the current stored value of this Node.
	 * @return the current value stored by this Node.
	 */
	public Double getValue() {
		return value;
	}

	/**
	 * Sets the current stored value of this Node.
	 * @param value the value that should replace the previous value stored by this Node.
	 */
	public void setValue(Double value) {
		this.value = value;
	}

	/** 
	 * Returns the threshold required to be stored in order for this Node to fire.
	 * @return the firing threshold.
	 */
	public Double getThreshold() {
		return threshold;
	}

	/**
	 * Sets the threshold required to be stored in order for this Node to fire.
	 * @param threshold the firing threshold.
	 */
	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}

	/** 
	 * Returns the number of times this Node has fired (in the current simulation).
	 * @return the number of times this Node has fired so far.
	 */
	public Integer getNumFires() {
		return numFires;
	}
	
	/** 
	 * Returns the 3-dimensional coordinate used to identify this Node.
	 * @return the 3-D ID.
	 */
	public ArrayList<Integer> getID() {
		return coordinateID;
	}
	
	/** 
	 * Returns the value of the sigmoid function after operating on the given input.
	 * @param x the input to the sigmoid function.
	 * @return the result of the sigmoid function.
	 */
	public Double sigmoid(Double x) {
		Double denominator = 1 + Math.pow(Math.E, x*-1);
		return 1 / denominator;
	}
	
	// closeness centrality determines how close a node is to other nodes in a network by measuring the sum of the shortest distances (geodesic paths) between that node and all other nodes in the network
	/*
	public Integer getClosenessCentrality() {
		return 0;
	}
	*/
	
	// betweenness centrality determines the relative importance of a node by measuring the amount of traffic flowing through that node to other nodes in the network. This is done my measuring the fraction of paths connecting all pairs of nodes and containing the node of interest
	/*
	public Double getBetweennessCentrality() {
		return 0.0;
	}
	*/
	
	/** 
	 * Returns the type of Node and its 3-dimensional coordinate used to identify this Node.
	 * @return a String representing the Node type and its ID.
	 */
	@Override
	public String toString() {
		return this.getClass().getName() + getID();
	}
	
	public abstract ArrayList<Edge> getInputs();
	
	public abstract ArrayList<Edge> getOutputs();
	
	public abstract Integer getDegreeCentrality();
	
	public abstract Double getClusteringCoefficient();
	
	public abstract void receiveActionPotential(Edge input);
	
	public abstract void sendActionPotential(Edge output);
	
	public abstract boolean fire();
	
	public abstract boolean sigmoidFire();
	
}
