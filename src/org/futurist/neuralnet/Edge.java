/**
 * @author Steven L. Moxley
 * @version 1.0
 */
package org.futurist.neuralnet;

import org.futurist.neuralnet.node.Node;

public class Edge {

	final Integer id;
	Node input;
	Node output;
	Double weight;
	
	/**
	 * Default constructor to create a network Edge (synapse) between Nodes (neurons) with the default weight of 100.0.
	 * @param i the unique ID assigned to this Edge.
	 * @param in the neuron whose axon sends an action potential to this synapse.
	 * @param out the neuron whose dendrite receives an action potential from this synapse.
	 */
	public Edge(Integer i, Node in, Node out) {
		id = i;
		input = in;
		output = out;
		weight = 100.00;
	}
	
	/**
	 * Constructor to create a network Edge (synapse) between Nodes (neurons) with the given weight.
	 * @param i the unique ID assigned to this Edge.
	 * @param in the neuron whose axon sends an action potential to this synapse.
	 * @param out the neuron whose dendrite receives an action potential from this synapse.
	 * @param w the weight given to this synaptic connection.
	 */
	public Edge(Integer i, Node in, Node out, Double w) {
		id = i;
		input = in;
		output = out;
		weight = w;
	}

	/** 
	 * Returns the unique ID assigned to this Edge.
	 * @return the ID.
	 */
	public Integer getID() {
		return id;
	}
	
	/** 
	 * Returns the neuron whose axon sends an action potential to this synapse.
	 * @return the Node representing the input neuron.
	 */
	public Node getInput() {
		return input;
	}
	
	/** 
	 * Sets the neuron whose axon sends an action potential to this synapse.
	 * @param input the Node representing the input neuron.
	 */
	public void setInput(Node input) {
		this.input = input;
	}

	/** 
	 * Returns the neuron whose dendrite receives an action potential from this synapse.
	 * @return the Node representing the output neuron.
	 */
	public Node getOutput() {
		return output;
	}

	/** 
	 * Sets the neuron whose dendrite receives an action potential from this synapse.
	 * @param output the Node representing the input neuron.
	 */
	public void setOutput(Node output) {
		this.output = output;
	}

	/** 
	 * Returns the weight given to this synaptic connection.
	 * @return the weight.
	 */
	public Double getWeight() {
		return weight;
	}

	/** 
	 * Returns the weight given to this synaptic connection.
	 * @param weight the weight.
	 */
	public void setWeight(Double weight) {
		this.weight = weight;
	}

	/** 
	 * Returns the type of Edge and its unique ID.
	 * @return a String representing the Edge type and its ID.
	 */
	@Override
	public String toString() { 
		return this.getClass().getName() + getID();
	}
	
}
