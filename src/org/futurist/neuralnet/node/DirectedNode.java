/**
 * @author Steven L. Moxley
 * @version 1.0
 */
package org.futurist.neuralnet.node;

import java.util.ArrayList;

import org.futurist.neuralnet.Edge;

public abstract class DirectedNode extends Node {
	
	ArrayList<Edge> inputs;
	ArrayList<Edge> outputs;
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 */
	public DirectedNode(ArrayList<Integer> i, Double v) { 
		super(i, v);
		inputs = new ArrayList<Edge>();
		outputs = new ArrayList<Edge>();
	}
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 * @param t the firing threshold of this Node.
	 * @param in the synaptic Edges from which this Node receives action potentials
	 * @param out the synaptic Edges to which this Node sends action potentials
	 */
	public DirectedNode(ArrayList<Integer> i, Double v, Double t, ArrayList<Edge> in, ArrayList<Edge> out) { 
		super(i, v, t);
		if(in == null) {
			inputs = new ArrayList<Edge>();
		} else {
			inputs = in;
		}
		if(out == null) {
			outputs = new ArrayList<Edge>();
		} else {
			outputs = in;
		}
	}
	
	/**
	 * Returns this Node's dendrites (input synapses that are directly connected neighbors).
	 * @return this Node's dendrites.
	 */
	public ArrayList<Edge> getInputs() {
		return inputs;
	}
	
	/**
	 * Add a directly connected dendrite (input synapse) to this Node's existing input <code>Edges</code>.
	 * @param e the dendrite <code>Edge</code> to add.
	 */
	public void addInput(Edge e) {
		if(!inputs.contains(e)) {
			inputs.add(e);
			//System.out.println("Added " + e + " as an input.");
		} else {
			System.out.println(e + " is already an input.");
		}
	}
	
	/**
	 * Remove a directly connected dendrite (input synapse) from this Node's existing input <code>Edges</code>.
	 * @param e the dendrite <code>Edge</code> to remove.
	 */
	public void removeInput(Edge e) {
		if(inputs.contains(e)) {
			inputs.remove(e);
			//System.out.println("Removed " + e + " as an input.");
		} else {
			System.out.println(e + " is not an input.");
		}
	}

	/**
	 * Returns this Node's axons (output synapses that are directly connected neighbors).
	 * @return this Node's axons.
	 */
	public ArrayList<Edge> getOutputs() {
		return outputs;
	}
	
	/**
	 * Add a directly connected axon (output synapse) to this Node's existing output <code>Edges</code>.
	 * @param e the axon <code>Edge</code> to add.
	 */
	public void addOutput(Edge e) {
		if(!outputs.contains(e)) {
			outputs.add(e);
			//System.out.println("Added " + e + " as an output.");
		} else {
			System.out.println(e + " is already an output.");
		}
	}
	
	/**
	 * Remove a directly connected axon (output synapse) from this Node's existing output <code>Edges</code>.
	 * @param e the axon <code>Edge</code> to remove.
	 */
	public void removeOutput(Edge e) {
		if(outputs.contains(e)) {
			outputs.remove(e);
			//System.out.println("Removed " + e + " as an output.");
		} else {
			System.out.println(e + " is not an output.");
		}
	}
	
	/**
	 * Returns this <code>Node's</code> degree centrality.
	 * @return the degree centrality.
	 */
	public Integer getDegreeCentrality() {
		return inputs.size();
	}
	
	/**
	 * Returns this <code>Node's</code> clustering coefficient.
	 * @return the clustering coefficient.
	 */
	public Double getClusteringCoefficient() {
		ArrayList<Node> neighborhood = new ArrayList<Node>(outputs.size());
		int neighborEdges = 0;
		
		for(Edge e : outputs) {
			neighborhood.add(e.getOutput());
		}
		
		for(Node n : neighborhood) {
			for(Edge e : n.getOutputs()) {
				if(!e.getOutput().equals(this) && neighborhood.contains(e.getOutput())) {
					neighborEdges++;
				}
			}
		}
		
		return new Double(neighborEdges / (Math.pow(neighborhood.size(), 2) - neighborhood.size()));
	}
	
	public abstract void receiveActionPotential(Edge input);
	
	public abstract void sendActionPotential(Edge output);
	
	public abstract boolean fire();
	
	public abstract boolean sigmoidFire();
	
}