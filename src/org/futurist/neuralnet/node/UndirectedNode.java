/**
 * @author Steven L. Moxley
 * @version 1.0
 */
package org.futurist.neuralnet.node;

import java.util.ArrayList;

import org.futurist.neuralnet.Edge;

public abstract class UndirectedNode extends Node {
	
	ArrayList<Edge> neighbors;
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 */
	public UndirectedNode(ArrayList<Integer> i, Double v) { 
		super(i, v);
		neighbors = new ArrayList<Edge>();
	}
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 * @param t the firing threshold of this Node.
	 * @param e the synaptic Edges to which this Node is connected 
	 */
	public UndirectedNode(ArrayList<Integer> i, Double v, Double t, ArrayList<Edge> e) { 
		super(i, v, t);
		if(e == null) {
			neighbors = new ArrayList<Edge>();
		} else {
			neighbors = e;
		}
	}

	/**
	 * Returns this Node's axons and dendrites (synapses are directly connected neighbors).  Note that this is the same as calling <code>getInputs()</code> or <code>getOutputs()</code> because this is an <code>UndirectedNode</code>.
	 * @return this Node's synapses.
	 */
	public ArrayList<Edge> getNeighbors() {
		return neighbors;
	}
	
	/**
	 * Returns this Node's dendrites (synapses that are directly connected neighbors).  Note that this is the same as calling <code>getOutputs()</code> because this is an <code>UndirectedNode</code>.
	 * @return this Node's synapses.
	 */
	public ArrayList<Edge> getInputs() {
		return getNeighbors();
	}
	
	/**
	 * Returns this Node's axons (synapses that are directly connected neighbors).  Note that this is the same as calling <code>getIntputs()</code> because this is an <code>UndirectedNode</code>.
	 * @return this Node's synapses.
	 */
	public ArrayList<Edge> getOutputs() {
		return getNeighbors();
	}
	
	/**
	 * Add a directly connected synaptic neighbor to this Node's existing <code>Edges</code>.  Note that since this is an <code>UndirectedNode</code>, no distinction is made between dendrites (inputs) and axons (outputs). 
	 * @param e the synaptic <code>Edge</code> to add.
	 */
	public void addNeighbor(Edge e) {
		if(!neighbors.contains(e)) {
			neighbors.add(e);
		} else {
			System.out.println(e + " is already a neighbor.");
		}
	}
	
	/**
	 * Remove a directly connected synaptic neighbor from this Node's existing <code>Edges</code>.  Note that since this is an <code>UndirectedNode</code>, no distinction is made between dendrites (inputs) and axons (outputs). 
	 * @param e the synaptic <code>Edge</code> to remove.
	 */
	public void removeNeighbor(Edge e) {
		if(neighbors.contains(e)) {
			neighbors.remove(e);
		} else {
			System.out.println(e + " is not a neighbor.");
		}
	}
	
	/**
	 * Returns this Node's degree centrality.
	 * @return the degree centrality.
	 */
	public Integer getDegreeCentrality() {
		return neighbors.size();
	}
	
	/**
	 * Returns this <code>Node's</code> clustering coefficient.
	 * @return the clustering coefficient.
	 */
	public Double getClusteringCoefficient() {
		ArrayList<Node> neighborhood = new ArrayList<Node>(neighbors.size());
		int neighborEdges = 0;
		
		for(Edge e : neighbors) {
			neighborhood.add(e.getOutput());
		}
		
		for(Node n : neighborhood) {
			for(Edge e : n.getOutputs()) {
				if(!e.getOutput().equals(this) && neighbors.contains(e.getOutput())) {
					neighborEdges++;
				}
			}
		}
		
		return new Double(neighborEdges / (Math.pow(neighbors.size(), 2) - neighbors.size()));
	}
	
	public abstract void receiveActionPotential(Edge input);
	
	public abstract void sendActionPotential(Edge output);
	
	public abstract boolean fire();
	
	public abstract boolean sigmoidFire();
	
}
