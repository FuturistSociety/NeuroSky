/**
 * @author Steven L. Moxley
 * @version 1.0
 */
package org.futurist.neuralnet.node;

import java.util.ArrayList;

import org.futurist.neuralnet.Edge;

public class UndirectedSumNode extends UndirectedNode {

	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 */
	public UndirectedSumNode(ArrayList<Integer> i, Double v) { 
		super(i, v);
	}
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 * @param t the firing threshold of this Node.
	 * @param e the synaptic Edges to which this Node is connected 
	 */
	public UndirectedSumNode(ArrayList<Integer> i, Double v, Double t, ArrayList<Edge> e) { 
		super(i, v, t, e);
	}
	
	/**
	 * Receives a signal from the given synapse.  Note that the <code>Edge</code> must already be connected to this Node but that any connected neighbor may send an action potential because this is an <code>UndirectedNode</code>.
	 * @param i the Edge containing the synaptic input to be received.
	 */
	public void receiveActionPotential(Edge i) {
		if(neighbors.contains(i)) {
			value += i.getInput().getValue() * i.getWeight();
		}		
	}
	
	/**
	 * Send a signal to the given synapse.  Note that the <code>Edge</code> must already be connected to this Node but that any connected neighbor may receive an action potential because this is an <code>UndirectedNode</code>.
	 * @param o the Edge containing the synaptic input to be sent.
	 */
	public void sendActionPotential(Edge o) {
		if(fire() && neighbors.contains(o)) {
			o.getOutput().receiveActionPotential(o);
		}
	}
	
	/**
	 * Determines whether or not this Node has a high enough stored value to meet the firing threshold and send an action potential.
	 * @return true if value >= threshold; returns false otherwise
	 */
	public boolean fire() {
		if(value >= threshold) {
			return true;
		} else {
			System.out.println("Action potential " + value + " is below the threshold of " + threshold + ".");
			return false;
		}
	}
	
	/**
	 * Determines whether or not the sigmoid of the value stored by this Node is enough to meet the firing threshold and send an action potential.
	 * @return true if sigmoid(value) >= threshold; returns false otherwise
	 */
	public boolean sigmoidFire() {
		if(sigmoid(value) >= threshold) {
			return true;
		} else {
			System.out.println("Action potential " + value + " is below the threshold of " + threshold + ".");
			return false;
		}
	}
	
}
