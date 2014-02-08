/**
 * @author Steven L. Moxley
 * @version 1.0
 */
package org.futurist.neuralnet.node;

import java.util.ArrayList;

import org.futurist.neuralnet.Edge;


public class DirectedSumNode extends DirectedNode {
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 */
	public DirectedSumNode(ArrayList<Integer> i, Double v) { 
		super(i, v);
	}
	
	/**
	 * Constructor to create an abstract Node.
	 * @param i the 3-dimensional coordinate ID assigned to this Node.
	 * @param v the initial value stored by this Node.
	 * @param t the firing threshold of this Node.
	 * @param in the synaptic Edges from which this Node receives action potentials
	 * @param out the synaptic Edges to which this Node sends action potentials
	 */
	public DirectedSumNode(ArrayList<Integer> i, Double v, Double t, ArrayList<Edge> in, ArrayList<Edge> out) { 
		super(i, v, t, in, out);
	}
	
	/**
	 * Receives a signal from the given synapse.  Note that the <code>Edge</code> must already be connected to this Node as an input neuron because this is a <code>DirectedNode</code>.
	 * @param i the Edge containing the synaptic input to be received.
	 */
	public void receiveActionPotential(Edge i) {
		if(inputs.contains(i)) {
			//System.out.print(this + "'s previous accumulated action potential was " + sum + " and is now " ); 
			value += i.getInput().getValue() * i.getWeight();
			//System.out.println(sum + " after receiving action potential from " + i + ".");
		} else {
			//System.out.println(this + "rejects " + i + "'s action potential because it is not connected!");
		}
	}
	
	/**
	 * Send a signal to the given synapse.  Note that the <code>Edge</code> must already be connected to this Node as an output neuron because this is an <code>DirectedNode</code>.
	 * @param o the Edge containing the synaptic input to be sent.
	 */
	public void sendActionPotential(Edge o) {
		for(Edge i : inputs) {
			receiveActionPotential(i);
		}
		if(sigmoidFire() && outputs.contains(o)) {
			//System.out.println(this + " is firing an action potential of " + value + "!");
			o.getOutput().receiveActionPotential(o);
			numFires++;
			value = 0.0;	// reset accumulated action potential level after firing
		}
	}
	
	/**
	 * Determines whether or not this Node has a high enough stored value to meet the firing threshold and send an action potential.
	 * @return true if value >= threshold; returns false otherwise
	 */
	public boolean fire() {
		if(value >= threshold) {
			//System.out.println(this + " has accumulated enough action potential to reach its threshold.");
			run();
			return true;
		} else {
			//System.out.println("Action potential " + sum + " is below the threshold of " + threshold + ".");
			return false;
		}
	}
	
	/**
	 * Determines whether or not the sigmoid of the value stored by this Node is enough to meet the firing threshold and send an action potential.
	 * @return true if sigmoid(value) >= threshold; returns false otherwise
	 */
	public boolean sigmoidFire() {
		if(sigmoid(value) >= threshold) {
			//System.out.println(this + " has accumulated enough action potential to reach its threshold.");
			return true;
		} else {
			//System.out.println("Action potential " + sum + " is below the threshold of " + threshold + ".");
			return false;
		}
	}
	
}