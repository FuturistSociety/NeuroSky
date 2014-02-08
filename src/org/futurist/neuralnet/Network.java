/**
 * @author Steven L. Moxley
 * @version 1.1
 */
package org.futurist.neuralnet;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.Random;

import javafx.scene.layout.GridPane;

import org.futurist.neuralnet.node.DirectedNode;
import org.futurist.neuralnet.node.DirectedSumNode;
import org.futurist.neuralnet.node.Node;
import org.futurist.neuralnet.node.UndirectedNode;

import org.apache.commons.math3.distribution.*;

public class Network extends Thread {

	public static final Integer DEFAULT_LENGTH = 100;
	public static final Integer DEFAULT_WIDTH = 100;
	public static final Integer DEFAULT_HEIGHT = 100;

	protected final Integer id;
	protected final Random rng;							// random node selector
	protected final AbstractRealDistribution dist;		// probability distribution that determines which nodes (neurons) to connect with an edge (synapse)
	protected Integer startNodes;						// number of nodes in the input layer/functional unit
	protected Integer layerNodes;						// number of nodes per hidden layer; UPGRADE: set to -1 for random number for each layer, -2...-x as per differential tests
	protected Integer endNodes;							// number of nodes in the output layer/functional unit
	protected Integer startEdges;						// number of connected synapses in the initial network
	protected Integer layers;							// number of hidden layers/hidden networks
	protected ArrayList<Node> nodes;					// all nodes in the network
	protected ArrayList<Edge> edges;					// all edges in the network
	protected ArrayList<Edge[]> functionalUnits;		// subset of edges that fire together to form a functional unity; a.k.a. hidden layer of neurons; funtionalUnits[0] is the input layer of neurons, and functionalUnits[functionalUnits.length] is the output layer of neurons
	protected Number[] idealValues;						// correct values that should be output by last layer of functional units; used for learning step
	protected DirectedNode[] inputNeurons;				// the first functional unit receives input that is fed to the rest of the network
	protected DirectedNode[] outputNeurons;				// the last functional unit outputs the values resulting from neuron firing and learning
	protected Double learnRate;							// learning weight; used for re-weighting edges
	protected Double learnScale;						// factor by which learning rate decays linearly w.r.t. distance from output layer
	protected final int x;								// physical length of network; the number of coordinates available to place Nodes along the x-axis
	protected final int y;								// physical width of network; the number of coordinates available to place Nodes along the y-axis
	protected final int z;								// physical height of network; the number of coordinates available to place Nodes along the z-axis
	protected GridPane pane;							// GUI
	protected boolean simulated;						// determines what type of statistics to print
	protected Double[][] adjacencyMatrix;				// synaptic connection weights between neuron nodes; adjacencyMatrix[i][j]=0 iff i=j; adjacencyMatrix[i][j]=w iff an Edge exists from i to j; adjacencyMatrix[i][j]=Double.MAX_VALUE otherwise
	protected Double[][] shortestPaths;					// shortest paths between neuron nodes; 
	protected Integer simRounds;						// number of simulation rounds to run
	protected Boolean simLearning;						// whether or not learning through backpropagation is enabled during the simulation rounds

	/**
	 * Default constructor to create a Network.
	 * @param id the unique ID of this Network.
	 * @param v the correct/ideal/target values that are used to train the network.
	 */
	public Network(Integer id, Number[] v) {
		this.id = id;
		rng = new Random();
		dist = new NormalDistribution();
		startNodes = 256;
		nodes = new ArrayList<Node>(startNodes);
		startEdges = startNodes * 3;
		edges = new ArrayList<Edge>(startEdges);
		layers = 3;
		functionalUnits = new ArrayList<Edge[]>(layers+2);
		idealValues = dataToDouble(v);
		endNodes = idealValues.length;
		learnRate = .25;
		x = DEFAULT_LENGTH;
		y = DEFAULT_WIDTH;
		z = DEFAULT_HEIGHT;
		simRounds = 5;
		simLearning = true;

		init();
	}

	/**
	 * Constructor to create a Network with the given parameters using the default 3-dimensional space.
	 * @param id the unique ID of this Network.
	 * @param d the statistical distribution drawn from to determine neurons' initial values and firing thresholds.
	 * @param inNodes the number of input neurons.
	 * @param lNodes the number of neurons in each hidden layer.
	 * @param outNodes the number of output neurons.
	 * @param numEdges the number of synapses.
	 * @param l the the number of layers.
	 * @param r the the learning rate used for reinforcement in backpropagation.
	 * @param s the scaling (or dampening) factor used to modulate the learning rate as backpropagation takes place.
	 * @param v the correct/ideal/target values that are used to train the network.
	 */
	public Network(Integer id, AbstractRealDistribution d, Integer inNodes, Integer lNodes, Integer outNodes, Integer numEdges, Integer l, Double r, Double s, Number[] v) {
		if(outNodes != v.length) {
			throw new InputMismatchException("The number of output neurons and the number of target values must match.");
		} else {
			this.id = id;
			rng = new Random();
			dist = d;
			startNodes = inNodes;
			layerNodes = lNodes;
			endNodes = outNodes;
			nodes = new ArrayList<Node>(startNodes);
			startEdges = numEdges;
			edges = new ArrayList<Edge>(startEdges);
			layers = l;
			functionalUnits = new ArrayList<Edge[]>(layers+2);
			idealValues = dataToDouble(v);
			learnRate = r;
			learnScale = s;
			x = DEFAULT_LENGTH;
			y = DEFAULT_WIDTH;
			z = DEFAULT_HEIGHT;
			simRounds = 5;
			simLearning = true;

			init();
		}
	}

	/**
	 * Constructor to create a Network with the given parameters using the given 3-dimensional space.
	 * @param id the unique ID of this Network.
	 * @param d the statistical distribution drawn from to determine neurons' initial values and firing thresholds.
	 * @param inNodes the number of input neurons.
	 * @param lNodes the number of neurons in each hidden layer.
	 * @param outNodes the number of output neurons.
	 * @param numEdges the number of synapses.
	 * @param l the the number of layers.
	 * @param r the the learning rate used for reinforcement in backpropagation.
	 * @param s the scaling (or dampening) factor used to modulate the learning rate as backpropagation takes place.
	 * @param v the correct/ideal/target values that are used to train the network.
	 * @param length the number of available locations in the x dimension (or points along the x-axis) where the Network may place a Node.
	 * @param width the number of available locations in the y dimension (or points along the y-axis) where the Network may place a Node.
	 * @param height the number of available locations in the z dimension (or points along the z-axis) where the Network may place a Node.
	 */
	public Network(Integer id, AbstractRealDistribution d, Integer inNodes, Integer lNodes, Integer outNodes, Integer numEdges, Integer l, Double r, Double s, Number[] v, Integer length, Integer width, Integer height) {
		if(outNodes != v.length) {
			throw new InputMismatchException("The number of output neurons and the number of target values must match.");
		} else {
			this.id = id;
			rng = new Random();
			dist = d;
			startNodes = inNodes;
			layerNodes = lNodes;
			endNodes = outNodes;
			nodes = new ArrayList<Node>(startNodes);
			startEdges = numEdges;
			edges = new ArrayList<Edge>(startEdges);
			layers = l;
			functionalUnits = new ArrayList<Edge[]>(layers+2);
			idealValues = dataToDouble(v);
			learnRate = r;
			learnScale = s;
			x = length;
			y = width;
			z = height;
			simRounds = 5;
			simLearning = true;

			init();
		}
	}

	/**
	 * Constructor to create a Network with the given parameters using the default 3-dimensional.
	 * @param i the unique ID of this Network.
	 * @param d the statistical distribution drawn from to determine neurons' initial values and firing thresholds.
	 * @param inNodes the number of input neurons.
	 * @param lNodes the number of neurons in each hidden layer.
	 * @param outNodes the number of output neurons.
	 * @param numEdges the number of synapses.
	 * @param l the the number of layers.
	 * @param r the the learning rate used for reinforcement in backpropagation.
	 * @param s the scaling (or dampening) factor used to modulate the learning rate as backpropagation takes place.
	 * @param v the correct/ideal/target values that are used to train the network.
	 */
	public Network(Integer i, AbstractRealDistribution d, Integer inNodes, Integer lNodes, Integer outNodes, Integer numEdges, Integer l, Double r, Double s, Number[] v, GridPane gui) {
		this(i, d, inNodes, lNodes, outNodes, numEdges, l, r, s, v);
		pane = gui;
	}

	private void init() {
		long startTime = System.nanoTime();

		// store free remaining <X, Y, Z> coordinates for placing Nodes
		// UPGRADE: find better than O(N^3) way of doing this
		ArrayList<ArrayList<Integer>> undrawnIDs = new ArrayList<ArrayList<Integer>>(x*y*z);
		for(int xIdx = 1; xIdx < x; xIdx++) {
			for(int yIdx = 1; yIdx < y; yIdx++) {
				for(int zIdx = 1; zIdx < z; zIdx++) {
					ArrayList<Integer> ID = new ArrayList<Integer>();
					ID.add(x);
					ID.add(y);
					ID.add(z);
					undrawnIDs.add(ID);
				}
			}
		}

		// create nodes for all hidden layers with values draw from the network's probability distribution
		for(int i = 0; i < layerNodes*layers; i++) {
			ArrayList<Integer> coordinates = undrawnIDs.get(rng.nextInt(undrawnIDs.size()));
			nodes.add(new DirectedSumNode(coordinates, dist.cumulativeProbability(i), dist.density(i), null, null));
			undrawnIDs.remove(coordinates);
		}

		// create edges for all hidden layers with weights from the network's probability distribution for each functional level
		for(int i = 0; i < layers; i++) {
			// make the functional unit (subset of nodes) that are connected at this level
			Edge[] unit = new Edge[layerNodes];

			for(int j = 0; j < layerNodes; j++) {
				Node n1 = nodes.get(rng.nextInt(nodes.size()));
				Node n2 = nodes.get(rng.nextInt(nodes.size()));
				while(n1 == n2) { n2 = nodes.get(rng.nextInt(nodes.size())); }
				Integer edgeID = j + (i*j);
				//Edge e = new Edge(edgeID, n1, n2);
				//Edge e = new Edge(edgeID, n1, n2, dist.density(edgeID));
				Edge e = new Edge(edgeID, n1, n2, dist.cumulativeProbability(edgeID));
				if(n1 instanceof UndirectedNode && n2 instanceof UndirectedNode) {
					//System.out.println(n1 + " and " + n2 + " are both UndirectedNodes!");
					((UndirectedNode) n1).addNeighbor(e);
					((UndirectedNode) n2).addNeighbor(e);
				} else if(n1 instanceof DirectedNode && n2 instanceof DirectedNode) {
					//System.out.println(n1 + " and " + n2 + " are both DirectedNodes!");
					((DirectedNode) n1).addOutput(e);
					((DirectedNode) n2).addInput(e);
				} else {
					System.out.println("ERROR: " + n1 + " and " + n2 + " are not the same type of node!");
				}
				unit[j] = e;
				edges.add(e);
			}
			functionalUnits.add(unit);
			//System.out.println("Initialized functional unit " + i + " with " + unit.length + " nodes.");
		}

		// create input layer neurons
		inputNeurons = new DirectedNode[startNodes];
		for(int i = 0; i < startNodes; i++) {
			ArrayList<Integer> coordinates = undrawnIDs.get(rng.nextInt(undrawnIDs.size()));
			DirectedSumNode inNode = new DirectedSumNode(coordinates, 0.0, 1.0, null, null);
			inputNeurons[i] = inNode;
			undrawnIDs.remove(coordinates);
		}

		// connect each input neuron to each neuron in the next layer
		for(DirectedNode inNode : inputNeurons) {
			for(Edge e : functionalUnits.get(0)) {
				e.setInput(inNode);
				inNode.addOutput(e);
			}
		}

		// create output layer neurons
		outputNeurons = new DirectedNode[endNodes];
		for(int i = 0; i < endNodes; i++) {
			ArrayList<Integer> coordinates = undrawnIDs.get(rng.nextInt(undrawnIDs.size()));
			DirectedSumNode outNode = new DirectedSumNode(coordinates, 0.0, 1.0, null, null);
			outputNeurons[i] = outNode;
			undrawnIDs.remove(coordinates);
		}

		// connect each output neuron to each neuron in the previous layer
		for(DirectedNode outNode : outputNeurons) {
			for(Edge e : functionalUnits.get(functionalUnits.size()-1)) {
				e.setOutput(outNode);
				outNode.addInput(e);
			}
		}

		// initialize the adjacency matrix
		adjacencyMatrix = new Double[nodes.size()][nodes.size()];
		for(int i = 0; i < nodes.size(); i++) {
			for(int j = 0; j < nodes.size(); j++) {
				if(i == j) {
					adjacencyMatrix[i][j] = 0.0;
				} else {
					adjacencyMatrix[i][j] = Double.MAX_VALUE;
				}
			}
		}

		// fill in the adjacency matrix
		for(int i = 0; i < nodes.size(); i++) {
			Node n1 = nodes.get(i);
			for(int j = 0; j < nodes.size(); j++) {
				Node n2 = nodes.get(j);
				for(Edge e : n1.getOutputs()) {
					if(e.getOutput().equals(n2)) {
						adjacencyMatrix[i][j] = e.getWeight();
					}
				}
				for(Edge e : n2.getInputs()) {
					if(e.getInput().equals(n2)) {
						adjacencyMatrix[i][j] = e.getWeight();
					}
				}
			}
		}

		// fill in the shortest paths matrix with the Floyd-Warshall algorithm
		shortestPaths = adjacencyMatrix;
		for(int i = 0; i < nodes.size(); i++) {
			for(int j = 0; j < nodes.size(); j++) {
				for(int k = 0; k < nodes.size(); k++) {
					Double distSum = shortestPaths[j][i] + shortestPaths[i][k];
					if(distSum < shortestPaths[j][k]) {
						shortestPaths[j][k] = distSum;
					}
				}
			}
		}

		simulated = false;
		double elapsedTimeInSec = (System.nanoTime() - startTime) * 1.0e-9;
		System.out.println("Finished initializing the neural network with " + nodes.size() + " neuron nodes and " + edges.size() + " synaptic edges in " + functionalUnits.size() + " functional units in " + elapsedTimeInSec + " seconds.");
		//printNodes();
		//printEdges();
	}

	/**
	 * Store new settings to determine the type of simulation to run.
	 * @param rounds the number of iterations to run before reporting the Network's state.
	 * @param learning whether or not learning through backpropagation should be enabled after reach round is run.
	 */
	public void setSimulationSettings(int rounds, boolean learning) {
		simRounds = rounds;
		simLearning = learning;
	}

	/**
	 * Run a simulation of the behavior of the <code>Network</code> with the stored settings.
	 */
	public void run() {
		long startTime = System.nanoTime();

		for(int r = 0; r < simRounds; r++) {
			/*
			 * UPGRADE: add JavaFX GUI
			for(Circle c : MainGUI.getNodeCircleMap().values()) {
				c.setFill(Color.GREY);
			}
			 */
			boolean step = true;
			while(!step) {
				System.out.print("Run next step in simulation? ");
				BufferedReader inputReader = new BufferedReader(new InputStreamReader(System.in));
				try {
					String answer = inputReader.readLine();
					if(answer.equalsIgnoreCase("yes") || answer.equalsIgnoreCase("y")) {
						step = true;
					}
				} catch (IOException e) {
				}
			}

			//System.out.println("Simulating round " + r + "...");
			for(Edge[] f : functionalUnits) {
				for(int i = 0; i < f.length; i++) {
					Node n = f[i].getInput(); 
					n.sendActionPotential(f[i]);

					if(pane != null) {
						if(n.fire()) {
							//MainGUI.getNodeCircleMap().get(n).setFill(Color.RED);
						}
					}
				}
			}

			if(simLearning && idealValues.length == functionalUnits.get(functionalUnits.size()-1).length) {
				backpropagate();
			}
		}
		simulated = true;
		double elapsedTimeInSec = (System.nanoTime() - startTime) * 1.0e-9;
		System.out.println("Finished simulating " + simRounds + " rounds of the neural network in " + elapsedTimeInSec + " seconds.");
	}

	/**
	 * Update a single given old layer/functional unit based on feedback from the change in performance at predicting given target values.
	 * @return the improved layer/functional unit after learning by adjusting weight according to sigmoid of performance.
	 */ 
	public Edge[] learn(Edge[] oldLayer, Number[] targetValues, Double scale) {
		Edge[] improvedLayer = oldLayer;
		for(int x = 0; x < improvedLayer.length; x++) {
			Double target = (Double) targetValues[x];
			Double error = Math.abs((improvedLayer[x].getOutput().getValue() - target) / target);
			improvedLayer[x].setWeight(improvedLayer[x].getWeight() + (sigmoid(error) * improvedLayer[x].getWeight() * (learnRate/scale)));  // new synaptic weight += error * old weight * scaled learning rate
		}
		return improvedLayer;
	}

	/**
	 * Send scaled feedback throughout the neural network for learning, beginning from the output layer/functional unit and ending at the initial input layer/functional unit.
	 */
	public void backpropagate() {
		for(int x = functionalUnits.size()-1; x > 1; x--) {
			Edge[] oldLayer = functionalUnits.get(x);
			if(oldLayer.equals(functionalUnits.get(functionalUnits.size()-1))) {
				functionalUnits.set(x, learn(oldLayer, idealValues, learnScale));
			} else {
				Edge[] targetLayer = functionalUnits.get(x+1);
				Double[] targetValues = new Double[targetLayer.length];
				for(int y = 0; y < targetLayer.length; y++) {
					targetValues[y] = targetLayer[y].getOutput().getValue();
				}
				functionalUnits.set(x, learn(oldLayer, targetValues, learnScale));
			}
		}
	}

	/**
	 * Prints the current value and firing threshold of each <code>Node</code> (neuron) in this <code>Network</code>.
	 */
	public void printNodes() {
		for(Node n : nodes) {
			System.out.println(n + " has a current value of " + n.getValue() + " and fires if its accumulated value is greater than or equal to " + n.getThreshold() + ".");
		}
	}

	/**
	 * Prints the input and output neurons of each <code>Edge</code> (synapse) in this <code>Network</code>.
	 */
	public void printEdges() {
		for(Edge e : edges) {
			System.out.println(e.getID() + " accepts action potential input from " + e.getInput() + " and transmits it to " + e.getOutput() + ".");
		}
	}

	/**
	 * Prints statistics of this <code>Network</code>.
	 */
	public void printStats() {
		System.out.println("There are " + nodes.size() + " neuron nodes and " + edges.size() + " synaptic edges in " + functionalUnits.size() + " functional units.");

		// Node statistics
		Double nodeInputSum = 0.0;
		Double nodeOutputSum = 0.0;
		Double nodeValueSum = 0.0;
		Double nodeThresholdSum = 0.0;
		Integer nodeDegreeSum = 0;		
		Double nodeClusterSum = 0.0;
		Double nodeClosenessSum = 0.0;
		//Double nodeBetweennesSum = 0.0;

		// calculate all Nodes' statistics
		for(Node n : nodes) {
			nodeInputSum += n.getInputs().size();
			nodeOutputSum += n.getOutputs().size();
			nodeValueSum += n.getValue();
			nodeThresholdSum += n.getThreshold();
			nodeDegreeSum += n.getDegreeCentrality();
			nodeClusterSum += n.getClusteringCoefficient();
		}

		// calculate Network statistics
		Double[] nodeCloseness = new Double[nodes.size()];
		//Double[] nodeBetweenness = new Double[nodes.size()];
		for(int i = 0; i < nodes.size(); i++) {
			Double closenessSum = 0.0;
			//Double betweennessSum = 0.0;
			for(int j = 0; j < nodes.size(); j++) {
				if(shortestPaths[i][j] != 0 && shortestPaths[i][j] != Double.MAX_VALUE)
					closenessSum += shortestPaths[i][j];
			}
			nodeCloseness[i] = closenessSum;
		}

		// print average Node statistics
		System.out.println("Each neuron has an average of " + nodeInputSum/nodes.size() + " synaptic inputs.");
		System.out.println("Each neuron has an average of " + nodeOutputSum/nodes.size() + " synaptic outputs.");
		System.out.println("Each neuron has an average value of " + nodeValueSum/nodes.size() + ".");
		System.out.println("Each neuron has an average threshold of " + nodeThresholdSum/nodes.size() + ".");
		System.out.println("The average degree centrality of all neurons in the network is " + nodeDegreeSum/nodes.size() + ".");
		System.out.println("The average clustering coefficient of all neurons in the network is " + nodeClusterSum/nodes.size() + ".");
		System.out.println("The average closeness centrality of all neurons in the network is " + nodeClosenessSum/nodes.size() + ".");
		//System.out.println("The average betweenness centrality of all neurons in the network is " + nodeBetweennesSum/nodes.size() + ".");

		// Edge statistics
		Double edgeWeightSum = 0.0;
		for(Edge e : edges) {
			edgeWeightSum += e.getWeight();
		}
		System.out.println("The average edge weight is " + edgeWeightSum/edges.size() + ".");

		// Network properties from http://en.wikipedia.org/wiki/Network_science#Network_Properties on April 21, 2012
		// The density D of a network is defined as a ratio of the number of edges E to the number of possible edges, given by the binomial coefficient \tbinom N2, giving D = \frac{2E}{N(N-1)}.
		Double density = (2*edges.size()) / (Math.pow(nodes.size(), 2) - nodes.size());
		System.out.println("The density of the neural network is: " + density + ".");

		// UPGRADE: shortest, mean, and longest path lengths
		//System.out.println("The shortest path between neurons is " + shortest + " synapses.");
		//System.out.println("The diameter of the neural network (longest path) is " + longest + ".");

		if(simulated) {
			Double nodeFireSum = 0.0;
			for(Node n : nodes) {
				nodeFireSum += n.getNumFires();
			}
			System.out.println("There were " + nodeFireSum + " neuron firings during the simulation, an average of " + nodeFireSum/nodes.size() + " per neuron.");
		}
		System.out.println();
	}

	/** 
	 * Returns the unique ID assigned to this <code>Network</code>.
	 * @return the ID.
	 */
	public Integer getID() {
		return id;
	}

	/** 
	 * Returns the statistical distribution drawn from to determine neurons' initial values and firing thresholds.
	 * @return the statistical distribution.
	 */
	public AbstractRealDistribution getDistribution() {
		return dist;
	}

	/** 
	 * Returns all neurons in this <code>Network</code>.
	 * @return the <code>Nodes</code> representing neurons.
	 */
	public ArrayList<Node> getNodes() {
		return nodes;
	}

	/** 
	 * Returns all synapses in this <code>Network</code>.
	 * @return the <code>Edges</code> representing synapses.
	 */
	public ArrayList<Edge> getEdges() {
		return edges;
	}

	/** 
	 * Returns the unique ID assigned to this <code>Edge</code>.
	 * @return the ID.
	 */
	public ArrayList<Edge[]> getFunctionalUnits() {
		return functionalUnits;
	}

	/** 
	 * Returns all input neurons in this <code>Network</code>.
	 * @return the <code>Nodes</code> representing input neurons.
	 */
	public Node[] getInputNeurons() {
		return inputNeurons;
	}

	/**
	 * Replaces existing input <code>Nodes</code> with the given ones.
	 * @param newInputs the new input <code>Nodes</code> representing neurons.
	 */
	public void setInputNeurons(Node[] newInputs) {
		Edge[] currentInputs = functionalUnits.get(0);
		if(newInputs.length == currentInputs.length) {
			for(int x = 0; x < currentInputs.length; x++) {
				currentInputs[x].setInput(newInputs[x]);
			}
		} else { 
			throw new InputMismatchException("Number of new input neurons must match number of existing input neurons!");
		}
	}

	/** 
	 * Returns all output neurons in this <code>Network</code>.
	 * @return the <code>Nodes</code> representing output neurons.
	 */
	public Node[] getOutputNeurons() {
		return outputNeurons;
	}

	/**
	 * Replaces existing output <code>Nodes</code> with the given ones.
	 * @param newOutputs the new output <code>Nodes</code> representing neurons.
	 */
	public void setOutputNeurons(Node[] newOutputs) {
		Edge[] currentOutputs = functionalUnits.get(functionalUnits.size()-1);
		if(newOutputs.length == currentOutputs.length) {
			for(int x = 0; x < currentOutputs.length; x++) {
				currentOutputs[x].setOutput(newOutputs[x]);
			}
		} else { 
			throw new InputMismatchException("Number of new input neurons must match number of existing input neurons!");
		}
	}

	/** 
	 * Returns the correct/ideal/target values that are used to train this <code>Network</code>.
	 * @return the target values.
	 */
	public Number[] getTargetValues() {
		return idealValues;
	}

	/**
	 * Returns the adjacency matrix of this <code>Network</code>.  adjacencyMatrix[i][j]=0 iff i=j; adjacencyMatrix[i][j]=w iff an Edge exists from i to j; adjacencyMatrix[i][j]=Double.MAX_VALUE otherwise.
	 * @return the adjacency matrix.
	 */
	public Double[][] getAdjacencyMatrix() {
		return adjacencyMatrix;
	}

	/** 
	 * Returns the unique ID assigned to this <code>Edge</code>.
	 * @return the ID.
	 */
	public Double[] getErrors() {
		Double[] errors = new Double[idealValues.length];
		String msg = "Difference between idealValue[x] - learnedOutput[x]: ";

		for(int i = 0; i < idealValues.length; i++) {
			Double err = idealValues[i].doubleValue() - getOutputNeurons()[i].getValue();
			errors[i] = err;
			msg += err + ", ";
		}
		msg = msg.substring(0, msg.lastIndexOf(", "));
		//System.out.println(msg);
		return errors;
	}

	/** 
	 * Returns the unique ID assigned to this <code>Edge</code>.
	 * @return the ID.
	 */
	public Double getAverageError() {
		Double[] errors = getErrors();
		Double sum = 0.0;
		for(Double e : errors) {
			sum += e;
		}
		return sum / errors.length;
	}

	/** 
	 * Returns the unique ID assigned to this <code>Edge</code>.
	 * @return the ID.
	 */
	public Random getRNG() {
		return rng;
	}

	/*
	 * UPGRADE: calculate the shortest path between 2 arbitrary nodes (up to the given maximum distance)
	public Edge[] getShortestPath(Node from, Node to, Integer maxHops) {

	}
	 */

	/** 
	 * Returns the value of the sigmoid function after operating on the given input.
	 * @param x the input to the sigmoid function.
	 * @return the result of the sigmoid function.
	 */
	public static Double sigmoid(Double x) {
		Double denominator = 1 + Math.pow(Math.E, x*-1);
		//Double denominator = 1 + Math.exp(-x);
		return 1 / denominator;
	}

	/**
	 * Utility method to transform any numeric data into an array of Doubles.
	 * @return the transformed data.
	 */
	public static Double[] dataToDouble(Number[] data) {
		Double[] result = new Double[data.length];
		if(data instanceof Integer[]) {
			for(int i = 0; i < data.length; i++) {
				result[i] = new Double(data[i].intValue());
			}
		} else if(data instanceof Long[]) {
			for(int i = 0; i < data.length; i++) {
				result[i] = new Double(data[i].longValue());
			}
		} else if(data instanceof Short[]) {
			for(int i = 0; i < data.length; i++) {
				result[i] = new Double(data[i].shortValue());
			}
		} else if(data instanceof Float[]) {
			for(int i = 0; i < data.length; i++) {
				result[i] = new Double(data[i].floatValue());
			}
		} else if(data instanceof Byte[]) {
			for(int i = 0; i < data.length; i++) {
				result[i] = new Double(data[i].byteValue());
			}
		} else {
			for(int i = 0; i < data.length; i++) {
				result[i] = new Double(data[i].doubleValue());
			}
		}
		return result;
	}
}
