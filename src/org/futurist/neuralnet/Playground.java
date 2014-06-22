/**
 * @author Steven L. Moxley
 * @version 1.2
 */
package org.futurist.neuralnet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

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
import org.futurist.neuralnet.Network;

public class Playground {

	public static void main(String[] args) {

		final int tests = 5;
		
		// test results
		ConcurrentHashMap<String, Double> defaultTest = new ConcurrentHashMap<String, Double>();
		ConcurrentHashMap<String, Double> distTest = new ConcurrentHashMap<String, Double>();
		ConcurrentHashMap<Integer, Double> numLevelsTest = new ConcurrentHashMap<Integer, Double>();
		ConcurrentHashMap<Integer, Double> numLayerNodesTest = new ConcurrentHashMap<Integer, Double>();
		ConcurrentHashMap<Integer, Double> numEdgesTest = new ConcurrentHashMap<Integer, Double>();
		ConcurrentHashMap<Double, Double> learnRateTest = new ConcurrentHashMap<Double, Double>();
		ConcurrentHashMap<Double, Double> learnScaleTest = new ConcurrentHashMap<Double, Double>();
		ConcurrentHashMap<Integer, Double> numRoundsTest = new ConcurrentHashMap<Integer, Double>();
		String outputFile = "C:\\Users\\smoxley\\Downloads\\network-stats.txt";

		// default values to use prior to testing
		Integer numLevels = 3;  // 1 hidden layer usually sufficient
		Integer inNodes = 10;  // number of input nodes
		Integer outNodes = 13;  // number of output nodes
		Integer lNodes = (inNodes * outNodes) / 2;  // number of nodes per hidden layer
		Integer numEdges = 2 * inNodes;
		AbstractRealDistribution dist = new ChiSquaredDistribution(inNodes*outNodes); // NormalDistribution() also good
		Double learnRate = 0.025;  // 1.7 and 1.11 also good
		Double learnScale = 1.25;  // linear dampening; 0.5 and 1.0 also good
		
		for(int t = 0; t < tests; t++) {
			System.out.println("Running test " + t + "...");

			// random ideal values for training/learning
			Double[] idealVals = new Double[outNodes];
			Random idealSeeder = new Random();
			for(int i = 0; i < idealVals.length; i++) {
				idealVals[i] = idealSeeder.nextDouble();
			}

			// default network
			Network defaultNet = new Network(0, dist, inNodes, lNodes, outNodes, numEdges, numLevels, learnRate, learnScale, idealVals, 100, 100, 100);
			//System.out.println("Ideal Values: " + defaultNet.getIdealValues().length + "\tOutput Nodes: " + defaultNet.getOutputNodes().size());
			defaultNet.run();
			if(defaultTest.contains(dist.toString())) {
				defaultTest.replace("Default Network", defaultNet.getAverageError() + defaultTest.get(dist.toString()));
			} else {
				defaultTest.put("Default Network", defaultNet.getAverageError());
			}
			//defaultNet.printNodes();
			//defaultNet.printEdges();
			File output = new File(outputFile);
			try {
				FileWriter writer = new FileWriter(output);
				writer.write(defaultNet.getStats());
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

			// test initial weighting by distribution		
			ArrayList<AbstractRealDistribution> dists = new ArrayList<AbstractRealDistribution>();
			dists.add(new BetaDistribution(inNodes*outNodes, numEdges));
			dists.add(new CauchyDistribution());
			dists.add(new ChiSquaredDistribution(inNodes*outNodes));
			dists.add(new ExponentialDistribution(inNodes*outNodes));
			dists.add(new FDistribution(inNodes*outNodes, numEdges));
			dists.add(new NormalDistribution());
			dists.add(new TDistribution(inNodes*outNodes));
			dists.add(new UniformRealDistribution());
			dists.add(new WeibullDistribution(inNodes*outNodes, numEdges));
			for(AbstractRealDistribution d : dists) {
				//System.out.println("Varying DISTRIBUTION, currently testing " + d.toString() + " with defaults of numLayers=" + numLevels + ", inputNeurons=" + inNodes + ", layerNodes=" + lNodes + ", outputNeurons=" + outNodes + ", numSynapses=" + numEdges + ", learningRate=" + learnRate + ", and learningScale=" + learnScale + "."); 
				Network nn = new Network(1, d, inNodes, lNodes, outNodes, numEdges, numLevels, learnRate, learnScale, idealVals, 100, 100, 100);
				//System.out.println("Ideal Values: " + nn.getTargetValues().length + "\tOutput Nodes: " + nn.getOutputNodes().size());
				nn.run();
				String distName = d.toString().split("@")[0];
				if(distTest.contains(distName)) {
					distTest.replace(distName, nn.getAverageError() + distTest.get(distName));
				} else {
					distTest.put(distName, nn.getAverageError());
				}
				//nn.printStats();
			}

			// test optimal number of functional units/hidden layers among 0, 1, 2, [inputs+outputs]/2, [inputs+outputs]
			ArrayList<Integer> numLayers = new ArrayList<Integer>();
			numLayers.add(3);
			numLayers.add(4);
			numLayers.add(5);
			numLayers.add(inNodes);
			numLayers.add(outNodes);
			numLayers.add((inNodes+outNodes) / 2);
			for(Integer i : numLayers) {
				//System.out.println("Varying HIDDEN LAYERS, currently testing " + i + " with defaults of " + dist.toString() + ", numNeurons=" + ", inputNeurons=" + inNodes + ", layerNodes=" + lNodes + ", outputNeurons=" + outNodes + ", numSynapses=" + numEdges + ", learningRate=" + learnRate + ", and learningScale=" + learnScale + "."); 
				Network nn = new Network(2, dist, inNodes, lNodes, outNodes, numEdges, i, learnRate, learnScale, idealVals, 100, 100, 100);
				nn.run();
				if(numLevelsTest.contains(i)) {
					numLevelsTest.replace(i, nn.getAverageError() + numLevelsTest.get(i));
				} else { 
					numLevelsTest.put(i, nn.getAverageError());
				}
				//nn.printStats();
			}

			// test optimal number of neurons per functional unit/hidden layer among [inputs+outputs]/2, [inputs+outputs]
			ArrayList<Integer> numNeurons = new ArrayList<Integer>();
			numNeurons.add(inNodes);
			numNeurons.add(outNodes);
			numNeurons.add(inNodes + outNodes);
			numNeurons.add((inNodes*outNodes) / 2);
			numNeurons.add(inNodes * numEdges);
			numNeurons.add(outNodes * numEdges);
			for(Integer i : numNeurons) {
				//System.out.println("Varying NUMBER OF NEURONS, currently testing " + i + " with defaults of " + dist.toString() + ", numLayers=" + numLevels + ", numSynapses=" + numEdges + ", learningRate=" + learnRate + ", and learningScale=" + learnScale + "."); 
				Network nn = new Network(3, dist, inNodes, i, outNodes, numEdges, numLevels, learnRate, learnScale, idealVals, 100, 100, 100);
				nn.run();
				if(numLayerNodesTest.contains(i)) {
					numLayerNodesTest.replace(i, nn.getAverageError() + numLayerNodesTest.get(i));
				} else {
					numLayerNodesTest.put(i, nn.getAverageError());
				}
				//nn.printStats();
			}

			// test optional number of synapses in the network among 2*numNodes, numNodes*numLevels, numNodes^2, 
			ArrayList<Integer> numSynapses = new ArrayList<Integer>();
			numSynapses.add(2 * inNodes);
			numSynapses.add(2 * outNodes);
			numSynapses.add(inNodes * numLevels);
			numSynapses.add(outNodes * numLevels);
			numSynapses.add(inNodes * outNodes);
			numSynapses.add(inNodes * outNodes * numLevels);
			numSynapses.add(new Double(Math.pow(inNodes, 2)).intValue());
			numSynapses.add(new Double(Math.pow(outNodes, 2)).intValue());
			for(Integer i : numSynapses) {
				//System.out.println("Varying NUMBER OF SYNAPSES, currently testing " + i + " with defaults of " + dist.toString() + ", numLayers=" + numLevels + ", inputNeurons=" + inNodes + ", layerNodes=" + lNodes + ", outputNeurons=" + outNodes + ", learningRate=" + learnRate + ", and learningScale=" + learnScale + "."); 
				Network nn = new Network(4, dist, inNodes, lNodes, outNodes, i, numLevels, learnRate, learnScale, idealVals, 100, 100, 100);
				nn.run();
				if(numEdgesTest.contains(i)) {
					numEdgesTest.replace(i, nn.getAverageError() + numEdgesTest.get(i));
				} else {
					numEdgesTest.put(i, nn.getAverageError());
				}
				//nn.printStats();
			}

			// test optimal learning rate from 0.01 to 2.00 stepping by 0.1
			for(Double i = 0.01; i < 2.00; i += 0.1) {
				//System.out.println("Varying LEARNING RATE, currently testing " + i + " with defaults of " + dist.toString() + ", numLayers=" + numLevels + ", inputNeurons=" + inNodes + ", layerNodes=" + lNodes + ", outputNeurons=" + outNodes + ", numSynapses=" + numEdges + ", and learningScale=" + learnScale + "."); 
				Network nn = new Network(5, dist, inNodes, lNodes, outNodes, numEdges, numLevels, i, learnScale, idealVals, 100, 100, 100);
				nn.run();
				if(learnRateTest.contains(i)) {
					learnRateTest.replace(i, nn.getAverageError() + learnRateTest.get(i));
				} else {
					learnRateTest.put(i, nn.getAverageError());
				}
				//nn.printStats();
			}

			// test optimal learning scale, varying according to 0.25n to 2.0n stepping by 0.25
			for(Double i = 0.25; i < 2.00; i += 0.25) {
				//System.out.println("Varying LEARNING SCALE, currently testing " + i + " with defaults of " + dist.toString() + ", numLayers=" + numLevels + ", inputNeurons=" + inNodes + ", layerNodes=" + lNodes + ", outputNeurons=" + outNodes + ", numSynapses=" + numEdges + ", and learningRate=" + learnRate + "."); 
				Network nn = new Network(5, dist, inNodes, lNodes, outNodes, numEdges, numLevels, learnRate, i, idealVals, 100, 100, 100);
				nn.run();
				if(learnScaleTest.contains(i)) {
					learnScaleTest.replace(i, nn.getAverageError() + learnScaleTest.get(i));
				} else {
					learnScaleTest.put(i, nn.getAverageError());
				}
				//nn.printStats();
			}

			// test optimal number of rounds from 2, 5, 10, 25, 100
			ArrayList<Integer> rounds = new ArrayList<Integer>();
			rounds.add(2);
			rounds.add(5);
			rounds.add(10);
			rounds.add(25);
			rounds.add(100);
			for(Integer i : rounds) {
				//System.out.println("Varying NUMBER OF ROUNDS, currently testing " + i + " with defaults of " + dist.toString() + ", numLayers=" + numLevels + ", inputNeurons=" + inNodes + ", layerNodes=" + lNodes + ", outputNeurons=" + outNodes + ", numSynapses=" + numEdges + ", learningRate=" + learnRate + ", and learningScale=" + learnScale + "."); 
				Network nn = new Network(6, dist, inNodes, lNodes, outNodes, numEdges, numLevels, learnRate, learnScale, idealVals, 100, 100, 100);
				nn.setSimulationSettings(i, true);
				nn.run();
				if(numRoundsTest.contains(i)) {
					numRoundsTest.replace(i, nn.getAverageError() + numRoundsTest.get(i));
				} else {
					numRoundsTest.put(i, nn.getAverageError());
				}
				//nn.printStats();
			}

			System.out.println();
		}
		
		System.out.print("DEFAULT NETWORK'S average error:\t");
		for(String p : defaultTest.keySet()) {
			System.out.println(defaultTest.get(p) / tests);
		}
		System.out.println();
		
		System.out.println("Varying DISTRIBUTION parameter average error:");
		for(String p : distTest.keySet()) {
			System.out.println("\t" + p + ":\t" + distTest.get(p) / tests);
		}
		System.out.println();
		
		System.out.println("Varying HIDDEN LAYERS parameter average error:");
		for(Integer p : numLevelsTest.keySet()) {
			System.out.println("\t" + p + ":\t" + numLevelsTest.get(p) / tests);
		}
		System.out.println();
		
		System.out.println("Varying NUMBER OF NEURONS parameter average error:");
		for(Integer p : numLayerNodesTest.keySet()) {
			System.out.println("\t" + p + ":\t" + numLayerNodesTest.get(p) / tests);
		}
		System.out.println();
		
		System.out.println("Varying NUMBER OF SYNAPSES parameter average error:");
		for(Integer p : numEdgesTest.keySet()) {
			System.out.println("\t" + p + ":\t" + numEdgesTest.get(p) / tests);
		}
		System.out.println();
		
		System.out.println("Varying LEARNING RATE parameter average error:");
		for(Double p : learnRateTest.keySet()) {
			System.out.println("\t" + p + ":\t" + learnRateTest.get(p) / tests);
		}
		System.out.println();
		
		System.out.println("Varying LEARNING SCALE parameter average error:");
		for(Double p : learnScaleTest.keySet()) {
			System.out.println("\t" + p + ":\t" + learnScaleTest.get(p) / tests);
		}		
		System.out.println();
		
		System.out.println("Varying NUMBER OF ROUNDS parameter average error:");
		for(Integer p : numRoundsTest.keySet()) {
			System.out.println("\t" + p + ":\t" + numRoundsTest.get(p) / tests);
		}
		System.out.println();
	}

}
