/**
 * @author Steven L. Moxley
 * @version 1.2
 */
package org.futurist.neuralnet.gui;

import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import org.futurist.neuralnet.Edge;
import org.futurist.neuralnet.Network;
import org.futurist.neuralnet.node.Node;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;

public class MainGUI extends Application {

	// GUI constants
	public static final int DEFAULT_LENGTH = 1024;
	public static final int DEFAULT_WIDTH = 1024;
	public static final int DEFAULT_HEIGHT = 768;

	// neural network constants
	public static int defaultNumInNodes = 10;
	public static int defaultNumOutNodes = 13;
	public static int defaultNumLayerNodes = (defaultNumInNodes * defaultNumOutNodes) / 2;
	public static int defaultNumEdges = defaultNumOutNodes * 2;
	public static int defaultNumLevels = 3;
	public static int defaultNumRounds = 2;

	Network network;
	protected static ConcurrentHashMap<Node, Circle> nodeCircleMap;
	protected static ConcurrentHashMap<Edge, Line> edgeLineMap; 

	public static int getDefaultWidth() {
		return DEFAULT_WIDTH;
	}

	public static int getDefaultHeight() {
		return DEFAULT_HEIGHT;
	}

	private void init(Stage primaryStage) {

		Group root = new Group();
		GridPane pane = new GridPane();
		pane.setPrefSize(DEFAULT_WIDTH, DEFAULT_HEIGHT);
		primaryStage.setScene(new Scene(pane));

		// create File menu item
		final MenuBar menuBar = new MenuBar();
		final MenuItem exitMenuItem = MenuItemBuilder.create().text("Exit").build();
		exitMenuItem.setOnAction(new EventHandler<ActionEvent>() {
			public void handle(ActionEvent t) {
				System.exit(0);
			}
		});

		// create File menu and add it to Pane
		Menu menu1 = MenuBuilder.create().text("File").items(exitMenuItem).graphic(new ImageView(new Image(MainGUI.class.getResourceAsStream("menuInfo.png")))).build();
		menuBar.getMenus().addAll(menu1);
		//pane.add(menuBar, 0, 0);

		double rate = 0.75;
		double scale = 1.0;
		Double[] idealVals = new Double[defaultNumOutNodes];
		Random idealSeeder = new Random();
		for(int i = 0; i < idealVals.length; i++) {
			idealVals[i] = idealSeeder.nextDouble();
		}
		
		network = new Network(1, new ChiSquaredDistribution(defaultNumInNodes*defaultNumOutNodes), defaultNumInNodes, defaultNumLayerNodes, defaultNumOutNodes, defaultNumEdges, defaultNumLevels, rate, scale, idealVals, pane);

		Random rng = network.getRNG();
		double nodeRadius = (pane.getHeight()*pane.getWidth()) / (network.getNodes().size()*2);
		nodeRadius = 25;
		System.out.println("Node radius: " + nodeRadius);

		nodeCircleMap = new ConcurrentHashMap<Node, Circle>();
		for(Node n : network.getNodes()) {
			int x = rng.nextInt(MainGUI.getDefaultWidth());
			int y = rng.nextInt(MainGUI.getDefaultHeight());
			Circle circle = new Circle(x, y, nodeRadius, Color.GREY);
			pane.add(circle, 0, 0);
			nodeCircleMap.put(n, circle);
		}

		edgeLineMap = new ConcurrentHashMap<Edge, Line>();
		for(Edge e : network.getEdges()) {
			Node input = e.getInput();
			Node output = e.getOutput();
			double x1 = nodeCircleMap.get(input).getCenterX();
			double y1 = nodeCircleMap.get(input).getCenterY();
			double x2 = nodeCircleMap.get(output).getCenterX();
			double y2 = nodeCircleMap.get(output).getCenterY();
			Line line = new Line(x1, y1, x2, y2);
			line.setStroke(Color.BLACK);
			pane.add(line, 0, 0);
			edgeLineMap.put(e, line);
		}

		pane.setOnKeyTyped(new EventHandler<KeyEvent>() {
			public void handle(KeyEvent e) {
				System.out.println(e.getCharacter());
				network.run();
			}
		});

		HBox hBox = new HBox();
		Button simButton = new Button("Run next simulation step");
		simButton.setText("Run next simulation step");
		simButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
			public void handle(MouseEvent e) {
				network.run();
			}
		});
		simButton.setVisible(true);
		hBox.getChildren().add(simButton);
		pane.add(hBox, 1, 1);

		root.getChildren().add(pane);
		network.run();

	}

	@Override public void start(Stage primaryStage) throws Exception {
		init(primaryStage);
		primaryStage.show();
	}

	public static ConcurrentHashMap<Node, Circle> getNodeCircleMap() {
		return nodeCircleMap;
	}

	public static ConcurrentHashMap<Edge, Line> getEdgeLineMap() {
		return edgeLineMap;
	}

	public static void main(String[] args) {
		launch(args);
	}
}