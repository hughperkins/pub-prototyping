package base;

import java.util.*;

import helpers.MathHelper;

class NeuralNetwork {
	final double eta = 2.0;   // learning rate
	final double alpha = 0.1; // momentum

	public double error = 0;

	public final int[] numNodesByLayer;
	public final double[][][] weightByInNodeByOutNodeByInLayer; // [inlayer][outnode][innode]
	public final double[][][] deltaWeightByInNodeByOutNodeByInLayer; // [inlayer][outnode][innode]

	public final double[][] valueByNodeByLayer;
	public final double[][] deltaByNodeByLayer;
	public final double[][] sumBackDeltaByNodeByLayer;

	public final int numLayers;

	Random random = new Random();

	NeuralNetwork( int[] numNodesByLayer ) {
		this.numNodesByLayer = numNodesByLayer;

		numLayers = numNodesByLayer.length;

		weightByInNodeByOutNodeByInLayer = new double[numLayers][][];
		deltaWeightByInNodeByOutNodeByInLayer = new double[numLayers][][];
		valueByNodeByLayer = new double[numLayers][];
		deltaByNodeByLayer = new double[numLayers][];
		sumBackDeltaByNodeByLayer = new double[numLayers][];
		for( int layer = 0; layer < numNodesByLayer.length; layer++ ) {
			weightByInNodeByOutNodeByInLayer[layer] = new double[numNodesByLayer[layer]][];
			deltaWeightByInNodeByOutNodeByInLayer[layer] = new double[numNodesByLayer[layer]][];
			valueByNodeByLayer[layer] = new double[numNodesByLayer[layer]];
			deltaByNodeByLayer[layer] = new double[numNodesByLayer[layer]];
			sumBackDeltaByNodeByLayer[layer] = new double[numNodesByLayer[layer]];
			for( int out = 0; out < numNodesByLayer[layer]; out++ ) {
				weightByInNodeByOutNodeByInLayer[layer][out] = new double[numNodesByLayer[layer]+1];
				deltaWeightByInNodeByOutNodeByInLayer[layer][out] = new double[numNodesByLayer[layer]+1];
				// add bias
				weightByInNodeByOutNodeByInLayer[layer][out][0] = random.nextDouble() - 0.5;
				// add other weights
				for( int in = 0; in < numNodesByLayer[layer]; in++ ) {
					weightByInNodeByOutNodeByInLayer[layer][out][in+1] = random.nextDouble() - 0.5;
				}
			}
		}
	}
	
	public void backPropagateDelta(final ArrayList<Integer> y) {
		// back-propagate
		// from target, for last layer
		deltaByNodeByLayer[numLayers - 1] = new double[numNodesByLayer[numLayers-1]];
		for( int node = 0; node < numNodesByLayer[numLayers - 1]; node++ ) {
			double nodevalue = valueByNodeByLayer[numLayers - 1][node];
			deltaByNodeByLayer[numLayers - 1][node] = ( y.get( node ) - nodevalue ) * nodevalue * ( 1 - nodevalue );
		}
		// back-propagate, other layers -> delta			  
		for( int layer = numLayers - 2; layer >= 0; layer-- ) {
			deltaByNodeByLayer[layer] = new double[numNodesByLayer[layer]];
			sumBackDeltaByNodeByLayer[layer] = new double[numNodesByLayer[layer]];
			for( int node = 0; node < numNodesByLayer[layer]; node++ ) {
				double sumBackDelta = 0;
				for( int forwardnode = 0; forwardnode < numNodesByLayer[layer + 1]; forwardnode++ ) {
					sumBackDelta += weightByInNodeByOutNodeByInLayer[layer][forwardnode][node + 1]
							* deltaByNodeByLayer[layer + 1][forwardnode];
				}
				sumBackDeltaByNodeByLayer[layer][node] = sumBackDelta;
				double nodevalue = valueByNodeByLayer[layer][node];
				deltaByNodeByLayer[layer][node] = sumBackDeltaByNodeByLayer[layer][node] 
						* nodevalue * (1 - nodevalue );
			}
		}
	}
	
	void trainOnePoint( final ArrayList<Integer> x, final ArrayList<Integer> y ) {
		getOutput( x );

		// calculate error
		for( int out = 0; out < numNodesByLayer[numLayers - 1]; out++ ) {
			double thisdiff = y.get(out) - valueByNodeByLayer[numLayers - 1][out];
			error += 0.5 * thisdiff * thisdiff;
		}

		backPropagateDelta(y);

		// update layer / layer + 1  weight
		for( int layer = numLayers - 2; layer >= 0; layer-- ) {
			for( int outnode = 0; outnode < numNodesByLayer[layer + 1]; outnode++ ) {
				deltaWeightByInNodeByOutNodeByInLayer[layer][outnode][0] =
						eta * 1 * deltaByNodeByLayer[layer + 1][outnode]
								+ alpha * deltaWeightByInNodeByOutNodeByInLayer[layer][outnode][0];
				for( int innode = 0; innode < numNodesByLayer[layer]; innode++ ) {
					deltaWeightByInNodeByOutNodeByInLayer[layer][outnode][innode + 1] =
							eta * valueByNodeByLayer[layer][innode] * deltaByNodeByLayer[layer + 1][outnode]
									+ alpha * deltaWeightByInNodeByOutNodeByInLayer[layer][outnode][innode + 1];
				}
				for( int innode = 0; innode < numNodesByLayer[layer] + 1; innode++ ) {
					weightByInNodeByOutNodeByInLayer[layer][outnode][innode] +=
							deltaWeightByInNodeByOutNodeByInLayer[layer][outnode][innode];
				}
			}
		}
	}

	public void trainOneIt(final ArrayList<ArrayList<Integer> > X, final ArrayList<ArrayList<Integer> > Y ) {
		error = 0;
		for( int i = 0; i < X.size(); i++ ) {
			trainOnePoint( X.get(i), Y.get(i) );			
		}
	}

	double[] getOutput( ArrayList<Integer> x ) {
		// move input values into first layer
		valueByNodeByLayer[0] = new double[numNodesByLayer[0]];
		for( int i = 0; i < numNodesByLayer[0]; i++ ) {
			valueByNodeByLayer[0][i] = x.get(i);
		}
		// forward propagate to other layers -> value
		for( int layer = 1; layer < numLayers; layer++ ) {
			valueByNodeByLayer[layer] = new double[numNodesByLayer[layer] ];
			for( int out = 0; out < numNodesByLayer[layer]; out++ ) {
				double outvalue = 0;
				////add bias
				double weight = weightByInNodeByOutNodeByInLayer[layer - 1][out][0];
				double srcvalue = 1;
				outvalue += srcvalue * weight;
				for( int in = 0; in < numNodesByLayer[layer - 1]; in++ ) {
					weight = weightByInNodeByOutNodeByInLayer[layer - 1][out][in + 1];
					srcvalue = valueByNodeByLayer[layer - 1][in];
					outvalue += srcvalue * weight;
				}
				valueByNodeByLayer[layer][out] = MathHelper.logisticSigmoid(outvalue);
			}
		}	
		return valueByNodeByLayer[numLayers-1];
	}
}
