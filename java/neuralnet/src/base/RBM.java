package base;

import java.util.*;

import helpers.*;

public class RBM {
	final double alpha = 0.1;

	String[] movies = new String[]{"f1", "f2","both","o1", "o2", "misc" };
	int[][] points = new int[][]{
			{1,1,1,0,0,0},
			{1,0,1,0,0,0},
			{1,1,1,0,0,0},
			{0,0,1,1,1,0},
			{0,0,1,1,1,0},
			{0,0,1,1,1,0}
	};

	final static int numHidden = 2;
	final int numVisible = movies.length;

	double[][] weights = new double[numHidden][numVisible];
	double[] hiddenbiasweights = new double[numHidden];
	double[] hiddenvisibleweights = new double[numVisible];

	int[]hidden = new int[numHidden];
	int[]visible = new int[numVisible];

	Random random = new Random();

	final int numIts = 10000;

	public void go() {
		//		for( int i = 0; i < numVisible; i++ ) {
		//			for( int j = 0; j < numHidden; j++ ) {
		//				weights[j][i] = 
		//			}
		//		}
		for( int it = 1; it <= numIts; it++ ) {
			for( int pointindex = 0; pointindex < points.length; pointindex++ ) {
				int[] point = points[pointindex];
				for( int i = 0; i < numVisible; i++ ) {
					visible[i] = point[i];
				}
				propagateToHidden();
				int[][]positivevh = getVH();
				propagateToVisible();
				int[][] negativevh = getVH();
				for( int i = 0; i < numVisible; i++ ) {
					for( int j = 0; j < numHidden; j++ ) {
						weights[j][i] = weights[j][i] + alpha * ( positivevh[j][i] - negativevh[j][i] );
					}
				}
			}
		}
		for( int i = 0; i < numVisible; i++ ) {
			String line = "";
			line += movies[i] + " ";
			for( int j = 0; j < numHidden; j++ ) {
				line += weights[j][i] + " ";
			}
			System.out.println(line);
		}
	}

	int[][] getVH(){
		int[][]vh = new int[numHidden][numVisible];
		for( int i = 0; i < numVisible; i++ ) {
			for( int j = 0; j < numHidden; j++ ) {
				vh[j][i] = hidden[j] * visible[i];
			}
		}
		return vh;
	}

	void propagateToVisible() {
		for( int i = 0; i < numVisible; i++ ) {
			double activationenergy = 0;
			for( int j = 0; j < numHidden; j++ ) {
				activationenergy += weights[j][i] * hidden[j];
			}
			double probability = MathHelper.logisticSigmoid(activationenergy);
			visible[i] = random.nextDouble() <= probability ? 1 : 0;
//			System.out.println("probability " + probability + " value " + visible[i] );
		}
	}

	void propagateToHidden() {
		for( int j = 0; j < numHidden; j++ ) {
			double activationenergy = 0;
			for( int i = 0; i < numVisible; i++ ) {
				activationenergy += weights[j][i] * visible[i];
			}
			double probability = MathHelper.logisticSigmoid(activationenergy);
			hidden[j] = random.nextDouble() <= probability ? 1 : 0;
		}
	}

	public static void main( String[] args ) {
		new RBM().go();
	}
}
