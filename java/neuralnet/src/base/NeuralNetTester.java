package base;

import java.util.ArrayList;

import helpers.*;

public class NeuralNetTester {
	final static int gridWidth = 16;
	final static int gridHeight = 16;
	final static int numSets = 5;
	
	void go() {
		ImagesDataLoader imagesDataLoader = new ImagesDataLoader(gridHeight, gridWidth);
		ArrayList<ArrayList<Integer>> X = imagesDataLoader.load(System.getProperty("user.home") + "/anndigits.png");
		ArrayList<ArrayList<Integer>> Y = ClassesDataLoader.loadClasses(System.getProperty("user.home") + "/anndigits.txt"); 
		final int totalDataPoints = X.size();
		int numTraining = totalDataPoints * (numSets - 1 ) / numSets; 
		int numTest = totalDataPoints / numSets; 
		int totalright = 0;
		for( int set = 0; set < numSets; set++ ) {
			ArrayList< ArrayList< Integer> > trainingX = new ArrayList<ArrayList<Integer>>();
			ArrayList< ArrayList< Integer> > testX = new ArrayList<ArrayList<Integer>>();
			ArrayList< ArrayList< Integer> > trainingY = new ArrayList<ArrayList<Integer>>();
			ArrayList< ArrayList< Integer> > testY = new ArrayList<ArrayList<Integer>>();
			
			for( int i = 0; i < numTraining; i++ ) {
				trainingX.add( X.get(i) );
				trainingY.add( Y.get(i) );
			}
			for( int i = 0; i < numTest; i++ ) {
				testX.add( X.get(numTraining + i) );
				testY.add( Y.get(numTraining + i) );				
			}
			NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{ 
					(int)(gridWidth * gridHeight), 100, 9 });
			for( int it = 0; it < 50; it++ ) {
   			   neuralNetwork.trainOneIt(trainingX, trainingY);
			}
			int right = 0;
			int wrong = 0;
			for( int i = 0; i < numTest; i++ ) {
				double[] result = neuralNetwork.getOutput(testX.get(i) );
				int thisnum = doublearraytoresult( result );
				if( testY.get(i).get( thisnum-1 ) == 1 ) {
					right++;
					totalright++;
				} else {
					wrong++;
				}
			}
			System.out.println("num correct " + right + " out of " + numTest );
		}
		System.out.println("total right " + totalright + " out of " + ( numTest * numSets ) );
	}
	
	int doublearraytoresult( double[] resultarray ) {
		double max = resultarray[0];
		int pos = 0;
		for( int i = 2; i < resultarray.length; i++ ) {
			if( resultarray[i] > max ) {
				max = resultarray[i];
				pos = i;
			}
		}
		return pos + 1;
	}
	
	public static void main(String[] args) {
		new NeuralNetTester().go();
	}
}
