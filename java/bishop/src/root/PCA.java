package root;

import helpers.GraphicalInterface;

import java.util.ArrayList;

import org.lwjgl.input.Keyboard;

import Jama.EigenvalueDecomposition;
import basictypes.Matrix;
import basictypes.Vector2d;
import basictypes.Vector2i;
import basictypes.Vector3i;

public class PCA {
	final int resolution = 128;
	GraphicalInterface graphicalInterface;

	ArrayList<Vector2i> points = new ArrayList<Vector2i>();
	ArrayList<Vector2d> whitepoints = new ArrayList<Vector2d>();	
	Vector2i mean;
	Matrix L;
	Matrix U;
	Matrix S;

	void calcMean(){
		mean = Vector2i.zero;
		for( Vector2i point : points ) {
			mean = mean.add(point);
		}		
		mean = mean.dividedBy(points.size());
		System.out.println("mean " + mean);
	}

	void calcS(){
		S = new Matrix(2, 2);
		calcMean();
		for( Vector2i point : points ) {
			Vector2i difffrommean = point.minus(mean);
			Matrix difffrommeanmatrix = Matrix.fromVectordd(difffrommean.asVectordd());
			System.out.println("difffrommean " + difffrommean );
			System.out.println("difffrommeanmatrix " + difffrommeanmatrix );
			S = S.add(difffrommeanmatrix.multiply(difffrommeanmatrix.getTranspose()));
		}
		System.out.println("S " + S );
		S = S.divide(points.size());
		System.out.println("S " + S );
		//		return S;
	}

	void whiten() {
		calcS();
		Jama.Matrix jS = new Jama.Matrix(S.values);
		EigenvalueDecomposition eigenvalueDecomposition = jS.eig();
		L = new Matrix( eigenvalueDecomposition.getD().getArray() );
		U = new Matrix( eigenvalueDecomposition.getV().getArray() );
		Matrix LHalf = new Matrix(L.values);
		LHalf.values[0][0] = 1 / Math.sqrt(LHalf.values[0][0]);
		LHalf.values[1][1] = 1 / Math.sqrt(LHalf.values[1][1]);
		Matrix UTranspose = U.getTranspose();
		whitepoints.clear();
		for( Vector2i point : points ) {
			Matrix diffmean = new Matrix( point.minus(mean).asVectordd() );
			Matrix whitepointmatrix = LHalf.multiply(UTranspose).multiply(diffmean);
			Vector2d whitepoint = Vector2d.fromVectordd(whitepointmatrix.asVectordd());
			whitepoints.add(whitepoint);
		}
	}

	void redraw() {
		for( Vector2i point : points ) {
			graphicalInterface.drawBuffer(point);
		}
		for( Vector2d point : whitepoints ) {
			Vector2d pointNormalized = point.multiplyBy(resolution / 2).add(new Vector2d(resolution/2,resolution/2));
			graphicalInterface.drawBuffer(new Vector3i(0,255,0), pointNormalized.floor());
			//			System.out.println(pointNormalized);
		}
		if( mean != null ) {
			Vector2d center = new Vector2d(resolution/2,resolution/2);
			for( int i = 0; i < 2; i++ ) {
				Vector2d uvector = new Vector2d(U.values[0][i],U.values[1][i]);
				Vector2d lineposd1 = center.minus(uvector.multiplyBy(resolution/4)); 
				Vector2d lineposd2 = center.add(uvector.multiplyBy(resolution/4)); 
				for(double u = 0; u < 1.0; u += 0.01 ) {
					Vector2d thispointd = lineposd1.multiplyBy(1 - u).add(lineposd2.multiplyBy(u));
					graphicalInterface.drawBuffer(graphicalInterface.colors[i + 1], thispointd.floor());
				}
			}
		}
		graphicalInterface.flipBuffers();
	}

	public void go() throws Exception{
		graphicalInterface = new GraphicalInterface(resolution, resolution);
		graphicalInterface.storeMouseDrags = true;
		while(true){
			if(graphicalInterface.mouseClicksPending()) {
				points.add(graphicalInterface.getNextMouseClick());
			}
			if( graphicalInterface.keyboardEventsPending()) {
				switch( graphicalInterface.getNextKeyboardEvent()) {
				case Keyboard.KEY_W:
					whiten();
					redraw();
					break;
				case Keyboard.KEY_C:
					points.clear();
					whitepoints.clear();
					mean = null;
					redraw();
					break;
				}
			}
			redraw();
			graphicalInterface.waitNextEvent();
		}
	}

	public static void main(String[] args ) throws Exception {
		new PCA().go();
	}
}
