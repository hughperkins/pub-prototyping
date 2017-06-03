package base;

import java.util.Arrays;

import org.lwjgl.input.Keyboard;

import basictypes.*;
import helpers.*;

public class HopfieldNet {
	static final int gridSize = 8;

	GraphicalInterface graphicalInterface;

	boolean[][] pixels = new boolean[gridSize][gridSize];

	int[][] weights = new int[gridSize*gridSize][gridSize*gridSize];

	public void go() {
		try {
			graphicalInterface = new GraphicalInterface(gridSize, gridSize);
			graphicalInterface.storeMouseDrags = true;
		} catch (Exception e) {
			e.printStackTrace();
		}

		while( true ) {
			while( graphicalInterface.mouseClicksPending() ) {
				Vector2i pos = graphicalInterface.getNextMouseClick();
				pixels[pos.y][pos.x] = true;
				System.out.println("click on " + pos );
			}
			while( graphicalInterface.keyboardEventsPending() ) {
				int click = graphicalInterface.getNextKeyboardEvent();
				if( click == Keyboard.KEY_M ) {
					System.out.println("memorize");
					memorize();
				}
				if( click == Keyboard.KEY_SPACE ) {
					System.out.println("test");
					test();
				}
				if( click == Keyboard.KEY_C ) {
					System.out.println("wipe grid");
					wipeGrid();
				}
				if( click == Keyboard.KEY_D ) {
					System.out.println("dump weights");
					dumpWeights();
				}
				if( click == Keyboard.KEY_P ) {
					System.out.println("dump pixels");
					dumpPixels();
				}
				if( click == Keyboard.KEY_E ) {
					System.out.println("dump energy");
					dumpEnergy();
				}
			}
			//			System.out.println("update interface");
			updateInterface();
		}
	}

	private void dumpPixels() {
		boolean[] gridarray = new boolean[gridSize * gridSize];
		for( int y = 0; y < gridSize; y++ ) {
			for( int x = 0; x < gridSize; x++ ) {
				gridarray[y*gridSize + x ] = pixels[y][x];
			}
		}
		System.out.println(Arrays.toString(gridarray) );
	}

	private void dumpEnergy() {
		int energy = 0;
		for(int y1 = 0; y1 < gridSize; y1++ ) {
			for( int x1 = 0; x1 < gridSize; x1++ ) {
				int v1 = pixels[y1][x1] ? 1 : -1;
				int weight1index = y1 * gridSize + x1;
				for(int y2 = 0; y2 < gridSize; y2++ ) {
					for( int x2 = 0; x2 < gridSize; x2++ ) {
						int weight2index = y2 * gridSize + x2;
						int v2 = pixels[y2][x2] ? 1 : -1;
						if( weight1index > weight2index ) {
							energy -= weights[weight1index][weight2index] * v1 * v2;
						}
					}
				}
			}
		}
		System.out.println("energy " + energy );
	}

	private void dumpWeights() {
		ArrayHelper.dump(weights, 2);
	}

	void updateInterface(){
		//		graphicalInterface.clear();
		for( int y = 0; y < gridSize; y++ ) {
			for( int x = 0; x < gridSize; x++ ) {
				if( pixels[y][x]) {
					graphicalInterface.draw(new Vector3i(255,255,255), new Vector2i(x,y));
				} else {
					graphicalInterface.erase(new Vector2i(x,y));
				}
			}
		}
	}

	private void test() {
		boolean updated = true;
		while( updated ) {
			boolean[][] newpixels = pixels.clone();
			updated = false;
			for(int y1 = 0; y1 < gridSize; y1++ ) {
				for( int x1 = 0; x1 < gridSize; x1++ ) {
					int weight1index = y1 * gridSize + x1;
					int sum = 0;
					for(int y2 = 0; y2 < gridSize; y2++ ) {
						for( int x2 = 0; x2 < gridSize; x2++ ) {
							int weight2index = y2 * gridSize + x2;
							int v2 = pixels[y2][x2] ? 1 : -1;
							int weight = 0;
							if( weight1index > weight2index ) {
								weight = weights[weight1index][weight2index];
							} else {
								weight = weights[weight2index][weight1index];
							}
							sum += weight * v2;
						}
					}
					boolean newvalue = sum >= 0;
					if( newpixels[y1][x1] != newvalue ) {
						newpixels[y1][x1] = newvalue;
						updated = true;
					}
				}
			}
			pixels = newpixels;
		}
	}

	private void memorize() {
		for(int y1 = 0; y1 < gridSize; y1++ ) {
			for( int x1 = 0; x1 < gridSize; x1++ ) {
				int weight1index = y1 * gridSize + x1;
				int v1 = pixels[y1][x1] ? 1 : -1;
				for(int y2 = 0; y2 < gridSize; y2++ ) {
					for( int x2 = 0; x2 < gridSize; x2++ ) {
						int weight2index = y2 * gridSize + x2;
						int v2 = pixels[y2][x2] ? 1 : -1;
						if( weight1index > weight2index ) {
							weights[weight1index][weight2index] +=  v1 * v2;
//						weights[weight2index][weight1index] +=  v1 * v2;
						}
					}
				}
			}
		}
	}

	private void wipeGrid() {
		for( int y = 0; y < gridSize; y++ ) {
			for( int x = 0; x < gridSize; x++ ) {
				pixels[y][x] = false;
			}
		}
	}

	public static void main( String[] args ) {
		new HopfieldNet().go();
	}
}
