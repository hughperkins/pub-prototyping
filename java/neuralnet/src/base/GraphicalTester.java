package base;
// copyright Hugh Perkins 2010,2011

import static org.lwjgl.opengl.GL11.*;

import java.util.*;

import org.lwjgl.input.*;
import org.lwjgl.opengl.Display;

import basictypes.*;
import helpers.*;

public class GraphicalTester {
	final static int gridHeight = 16;
	final static int gridWidth = 16;
	double aspectRatio;

	final static Vector3i[] colors = new Vector3i[]{
		new Vector3i(255,0,0),
		new Vector3i(0,255,0),
		new Vector3i(0,0,255),
		new Vector3i(255,255,0),
		new Vector3i(255,0,255),
		new Vector3i(0,255,255)
	};

	final HashSet<Vector2i> features = new HashSet<Vector2i>();

	final ArrayList< ArrayList<Integer> > X = new ArrayList<ArrayList<Integer>>();
	final ArrayList<ArrayList<Integer> > Y = new ArrayList<ArrayList<Integer>>();

	final FontDrawer fontDrawer = new FontDrawer();
	NeuralNetwork neuralNetwork; // = new NeuralNetwork( new int[]{ 
			//(int)(gridWidth * gridHeight), 4 } );

	boolean iterate = false;

	int mousex;
	int mousey;

	void iterate() {
		System.out.println("iterate");
		neuralNetwork = new NeuralNetwork( new int[]{ 
				(int)(gridWidth * gridHeight), 4, 4 } );
		for( int i = 0; i < 100; i++ ) {
			neuralNetwork.trainOneIt(X, Y);
		}
		updateStatus();
	}

	public static void main(String args[])throws Exception {
		new GraphicalTester().run();
	}

	public void run()throws Exception {
		init();
		while (true) {
			updateModel();
			render();
		}
	}

	ArrayList<Integer> featuresToX(HashSet<Vector2i> features ) {
		int xlength = gridHeight * gridWidth;
		ArrayList<Integer> newpoint = new ArrayList<Integer>(xlength);
		CollectionsHelper.resize(newpoint, xlength, 0);
		for( Vector2i point : features ) {
			int thisoffset = point.y * gridWidth + point.x; 
			newpoint.set( thisoffset, 1 );
		}
		return newpoint;
	}

	void handleKeyboard(){
		if( Keyboard.next() && Keyboard.getEventKeyState() ) {
			if( Keyboard.getEventKey() == Keyboard.KEY_ESCAPE) {
				System.exit(0);
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_SPACE) {
				iterate();
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_I) {
				iterate = !iterate;
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_Q) {
				dataChanged();
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_A) {
				dataChanged();
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_Y) {
				int thisnum = guesseddigit;
				ArrayList<Integer> newpoint = featuresToX(features);
				X.add(newpoint);
				ArrayList<Integer> newtarget = new ArrayList<Integer>();
				CollectionsHelper.resize(newtarget, 9, 0);
				newtarget.set(thisnum - 1, 1);
				Y.add(newtarget);
				features.clear();
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_D) {
				ImagesDataLoader imagesDataLoader = new ImagesDataLoader(gridHeight, gridWidth);
				imagesDataLoader.save(System.getProperty("user.home") + "/anndigits.png", X );
				ClassesDataLoader.saveClasses(System.getProperty("user.home") + "/anndigits.txt", Y);
			}
			if( Keyboard.getEventKey() == Keyboard.KEY_C) {
				features.clear();
				dataChanged();
			}
			if( Keyboard.getEventKey() >= Keyboard.KEY_1 &&
					Keyboard.getEventKey() <= Keyboard.KEY_9 ) {
				int thisnum = Keyboard.getEventKey() - Keyboard.KEY_1 + 1;
				ArrayList<Integer> newpoint = featuresToX(features);
				X.add(newpoint);
				ArrayList<Integer> newtarget = new ArrayList<Integer>();
				CollectionsHelper.resize(newtarget, 9, 0);
				newtarget.set(thisnum - 1, 1);
				Y.add(newtarget);
				//features.clear();
			}
		}		
	}

	void dataChanged(){
		float[]X = new float[features.size()];
		float[]T = new float[features.size()];
		int i = 0;
		for(Vector2i point : features ) {
			X[i] = point.x / (float)gridWidth;
			T[i] = point.y / (float)gridWidth;
			i++;
		}
		updateStatus();
	}

	void handleMouse(){
		int displayWidth = Display.getDisplayMode().getWidth();
		int displayHeight = Display.getDisplayMode().getHeight();
		//int x = (int)(Mouse.getX() * gridWidth / Display.getDisplayMode().getWidth() / aspectRatio);
		int x = (int)((Mouse.getX() - (displayWidth - displayHeight ) / 2 ) * gridWidth / (double)Display.getDisplayMode().getWidth() * aspectRatio );
		int y = Mouse.getY() * gridHeight / Display.getDisplayMode().getHeight();
		mousex = x;
		mousey = y;
		if( Mouse.isButtonDown(0)) {
			if( x >= 0 && y >= 0 && x < gridWidth && y < gridHeight ) {
				features.add(new Vector2i(mousex, mousey));
				dataChanged();
			}
		}
		if( Mouse.isButtonDown(1)) {
			features.remove(new Vector2i(mousex, mousey));
			dataChanged();
		}
	}

	void updateModel(){
		handleMouse();
		handleKeyboard();
		if(features.size()>= 2 ) {
			if( iterate ) {
			}
		}
	}


	float gridPosToGlPosX(int x ) {
		//gridWidth = gridHeight * Display.getDisplayMode().getWidth() / Display.getDisplayMode().getHeight();
		//float result = (float)( (x / gridWidth * 2 - 1));// / aspectRatio ); 
		float result = (float)(((float)x / gridWidth * 2 - 1 ) / aspectRatio); 
		return result;
	}
	float gridPosToGlPosY(int y ) {
		float result = (float)y / gridHeight * 2 - 1; 
		return result;
	}

	void drawPixel(int x, int y ){
		glVertex3f(gridPosToGlPosX(x), gridPosToGlPosY(y+1),0);
		glVertex3f(gridPosToGlPosX(x+1), gridPosToGlPosY(y+1),0);
		glVertex3f(gridPosToGlPosX(x+1), gridPosToGlPosY( y),0);		
		glVertex3f(gridPosToGlPosX(x), gridPosToGlPosY(y),0);
	}

	void glColor(Vector3i color ) {
		glColor3f(color.x / 255f, color.y / 255f, color.z / 255f );
	}

	void drawString(int x, int y, String text ){
		fontDrawer.drawString(gridPosToGlPosX(x), gridPosToGlPosY(y), 0.02f, text );
	}

	int textPos = 0;

	void initTextPos() {
		textPos = 0;
	}

	void write( String text ) {
		fontDrawer.drawString( -0.98f, 0.9f - (float)textPos / 20, 0.05f, text );
		textPos ++;
	}

	void updateStatus() {
		if( neuralNetwork == null ) {
			return;
		}
		ArrayList<Integer> x = featuresToX(features);
		double[] resultarray = neuralNetwork.getOutput(x);
		double max = resultarray[0];
		int maxpos = 0;
		for( int i = 1; i < resultarray.length; i++ ) {
			if( resultarray[i] > max ) {
				max = resultarray[i];
				maxpos = i;
			}
		}
		guesseddigit = maxpos + 1;
	}

	int guesseddigit = 0;

	void drawStatus() {
		initTextPos();
		write("Guess: " + guesseddigit );
		write("Number training points: " + X.size() );
		if( neuralNetwork != null ) {
			write("Error: " + neuralNetwork.error );
		}
	}

	Vector2d getNodePos( int layer, int node ) {
		return new Vector2d(-0.9 + layer * 0.2, - node * 0.1 + 0.5 );
	}

	void renderNet(){
		if( neuralNetwork == null ) {
			return;
		}
		ArrayList<Integer> x = featuresToX(features);
		double[] resultarray = neuralNetwork.getOutput(x);
		int index = X.indexOf(x);
		if( index != -1 ) {
			neuralNetwork.backPropagateDelta(Y.get(index));
		}
		Vector2d nodesize = new Vector2d( 0.05, 0.05 );
		int numLayers = neuralNetwork.numLayers;
		for( int layer = 0; layer < numLayers; layer++ ) {
			// draw nodes
			for( int node = -1; node < neuralNetwork.numNodesByLayer[layer]; node++ ) {
				Vector2d nodepos = getNodePos(layer, node);
				// draw circle
				double nodevalue = 1;
				if( node != - 1 ) {
					nodevalue = neuralNetwork.valueByNodeByLayer[layer][node];
				}
				if( nodevalue > 0 ) {
   				   glColor3f((float)(1-nodevalue),1,(float)(1-nodevalue));
				} else {
   				   glColor3f(1,(float)(1-nodevalue),(float)(1-nodevalue));					
				}
				GraphicsHelper.draw2dSquare(nodepos, nodesize);
				// write value
				glColor3f(0,0,0);/*
				String text = "b=1";
				if( node != - 1 ) {
					text = "" + nodevalue;
				}
				if( text.length() > 5 ) {
					text = text.substring(0, 5);
				}
				fontDrawer.drawString( (float)(nodepos.x - nodesize.x * 0.4f),
						(float)(nodepos.y - nodesize.y * 0.3f ), 
						0.03f,
						text );*/
				glColor3f(1,1,1);/*
				if( node != -1 ) {
					fontDrawer.drawString( (float)(nodepos.x - nodesize.x * 0.4f),
							(float)(nodepos.y - nodesize.y * 0.3f - 0.06f ), 
							0.03f,
							"d " + left("" + neuralNetwork.deltaByNodeByLayer[layer][node],5 ));					
					if( layer != numLayers - 1 ) {
						fontDrawer.drawString( (float)(nodepos.x - nodesize.x * 0.4f),
								(float)(nodepos.y - nodesize.y * 0.3f - 0.12f ), 
								0.03f,
								"sbd " + left("" + neuralNetwork.sumBackDeltaByNodeByLayer[layer][node],5 ));										
					}
				}*/
			}
			if( layer < numLayers - 1 ) {
				// draw weights
				for( int out = 0; out < neuralNetwork.numNodesByLayer[layer+1]; out++ ) {
					Vector2d outpos = getNodePos(layer + 1, out);
					for( int in = -1; in < neuralNetwork.numNodesByLayer[layer]; in++ ) {
						float nodevalue = 1;
						if( in != - 1 ) {
							nodevalue = (float)neuralNetwork.valueByNodeByLayer[layer][in];
						}
						float weight = (float)neuralNetwork.weightByInNodeByOutNodeByInLayer[layer][out][in+1];
						float weightransmit = nodevalue * weight;
						Vector2d inpos = getNodePos(layer, in);
						if( weightransmit > 0 ) {
		   				   glColor3f((float)(1-weightransmit),1,(float)(1-weightransmit));
						} else {
		   				   glColor3f(1,(float)(1-weightransmit),(float)(1-weightransmit));					
						}
						glBegin(GL_LINES);
						glVertex3f((float)inpos.x,(float)inpos.y,0);
						glVertex3f((float)outpos.x,(float)outpos.y,0);
						glEnd();
						glColor3f(1,1,1);
						/*
						fontDrawer.drawString(
								(float)(( 3 * inpos.x + outpos.x ) / 4),
								(float)(( 3 * inpos.y + outpos.y ) / 4),
								0.03f,
								( "" + neuralNetwork.weightByInNodeByOutNodeByInLayer[layer][out][in+1] ).substring(0, 5)
								);*/
					}
				}
			}
		}
	}
	
	String left( String in, int maxchars ) {
		if( in.length() <= maxchars ) {
			return in;
		}
		return in.substring(0, maxchars );
	}

	void drawDrawingBoundary() {
		glBegin( GL_LINES );
		glVertex3f(gridPosToGlPosX(0), gridPosToGlPosY(0),0);
		glVertex3f(gridPosToGlPosX(0), gridPosToGlPosY(gridHeight ),0);
		glVertex3f(gridPosToGlPosX(gridWidth ), gridPosToGlPosY(0),0);
		glVertex3f(gridPosToGlPosX(gridWidth ), gridPosToGlPosY(gridHeight ),0);
		glEnd();
	}
	
	private void render() {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glColor4f(1,1,1,1);

		//		System.out.println("render");
		glColor3f(1,1,1);
		drawDrawingBoundary();
		glBegin(GL_QUADS);  
		glColor3f(1,1,1);
		for( Vector2i point : features ) {
			//glColor3f( point.x/ (float)gridWidth, point.y / (float)gridHeight, 1 - point.x/ (float)gridWidth - point.y / (float)gridHeight );
			glColor3f(0.5f,0.5f,0.5f);
			drawPixel( point.x, point.y );
		}
		glColor3f(1,1,1);
		glEnd();
		drawStatus();
		renderNet();
		Display.update();
	}

	void loadDump(){
		ImagesDataLoader imagesDataLoader = new ImagesDataLoader(gridHeight, gridWidth);
		ArrayList<ArrayList<Integer>> X = imagesDataLoader.load(System.getProperty("user.home") + "/anndigits.png");
		ArrayList<ArrayList<Integer>> Y = ClassesDataLoader.loadClasses(System.getProperty("user.home") + "/anndigits.txt"); 
		this.X.clear();
		this.X.addAll( X );
		this.Y.clear();
		this.Y.addAll( Y );
		System.out.println("x size " + X.size() + " y size " + Y.size() );
	}

	private void init() throws Exception {
		loadDump();

		Display.create();
		fontDrawer.init();
		glShadeModel(GL_SMOOTH); // Enable Smooth Shading
		glEnable(GL_TEXTURE_2D);
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		//gridWidth = gridHeight * Display.getDisplayMode().getWidth() / Display.getDisplayMode().getHeight();
		aspectRatio = (double)Display.getDisplayMode().getWidth() / Display.getDisplayMode().getHeight();
		System.out.println("aspectratio " + aspectRatio);
	}
}
