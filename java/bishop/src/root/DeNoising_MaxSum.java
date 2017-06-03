package root;

import java.awt.image.*;
import java.io.*;

import javax.imageio.ImageIO;

import basictypes.Vector2i;

import helpers.ImageHelper;

public class DeNoising_MaxSum {
	final double h = 0; // bias to overall pixel value
	final double beta = 2.0; // bias to neighbors
	final double eta = 2; // bias to noisy

	int[][] noisyImage;
	int width;
	int height;
	int[][] cleanImage;

	final int numIts = 100;
	
	double calcEnergy( int x, int y, int value ) {
		double energy = value * h;
		for( Vector2i direction : Vector2i.directions4 ) {
			int newx = x + direction.x;
			int newy = y + direction.y;
			if( newx < 0 || newy < 0 || newx >= width || newy >= height ) {
				continue;
			}
			energy -= beta * value * cleanImage[newy][newx];
		}
		energy -= eta * noisyImage[y][x] * value;
		return -energy;
	}

	void doCleaning() {
		for( int it = 0; it <= numIts; it++ ) {
			boolean changed = false;
			for( int y = 0; y < height; y++ ) {
				for( int x = 0; x < width; x++ ) {
					double negEnergy = calcEnergy( x, y, -1 ); 
					double posEnergy = calcEnergy( x, y, 1 );
					if( posEnergy > negEnergy && cleanImage[y][x] != 1 ) {
						cleanImage[y][x] = 1;
						changed = true;
					} else if( posEnergy < negEnergy && cleanImage[y][x] != -1 ) {
						cleanImage[y][x] = -1;
						changed = true;
					}
				}
			}
			System.out.println("changed " + changed );
			if(!changed ) {
				break;
			}
		}
	}

	public void go() throws Exception {
		BufferedImage noisyImageImage = ImageHelper.loadImageFromAbsResource("/noisyimage.png");
		width = noisyImageImage.getWidth();
		height = noisyImageImage.getHeight();
		System.out.println("width: " + noisyImageImage.getWidth() + " height " + noisyImageImage.getHeight() );
		noisyImage = new int[ noisyImageImage.getHeight()][ noisyImageImage.getWidth() ];
		int onecount = 0;
		int zerocount = 0;
		byte[] rasterbytes = ((DataBufferByte)noisyImageImage.getRaster().getDataBuffer()).getData();
		System.out.println("rasterbytes size: " + rasterbytes.length + " " + rasterbytes.length / width / height 
				+ noisyImageImage.getType() );
		for( int y = 0; y < height; y++ ) {
			for( int x = 0; x < width; x++ ) {
				int value = noisyImageImage.getRGB(x, y);
				if( value != -1) {
					noisyImage[y][x] = 1;
					onecount++;
				} else {
					noisyImage[y][x] = -1;
					zerocount++;
				}
			}
		}
		System.out.println("zerocount " + zerocount + " onecount " + onecount );

		cleanImage = new int[height][width];
		for( int y = 0; y < height; y++ ) {
			for( int x = 0; x < width; x++ ) {
				cleanImage[y][x] = noisyImage[y][x];
			}
		}

		doCleaning();

		for( int y = 0; y < height; y++ ) {
			for( int x = 0; x < width; x++ ) {
				if( cleanImage[y][x] == 1 ) {
					noisyImageImage.setRGB(x, y, -16777216);
				} else {
					noisyImageImage.setRGB(x, y, -1);					
				}
			}
		}
		FileOutputStream fileOutputStream = new FileOutputStream(new File("/tmp/cleanedimage_maxsum.png"));
		ImageIO.write(noisyImageImage, "PNG", fileOutputStream );
	}

	public static void main( String[] args ) throws Exception {
		new DeNoising_MaxSum().go();
	}
}
