package root;

import java.awt.image.*;
import java.io.*;
import java.util.Random;

import javax.imageio.ImageIO;

import helpers.ImageHelper;

public class NoiseGenerator {
	public void go() throws Exception {
		BufferedImage cleanImageImage = ImageHelper.loadImageFromAbsResource("/imageclean.png");
		int width = cleanImageImage.getWidth();
		int height = cleanImageImage.getHeight();
		System.out.println("width: " + cleanImageImage.getWidth() + " height " + cleanImageImage.getHeight() );

		Random random = new Random();
		byte[] rasterbytes = ((DataBufferByte)cleanImageImage.getRaster().getDataBuffer()).getData();
		for( int y = 0; y < height; y++ ) {
			for( int x = 0; x < width; x ++ ) {
				int value = cleanImageImage.getRGB(x, y);
//				System.out.println(value);
				if( random.nextInt(10) == 0 ) {
					if( value == -1 ) {
						cleanImageImage.setRGB(x, y, -16777216);
					} else {
						cleanImageImage.setRGB(x, y, -1);						
					}
//					cleanImageImage.se
				}
			}
		}
		FileOutputStream fileOutputStream = new FileOutputStream(new File("/tmp/noisyimage.png"));
		ImageIO.write(cleanImageImage, "PNG", fileOutputStream );
	}
	
	public static void main(String[] args ) throws Exception {
		new NoiseGenerator().go();
	}
}
