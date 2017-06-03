package base;

import static org.lwjgl.opengl.GL11.*;

import java.util.*;

import basictypes.Vector2d;

public class GraphicsHelper {
	public static final void draw2dSquare(Vector2d center, Vector2d size ){
		draw2dSquare(center.x - size.x / 2, center.y - size.y / 2, size.x, size.y );
	}

	public static final void draw2dSquare(double x, double y, double width, double height){
		glBegin(GL_TRIANGLE_FAN);		
		glTexCoord2f(0,0);
		glVertex2f((float)x,(float)y);
		glTexCoord2f(1,0);
		glVertex2f((float)(x + width),(float)y);		
		glTexCoord2f(1,1);
		glVertex2f((float)(x + width),(float)(y + height));
		glTexCoord2f(0,1);
		glVertex2f((float)x,(float)(y + height));
		glEnd();		
	}

}
