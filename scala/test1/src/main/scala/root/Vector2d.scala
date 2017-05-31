package root

class Vector2d(xin: Double, yin: Double) {
   def x = xin
   def y = yin
   def this() = this(0,0)
   override def toString() = {
      "Vector2d( " + xin + ", " + yin + ")";
   }
   def +(second: Vector2d) : Vector2d = {
      new Vector2d(x + second.x, y + second.y)
   }   
}
