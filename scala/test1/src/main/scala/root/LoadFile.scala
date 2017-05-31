package root
import collection.mutable
import helpers.Timer
import breeze.linalg._
import breeze.stats.distributions._

object LoadFile {
   def load(filename: String, K: Int) : ( mutable.ArrayBuffer[ SparseVector[Double] ],  mutable.ArrayBuffer[Int] ) = {
      println( filename )
      val source = io.Source.fromFile(filename)
      var i = 0
      val X = mutable.ArrayBuffer[ SparseVector[Double] ]()
      val Y = mutable.ArrayBuffer[Int]()
      for( line <- source.getLines ) {
         val splitline = line.split(" ")
         var ystring = splitline(0)
         if( ystring(0) == '+' ) {
            ystring = ystring.substring(1)
         }
         val y = ystring.toInt
         val splitlinelength = splitline.length
         val x = SparseVector.zeros[Double](K)
         for( j <- 1 until splitlinelength ) {
            val splittokens = splitline(j).split(":")
            val fid = splittokens(0).toInt - 1
            val fval = splittokens(1).toDouble
            x(fid) = fval
         }
         X += x
         Y += y
         i += 1
      }
      source.close
      println( "lines " + i )
      println( X(2) )
      println( X(4) )
      println( Y(2) )
      println( Y(4) )
      (X,Y)
   }

//   def main( args: Array[String] ) {
 //     load("/tsinghua/mldata/splice/splice",60)
  // }
}

