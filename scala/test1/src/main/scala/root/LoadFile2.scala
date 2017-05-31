package root
import collection.mutable
import helpers.Timer
import breeze.linalg._
import breeze.stats.distributions._
import MatrixHelper._

object LoadFile2 {
   // X has points as columns, with feature index as row number
   def load(filename: String, K: Int) : ( CSCMatrix[Double],  DenseVector[Double] ) = {
      println( filename )
      var n = 0
      var source = io.Source.fromFile(filename)
      for( line <- source.getLines ) {
         n += 1
      }
      val N = n
      source.close
      source = io.Source.fromFile(filename)
      var i = 0
      val X = spzeros(K,N)
      val Y = zeros(N)
      for( line <- source.getLines ) {
         val splitline = line.split(" ")
         var ystring = splitline(0)
         if( ystring(0) == '+' ) {
            ystring = ystring.substring(1)
         }
         Y(i) = ystring.toInt
         val splitlinelength = splitline.length
         for( j <- 1 until splitlinelength ) {
            val splittokens = splitline(j).split(":")
            val fid = splittokens(0).toInt - 1
            val fval = splittokens(1).toDouble
            X(fid,i) = fval
         }
         i += 1
      }
      source.close
      println( "lines " + i )
      (X,Y)
   }

//   def main( args: Array[String] ) {
 //     load("/tsinghua/mldata/splice/splice",60)
  // }
}

