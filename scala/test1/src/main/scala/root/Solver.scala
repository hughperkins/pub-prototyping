// Copyright Hugh Perkins 2012
// You can use this under the terms of the Apache Public License 2.0
// http://www.apache.org/licenses/LICENSE-2.0

package root

import breeze.linalg._

object Solver {
   // solve Ax = b, for x, where A = choleskyMatrix * choleskyMatrix.t
   // choleskyMatrix should be lower triangular
   def solve( choleskyMatrix: DenseMatrix[Double], b: DenseVector[Double] ) : DenseVector[Double] = {
      val C = choleskyMatrix
      val size = C.rows
      if( C.rows != C.cols ) {
          // throw exception or something
      }
      if( b.length != size ) {
          // throw exception or something
      }
      // first we solve C * y = b
      // (then we will solve C.t * x = y)
      val y = DenseVector.zeros[Double](size)
      // now we just work our way down from the top of the lower triangular matrix
      for( i <- 0 until size ) {
         var sum = 0.
         for( j <- 0 until i ) {
            sum += C(i,j) * y(j)
         }
         y(i) = ( b(i) - sum ) / C(i,i)
      }
      println(y)
      // now calculate x
      val x = DenseVector.zeros[Double](size)
      val Ct = C.t
      // work up from bottom this time
      for( i <- size -1 to 0 by -1 ) {
         var sum = 0.
         for( j <- i + 1 until size ) {
            sum += Ct(i,j) * x(j)
         }
         x(i) = ( y(i) - sum ) / Ct(i,i)
      }
      x
   }
}

