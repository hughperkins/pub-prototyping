package root
import math._
import collection.mutable.ArrayBuffer
import helpers.Timer
import breeze.linalg._
import breeze.stats.distributions._

object EmSvm {
   def go( K : Int, X: ArrayBuffer[ SparseVector[Double] ], Y:  ArrayBuffer[Int] ) {
      val nu = 1
      val iterations = 1
      val nu_a = 1
      val nu_b = 1

      //tic
      val N = X.length

      val Xy = new ArrayBuffer[ SparseVector[Double] ](N)
      for( n <- 0 until N ) {
         Xy(n) = X(n) * Y(n).toDouble
      }

      var beta = SparseVector.zeros[Double](K)

      val Xsqrtlambda = new ArrayBuffer[ SparseVector[Double ] ](N)
      val lambda = DenseVector.zeros[Double](N)
      for( it <- 0 until iterations ) {
         for( n <- 0 until N ) {
            lambda(n) = ( 1 - ( Xy(n) dot beta ) ).abs
            Xsqrtlambda(n) = X(n) * sqrt(1/lambda(n))
         }
         val xlambdax = DenseMatrix.zeros[Double](K,K)
         for( i <- 0 until K ) {
            for( j <- 0 until K ) {
               xlambdax(i,j) = Xsqrtlambda(i) dot Xsqrtlambda(j)
            }
         }/*
         val BInv = DenseMatrix.eye[Double](K) / ( nu*nu) + xlambdax
         val b = DenseVector.zeros[Double](K)
         for( n <- 0 until N ) {
            b(n) = Xy(n) dot lambda.map( x => 1 + 1 / x )
         }*/
        // beta = 
      }

/*
      for iteration=1:iterations
          lambda = abs( ones(n,1) - Xy * beta );   
          %'lambda', lambda(1:5,1)
          LambdaInv = spdiags(1./lambda,0,n,n);
          %'LambdaInv',LambdaInv(1:5,1:5)
          BInv = eye(k) / nu / nu + X' * LambdaInv * X;
          %'BInv',BInv(1:5,1:4)
          b = Xy' * ( ones(n,1) + 1./ lambda);
          %'b',b(1:5,1)

          beta = BInv \ b;
          %'beta', beta(1:5,1)
          
          nu = 1 / sqrt( ( nu_b + sum( beta.^ 2 ) ) / (k/2 + nu_a  - 1)  );
          %nu
      end
      learntime = toc;*/

   }
}

