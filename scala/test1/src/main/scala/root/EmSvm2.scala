package root
import math._
import collection.mutable.ArrayBuffer
import TicToc._
import breeze.linalg._
import breeze.stats.distributions._
import root.MatrixHelper._

object EmSvm2 {
   def go( Xt: CSCMatrix[Double], Y:  DenseVector[Double] ) {
      val iterations = 1
      val nu_a = 1
      val nu_b = 1

      val X = transpose(Xt)
      val N = X.rows
      val K = X.cols

      var beta = zeros(K,1)
      var nu = 1.

      tic()
      val Xy = spdiag(Y) * X 
      toc("multiply X by spdiag(Y)")

      for( it <- 0 until iterations ) {
        var lambda = ( Xy * beta  ).map( u => abs( 1 - u ) )
        toc("lambda")
//        var lambda2 = absm( ones(N,1) - ( Xy * beta  ) )
//        toc("lambda2")
        var lambdainv = spdiag( ones(N,1) :/ lambda )
        toc("lambdainv")
        var Xtlambdainv = Xt * lambdainv
        toc("xtlambdainv")
        var xlambda1fromleft = Xtlambdainv * X
        toc("xlambdaxfromleft")

        var Xtlambdainvfull = full(Xtlambdainv)
        var Xfull = full(X)
        toc("make full")
        var xlambda1fromleftfull = Xtlambdainvfull * Xfull
        toc("xlambdaxfromleftfull")
        
        var Binv = mul( eye(K), (1/nu*nu) ) + Xt * lambdainv * X        
        toc("binv")
        var b = transpose(Xy) * lambda.map( x => 1 + 1 / x ) 
        toc("b")
        beta = Binv \ b
        toc("beta")
        nu = 1. / sqrt( ( nu_b + sum( powm( beta, 2 ) ) ) / (K/2 + nu_a  - 1)  )
        toc("it " + it)
      }      
      //println(beta)
   }
}

//        var Xsqrtlambda = X * spdiag( lambda.map( x => sqrt( 1/x ) ) )
//        toc("sqrtlambda")
//        println(shape(Xsqrtlambda))
//        var xlambdax = Xsqrtlambda * transpose( Xsqrtlambda )
//        toc("xlambdax")
//        var Binv = mul( eye(K), (1/nu*nu) ) + xlambdax
