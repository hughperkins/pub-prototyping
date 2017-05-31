package root
import helpers.Timer
import breeze.linalg._
import breeze.stats.distributions._

object TestLda {
   def matrixRand( maxExclusive: Integer, rows: Integer, cols: Integer = 1 ) : DenseMatrix[Int] = {
      val result = DenseMatrix.zeros[Int](rows,cols)
      for( row <- 0 until rows ) {
         for( col <- 0 until cols ) {
            result(row,col) = scala.util.Random.nextInt(maxExclusive)
         }
      }
      result
   }

   def go() {
      val burnin = 30
	   val samples = 5
	   val N = 100; // num docs
	   val M = 30; // num words per doc
	   val J = 5; // vocab size
	   val alpha = 0.1;
	   val beta = 0.1;
	   val K = 3;
	   val iterations = burnin + samples;
	   val topics = DenseMatrix(
			   (0.5,0.5,0.,0.,0.),
			   (0.5,0.,0.,0.,0.5),
			   (0.,0.333,0.333,0.334,0.)
	   )

		val X = DenseMatrix.zeros[Int](N,M)
      val alphaArray = DenseVector.fill[Double](K)(alpha)
		for( n <- 0 until N ) {
			val Theta = Dirichlet(alphaArray).draw()
			for( m <- 0 until M ) {
				val thisz = Multinomial(Theta).draw()
				val thisword = Multinomial(topics(thisz,::)).draw()._2
				X(n,m) = thisword
			}
		}
		println( X )

		val Z = matrixRand(K, N, M)
		println(Z)

		val c_k = DenseVector.zeros[Int](K)
      val c_jk = DenseMatrix.zeros[Int](J,K)
      val c_nk = DenseMatrix.zeros[Int](N,K)
      for( n <- 0 until N ) {
         for( m <- 0 until M ) {
            val thisk = Z(n,m)
            val thisj = X(n,m)
				c_k(thisk) += 1
				c_jk(thisj,thisk) += 1
				c_nk(n,thisk) += 1
			}
		}
      println(c_k)
      println(c_jk)
      println(c_nk)

		val avg_c_k = DenseVector.zeros[Int](K)
		val avg_c_jk = DenseMatrix.zeros[Int](J,K)
		val timer = new Timer()
		for( it <- 0 until iterations ) {
			for( n <- 0 until N ) {
				for( m <- 0 until M ) {
					val oldk = Z(n,m)
					val thisj = X(n,m)
					c_k(oldk) -= 1
					c_nk(n,oldk) -= 1
					c_jk(thisj,oldk) -= 1

					val prob_by_k_unnorm = DenseVector.zeros[Double](K)
					for( k <- 0 until K ) {
						prob_by_k_unnorm(k) = ( ( c_nk(n,k) + alpha ) 
								* ( c_jk(thisj,k) + beta ) 
								/ ( c_k(k) + J * beta ) )
					}
					val prob_by_k = prob_by_k_unnorm / sum(prob_by_k_unnorm)
					val newk = Multinomial(prob_by_k).draw()

					Z(n,m) = newk
					c_k(newk) += 1
					c_nk(n,newk) += 1
					c_jk(thisj,newk) += 1
				}
			}
			if( it > burnin ) {
				avg_c_k += c_k / samples
				avg_c_jk += c_jk / samples
			}
			val time = timer.timeCheckMilliseconds()
			println("it: " + it + " " + time + " ms")
		}

		val p_given_k_of_j = DenseMatrix.zeros[Double](K,J)
		for( k <- 0 until K ) {
			for( j <- 0 until J ) {
				p_given_k_of_j(k,j) = ( avg_c_jk(j,k) + beta ) / ( avg_c_k(k) + J * beta )
			}
		}
		println(p_given_k_of_j)
      println(p_given_k_of_j.map( x => x > 0.1) )
   }
}

