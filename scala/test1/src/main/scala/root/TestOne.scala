package root
import helpers.Timer
import breeze.linalg._
import TicToc._
import MatrixHelper._

object TestOne {        
    def main( args: Array[String]) {
        //println("foobar 2")
        //testSolver()  
        //test1()
//        testMult2
        //testSparseSparseMult
        testEmSvm2
        //testsprand
     // testsprandmult()
//       testDenseSparseMult();
    }
    
    def testsprandmult() {
        var N = 1000
        var K = 100
        println("N " + N + " K " + K)
        tic()
//        println(MatrixHelper.sprand(3,3,0.3))
        //var a = MatrixHelper.sprand(N,N,0.001)
//        var a = CSCMatrix.rand(K,N)
//        toc("created rand sparse matrix")
        var a = sprand(K,N,0.00003)
        println(a.colPtrs(a.cols));
        println(a)
        //println(a.colPtrs.deep.mkString("\n" ) );
        //println(a.rowIndices.mkString("\n" ) );
        toc("created sprand sparse matrix")
        var bv = DenseVector.rand(N)
        var b = spdiag( bv );
        println(bv)
        //println(a)
        //println(b)
        toc("created diag rand sparse matrix")
        //var c = MatrixHelper.mul(a,b)
        //toc("multiplied a by b using MatrixHelper.mul")
        var c2 = a * b
        toc("a * b")
        //println( (c - c2).sum() );
        //println(c)
    }

    def testsprand() {
       println(MatrixHelper.sprand(5,5,0.33333))
       val A = MatrixHelper.sprand(1000,1000,0.3333)
       //var count = 0
       //A.foreach( x => if( x > 0 ){count += 1} )
       println( A.activeSize )
       println( A.colPtrs(1000) )
    }

    def testEmSvm2() {
      println("loading file")
      val (_X,_Y) = LoadFile2.load("/tsinghua/mldata/splice/splice",60)
      val X = _X; val Y = _Y
      println( X.rows )
      println( X.cols )
      println( Y.size )
      println( X(0 to 59, 0 to 0 ) )
      println( Y(0) )
      println("call emsvm2")
      EmSvm2.go( X, Y )
    }
/*
    def testSparseSparseMult() {
        var A = CSCMatrix.zeros[Double](2,3)
        A(0,0) = 2.
        A(0,1) = 3.
        A(1,1) = 4.
        println("A")
        println(A)
        var B = CSCMatrix.zeros[Double](3,2)
        B(0,0) = 2.
        B(0,1) = 3.
        B(1,0) = 5.
        B(1,1) = 4.
        println("B")
        println(B)
        var C = root.MatrixHelper.mul( A, B)
        println("C")
        println(C)    

        A = CSCMatrix.zeros[Double](10000,10000)
        B = CSCMatrix.zeros[Double](10000,10000)
        val timer = new helpers.Timer
        C = MatrixHelper.mul(A,B)
        timer.printTimeCheckMilliseconds

        println("sprand * spdiag:")
        val N = 100000
        val K = 100  
        val fillFactor = 0.01
        A = MatrixHelper.sprand(K,N,fillFactor)
        timer.printTimeCheckMilliseconds
        B = MatrixHelper.spdiag( DenseVector.rand(N) )
        timer.printTimeCheckMilliseconds
        C = MatrixHelper.mul(A,B)
        timer.printTimeCheckMilliseconds
    }
*/
    /*
    def testDenseSparseMult() {
        var A = DenseMatrix( (1.,2.),(3.,4.))
        var B = CSCMatrix.zeros[Double](2,2)
        B(0,0) = 2.
        B(0,1) = 3.
        B(1,0) = 5.
        B(1,1) = 4.
        var C = root.MatrixHelper.mul( A, B)
        println(C)

        B = CSCMatrix.zeros[Double](2,2)
        B(0,0) = 2.
        B(1,1) = 4.
        C = root.MatrixHelper.mul( A, B)
        println(C)
        
        var N = 100000
        var K = 100
        tic
        var a = DenseMatrix.rand(K,N)
        toc("after rand")
        var b = MatrixHelper.spdiag(DenseVector.rand(N))
        toc("after rand diag")
        var c = a * b
        toc("after multiply using * operator")
        var c2 = MatrixHelper.mul(a,b)
        toc("after multiply using mul")
        for( row <- 0 until c.rows ) {
          for( col <- 0 until c.cols ) {
            if( c(row,col) != c2(row,col)) {
              println("discrepancy " + row + " " + col + " " + c(row,col) + " " + c2(row,col))
            }
          }
        }

        var bt = transpose(b)
        toc("transpose b")
        var c3 = bt * a.t
        toc("b' * a'")
        for( row <- 0 until c.rows ) {
          for( col <- 0 until c.cols ) {
            if( c2(row,col) != c3(col,row)) {
              println("discrepancy " + row + " " + col + " " + c2(row,col) + " " + c3(col,row))
            }
          }
        }
    }
*/
    def testMult() {
       val timer = new Timer
       val N = 100000
       val K = 100
       val A = DenseMatrix.rand(N,K)
       val b = DenseVector.rand(N)
       timer.printTimeCheckMilliseconds
       val c = A.t * MatrixHelper.spdiag(b) * A
       timer.printTimeCheckMilliseconds
    }

    def testMult2() {
       val N = 100000
       val K = 100
       val A = DenseMatrix.rand(N,K)
       val b = DenseVector.rand(N)
       val c = spdiag(b)
       val timer = new Timer
       val d = A.t * c
       val e = d * A
       timer.printTimeCheckMilliseconds
    }

    def testSolver() {
        val A = DenseMatrix.rand(5,5)
        val As = A * A.t
        val C = cholesky(As)
        val b = DenseVector.rand(5)
        println(As)
        val x = Solver.solve(C,b)
        println(x)
        println(b)
        println(As * x)
        println(As * b)

        val x2 = As \ b
        println(x2)
        println(As * x2)
    }

    def test1() {
        println("hello")
        val testClass = new TestClass
        var a = 3
        a += 2
        testClass.foo
        println( 3 + 5 )
        val v1 = new Vectordd(Array(3,5))
        println( v1 )
        val timer = new Timer
        println( v1 + new Vectordd(Array(7,11) ) )
        println( v1 - new Vectordd(Array(7,11) ) )
        println( v1 * new Vectordd(Array(7,11) ) )
        println( v1 :* new Vectordd(Array(7,11) ) )
        timer.printTimeCheckMilliseconds()
        
      //  val X = DenseMatrix.zeros[Double](3,5)
       // println( X )
       // println("")
       // X(1,1) = 5
        //X(2,2) = 7
        //println( X )

//    var K = 100
//var N = 100000
//var A = DenseMatrix.rand(N,K)
//var B = DenseMatrix.rand(K,N)
//println("multiplying now")
//timer.printTimeCheckMilliseconds(); B * A; timer.printTimeCheckMilliseconds()

   }
}
