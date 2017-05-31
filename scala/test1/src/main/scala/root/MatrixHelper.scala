// Copyright Hugh Perkins 2012
// You can use this under the terms of the Apache Public License 2.0
// http://www.apache.org/licenses/LICENSE-2.0

package root

import breeze.linalg._
import collection.mutable.ArrayBuffer
import collection.mutable.ListBuffer
import helpers.Timer

object MatrixHelper {
  def full(a: CSCMatrix[Double]) : DenseMatrix[Double] = {
    val result = zeros(a.rows,a.cols)
    for( ((row,col),value) <- a.activeIterator ) {
    	result(row,col) = value
    }
    result
  }
  
  def full(a: SparseVector[Double]) : DenseVector[Double] = {
    val result = zeros(a.size)
    for( (row,value) <- a.activeIterator ) {
    	result(row) = value
    }
    result
  }

  def shape(a : DenseVector[Double] ): DenseMatrix[Int] = { 
     DenseMatrix.fill(1,1)(a.size)
  }

  def shape(a : Matrix[Double] ): DenseMatrix[Int] = { 
     DenseMatrix((a.rows,a.cols))
  }
  
  def powm(a : DenseMatrix[Double], n: Double) : DenseMatrix[Double] = { 
    a.map( x => Math.pow(x, n) )    
  }
  
  def sqrtm(a : DenseMatrix[Double]) : DenseMatrix[Double] = { 
    a.map(x => Math.sqrt(x))    
  }
  
  def absm(a : DenseMatrix[Double] ) : DenseMatrix[Double] = {
    a.map(x => Math.abs(x))
  } 
  
   def ones(r: Int) : DenseVector[Double] = {
     DenseVector.ones[Double](r)
   }
  
   def ones(r: Int, c: Int) : DenseMatrix[Double] = {
     DenseMatrix.fill[Double](r,c)(1)
   }
  
   def zeros(r: Int) : DenseVector[Double] = {
     DenseVector.zeros[Double](r)
   }
  
   def zeros(r: Int, c: Int) : DenseMatrix[Double] = {
     DenseMatrix.zeros[Double](r,c)
   }
  
   def spzeros(r: Int) : SparseVector[Double] = {
     SparseVector.zeros[Double](r)
   }
  
   def spzeros(r: Int, c: Int) : CSCMatrix[Double] = {
     CSCMatrix.zeros[Double](r,c)
   }
  
   // transposes column sparse into new column sparse
   // (we want this if for example we want to multiple two sparse matrices together)
   def transpose( A: CSCMatrix[Double] ) : CSCMatrix[Double] = {
      // I guess we pump the data into an array of row,col,value structs
      // then resort that array, in row order
      // then pump out to the new sparse matrix
      // ?
      val inrows = A.rows
      val incols = A.cols
      val outrows = A.cols
      val outcols = A.rows
      val capacity = A.colPtrs(incols)
      //println("inrows " + inrows + " incols " + incols + " capacity " + capacity )
      var incol = 0
      val tuplearray = new Array[Tuple3[Int,Int,Double]](capacity)
      var i = 0
      while( incol < incols ) {
          val rowStartIndex = A.colPtrs(incol)
          val rowEndIndex = A.colPtrs(incol + 1) - 1
          val rowCapacity = rowEndIndex - rowStartIndex + 1
          var ri = 0
          while( ri < rowCapacity ) {
             val inrow = A.rowIndices(rowStartIndex + ri)
             val value = A.data(rowStartIndex + ri)
             tuplearray(i) = ( inrow, incol, value )
             ri += 1
             i += 1
          }
          incol += 1
      }
      // sort...
      util.Sorting.stableSort( tuplearray, (a: Tuple3[Int,Int,Double], b: Tuple3[Int,Int,Double]) => a._1 < b._1 )
      // now pump into the result...
      val result = CSCMatrix.zeros[Double](outrows,outcols)
      result.reserve( capacity )
      i = 0
      var lastInRow = -1
      while( i < capacity ) {
         val (inrow,incol,value) = tuplearray(i)
         if( inrow != lastInRow ) {
            result.colPtrs(inrow) = i
            lastInRow = inrow
         }
         result.rowIndices(i) = incol
         result.data(i) = value
         i += 1
      }
      result.colPtrs(lastInRow+1) = i
      result
   }

   // return a column of a CSCMatrix, as a SparseVector
   // note that this gives a *copy*, not a view.  (It should probably
   // give a view at some point in the near future)
   def colSlice( A: CSCMatrix[Double], colIndex: Int ) : SparseVector[Double] = {
      val size = A.rows
      val rowStartIndex = A.colPtrs(colIndex)
      val rowEndIndex = A.colPtrs(colIndex + 1) - 1
      val capacity = rowEndIndex - rowStartIndex + 1
      val result = SparseVector.zeros[Double](size)
      result.reserve(capacity)
      var i = 0
      while( i < capacity ) {
         val thisindex = rowStartIndex + i
         val row = A.rowIndices(thisindex)
         val value = A.data(thisindex)
         result(row) = value
         i += 1
      }
      result
   }
/*
   // it's only efficient to put the sparse matrix on the right hand side, since 
   // it is a column-sparse matrix
   def mul( A: DenseMatrix[Double], B: CSCMatrix[Double] ) : DenseMatrix[Double] = {
      if( A.cols != B.rows ) {
         // throw exception or something
      }
      val resultRows = A.rows
      val resultCols = B.cols
      var row = 0
      val result = DenseMatrix.zeros[Double](resultRows, resultCols )
      while( row < resultRows ) {
         var col = 0
         while( col < resultCols ) {
            val rightRowStartIndex = B.colPtrs(col)
            val rightRowEndIndex = B.colPtrs(col + 1) - 1
            val numRightRows = rightRowEndIndex - rightRowStartIndex + 1
            var ri = 0
            var sum = 0.
            while( ri < numRightRows ) {
               val inner = B.rowIndices(rightRowStartIndex + ri)
               val rightValue = B.data(rightRowStartIndex + ri)
               sum += A(row,inner) * rightValue
               ri += 1
            }
            result(row,col) = sum
            col += 1
         }
         row += 1
      }
      result
   }
   */   
/*

   // multiply two sparse matrices, and get a sparse matrix out
   def mul_old( A: CSCMatrix[Double], B: CSCMatrix[Double] ) : CSCMatrix[Double] = {
      // we will transpose the first matrix, keeping it in sparse form
      // so each column of the transposed matrix is now the rows of the 
      // original matrix
      val resultRows = A.rows
      val resultCols = B.cols
      var row = 0
      val AT = transpose(A)
      // we first store the results into an array of tuples, that we can extend as necessary
      // then we write these results into the final sparse matrix
      val resultarray = new ListBuffer[ Tuple3[ Int, Int, Double ] ]
      var i = 0
      var resultCol = 0 
      while( resultCol < resultCols ) {
         var resultRow = 0
         while( resultRow < resultRows ) {
            val leftRow = colSlice(AT, resultRow )
            val rightCol = colSlice(B, resultCol )
            val value = leftRow dot rightCol
            if( value != 0 ) {
               resultarray += Tuple3( resultRow, resultCol, value )
               i += 1
            }
            resultRow += 1
         }
         resultCol += 1
      }
      // now write the list to the result
      val result = CSCMatrix.zeros[Double](resultRows,resultCols)
      result.reserve( resultarray.size )
      i = 0
      var lastCol = -1
      for( (row,col,value) <- resultarray ) {
         if( col != lastCol ) {
            result.colPtrs(col) = i
            lastCol = col
         }
         result.rowIndices(i) = row
         result.data(i) = value
         i += 1
      }
      result.colPtrs(lastCol+1) = i
      result
   }   
*/
   def getActiveColumns( A: CSCMatrix[Double] ) : ListBuffer[Int] = {
      val activeColumns = new ListBuffer[ Int ]
      var i = 0
      var cols = A.cols
      while( i < cols ) {
         if( A.colPtrs(i) != A.colPtrs(i+1) ) {
            activeColumns += i
         }
         i += 1
      }
      activeColumns
   }

   // this works for fillFactor << 1 for now, or exactly 1
   // for fillFactors more than about 0.1, the effective fill factor will be 
   // noticeably lower than the requested fill factor
   def sprand( rows: Int, cols: Int, fillFactor: Double = 0.01 ) : CSCMatrix[Double] = {
      // we first store the results into an array of tuples, that we can extend as necessary
      // then we write these results into the final sparse matrix
      // we could loop over entire matrix, but really slow
      // we could fill in random coordinates, but really slow, and out of order
      // maybe we can use exponential distribution?
      // note that if fillFactor is near one, we should probably fill all first, then eliminate some
      // after?
      val expdist = breeze.stats.distributions.Exponential(fillFactor)
      val resultarray = new ListBuffer[ Tuple3[ Int, Int, Double ] ]
      var cum = 0.
      val totalSize = rows * cols
      var lastCumInt = -1
      cum += expdist.draw()
      while( cum < totalSize ) {
         val cumInt = cum.toInt
         if( cumInt != lastCumInt ) { // this means our formula is wrong, since we skip many in this way
             val col = ( cumInt / rows ).toInt
             val row = cumInt - col * rows
             resultarray += Tuple3( row, col, math.random )
             lastCumInt = cumInt
         }
         if( fillFactor != 1 ) {
        	 cum += expdist.draw()
         } else {
           cum += 1
         }
      }
      val result = lilToCsc(rows,cols,resultarray )
      result
   }

   // convert from list of (row,col,value) tuples to CSCMatrix format
   def lilToCsc( rows: Int, cols: Int, lil: ListBuffer[Tuple3[Int,Int,Double]] ) : CSCMatrix[Double] = {
      val result = CSCMatrix.zeros[Double](rows,cols)
      result.reserve( lil.size )
      var i = 0
      var lastCol = -1
      for( (row,col,value) <- lil ) {
         var thiscol = lastCol + 1
         while( thiscol <= col ) {
            result.colPtrs(thiscol) = i
            thiscol += 1
         }
         lastCol = col
         result.rowIndices(i) = row
         result.data(i) = value
         i += 1
      }
     var thiscol = lastCol + 1
     while( thiscol < cols ) {
        result.colPtrs(thiscol) = i
        thiscol += 1
     }
      lastCol = cols - 1
      println("lastCol " + lastCol + " i " + i)
      result.colPtrs(lastCol+1) = i
      result
   }
/*
   // multiply two sparse matrices, and get a sparse matrix out
   def mul( A: CSCMatrix[Double], B: CSCMatrix[Double] ) : CSCMatrix[Double] = {
      if( A.cols != B.rows ) {
         // throw exception or something
      }
      // we will transpose the first matrix, keeping it in sparse form
      // so each column of the transposed matrix is now the rows of the 
      // original matrix
      val resultRows = A.rows
      val resultCols = B.cols
      var row = 0
      val AT = transpose(A)
      // we first store the results into an array of tuples, that we can extend as necessary
      // then we write these results into the final sparse matrix
      val resultarray = new ListBuffer[ Tuple3[ Int, Int, Double ] ]
      var resultCol = 0 
      // get list of non empty columns of left and right
      val left = AT
      val right = B
      val leftColumns = resultRows
      val leftActiveColumnList = getActiveColumns(left)
      val rightActiveColumnList = getActiveColumns(right)
      // go through each active left col and active right col
      // then multiply together
      for( rightCol <- rightActiveColumnList ) {
         for( leftCol <- leftActiveColumnList ) {
            val leftSlice = colSlice(left, leftCol )
            val rightSlice = colSlice(right, rightCol )
            val value = leftSlice dot rightSlice
            if( value != 0 ) {
               resultarray += Tuple3( leftCol, rightCol, value )
            }            
         }
      }
      lilToCsc(resultRows,resultCols,resultarray )
   }   
*/
   def spdiag( a: Tensor[Int,Double] ) : CSCMatrix[Double] = {
      val size = a.size
      val result = CSCMatrix.zeros[Double](size,size)
      result.reserve(a.size)
      var i = 0
      while( i < size ) {
         result.rowIndices(i) = i
         result.colPtrs(i) = i
         result.data(i) = a(i)
         //result(i,i) = a(i)
         i += 1
      }
      //result.activeSize = size
      result.colPtrs(i) = i
      result
   }
   def mul(a: DenseMatrix[Double], s: Double ) : DenseMatrix[Double] = {
      a.map(x => s * x)
   }
   def speye( n: Int ) : CSCMatrix[Double] = {
      val result = CSCMatrix.zeros[Double](n,n)
      result.reserve(n)
      var i = 0
      while( i < n ) {
         result.rowIndices(i) = i
         result.colPtrs(i) = i
         result.data(i) = 1
         i += 1
      }
      result.colPtrs(i) = i
     
      result
   }
   def eye( n: Int ) : DenseMatrix[Double] = {
      val result = zeros(n,n)
      var i = 0
      while( i < n ) {
        result(i,i) = 1
        i += 1
      }     
      result
   }
   def spdiag( a: Matrix[Double] ) : CSCMatrix[Double] = {
     if( a.cols != 1 ) {
       throw new RuntimeException("spdiag expects matrix with single column")
     }
      val size = a.size
      val result = CSCMatrix.zeros[Double](size,size)
      result.reserve(a.size)
      var i = 0
      while( i < size ) {
         result.rowIndices(i) = i
         result.colPtrs(i) = i
         result.data(i) = a(i,0)
         //result(i,i) = a(i)
         i += 1
      }
      //result.activeSize = size
      result.colPtrs(i) = i
      result
   }
}

