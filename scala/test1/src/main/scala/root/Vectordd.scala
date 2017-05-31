package root

class Vectordd(valuesin: Array[Double]) {
   def values = valuesin
   def D = valuesin.length
   def this() = this( new Array[Double](0))
   override def toString() = {
      var result = "Vectordd("
        for( value <- values ) {
          result += " " + value
        }
      result += " )"
      result
   }
   def +(second: Vectordd) : Vectordd = {
     val result = new Array[Double](D)
     for( i <- 0 until D ) {
       result(i) = values(i) + second.values(i)
     }
     new Vectordd(result)
   }   
   def -(second: Vectordd) : Vectordd = {
     val result = new Array[Double](D)
     for( i <- 0 until D ) {
       result(i) = values(i) - second.values(i)
     }
     new Vectordd(result)
   }   
   def :*(second: Vectordd) : Vectordd = {
     val result = new Array[Double](D)
     for( i <- 0 until D ) {
       result(i) = values(i) * second.values(i)
     }
     new Vectordd(result)
   }   
   def *(second: Vectordd) : Double = {
     var result = 0.0
     for( i <- 0 until D ) {
       result += values(i) * second.values(i)
     }
     result
   }   
}
