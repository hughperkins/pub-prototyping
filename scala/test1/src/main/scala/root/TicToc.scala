// Copyright Hugh Perkins 2012
// You can use this under the terms of the Apache Public License 2.0
// http://www.apache.org/licenses/LICENSE-2.0

package root

object TicToc {
  var startTime = 0l
	def tic() {
	  startTime = System.nanoTime()
	}
	def toc() {
	  var milliseconds = ( ( System.nanoTime() - startTime ) / 1000 / 1000 ).toInt;
	  println("Elapsed: " + milliseconds + " ms")
	  startTime = System.nanoTime()
	}
	def toc(message : String ) {
	  var milliseconds = ( ( System.nanoTime() - startTime ) / 1000 / 1000 ).toInt;
	  println(message + ": " + milliseconds + " ms")
	  startTime = System.nanoTime()
	}
}