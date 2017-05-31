name := "test1"

version := "1.0"

fork := true

javaOptions in run += "-client -Xmx512M"

libraryDependencies  ++= Seq(
            // other dependencies here
            // pick and choose:
            "org.scalanlp" %% "breeze-math" % "0.2-SNAPSHOT"
            //"org.scalanlp" %% "breeze-learn" % "0.2-SNAPSHOT",
            //"org.scalanlp" %% "breeze-process" % "0.2-SNAPSHOT",
            //"org.scalanlp" %% "breeze-viz" % "0.2-SNAPSHOT"
)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.2-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)


