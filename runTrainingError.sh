#!/bin/sh

echo "Compiling..."
mvn compile > /dev/null
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out > /dev/null
mkdir output > /dev/null

echo "starting training error experiment"
java -Xmx4g -cp ./target/classes:`cat classpath.out` edu.umd.cs.bachfna13.WikiTrainingError > /dev/null



