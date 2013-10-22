package edu.umd.cs.bachfna13;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.umd.cs.bachfna13.learning.SVMStructRank;
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication
import edu.umd.cs.psl.application.learning.weight.em.BernoulliMeanFieldEM
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.UniqueID

/*
 * Initializes DataStore, ConfigBundle, and PSLModel
 */
Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle("toyrank")
cb.setProperty("rdbmsdatastore.usestringids", true)

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = cb.getString("dbpath", defaultPath + File.separator + "toyrank")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)

PSLModel m = new PSLModel(this, data)

/*
 * Defines Predicates
 */
m.add predicate: "Link", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Category", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

/* Partition numbers */
Partition backgroundPart = new Partition(0);
Partition targetLinksPart = new Partition(1);
Partition trueLinksPart = new Partition(2);

/* Constants */
UniqueID a = data.getUniqueID("a");
UniqueID b = data.getUniqueID("b");
UniqueID c = data.getUniqueID("c");
UniqueID d = data.getUniqueID("d");
UniqueID w = data.getUniqueID("w");
UniqueID x = data.getUniqueID("x");
UniqueID y = data.getUniqueID("y");
UniqueID z = data.getUniqueID("z");

UniqueID catA = data.getUniqueID("catA");
UniqueID catB = data.getUniqueID("catB");

Database db = data.getDatabase(backgroundPart);
for (node in ([a, b, c, d] as Set))
	db.getAtom(category, node, catA).setValue(1.0).commitToDB();
for (node in ([w, x, y, z] as Set))
	db.getAtom(category, node, catB).setValue(1.0).commitToDB();
db.close();

db = data.getDatabase(targetLinksPart);
for (source in ([a, b, c, d, w, x, y, z] as Set))
	for (sink in ([a, b, c, d, w, x, y, z] as Set))
		db.getAtom(link, source, sink).setValue(0.0).commitToDB();
db.close();

//db = data.getDatabase(trueLinksPart);
//for (source in ([a, b, c, d] as Set))
//	for (sink in ([a, b, c, d] as Set))
//		db.getAtom(link, source, sink).setValue(1.0).commitToDB();
//for (source in ([w, x, y, z] as Set))
//	for (sink in ([w, x, y, z] as Set))
//		db.getAtom(link, source, sink).setValue(1.0).commitToDB();
//db.close();

db = data.getDatabase(trueLinksPart);
for (source in ([a] as Set))
	for (sink in ([c] as Set))
		db.getAtom(link, source, sink).setValue(1.0).commitToDB();
for (source in ([w] as Set))
	for (sink in ([y] as Set))
		db.getAtom(link, source, sink).setValue(1.0).commitToDB();
db.close();

/* Declares rules */
def sq = true
def initialWeight = 2.0;
for (sourceCat in ([catA, catB] as Set))
	for (sinkCat in ([catA, catB] as Set))
		m.add rule: (category(X, sourceCat) & category(Y, sinkCat)) >> link(X, Y), squared: sq, weight: initialWeight

m.add rule: ~link(X,Y), squared: sq, weight: initialWeight / 1.0;
		
/* Learns model */
rvDB = data.getDatabase(targetLinksPart, [category] as Set, backgroundPart);
obsvDB = data.getDatabase(trueLinksPart, [link] as Set);
WeightLearningApplication wl = new SVMStructRank(m, rvDB, obsvDB, cb);
//WeightLearningApplication wl = new MaxLikelihoodMPE(m, rvDB, obsvDB, cb);
wl.learn();
wl.close();
rvDB.close();
obsvDB.close();

System.out.println("Completed learning.");
System.out.println(m);
