package edu.umd.cs.bachfna13;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import edu.umd.cs.bachfna13.learning.SVMStructRank;
import edu.umd.cs.psl.application.inference.MPEInference;
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
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator;
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.util.database.Queries;

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
m.add predicate: "Random1", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random2", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random3", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random4", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random5", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random6", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random7", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random8", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random9", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Random0", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]


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

Random rand = new Random(0);
for (source in [a, b, c, d, w, x, y, z]) {
	for (sink in [a, b, c, d, w, x, y, z]) {
		for (Predicate pred in [Random1, Random2, Random3, Random4, Random5, Random6, Random7, Random8, Random9, Random0]) {
			db.getAtom(pred, source, sink).setValue(rand.nextDouble()).commitToDB();
		}
	}
}
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
for (source in ([b] as Set))
	for (sink in ([c] as Set))
		db.getAtom(link, source, sink).setValue(1.0).commitToDB();
for (source in ([w] as Set))
	for (sink in ([y] as Set))
		db.getAtom(link, source, sink).setValue(1.0).commitToDB();
db.close();

/* Declares rules */
def sq = true
def initialWeight = 0.0;
//for (sourceCat in ([catA, catB] as Set))
//	for (sinkCat in ([catA, catB] as Set))
//		m.add rule: (category(X, sourceCat) & category(Y, sinkCat)) >> link(X, Y), squared: sq, weight: initialWeight
m.add rule: (category(X, catA) & category(Y, catA)) >> link(X, Y), squared: sq, weight: 0.0
m.add rule: (category(X, catA) & category(Y, catB)) >> link(X, Y), squared: sq, weight: initialWeight
m.add rule: (category(X, catB) & category(Y, catA)) >> link(X, Y), squared: sq, weight: initialWeight
m.add rule: (category(X, catB) & category(Y, catB)) >> link(X, Y), squared: sq, weight: 0.0
m.add rule: Random1(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random2(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random3(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random4(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random5(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random6(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random7(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random8(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random9(A,B) >> link(A,B), squared: sq, weight: initialWeight
m.add rule: Random0(A,B) >> link(A,B), squared: sq, weight: initialWeight

m.add rule: ~link(X,Y), squared: sq, weight: initialWeight / 1.0;

/* Learns model */
rvDB = data.getDatabase(targetLinksPart, [category, Random1, Random2, Random3, Random4, Random5, Random6, Random7, Random8, Random9, Random0] as Set, backgroundPart);
obsvDB = data.getDatabase(trueLinksPart, [link] as Set);
cb.setProperty("votedperceptron.l2regularization", 1.0);
WeightLearningApplication wl = new SVMStructRank(m, rvDB, obsvDB, cb);
//WeightLearningApplication wl = new MaxLikelihoodMPE(m, rvDB, obsvDB, cb);
wl.learn();
wl.close();

System.out.println("Completed learning.");
System.out.println(m);

for (RandomVariableAtom atom : Queries.getAllAtoms(rvDB, link)) {
	atom.setValue(0.0);
	atom.commitToDB();
}

MPEInference mpe = new MPEInference(m, rvDB, cb)
mpe.mpeInference()

def comparator = new SimpleRankingComparator(rvDB)
comparator.setBaseline(obsvDB)

def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
double [] score = new double[metrics.size() + 1]

for (int i = 0; i < metrics.size(); i++) {
	comparator.setRankingScore(metrics.get(i))
	score[i] = comparator.compare(Link)
}

comparator = new DiscretePredictionComparator(rvDB)
comparator.setBaseline(obsvDB)

DiscretePredictionStatistics stats = comparator.compare(Link)
score[3] = stats.accuracy;

log.warn("Area under positive-class PR curve: " + score[0])
log.warn("Area under negative-class PR curve: " + score[1])
log.warn("Area under ROC curve: " + score[2])
log.warn("Rounded accuracy: " + score[3]);

obsvDB.close();
