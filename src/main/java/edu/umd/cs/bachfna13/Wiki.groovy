package edu.umd.cs.bachfna13

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import edu.umd.cs.bachfna13.util.DataOutputter;
import edu.umd.cs.bachfna13.util.ExperimentConfigGenerator;
import edu.umd.cs.bachfna13.util.FoldUtils;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.loading.Inserter;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


/*** CONFIGURATION PARAMETERS ***/

dataPath = "./data/wiki/"
numCategories = 30
labelFile = "labels.txt"
linkFile = "links.txt"
candidateFile = "candidates.txt"
wordFile = "document.txt"
dictFile = "words.txt"
sq = true
if (args.length > 0)
	sq = Boolean.parseBoolean(args[0]);
usePerCatRules = true
folds = 5 // number of folds
if (args.length > 1)
	seedRatio = Double.parseDouble(args[1]);
Random rand = new Random(0) // used to seed observed data
targetSize = 200
explore = 0.05

Logger log = LoggerFactory.getLogger(this.class)

ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle("wiki")

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = cb.getString("dbpath", defaultPath + File.separator + "pslWiki")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), cb)



/**
 * SET UP CONFIGS
 */

ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("wiki");

/*
 * SET MODEL TYPES
 *
 * Options:
 * "quad" HLEF
 * "bool" MLN
 */
configGenerator.setModelTypes([(sq) ? "quad" : "linear"]);

/*
 * SET LEARNING ALGORITHMS
 *
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 */
configGenerator.setLearningMethods(["MLE"]);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 5.0]);

/* MM options */
configGenerator.setMaxMarginSlackPenalties([(double) 0.1]);
configGenerator.setMaxMarginLossBalancingTypes([LossBalancingType.NONE]);
configGenerator.setMaxMarginNormScalingTypes([NormScalingType.NONE]);
configGenerator.setMaxMarginSquaredSlackValues([false]);

List<ConfigBundle> configs = configGenerator.getConfigs();





/*
 * DEFINE MODEL
 */

PSLModel m = new PSLModel(this, data)

// rules
m.add predicate: "HasCat", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "HasWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Link", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Candidate", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Cat", types: [ArgumentType.UniqueID]

// prior
m.add rule : ~(Link(A,B)), weight: 0.001, squared: sq

for (int i = 0; i < numCategories; i++)  {
	UniqueID cat1 = data.getUniqueID(i)
	for (int j = 0; j < numCategories; j++) {
		UniqueID cat2 = data.getUniqueID(j)
		// per-cat rules
		m.add rule : ( HasCat(A, cat1) &  HasCat(B, cat2)) >> Link(A,B), weight: 1.0, squared: sq
	}
	
	// triangle rules
	// blocked to reduce cubic blowup
	m.add rule: (Link(A,B) & Link(B,C) & HasCat(B, cat1) & Candidate(A,C)) >> Link(A,C), weight: 1.0, squared: sq
}



/* get all default weights */
Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());


/*** LOAD DATA ***/
Partition fullObserved =  new Partition(0)
Partition groundTruth = new Partition(1)

log.debug("Starting initial loading");

def inserter
inserter = data.getInserter(Link, groundTruth)
InserterUtils.loadDelimitedData(inserter, dataPath + linkFile)
log.debug("Loaded links");

inserter = data.getInserter(HasCat, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + labelFile)
log.debug("Loaded categories");

inserter = data.getInserter(Candidate, fullObserved)
InserterUtils.loadDelimitedData(inserter, dataPath + candidateFile)
log.debug("Loaded candidate links");

//def dictionary = loadDictionary(dataPath + dictFile)
//log.debug("Loaded dictionary {}", dictionary);
//
//inserter = data.getInserter(HasWord, fullObserved)
//insertWords(inserter, dataPath + wordFile, dictionary)
//log.debug("Loaded words");

log.debug("Initial loading complete");

trainReadPartitions = new ArrayList<Partition>()
testReadPartitions = new ArrayList<Partition>()
trainWritePartitions = new ArrayList<Partition>()
testWritePartitions = new ArrayList<Partition>()
trainLabelPartitions = new ArrayList<Partition>()
testLabelPartitions = new ArrayList<Partition>()

def keys = new HashSet<Variable>()
ArrayList<Set<Integer>> trainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingKeys = new ArrayList<Set<Integer>>()
def queries = new HashSet<DatabaseQuery>()


/*
 * DEFINE PRIMARY KEY QUERIES FOR FOLD SPLITTING
 */
Variable document = new Variable("Document")
Variable linkedDocument = new Variable("LinkedDoc")
keys.add(document)
keys.add(linkedDocument)
queries.add(new DatabaseQuery(Link(document, linkedDocument).getFormula()))
queries.add(new DatabaseQuery(Candidate(document, linkedDocument).getFormula()))
queries.add(new DatabaseQuery(HasCat(document, A).getFormula()))
queries.add(new DatabaseQuery(HasWord(document, W).getFormula()))

def partitionDocuments = new HashMap<Partition, Set<GroundTerm>>()

for (int i = 0; i < folds; i++) {
	trainReadPartitions.add(i, new Partition(i + 2))
	testReadPartitions.add(i, new Partition(i + folds + 2))

	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))

	trainLabelPartitions.add(i, new Partition(i + 4*folds + 2))
	testLabelPartitions.add(i, new Partition(i + 5*folds + 2))

	//	Set<GroundTerm> [] documents = FoldUtils.generateRandomSplit(data, trainTestRatio,
	//			fullObserved, groundTruth, trainReadPartitions.get(i),
	//			testReadPartitions.get(i), trainLabelPartitions.get(i),
	//			testLabelPartitions.get(i), queries, keys, filterRatio)
	Set<GroundTerm> [] documents = FoldUtils.generateSnowballSplit(data, fullObserved, groundTruth,
			trainReadPartitions.get(i), testReadPartitions.get(i), trainLabelPartitions.get(i),
			testLabelPartitions.get(i), queries, keys, targetSize, Link, explore)

	partitionDocuments.put(trainReadPartitions.get(i), documents[0])
	partitionDocuments.put(testReadPartitions.get(i), documents[1])
}

List<List<Double []>> results = new ArrayList<List<Double []>>()
for (int i = 0; i < configs.size(); i++)
	results.add(new ArrayList<Double []>())

for (int fold = 0; fold < folds; fold++) {

	/*** POPULATE DBs ***/

	Database db;
	DatabasePopulator dbPop;

	/* Populate Link */
	Database trainDB = data.getDatabase(trainWritePartitions.get(fold))
	Database testDB = data.getDatabase(testWritePartitions.get(fold))

	Variable Doc1 = new Variable("Document1");
	Variable Doc2 = new Variable("Document2");
	def substitutions = new HashMap<Variable, Set<GroundTerm>>();

	substitutions.put(Doc1, partitionDocuments.get(trainReadPartitions.get(fold)))
	substitutions.put(Doc2, partitionDocuments.get(trainReadPartitions.get(fold)))
	dbPop = new DatabasePopulator(trainDB);
	dbPop.populate(new QueryAtom(Link, Doc1, Doc2), substitutions);

	substitutions.put(Doc1, partitionDocuments.get(testReadPartitions.get(fold)))
	substitutions.put(Doc2, partitionDocuments.get(testReadPartitions.get(fold)))
	dbPop = new DatabasePopulator(testDB);
	dbPop.populate(new QueryAtom(Link, Doc1, Doc2), substitutions);
	
	DataOutputter.outputPredicate("output/wiki/training/observed"+fold+".txt" , trainDB, HasCat, ",", false, "nodeid,label")
	
	testDB.close();
	trainDB.close();

	toClose = [Link] as Set
	Database labelsDB = data.getDatabase(trainLabelPartitions.get(fold), toClose)

	groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [Link] as Set)
	DataOutputter.outputPredicate("output/wiki/groundTruth" + fold + ".node" , groundTruthDB, Link, ",", false, "node,neighbor")
	groundTruthDB.close()

	/*** EXPERIMENT ***/

	log.debug("Setup done. Starting learning experiments")

	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		trainDB = data.getDatabase(trainWritePartitions.get(fold), [HasCat, HasWord] as Set, trainReadPartitions.get(fold))
		learn(m, trainDB, labelsDB, config, log)
		trainDB.close()

		System.out.println("Learned model " + config.getString("name", "") + "\n" + m.toString())

		/* Inference on test set */
		testDB = data.getDatabase(testWritePartitions.get(fold), [HasCat, HasWord] as Set, testReadPartitions.get(fold))
		Set<GroundAtom> allAtoms = Queries.getAllAtoms(testDB, Link)
		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
			atom.setValue(0.0)
		MPEInference mpe = new MPEInference(m, testDB, config)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())
		testDB.close();
		/*
		 * Evaluation
		 */
		Database resultsDB = data.getDatabase(testWritePartitions.get(fold))
		def comparator = new SimpleRankingComparator(resultsDB)
		def groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [Link] as Set)
		comparator.setBaseline(groundTruthDB)


		def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
		double [] score = new double[metrics.size()]

		for (int i = 0; i < metrics.size(); i++) {
			comparator.setRankingScore(metrics.get(i))
			score[i] = comparator.compare(Link)
		}
		System.out.println("Area under positive-class PR curve: " + score[0])
		System.out.println("Area under negative-class PR curve: " + score[1])
		System.out.println("Area under ROC curve: " + score[2])

		results.get(configIndex).add(score);
		resultsDB.close()
		groundTruthDB.close()
	}
}


for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
	def methodStats = results.get(configIndex)
	configName = configs.get(configIndex).getString("name", "");
	sum = new double[3];
	sumSq = new double[3];
	for (int fold = 0; fold < folds; fold++) {
		def score = methodStats.get(fold)
		for (int i = 0; i < 3; i++) {
			sum[i] += score[i];
			sumSq[i] += score[i] * score[i];
		}
		System.out.println("Method " + configName + ", fold " + fold +", auprc positive: "
				+ score[0] + ", negative: " + score[1] + ", auROC: " + score[2])
	}

	mean = new double[3];
	variance = new double[3];
	for (int i = 0; i < 3; i++) {
		mean[i] = sum[i] / folds;
		variance[i] = sumSq[i] / folds - mean[i] * mean[i];
	}

	System.out.println();
	System.out.println("Method " + configName + ", auprc positive: (mean/variance) "
			+ mean[0] + "  /  " + variance[0] );
	System.out.println("Method " + configName + ", auprc negative: (mean/variance) "
			+ mean[1] + "  /  " + variance[1] );
	System.out.println("Method " + configName + ", auROC: (mean/variance) "
			+ mean[2] + "  /  " + variance[2] );
	System.out.println();
}


public void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("learningmethod", "")) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}


public void insertWords(Inserter inserter, String wordFile, Map<Integer, String> dictionary) {
	// load words
	Scanner wordScanner = new Scanner(new FileReader(wordFile));
	while (wordScanner.hasNext()) {
		String line = wordScanner.nextLine();
		String [] tokens = line.split("\t");
		Integer docID = Integer.decode(tokens[0]);
		Map<Integer, Double> docWords = parseWords(tokens[1]);
		for (Map.Entry<Integer, Double> e : docWords.entrySet()) {
			Integer wordID = e.getKey();
			Double count = (double) e.getValue();
			String word = dictionary.get(wordID);
			if (word != null)
				inserter.insert(docID, wordID);
		}
	}
	wordScanner.close();
}


public Map<Integer, Double> parseWords(String string) {
	String [] tokens = string.split(" ");
	Map<Integer, Double> words = new HashMap<Integer, Double>(1000);
	for (int i = 0; i < tokens.length; i++) {
		if (tokens[i].length() > 1) {
			String [] subTokens = tokens[i].split(":");
			words.put(Integer.decode(subTokens[0]), Double.parseDouble(subTokens[1]));
		}
	}
	return words;
}

public Map<Integer, String> loadDictionary(String dictFile) {
	Map<Integer,String> dictionary = new HashMap<Integer, String>();

	Scanner wordScanner = new Scanner(new FileReader(dictFile));
	while (wordScanner.hasNext()) {
		String line = wordScanner.nextLine();
		String [] tokens = line.split("\t");
		String word = tokens[0];
		Integer id = Integer.decode(tokens[1]);
		Integer count = Integer.decode(tokens[2]);

		if (count > 100 && count < 1000)
			dictionary.put(id, word);
	}
	wordScanner.close();

	return dictionary;
}