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
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron;
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
import edu.umd.cs.psl.parser.PSLModelLoader;
import edu.umd.cs.psl.parser.PSLParser;
import edu.umd.cs.psl.reasoner.admm.ADMMReasoner;
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


/*** CONFIGURATION PARAMETERS ***/

dataPath = "./data/wiki/"
numCategories = 30
labelFile = "labels.txt"
linkFile = "links.txt"
threshold = "0.4"
candidateFile = "candidates." + threshold + ".txt"
simFile = "similar."+ threshold + ".txt"
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

log.warn("Starting run at {}", new Date())

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
configGenerator.setLearningMethods(["OMM", "MLE"]);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 5.0]);
configGenerator.setRegularizationParameters([(double) 1.0, (double) 0.1, (double) 0.01]);

/* MM options */
configGenerator.setMaxMarginSlackPenalties([(double) 1.0]);
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
m.add predicate: "Similar", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Link", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "Candidate", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

double initialWeight = 0.0

// prior
m.add rule : ~(Link(A,B)), weight: 0.1, squared: sq
m.add rule : ( Similar(A,B)) >> Link(A, B), weight: initialWeight, squared: sq
m.add rule : ( HasCat(A,C) & HasCat(B,C) ) >> Link(A,B), weight: initialWeight, squared: sq

for (int i = 0; i < numCategories; i++)  {
	UniqueID cat1 = data.getUniqueID(i)
	for (int j = 0; j < numCategories; j++) {
		UniqueID cat2 = data.getUniqueID(j)
		// per-cat rules
		m.add rule : ( HasCat(A, cat1) &  HasCat(B, cat2)) >> Link(A,B), weight: initialWeight, squared: sq
		m.add rule : ( HasCat(A, cat1) &  HasCat(B, cat2)) >> ~Link(A,B), weight: initialWeight, squared: sq
	}

	// triangle rules
	// blocked to reduce cubic blowup
	m.add rule: (Link(A,B) & Link(B,C) & HasCat(B, cat1) & Candidate(A,C)) >> Link(A,C), weight: initialWeight, squared: sq
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

inserter = data.getInserter(Similar, fullObserved)
InserterUtils.loadDelimitedDataTruth(inserter, dataPath + simFile)
log.debug("Loaded similarities");


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
queries.add(new DatabaseQuery(Similar(document, linkedDocument).getFormula()))
queries.add(new DatabaseQuery(HasCat(document, A).getFormula()))

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

	Variable Doc1 = new Variable("Document1");
	Variable Doc2 = new Variable("Document2");
	def substitutions = new HashMap<Variable, Set<GroundTerm>>();

	substitutions.put(Doc1, partitionDocuments.get(trainReadPartitions.get(fold)))
	substitutions.put(Doc2, partitionDocuments.get(trainReadPartitions.get(fold)))
	dbPop = new DatabasePopulator(trainDB);
	dbPop.populate(new QueryAtom(Link, Doc1, Doc2), substitutions);

	trainDB.close();

	toClose = [Link] as Set
	Database labelsDB = data.getDatabase(trainLabelPartitions.get(fold), toClose)

	DataOutputter.outputPredicate("output/wiki/groundTruth" + fold + ".link" , labelsDB, Link, ",", false, "node,neighbor")

	/*** EXPERIMENT ***/

	log.debug("Setup done. Starting learning experiments")

	def observedToClose = [HasCat, Similar, Candidate] as Set

	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		config.setProperty(ADMMReasoner.MAX_ITER_KEY, 100);
		trainDB = data.getDatabase(trainWritePartitions.get(fold), observedToClose, trainReadPartitions.get(fold))
		learn(m, trainDB, labelsDB, config, log)
		trainDB.close()
		config.setProperty(ADMMReasoner.MAX_ITER_KEY, ADMMReasoner.MAX_ITER_DEFAULT);

		log.debug("Learned model " + config.getString("name", "") + "\n" + m.toString())
		PSLModelLoader.outputModel("output/wiki/models/" + config.getString("name", "") + "." + fold + ".psl", m)

		/* Inference on training set */
		trainDB = data.getDatabase(trainWritePartitions.get(fold), observedToClose, trainReadPartitions.get(fold))
		Set<GroundAtom> allAtoms = Queries.getAllAtoms(trainDB, Link)
		for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
			atom.setValue(0.0)
		MPEInference mpe = new MPEInference(m, trainDB, config)
		FullInferenceResult result = mpe.mpeInference()
		log.debug("Objective: " + result.getTotalWeightedIncompatibility())
		trainDB.close();


		/*
		 * Evaluation
		 */
		Database resultsDB = data.getDatabase(trainWritePartitions.get(fold))
		def comparator = new SimpleRankingComparator(resultsDB)
		def groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [Link] as Set)
		comparator.setBaseline(groundTruthDB)

		DataOutputter.outputPredicate("output/wiki/predictions/" + config.getString("name", "") + "." + fold + ".txt", resultsDB, Link, ",", true, "from,to")
		
		def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
		double [] score = new double[metrics.size() + 1]

		for (int i = 0; i < metrics.size(); i++) {
			comparator.setRankingScore(metrics.get(i))
			score[i] = comparator.compare(Link)
		}

		comparator = new DiscretePredictionComparator(resultsDB)
		comparator.setBaseline(groundTruthDB)

		DiscretePredictionStatistics stats = comparator.compare(Link)
		score[3] = stats.accuracy;
	
		log.warn("Area under positive-class PR curve: " + score[0])
		log.warn("Area under negative-class PR curve: " + score[1])
		log.warn("Area under ROC curve: " + score[2])
		log.warn("Rounded accuracy: " + score[3]);

		
		def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
		log.warn("Method " + config.getString("name", "") + ", fold " + fold +", acc " + stats.getAccuracy() +
				", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
				", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
				", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
		
		results.get(configIndex).add(score);
		resultsDB.close()
		groundTruthDB.close()
	}
}


log.warn("Finished at {}", new Date());



for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
	def methodStats = results.get(configIndex)
	configName = configs.get(configIndex).getString("name", "");
	sum = new double[4];
	sumSq = new double[4];
	for (int fold = 0; fold < folds; fold++) {
		def score = methodStats.get(fold)
		for (int i = 0; i < 4; i++) {
			sum[i] += score[i];
			sumSq[i] += score[i] * score[i];
		}
		log.warn("Method " + configName + ", fold " + fold +", auprc positive: "
				+ score[0] + ", negative: " + score[1] + ", auROC: " + score[2]
				+ ", rounded accuracy: " + score[3])
	}

	mean = new double[4];
	variance = new double[4];
	for (int i = 0; i < 4; i++) {
		mean[i] = sum[i] / folds;
		variance[i] = sumSq[i] / folds - mean[i] * mean[i];
	}


	log.warn("Method " + configName + ", auprc positive: (mean/variance) "
			+ mean[0] + "  /  " + variance[0] );
	log.warn("Method " + configName + ", auprc negative: (mean/variance) "
			+ mean[1] + "  /  " + variance[1] );
	log.warn("Method " + configName + ", auROC: (mean/variance) "
			+ mean[2] + "  /  " + variance[2] );
	log.warn("Method " + configName + ", rounded accuracy: "
			+ mean[3] + "  /  " + variance[3] );
}


public void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("learningmethod", "")) {
		case "OMM":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
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