/**
 * Name: TONG Haowen Joel
 * Matric ID: A0108165J
 * <p/>
 * CS4248 Assignment 2
 * <p/>
 * Oct 23, 2015
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by joel on 10/20/15.
 * <p/>
 * This is the public entry point for the App.
 * <p/>
 * Please set flags accordingly for trainer or tester.
 */
public class sctrain {

  /* labels */
  public static final int APP_TRAIN = 0;
  public static final int APP_TEST = 1;

  // NOTE: Please set accordingly.
  public static final int APP_TYPE = APP_TRAIN;

  /**
   * Main entry point of program for both trainer and tester.
   *
   * @param args
   */
  public static void main(String[] args) throws IOException {
    if (args.length < 2) {
      throw new IllegalArgumentException("Wrong number of arguments!");
    }
    String word1 = args[0];
    String word2 = args[1];
    if (APP_TYPE == APP_TRAIN) {
      if (args.length < 4) {
        throw new IllegalArgumentException("Wrong number of arguments!");
      }
      String trainFile = args[2];
      String modelFile = args[3];
      SCTrainer scTrainer = new SCTrainer(word1, word2, trainFile, modelFile);
      scTrainer.train();
      scTrainer.write();

    } else if (APP_TYPE == APP_TEST) {
      if (args.length < 5) {
        throw new IllegalArgumentException("Wrong number of arguments!");
      }
      String testFile = args[2];
      String modelFile = args[3];
      String answerFile = args[4];
      System.out.println(args[2] + "," + args[3] + "," + args[4]);
      SCTester scTester = new SCTester(testFile, answerFile, modelFile);
      scTester.runTest();
    } else {
      throw new IllegalArgumentException("Oops.  No App behavior specified!");
    }
  }


  /**
   * Created by joel on 10/13/15.
   * <p/>
   * interface for classification.
   */
  public static interface Classifier {

    public void loadDataset(List<Record> records);

    public Model train();

    public double test();
  }

  /**
   * Created by joel on 10/14/15.
   * <p/>
   * The Machine packages the @link{Model} and @link{LogisticRegressionClassifier}
   * into a higher-level, easy to use format.
   */
  public static class CS4248Machine implements Machine {

    private List<String> mappings;
    private List<String> vocabulary;
    private List<String> collocationNGrams;
    private List<String> features;
    private LogisticRegressionClassifier classifier;
    private Model model;

    private double learningRate;
    private float learningMinThreshold;
    private int wordDiffMinThreshold;
    private int stopWordsStart;
    private int stopWordsEnd;
    private int nGramSize;
    private int folds;
    private double learningDecay;
    private double terminationThreshold;
    private long timeoutPerDimen;
    private int featureCountMin;

    /**
     * Sets the parameters of the machine.
     *
     * @param learningRate         Rate of learning.
     * @param learningDecay        Rate to decay the rate of learning.
     * @param terminationThreshold Deprecated.
     * @param timeoutPerDimen      Usually, do not time out unless needed.
     * @param learningMinThreshold Minimum threshold for w_n to be considered equal to w_n-1
     * @param wordDiffMinThreshold Stop word minimum threshold.  Less than threshold stop words are removed.
     * @param collocationStart     start index of collocation
     * @param collocationEnd       stop index of collocation
     * @param nGramSize            Size of Ngram used to chunk collocation
     * @param folds                Number of folds used for training.
     * @param featureCountMin      Minimum frequency count per feature; sparse features are removed.
     */
    public void setParam(double learningRate,
                         double learningDecay,
                         double terminationThreshold,
                         long timeoutPerDimen,
                         float learningMinThreshold,
                         int wordDiffMinThreshold,
                         int collocationStart,
                         int collocationEnd,
                         int nGramSize,
                         int folds,
                         int featureCountMin
    ) {
      this.learningRate = learningRate;
      this.learningDecay = learningDecay;
      this.terminationThreshold = terminationThreshold;
      this.timeoutPerDimen = timeoutPerDimen;
      this.learningMinThreshold = learningMinThreshold;
      this.wordDiffMinThreshold = wordDiffMinThreshold;
      this.stopWordsStart = collocationStart;
      this.stopWordsEnd = collocationEnd;
      this.nGramSize = nGramSize;
      this.folds = folds;
      this.featureCountMin = featureCountMin;
    }

    /**
     * Instantiates a new machine.
     */
    public CS4248Machine() {
      classifier = new LogisticRegressionClassifier();
    }

    /**
     * Trains the dataset with file.
     *
     * @param datasetFileName
     * @throws FileNotFoundException
     */
    @Override
    public void train(String datasetFileName) throws FileNotFoundException {
      List<RawRecord> trainset = RawRecord.parse(datasetFileName);
      List<Record> records = new ArrayList<>();

      // reduce features by including relevant stop words
      Set<String> stopWords = optimizeStopWords(
          trainset,
          Util.loadStopWords(),
          wordDiffMinThreshold
      );

      RawRecord.removeTokens(trainset, stopWords);
      //    System.out.println(stopWords.size());
      //    App.RawRecord.print(trainset);

      mapLabels(trainset); // generate label mapping

      // TODO comment out this dump statement
      vocabulary = RawRecord.makeVocabularyList(trainset);
      collocationNGrams = RawRecord.makeCollocationsNGramsList(nGramSize, trainset, stopWordsStart, stopWordsEnd);

      generateFeatures();
      this.features = winnowFeatures(features, trainset);
      System.out.println("PrunedFeatureSize=" + features.size());

      for (RawRecord r : trainset) {
        records.add(convToRecord(r, features));
      }
      Collections.shuffle(records); // shuffle the collection to ensure unbiased ordering

      model = trainNFolds(records, folds);
    }

    /**
     * Feature-reduction function, to winnow features down.
     *
     * @param features
     * @param records
     * @return
     */
    private List<String> winnowFeatures(List<String> features, List<RawRecord> records) {
      Map<String, Integer> freqTable = makeFeatureFrequencyTable(features, records);
      List<String> prunedFeatures = new ArrayList<>(features.size());
      int count = 0;
      for (String f : features) {
        Integer freq = freqTable.get(f);
        if (freq >= featureCountMin) {
          prunedFeatures.add(f);
        }
      }
      int reducedStatistic = features.size() - prunedFeatures.size();
      System.out.println("There are " + reducedStatistic + " redundant features.");
      return prunedFeatures;
    }

    /**
     * Make feature histogram table, for winnowing.
     *
     * @param features
     * @param records
     * @return
     */
    private Map<String, Integer> makeFeatureFrequencyTable(List<String> features, List<RawRecord> records) {
      Map<String, Integer> freqTable = new HashMap<>(features.size());

      for (String f : features) {
        freqTable.put(f, 0);
      }

      for (RawRecord r : records) {
        // make word token frequency
        for (String token : r.getTokens()) {
          if (freqTable.containsKey(token)) {
            freqTable.put(token, freqTable.get(token) + 1);
          }
        }
        // then make nGram frequency
        List<String> collocation = Util.getCollocation(r.getTokens(), stopWordsStart, stopWordsEnd, r.getIdx());
        List<String> nGrams = Util.getNGrams(nGramSize, collocation);
        for (String nGram : nGrams) {
          if (freqTable.containsKey(nGram)) {
            freqTable.put(nGram, freqTable.get(nGram) + 1);
          }
        }
      }

      return freqTable;
    }

    /**
     * Generates features used, including all tokens and Ngrams.
     * <p/>
     * Needs winnowing.
     */
    private void generateFeatures() {
      features = new ArrayList<>();
      features.addAll(vocabulary);
      features.addAll(collocationNGrams);
      StringBuffer sb = new StringBuffer();
      sb.append("<features>\n");
      for (String f : features) {
        sb.append(f + ",");
      }
      sb.append("\n</features>\n");
      sb.append("There are " + features.size() + " features.\n");
      System.out.println(sb.toString());
    }

    /**
     * Train using a list of records.
     *
     * @param records
     * @return
     */
    private Model train(List<Record> records) {
      classifier.loadDataset(records);
      Model model = classifier.train(learningMinThreshold, learningRate, learningDecay, terminationThreshold, timeoutPerDimen);
      return model;
    }

    /**
     * Train using records and and initial set of weights.
     *
     * @param records
     * @param initialWeights
     * @return
     */
    private Model train(List<Record> records, Vector initialWeights) {
      classifier.loadDataset(records);
      Model model = classifier.train(initialWeights, learningMinThreshold, learningRate, learningDecay, terminationThreshold, timeoutPerDimen);
      return model;
    }

    /**
     * Train and evaluate using N folds.
     *
     * @param records
     * @param nFolds
     * @return
     */
    private Model trainNFolds(List<Record> records, int nFolds) {
      Model bestModel = null;
      double bestAccuracy = 0d;
      int stepSize = records.size() / nFolds;
      for (int i = 0; i < nFolds; i++) {
        System.out.println("Training fold " + i + "..");
        int start = stepSize * i;
        int end = Math.min(start + stepSize - 1, records.size());
        System.out.println("Start=" + start + " End=" + end);
        // generate the train and test sets
        List<Record> currTestSet = records.subList(start, end);
        List<Record> currTrainSet = new ArrayList<>();
        currTrainSet.addAll(records.subList(0, start));
        currTrainSet.addAll(records.subList(end, records.size() - 1));
        Model currModel = train(currTrainSet);
        double currAccuracy = currModel.testAccuracy(currTestSet);
        System.out.println("currAccuracy=" + currAccuracy);
        if (currAccuracy > bestAccuracy) {
          bestAccuracy = currAccuracy;
          bestModel = currModel;
        }
      }
      System.out.println("Using model with accuracy " + bestAccuracy);
      // then further train best model on all samples
      System.out.println("Training with all weights..");
      bestModel = train(records, bestModel.getWeights());
      return bestModel;
    }

    /**
     * Remove undistinct stop words from train set.
     *
     * @param trainset
     * @param stopWordsAll
     * @param minThreshold
     * @return
     */
    private Set<String> optimizeStopWords(List<RawRecord> trainset, Set<String> stopWordsAll, int minThreshold) {
      HashMap<String, List<RawRecord>> partitions = RawRecord.segment(trainset);
      // convert all label partitions to array list for processing
      List<String> keys = new ArrayList<>(2);
      for (String label : partitions.keySet()) {
        keys.add(label);
      }
      Set<String> optimizedStopWords = Util.selectDistinctStopWords(
          stopWordsAll,
          partitions.get(keys.get(0)),
          partitions.get(keys.get(1)),
          minThreshold
      );
      return optimizedStopWords;
    }

    /**
     * Test the trained model on a given test set.
     *
     * @param questionFilename
     * @param answerFilename
     * @return
     * @throws FileNotFoundException
     */
    @Override
    public Map<String, PredictionResult> test(String questionFilename, String answerFilename) throws FileNotFoundException {
      List<RawRecord> testSet = RawRecord.parse(questionFilename, answerFilename);
      Map<String, PredictionResult> results = new HashMap<>();
      for (RawRecord record : testSet) {
        Record curr = convToRecord(record, features);
        String predicted = getIntLabel(model.evaluate(curr.getVectors()));
        String actual = record.getLabel();
        if (!results.containsKey(actual)) {
          results.put(actual, new PredictionResult());
        }
        if (predicted.equals(actual)) {
          PredictionResult pr = results.get(actual);
          pr.incCorrect();
          pr.incTotal();
        } else {
          PredictionResult pr = results.get(actual);
          pr.incTotal();
        }
      }
      return results;
    }

    /**
     * Maps labels
     *
     * @param trainset
     */
    private void mapLabels(List<RawRecord> trainset) {
      mappings = new ArrayList<>(2);
      mappings.add(trainset.get(0).getLabel());
      for (RawRecord r : trainset) {
        if (!mappings.get(0).equals(r.getLabel())) {
          mappings.add(r.getLabel());
          break;
        }
      }
    }

    /**
     * Gets an Int label as a String.
     *
     * @param labelRef
     * @return
     */
    private String getIntLabel(int labelRef) {
      return mappings.get(labelRef);
    }

    /**
     * Gets the a String label by index.
     *
     * @param label
     * @return
     */
    private int getLabelInt(String label) {
      for (int i = 0; i < mappings.size(); i++) {
        String curr = mappings.get(i);
        if (label.equals(curr)) {
          return i;
        }
      }
      return -1;
    }

    /**
     * Covnerts a {@link RawRecord} to a low-level {@link Record}, based on a feature
     * index.
     *
     * @param in       The record to convert
     * @param features The feature index
     * @return
     */
    private Record convToRecord(RawRecord in, List<String> features) {
      //    App.Vector v = App.Vector.zero(features.size() + collocationNGrams.size());
      Vector v = Vector.zero(features.size());
      // first process individual word tokens ---
      for (String token : in.getTokens()) {
        for (int i = 0; i < features.size(); i++) {
          if (token.equals(features.get(i))) {
            v.set(i, 1);
            break;
          }
        }
      }

      // then process collocationNGrams ---
      //TODO move out start and end indexes
      // TODO refactor loops here
      List<String> collocation = Util.getCollocation(in.getTokens(), stopWordsStart, stopWordsEnd, in.getIdx());
      List<String> nGrams = Util.getNGrams(nGramSize, collocation);
      for (String nGram : nGrams) {
        for (int i = features.size() - 1; i >= 0; i--) {
          if (nGram.equals(features.get(i))) {
            v.set(i, 1);
            break;
          }
        }
      }

      return new Record(getLabelInt(in.getLabel()), v);
    }

    /**
     * Gets the native model.
     *
     * @return
     */
    public Model getModel() {
      return model;
    }

    /**
     * Sets the native model.
     *
     * @param model
     */
    public void setModel(Model model) {
      this.model = model;
    }

    /**
     * Writes a model to a file
     * Format follows
     * <p/>
     * feature1:weight1
     * feature2:weight2
     * ... ...
     *
     * @param modelFile
     * @throws IOException
     */
    public void writeToFile(String modelFile) throws IOException {
      FileWriter fw = new FileWriter(new File(modelFile));
      fw.append(mappings.get(0) + " " + mappings.get(1) + "\n");
      for (int i = 0; i < features.size(); i++) {
        String f = features.get(i);
        fw.append(f + ":" + model.getWeights().get(i) + "\n");
      }
      fw.close();
    }

    /**
     * Builds model from a file.
     *
     * @param modelFile
     * @throws IOException
     */
    public void readFromFile(String modelFile) throws IOException {
      features = new ArrayList<>(8000); // allocate at least 8000 so no need to recopy
      List<Double> weights = new ArrayList<>(8000);
      Scanner scanner = new Scanner(new File(modelFile));

      // generate labels
      makeMapLabels(scanner.nextLine().split(" "));

      while (scanner.hasNext()) {
        String[] tokens = scanner.nextLine().split(":");
        if (tokens.length == 2) {
          features.add(tokens[0]);
          weights.add(Double.parseDouble(tokens[1]));
        }
      }

      System.out.println(weights.size() + " features read.");

      Vector weightsV = Vector.zero(features.size());
      for (int i = 0; i < weights.size(); i++) {
        weightsV.set(i, weights.get(i));
      }
      model = new Model(weightsV);
    }

    /**
     * Util function.
     *
     * @param labels
     */
    public void makeMapLabels(String[] labels) {
      mappings = new ArrayList<>();
      mappings.add(labels[0]);
      mappings.add(labels[1]);
    }
  }

  /**
   * Created by joel on 10/13/15.
   * <p/>
   * Implements a Logisitic Regression Classifier.
   * <p/>
   * For generalizability, features are generalized to a set of low-lying weights,
   * rather than custom Feature classes per-se.
   */
  public static class LogisticRegressionClassifier implements Classifier {

    public static final long NO_TIMEOUT = -1;

    List<Record> records;
    float minThreshold = 2;
    private double alpha;
    private double learningDecay;
    private double terminationThreshold;
    private long timeoutPerDimen;

    public LogisticRegressionClassifier() {
    }

    /**
     * Loads a dataset of low-level Records.
     *
     * @param records
     */
    public void loadDataset(List<Record> records) {
      this.records = records;
      this.alpha = 2;
      this.learningDecay = 0.8;
      this.terminationThreshold = 0.000000001;
      this.timeoutPerDimen = NO_TIMEOUT;
    }

    /**
     * The heaviside function.
     *
     * @param raw
     * @return
     */
    public static int heaviside(double raw) {
      if (raw >= 0) {
        return 1;
      } else {
        return 0;
      }
    }

    /**
     * Get dimensions used in classifier.
     *
     * @return
     */
    private int getDimen() {
      if (records.size() < 1) {
        return 0;
      }
      return records.get(0).getDimen();
    }

    /**
     * Train the dataset with initial zero vector.
     *
     * @return
     */
    public Model train() {
      Vector weights = Vector.zero(getDimen());
      return train(weights);
    }

    /**
     * Train the dataset with initial custom vector.
     *
     * @param weights
     * @return
     */
    public Model train(Vector weights) {
      // init weights to zero

      // use stochastic GA
      trainWeightStochastic(records, weights, alpha, learningDecay, terminationThreshold, timeoutPerDimen);
//      trainWeightBatch(records, weights, alpha, learningDecay, terminationThreshold, timeoutPerDimen);
      return new Model(weights);
    }

    /**
     * Train using batch method.  Deprecated.
     *
     * @param records
     * @param existingWeights
     * @param alpha
     * @param learningDecay
     * @param terminationThreshold
     * @param timeoutPerDimen
     */
    @Deprecated
    private void trainWeightBatch(List<Record> records,
                                  Vector existingWeights,
                                  double alpha,
                                  double learningDecay,
                                  double terminationThreshold,
                                  long timeoutPerDimen) {
      //    long startTime = new Date().getTime();
      // for each weight
      for (int i = 0; i < getDimen(); i++) {
        double diff = 999;
        double currAlpha = alpha;
        while (diff > terminationThreshold) {
          //        boolean hasTimeout = ((new Date().getTime() - startTime) < timeoutPerDimen || timeoutPerDimen != NO_TIMEOUT)
          currAlpha *= learningDecay;
          double newWeight = existingWeights.get(i) + currAlpha / records.size() * batchSum(i, records, existingWeights);
          diff = Math.abs(newWeight - existingWeights.get(i));
          existingWeights.set(i, newWeight);
        }
      }
      System.out.println("Exited!");
    }

    /**
     * Helper function.  Deprecated.
     *
     * @param currIdx
     * @param records
     * @param existingWeights
     * @return
     */
    @Deprecated
    private double batchSum(int currIdx, List<Record> records, Vector existingWeights) {
      double sum = 0;
      for (Record r : records) {
        sum += r.getVectors().get(currIdx) * (r.getLabel() - (r.getLabel() - 1 / (1 + Math.exp(-1 * existingWeights.dot(r.getVectors())))));
      }
      return sum;
    }

    /**
     * Train using stochastic method.
     * <p/>
     * This method enforces more rigid implementation,
     * by going through each record until all weights
     * are sufficiently satisfied to perform with given weight.
     * <p/>
     * It also implements higher learning rates for earlier iterations, to
     * reach coarse max faster, and finer learning rates for later iterations,
     * to finetune learning rates. (see Daniel T Larose, Discoering Knowledge in Data)
     * <p/>
     * It is still faster than batch.
     *
     * @param records              A list of low-level records to train on.
     * @param existingWeights      Existing weights to use for training.
     * @param alpha                Learning rate.
     * @param learningDecay        Decay rate for learning rate.
     * @param terminationThreshold When to terminate each weight iteration.
     * @param timeoutPerDimen      Maximum timeout, do not use.
     */
    private void trainWeightStochastic(List<Record> records,
                                       Vector existingWeights,
                                       double alpha,
                                       double learningDecay,
                                       double terminationThreshold,
                                       long timeoutPerDimen) {
      //    long startTime = new Date().getTime();
      // for each weight
      for (int i = 0; i < getDimen(); i++) {
        double diff = 999;
        double currAlpha = alpha;
        while (diff > terminationThreshold) {
          //        boolean hasTimeout = ((new Date().getTime() - startTime) < timeoutPerDimen || timeoutPerDimen != NO_TIMEOUT)
          currAlpha *= learningDecay;
          for (int x = 0; x < records.size(); x++) {
            Record r = records.get(x);
            double actualX = r.getVectors().get(i);
            double newWeight = existingWeights.get(i) + currAlpha * actualX * (r.getLabel() - 1 / (1 + Math.exp(-1 * existingWeights.dot(r.getVectors()))));
            diff = Math.abs(newWeight - existingWeights.get(i));
            existingWeights.set(i, newWeight);
          }
        }
      }
      System.out.println("Exited!");
    }

    /**
     * Trains on the dataset, outputting a model.
     * @param minThreshold
     * @param alpha
     * @param learningDecay
     * @param terminationThreshold
     * @param timeoutPerDimen
     * @return
     */
    public Model train(float minThreshold, double alpha, double learningDecay, double terminationThreshold, long timeoutPerDimen) {
      this.minThreshold = minThreshold;
      this.alpha = alpha;
      this.learningDecay = learningDecay;
      this.terminationThreshold = terminationThreshold;
      this.timeoutPerDimen = timeoutPerDimen;
      return train();
    }

    /**
     * Train with a given set of initial weights.
     * @param initialWeights
     * @param minThreshold
     * @param alpha
     * @param learningDecay
     * @param terminationThreshold
     * @param timeoutPerDimen
     * @return
     */
    public Model train(Vector initialWeights, float minThreshold, double alpha, double learningDecay, double terminationThreshold, long timeoutPerDimen) {
      this.minThreshold = minThreshold;
      this.alpha = alpha;
      this.learningDecay = learningDecay;
      this.terminationThreshold = terminationThreshold;
      this.timeoutPerDimen = timeoutPerDimen;
      return train(initialWeights);
    }

    /**
     * Deprecated.
     * @return
     */
    public double test() {
      return 0;
    }
  }

  /**
   * Created by joel on 10/14/15.
   *
   * Interface for Machine.
   *
   */
  public static interface Machine {

    void train(String datasetFileName) throws FileNotFoundException;

    Map<String, PredictionResult> test(String questionFilename, String answerFilename) throws FileNotFoundException;
  }

  /**
   * Created by joel on 10/14/15.
   *
   * Specifies the model.
   *
   */
  public static class Model {
    private Vector weights;

    public Model(Vector weights) {
      this.weights = weights;
    }

    /**
     * Gets weights.
     * @return
     */
    public Vector getWeights() {
      return weights;
    }

    /**
     * Sets weights.
     * @param weights
     */
    public void setWeights(Vector weights) {
      this.weights = weights;
    }

    /**
     * Evaluate a given set of vectors and output a label,
     * as according to lecture notes. (0 or a 1).
     * @param vectors
     * @return label 0 or label 1.
     */
    public int evaluate(Vector vectors) {
      return LogisticRegressionClassifier.heaviside(weights.dot(vectors));
    }

    @Override
    public String toString() {
      return "App.Model{" +
          "weights=" + weights +
          '}';
    }

    /**
     * A low-level native method used to test accuracy
     * <p/>
     * Necessary for N-fold cross validation
     *
     * @param testSet A list of records to test on.
     * @return The accuracy of the model.
     */
    public double testAccuracy(List<Record> testSet) {
      double total = 0;
      double correct = 0;
      for (Record r : testSet) {
        int prediction = evaluate(r.getVectors());
        if (r.getLabel() == prediction) {
          correct++;
        }
        total++;
      }
      return correct / total;
    }

  }

  /**
   * Created by joel on 10/15/15.
   *
   * Helper function to store prediction results.
   *
   * Useful for getting per label.
   *
   */
  public static class PredictionResult {

    private Integer total;
    private Integer correct;

    public PredictionResult(Integer total, Integer correct) {
      this.total = total;
      this.correct = correct;
    }

    public PredictionResult() {
      this.total = 0;
      this.correct = 0;
    }

    /**
     * Gets labels.
     * @param results
     * @return
     */
    public static List<String> getLabels(Map<String, PredictionResult> results) {
      List<String> labels = new ArrayList<>(2);
      for (String key : results.keySet()) {
        labels.add(key);
      }
      Collections.sort(labels); // always ensure alphabetical
      return labels;
    }

    /**
     * Increases total
     */
    public void incTotal() {
      total++;
    }

    /**
     * Increases correct
     */
    public void incCorrect() {
      correct++;
    }

    /**
     * Retrieves accuracy.
     * @return
     */
    public double getAccuracy() {
      return ((double) correct) / total;
    }

    /**
     * Pretty-print a list of PredictionResults.
     * @param results
     */
    public static void printResults(Map<String, PredictionResult> results) {
      StringBuffer sb = new StringBuffer();
      double correct = 0;
      double total = 0;
      sb.append("--- Results ---\n");
      Iterator it = results.entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<String, PredictionResult> curr = (Map.Entry<String, PredictionResult>) it.next();
        sb.append("Label=" + curr.getKey() + " Accuracy=" + curr.getValue().getAccuracy() + "\n");
        correct += curr.getValue().correct;
        total += curr.getValue().total;
      }
      sb.append("TOTAL ACCURACY=" + correct / total + "\n");
      sb.append("--- /Results ---\n");
      System.out.println(sb.toString());
    }

  }

  /**
   * Created by joel on 10/14/15.
   *
   * A RawRecord is a higher-level version of a @link{Record}, and stores
   * label info.
   *
   */
  public static class RawRecord {

    private String label;
    private String id;
    private List<String> tokens;
    private int idx;                // where label was found

    public RawRecord(String label, String id, String text) {
      this.label = label;
      this.id = id;
      this.tokens = tokenize(text);
    }

    public RawRecord(String label, String id, List<String> tokens) {
      this.label = label;
      this.id = id;
      this.tokens = tokens;
      this.idx = findIdx();
    }

    private static List<String> tokenize(String text) {
      return new ArrayList<>(Arrays.asList(text.split("[\\s']")));
    }

    /**
     * Parses a file into a list of Raw Records.
     * @param filename
     * @return
     * @throws FileNotFoundException
     */
    public static List<RawRecord> parse(String filename) throws FileNotFoundException {
      List<RawRecord> result = new ArrayList<>();
      Scanner scanner = new Scanner(new File(filename));
      while (scanner.hasNextLine()) {
        List<String> currTokens = tokenize(scanner.nextLine());
        String currId = currTokens.remove(0);
        String foundLabel = getLabelAndClean(currTokens, true);
        result.add(new RawRecord(foundLabel, currId, currTokens));
      }
      return result;
    }

    /**
     * Helper function.
     * @param in
     * @return
     */
    private static Map<String, String> makeAnswerMap(Scanner in) {
      Map<String, String> result = new HashMap<>();
      while (in.hasNextLine()) {
        List<String> tokens = tokenize(in.nextLine());
        String id = tokens.get(0);
        String label = tokens.get(1);
        result.put(id, label);
      }
      return result;
    }

    /**
     * Helper function
     * @param in
     * @return
     */
    private static Map<String, List<String>> makeQuestionMap(Scanner in) {
      Map<String, List<String>> result = new HashMap<>();
      while (in.hasNextLine()) {
        List<String> tokens = tokenize(in.nextLine());
        String id = tokens.remove(0);
        getLabelAndClean(tokens, false);
        result.put(id, tokens);
      }
      return result;
    }

    /**
     * Parse a test file.
     * @param questionFilename
     * @param answerFilename
     * @return
     * @throws FileNotFoundException
     */
    public static List<RawRecord> parse(String questionFilename, String answerFilename) throws FileNotFoundException {
      List<RawRecord> result = new ArrayList<>();
      Map<String, List<String>> questions = makeQuestionMap(new Scanner(new File(questionFilename)));
      Map<String, String> answers = makeAnswerMap(new Scanner(new File(answerFilename)));

      Iterator it = answers.entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry curr = (Map.Entry) it.next();
        String id = (String) curr.getKey();
        String label = (String) curr.getValue();
        List<String> tokens = questions.get(id);
        RawRecord record = new RawRecord(label, id, tokens);
        result.add(record);
      }

      return result;
    }

    /**
     * Finds the index of the label.
     *
     * @return
     */
    private int findIdx() {
      for (int i = 0; i < tokens.size(); i++) {
        String token = tokens.get(i);
        if (token.equals(">>")) {
          return i + 1;
        }
      }
      return -1;
    }

    /**
     * Cleans up string and formats it to nice specification.
     *
     * @param currTokens
     * @return
     */
    private static String getLabelAndClean(List<String> currTokens, boolean hasLabel) {
      String foundLabel = "";
      List<String> detected = new ArrayList<>(2);
      for (int i = 0; i < currTokens.size(); i++) {
        String curr = currTokens.get(i).toLowerCase().replaceAll("[\"\':;,./?!@-_]", "");
        currTokens.set(i, curr); // to lower case as well
        if (curr.equals(">>")) {
          if (hasLabel) { // if the label is within text
            foundLabel = currTokens.get(i + 1); // the label
            detected.add(foundLabel); // remove target word
          }
        }
      }
      currTokens.removeAll(detected);
      return foundLabel;
    }

    /**
     * Removes tokens
     * @param rawRecords
     * @param stopWords
     */
    public static void removeTokens(List<RawRecord> rawRecords, Set<String> stopWords) {
      for (RawRecord record : rawRecords) {
        record.removeTokens(stopWords);
      }
    }

    /**
     * Removes punctuation
     * @param rawRecords
     */
    public static void removePunctuation(List<RawRecord> rawRecords) {
      for (RawRecord record : rawRecords) {
        record.removePunctuation();
      }
    }

    /**
     * Creates a vocabulary list.
     * @param records
     * @return
     */
    public static HashMap<String, Integer> makeVocabulary(List<RawRecord> records) {
      HashMap<String, Integer> vocabulary = new HashMap<>();
      for (RawRecord record : records) {
        List<String> tokens = record.getTokens();
        for (String token : tokens) {
          if (vocabulary.containsKey(token)) {
            vocabulary.put(token, vocabulary.get(token) + 1);
          } else {
            vocabulary.put(token, 1);
          }
        }
      }
      return vocabulary;
    }

    /**
     * Creates a vocabulary list.
     * @param records
     * @return
     */
    public static List<String> makeVocabularyList(List<RawRecord> records) {
      HashMap<String, Integer> vocabulary = makeVocabulary(records);
      List<String> result = new ArrayList<>();
      for (String key : vocabulary.keySet()) {
        result.add(key);
      }
      return result;
    }

    /**
     * Get all collations in a list of {@link RawRecord} as a {@link List}.
     *
     * @param records
     * @param start
     * @param end
     * @return
     */
    public static List<String> makeCollocationsNGramsList(int n, List<RawRecord> records, int start, int end) {
      List<String> result;
      Set<String> collocations = getAllCollocationsAsNGram(n, records, start, end);
      result = new ArrayList<>(collocations.size()); // allocate just enough memory
      result.addAll(collocations);
      return result;
    }

    public void removePunctuation() {
      Set<String> punctuation = new HashSet<>(Arrays.asList(new String[]{
          "!",
          "@",
          "\'",
          "\"",
          ",",
          ".",
          ":",
          ";",
          "(",
          ")",
          "?",
          "!",
          "-",
          "--",
          "[",
          "]"
      }));
      removeTokens(punctuation);
    }

    public void removeTokens(Set<String> tokens) {
      List<String> found = new ArrayList();
      for (String token : this.tokens) {
        if (tokens.contains(token)) {
          found.add(token);
        }
      }
      this.tokens.removeAll(found);
    }

    public static List<RawRecord> splice(List<RawRecord> records, String label) {
      List<RawRecord> result = new ArrayList<>(records.size());
      for (RawRecord record : records) {
        if (record.getLabel().equals(label)) {
          result.add(record);
        }
      }
      return result;
    }

    /**
     * Gets all collocations in a given trainset.
     *
     * @param records
     * @param start
     * @param end
     * @return
     */
    public static Set<String> getAllCollocationsAsNGram(int n, List<RawRecord> records, int start, int end) {
      Set<String> collocations = new HashSet<>(records.size());
      for (RawRecord record : records) {
        List<String> collocation = Util.getCollocation(record.tokens, start, end, record.getIdx());
        // TODO refactor trigrams to Ngram customizable
        List<String> nGrams = Util.getNGrams(n, collocation);
        collocations.addAll(nGrams);
      }

      //Monkey Patch return collocation straight instead
      //    for (App.RawRecord record : records) {
      //      int max = record.getTokens().size() - 1;
      //      int realStart = Math.min(Math.max(0, record.getIdx() + start), max);
      //      int realEnd = Math.max(0, Math.min(max, record.getIdx() - start));
      //      String collocation = App.Util.join(" ", record.getTokens().subList(realStart, realEnd));
      //      collocations.add(collocation);
      //    }

      return collocations;
    }

    /**
     * Partitions a list of records by their labels.
     *
     * @param allRecords
     * @return
     */
    public static HashMap<String, List<RawRecord>> segment(List<RawRecord> allRecords) {
      HashMap<String, List<RawRecord>> result = new HashMap<>();
      for (RawRecord r : allRecords) {
        String label = r.getLabel();
        if (!result.containsKey(label)) {
          result.put(label, new ArrayList<RawRecord>());
        }
        result.get(label).add(r);
      }
      return result;
    }

    public static void print(List<RawRecord> rawRecords) {
      StringBuffer sb = new StringBuffer();
      for (RawRecord record : rawRecords) {
        sb.append(record.toString() + "\n");
      }
      System.out.println("--- App.Record Dump ---");
      System.out.println(sb.toString());
      System.out.println("--- /App.Record Dump ---");
    }

    @Override
    public String toString() {
      return "App.RawRecord{" +
          "label='" + label + '\'' +
          ", id='" + id + '\'' +
          ", tokens=" + tokens +
          ", idx=" + idx +
          '}';
    }

    public String getLabel() {
      return label;
    }

    public void setLabel(String label) {
      this.label = label;
    }

    public String getId() {
      return id;
    }

    public void setId(String id) {
      this.id = id;
    }

    public List<String> getTokens() {
      return tokens;
    }

    public void setTokens(List<String> tokens) {
      this.tokens = tokens;
    }

    public int getIdx() {
      return idx;
    }
  }

  /**
   *
   * A Record is the lowest-level element used by the @link{LogisticRegressionClassifier}.
   *
   * Created by joel on 10/13/15.
   */
  public static class Record {

    private int label;      // Y
    private Vector vectors; // x

    public Record(int label, Vector vectors) {
      this.label = label;
      this.vectors = vectors;
    }

    public Record() {
    }

    public int getLabel() {
      return label;
    }

    public void setLabel(int label) {
      this.label = label;
    }

    public int getDimen() {
      return vectors.size();
    }

    public Vector getVectors() {
      return vectors;
    }

    public void setVectors(Vector vectors) {
      this.vectors = vectors;
    }
  }

  /**
   *
   * High-level function for creating a test app.
   *
   * Created by joel on 10/20/15.
   */
  public static class SCTester {

    private CS4248Machine machine;

    private String testFile;
    private String answerFile;
    private String modelFile;

    public SCTester(String testFile, String answerFile, String modelFile) throws IOException {
      this.testFile = testFile;
      this.answerFile = answerFile;
      this.modelFile = modelFile;
      createModel();
    }

    private void createModel() throws IOException {
      machine = new CS4248Machine();
      machine.readFromFile(modelFile);
    }

    public void runTest() throws FileNotFoundException {
      PredictionResult.printResults(machine.test(testFile, answerFile));
    }

  }

  /**
   *
   * High-level function for creating a train app.
   *
   * Created by joel on 10/20/15.
   */
  public static class SCTrainer {

    public String word1;
    public String word2;
    public String trainFile;
    public String modelFile;
    public CS4248Machine machine;

    public SCTrainer(String word1, String word2, String trainFile, String modelFile) {
      this.word1 = word1;
      this.word2 = word2;
      this.trainFile = trainFile;
      this.modelFile = modelFile;
      init();
    }

    /**
     * Trains with a set of parameters found.
     */
    private void init() {
      int featureCountMin = 3; // each feature must appear more than or equal to n times in corpus
      int numFolds = 3;        // number of folds used
      int nGramSize = 3;       // size of Ngram chunks in collocation
      double learningRate = 2;  // learning rate
      double learningDecay = 0.8; // decay coefficient for learning rate
      double terminationThreshold = 0.0000000001; // how similar should weights be before termination
      long timeoutPerDimen = LogisticRegressionClassifier.NO_TIMEOUT; // do not wait for timeout
      float learningMinThreshold = 2; // deprecated DO NOT USE
      int wordDiffMinThreshold = 20; // Remove stop words that appear less than this threshold, between labels
      Util.Pair stopWordsRef = new Util.Pair(-4, 4); // the collocation to use

      machine = new CS4248Machine();
      machine.setParam(
          learningRate,
          learningDecay,
          terminationThreshold,
          timeoutPerDimen,
          learningMinThreshold,
          wordDiffMinThreshold,
          stopWordsRef.a,
          stopWordsRef.b,
          nGramSize,
          numFolds,
          featureCountMin
      );

    }

    public void train() throws FileNotFoundException {
      machine.train(trainFile);
    }

    public void write() throws IOException {
      machine.writeToFile(modelFile);
    }

  }

  /**
   * Created by joel on 10/20/15.
   *
   * Stop words pasted into program.
   *
   */
  public static class StopWords {

    public static String stopWords = "about\n" +
        "above\n" +
        "across\n" +
        "after\n" +
        "afterwards\n" +
        "again\n" +
        "albeit\n" +
        "all\n" +
        "almost\n" +
        "alone\n" +
        "along\n" +
        "already\n" +
        "also\n" +
        "although\n" +
        "always\n" +
        "among\n" +
        "amongst\n" +
        "an\n" +
        "and\n" +
        "another\n" +
        "any\n" +
        "anyhow\n" +
        "anyone\n" +
        "anything\n" +
        "anywhere\n" +
        "are\n" +
        "around\n" +
        "as\n" +
        "at\n" +
        "b\n" +
        "be\n" +
        "became\n" +
        "because\n" +
        "become\n" +
        "becomes\n" +
        "becoming\n" +
        "been\n" +
        "before\n" +
        "beforehand\n" +
        "behind\n" +
        "being\n" +
        "below\n" +
        "beside\n" +
        "besides\n" +
        "between\n" +
        "beyond\n" +
        "both\n" +
        "but\n" +
        "by\n" +
        "c\n" +
        "can\n" +
        "cannot\n" +
        "co\n" +
        "could\n" +
        "d\n" +
        "down\n" +
        "during\n" +
        "e\n" +
        "each\n" +
        "eg\n" +
        "either\n" +
        "else\n" +
        "elsewhere\n" +
        "enough\n" +
        "etc\n" +
        "even\n" +
        "ever\n" +
        "every\n" +
        "everyone\n" +
        "everything\n" +
        "everywhere\n" +
        "except\n" +
        "f\n" +
        "few\n" +
        "first\n" +
        "for\n" +
        "former\n" +
        "formerly\n" +
        "from\n" +
        "further\n" +
        "g\n" +
        "h\n" +
        "had\n" +
        "has\n" +
        "have\n" +
        "he\n" +
        "hence\n" +
        "her\n" +
        "here\n" +
        "hereafter\n" +
        "hereby\n" +
        "herein\n" +
        "hereupon\n" +
        "hers\n" +
        "herself\n" +
        "him\n" +
        "himself\n" +
        "his\n" +
        "how\n" +
        "however\n" +
        "i\n" +
        "ie\n" +
        "if\n" +
        "in\n" +
        "inc\n" +
        "indeed\n" +
        "into\n" +
        "is\n" +
        "it\n" +
        "its\n" +
        "itself\n" +
        "j\n" +
        "k\n" +
        "l\n" +
        "last\n" +
        "latter\n" +
        "latterly\n" +
        "least\n" +
        "less\n" +
        "ltd\n" +
        "m\n" +
        "many\n" +
        "may\n" +
        "meanwhile\n" +
        "might\n" +
        "more\n" +
        "moreover\n" +
        "most\n" +
        "mostly\n" +
        "much\n" +
        "must\n" +
        "n\n" +
        "namely\n" +
        "neither\n" +
        "never\n" +
        "nevertheless\n" +
        "next\n" +
        "no\n" +
        "nobody\n" +
        "none\n" +
        "noone\n" +
        "nor\n" +
        "not\n" +
        "nothing\n" +
        "now\n" +
        "nowhere\n" +
        "o\n" +
        "of\n" +
        "off\n" +
        "often\n" +
        "on\n" +
        "once\n" +
        "one\n" +
        "only\n" +
        "onto\n" +
        "or\n" +
        "other\n" +
        "others\n" +
        "otherwise\n" +
        "our\n" +
        "ours\n" +
        "ourselves\n" +
        "out\n" +
        "over\n" +
        "own\n" +
        "p\n" +
        "per\n" +
        "perhaps\n" +
        "q\n" +
        "r\n" +
        "rather\n" +
        "s\n" +
        "same\n" +
        "seem\n" +
        "seemed\n" +
        "seeming\n" +
        "seems\n" +
        "several\n" +
        "she\n" +
        "should\n" +
        "since\n" +
        "so\n" +
        "some\n" +
        "somehow\n" +
        "someone\n" +
        "something\n" +
        "sometime\n" +
        "sometimes\n" +
        "somewhere\n" +
        "still\n" +
        "such\n" +
        "t\n" +
        "than\n" +
        "that\n" +
        "the\n" +
        "their\n" +
        "them\n" +
        "themselves\n" +
        "then\n" +
        "thence\n" +
        "there\n" +
        "thereafter\n" +
        "thereby\n" +
        "therefore\n" +
        "therein\n" +
        "thereupon\n" +
        "these\n" +
        "they\n" +
        "this\n" +
        "those\n" +
        "though\n" +
        "through\n" +
        "throughout\n" +
        "thru\n" +
        "thus\n" +
        "together\n" +
        "too\n" +
        "toward\n" +
        "towards\n" +
        "u\n" +
        "under\n" +
        "until\n" +
        "up\n" +
        "upon\n" +
        "v\n" +
        "very\n" +
        "via\n" +
        "w\n" +
        "was\n" +
        "well\n" +
        "were\n" +
        "what\n" +
        "whatever\n" +
        "whatsoever\n" +
        "when\n" +
        "whence\n" +
        "whenever\n" +
        "whensoever\n" +
        "where\n" +
        "whereafter\n" +
        "whereas\n" +
        "whereat\n" +
        "whereby\n" +
        "wherefrom\n" +
        "wherein\n" +
        "whereinto\n" +
        "whereof\n" +
        "whereon\n" +
        "whereto\n" +
        "whereunto\n" +
        "whereupon\n" +
        "wherever\n" +
        "wherewith\n" +
        "whether\n" +
        "which\n" +
        "whichever\n" +
        "whichsoever\n" +
        "while\n" +
        "whilst\n" +
        "whither\n" +
        "who\n" +
        "whoever\n" +
        "whole\n" +
        "why\n" +
        "will\n" +
        "with\n" +
        "within\n" +
        "without\n" +
        "would\n" +
        "x\n" +
        "yet\n" +
        "z";

  }

  /**
   * Created by joel on 10/14/15.
   *
   * A Util class.
   *
   */
  public static class Util {

    /**
     * Load stop words from file
     *
     * @param filename
     * @return
     * @throws FileNotFoundException
     */
    public static Set<String> loadStopWords(String filename) throws FileNotFoundException {
      Set<String> result = new HashSet<>();
      Scanner scanner = new Scanner(new File(filename));
      while (scanner.hasNextLine()) {
        String word = scanner.nextLine();
        result.add(word);
      }
      return result;
    }

    /**
     * Load stop words from memory
     *
     * @return
     */
    public static Set<String> loadStopWords() {
      String[] tokens = StopWords.stopWords.split(" ");
      Set<String> result = new HashSet<>(Arrays.asList(tokens));
      return result;
    }

    /**
     * Function to select distinct stop words, based on analysis of 2 individual corpuses.
     * If there are huge differences greater than minThreshold, then these words are used.
     *
     * @param stopWords
     * @param corpus1
     * @param corpus2
     * @param minThreshold
     * @return
     */
    public static Set<String> selectDistinctStopWords(Set<String> stopWords, List<RawRecord> corpus1, List<RawRecord> corpus2, int minThreshold) {

      Map<String, Integer> distrCorpus1 = RawRecord.makeVocabulary(corpus1);
      Map<String, Integer> distrCorpus2 = RawRecord.makeVocabulary(corpus2);
      Set<String> result = new HashSet<>();

      for (String word : stopWords) {
        int count1, count2, diff;
        count1 = count2 = 0;
        if (distrCorpus1.containsKey(word)) {
          count1 = distrCorpus1.get(word);
        }
        if (distrCorpus2.containsKey(word)) {
          count2 = distrCorpus2.get(word);
        }
        diff = Math.abs(count1 - count2);
        if (diff <= minThreshold) {
          result.add(word);
        }
      }

      return result;
    }

    /**
     *
     * Get N-Grams from collocation.
     *
     * @param tokens
     * @param start
     * @param end
     * @param ref
     * @return
     * @deprecated
     */
    public static List<String> getNGrams(List<String> tokens, int start, int end, int ref) {
      List<String> collocations = new ArrayList<>();
      int max = ref + end;
      int min = ref - start;
      if (max < tokens.size()) {
        max = tokens.size() - end;
      }
      if (min > 0) {
        min = start;
      }
      for (int i = min; i < max; i++) {
        String collocation = Util.join(" ", tokens.subList(i - start, i + end));
        collocations.add(collocation);
      }
      return collocations;
    }

    /**
     * Retrieves a collocation, includes fall-back options if unavailable.
     * @param tokens
     * @param start
     * @param end
     * @param ref
     * @return
     */
    public static List<String> getCollocation(List<String> tokens, int start, int end, int ref) {
      int max = ref + end;
      int min = ref + start;
      if (max >= tokens.size()) {
        max = tokens.size() - 1;
      }
      if (min < 0) {
        min = 0;
      }
      if (min >= tokens.size()) {
        min = tokens.size() - 1;
      }
      if (max < 0) {
        max = 0;
      }
      //    String collocation = App.Util.join(" ", tokens.subList(min, max));
      return tokens.subList(min, max);
    }

    /**
     * Joins a list of tokens into a String with given separator.
     * @param separator
     * @param tokens
     * @return
     */
    public static String join(String separator, List<String> tokens) {
      if (tokens.size() < 1) return "";
      StringBuffer sb = new StringBuffer();
      sb.append(tokens.get(0));
      for (int i = 1; i < tokens.size(); i++) {
        sb.append(" ");
        sb.append(tokens.get(i));
      }
      return sb.toString();
    }

    /**
     * Retrieves a list of Ngrams in a given collocation.
     * @param n
     * @param collocation
     * @return
     */
    public static List<String> getNGrams(int n, List<String> collocation) {
      List<String> nGrams = new ArrayList<>();
      for (int i = 0; i < collocation.size() - n; i++) {
        nGrams.add(Util.join(" ", collocation.subList(i, i + n)));
      }
      return nGrams;
    }

    public static class Pair {
      int a;
      int b;

      public Pair(int a, int b) {
        this.a = a;
        this.b = b;
      }
    }
  }

  /**
   * Class for handling Vectors.
   * <p/>
   * Created by joel on 10/13/15.
   */
  public static class Vector {

    protected double[] vectors;

    /**
     * Initializer from varargs
     *
     * @param vectors
     */
    public Vector(Double... vectors) {
      this.vectors = new double[vectors.length];
      for (int i = 0; i < vectors.length; i++) {
        this.vectors[i] = vectors[i];
      }
    }

    /**
     * Initializer.
     *
     * @param vectors
     */
    public Vector(double[] vectors) {
      this.vectors = vectors;
    }

    public Vector clone() {
      double[] tempVec = Arrays.copyOf(this.vectors, this.vectors.length);
      return new Vector(tempVec);
    }

    /**
     * Gets at index.
     *
     * @param idx
     * @return
     */
    public double get(int idx) {
      return vectors[idx];
    }

    /**
     * Sets at index.
     *
     * @param idx
     * @param val
     */
    public void set(int idx, double val) {
      vectors[idx] = val;
    }

    /**
     * Returns size of the vector.
     *
     * @return
     */
    public int size() {
      return vectors.length;
    }

    /**
     * Calculates the dot product between 2 vectors
     *
     * @param b
     * @return
     */
    public double dot(Vector b) {
      double result = 0;
      for (int i = 0; i < this.size(); i++) {
        result += this.vectors[i] * b.vectors[i];
      }
      return result;
    }

    /**
     * Initializes a vector accordingly.
     *
     * @param size
     * @param initVal
     * @return
     */
    public static Vector init(int size, double initVal) {
      double[] vec = new double[size];
      for (int i = 0; i < size; i++) {
        vec[i] = initVal + 0;
      }
      return new Vector(vec);
    }

    /**
     * Returns a zero vector.
     *
     * @param size
     * @return
     */
    public static Vector zero(int size) {
      return init(size, 0);
    }

    public static Vector randSeed(int min, int max, int size) {
      Vector v = init(size, 0);
      for (int i = 0; i < v.size(); i++) {
        v.set(i, min + Math.random() * (max - min));
      }
      return v;
    }

    /**
     * Adds another vector.
     *
     * @param b
     * @return
     */
    public Vector add(Vector b) {
      Vector result = Vector.init(b.size(), 0);
      for (int i = 0; i < this.size(); i++) {
        result.vectors[i] = this.vectors[i] + b.vectors[i];
      }
      return result;
    }

    /**
     * Subtracts another vector.
     *
     * @param b
     * @return
     */
    public Vector subtract(Vector b) {
      Vector result = Vector.init(b.size(), 0);
      for (int i = 0; i < this.size(); i++) {
        result.vectors[i] = this.vectors[i] - b.vectors[i];
      }
      return result;
    }

    /**
     * Calculates the normal of a vector
     *
     * @return
     */
    public double norm() {
      double result = 0;
      for (double vec : this.vectors) {
        result += vec * vec;
      }
      return Math.sqrt(result);
    }

    @Override
    public String toString() {
      return "App.Vector{" +
          "vectors=" + Arrays.toString(vectors) +
          '}';
    }
  }
}
