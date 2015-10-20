import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248Machine implements Machine {

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

  public void setParam(double learningRate,
                       double learningDecay,
                       double terminationThreshold,
                       long timeoutPerDimen,
                       float learningMinThreshold,
                       int wordDiffMinThreshold,
                       int stopWordsStart,
                       int stopWordsEnd,
                       int nGramSize,
                       int folds) {
    this.learningRate = learningRate;
    this.learningDecay = learningDecay;
    this.terminationThreshold = terminationThreshold;
    this.timeoutPerDimen = timeoutPerDimen;
    this.learningMinThreshold = learningMinThreshold;
    this.wordDiffMinThreshold = wordDiffMinThreshold;
    this.stopWordsStart = stopWordsStart;
    this.stopWordsEnd = stopWordsEnd;
    this.nGramSize = nGramSize;
    this.folds = folds;
  }

  public CS4248Machine() {
    classifier = new LogisticRegressionClassifier();
  }

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
//    RawRecord.print(trainset);

    mapLabels(trainset); // generate label mapping

    // TODO comment out this dump statement
    vocabulary = RawRecord.makeVocabularyList(trainset);
    collocationNGrams = RawRecord.makeCollocationsNGramsList(nGramSize, trainset, stopWordsStart, stopWordsEnd);

    generateFeatures();

    for (RawRecord r : trainset) {
      records.add(convToRecord(r, features));
    }
    Collections.shuffle(records); // shuffle the collection to ensure unbiased ordering

    model = trainNFolds(records, folds);
  }

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

  private Model train(List<Record> records) {
    classifier.loadDataset(records);
    Model model = classifier.train(learningMinThreshold, learningRate, learningDecay, terminationThreshold, timeoutPerDimen);
    return model;
  }

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
    return bestModel;
  }

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
    Vector v = Vector.zero(features.size() + collocationNGrams.size());
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

  public Model getModel() {
    return model;
  }

  public void setModel(Model model) {
    this.model = model;
  }

  /**
   * Writes a model to a file
   * Format follows
   *
   * feature1:weight1
   * feature2:weight2
   * ... ...
   *
   * @param modelFile
   * @throws IOException
   */
  public void writeToFile(String modelFile) throws IOException {
    FileWriter fw = new FileWriter(new File(modelFile));
    for (int i = 0; i < features.size(); i++) {
      String f = features.get(i);
      fw.append(f + ":" + model.getWeights().get(i) + "\n");
    }
    fw.close();
  }

  /**
   * Builds model from a file.
   * @param modelFile
   * @throws IOException
   */
  public void readFromFile(String modelFile) throws IOException {
    // TODO
  }
}
