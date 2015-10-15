import java.io.FileNotFoundException;
import java.util.*;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248Machine implements Machine {

  private List<String> mappings;
  private List<String> vocabulary;
  private LogisticRegressionClassifier classifier;
  private Model model;

  public CS4248Machine() {
    classifier = new LogisticRegressionClassifier();
  }

  @Override
  public void train(String datasetFileName, String stopWordsFileName, int minThreshold, float learningMinThreshold, double learningRate) throws FileNotFoundException {
    List<RawRecord> trainset = RawRecord.parse(datasetFileName);
    List<Record> records = new ArrayList<>();
    mapLabels(trainset); // generate label mapping
    vocabulary = RawRecord.makeVocabularyList(trainset);
    for (RawRecord r : trainset) {
      records.add(convToRecord(r, vocabulary));
    }

    // reduce features by including relevant stop words
    Set<String> stopWords = optimizeStopWords(
        trainset,
        Util.loadStopWords(stopWordsFileName),
        minThreshold
    );
    RawRecord.removeTokens(trainset, stopWords);

    classifier.loadDataset(records);
    model = classifier.train(learningMinThreshold, learningRate);
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
        partitions.get(keys.get(0)),
        minThreshold
    );
    return optimizedStopWords;
  }

  @Override
  public Map<String, PredictionResult> test(String questionFilename, String answerFilename) throws FileNotFoundException {
    List<RawRecord> testSet = RawRecord.parse(questionFilename, answerFilename);
    Map<String, PredictionResult> results = new HashMap<>();
    for (RawRecord record : testSet) {
      Record curr = convToRecord(record, vocabulary);
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

  private String getIntLabel(int labelRef) {
    return mappings.get(labelRef);
  }

  private int getLabelInt(String label) {
    for (int i = 0; i < mappings.size(); i++) {
      String curr = mappings.get(i);
      if (label.equals(curr)) {
        return i;
      }
    }
    return -1;
  }

  private Record convToRecord(RawRecord in, List<String> vocabulary) {
    Vector v = Vector.zero(vocabulary.size());
    for (String token : in.getTokens()) {
      for (int i = 0; i < vocabulary.size(); i++) {
        if (token.equals(vocabulary.get(i))) {
          v.set(i, 1);
          break;
        }
      }
    }
    return new Record(getLabelInt(in.getLabel()), v);
  }
}
