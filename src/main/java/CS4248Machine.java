import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

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
  public void train(String datasetFileName) throws FileNotFoundException {
    List<RawRecord> trainset = RawRecord.parse(datasetFileName);
    List<Record> records = new ArrayList<>();
    mapLabels(trainset); // generate label mapping
    vocabulary = RawRecord.makeVocabularyList(trainset);
    for (RawRecord r : trainset) {
      records.add(convToRecord(r, vocabulary));
    }
    classifier.loadDataset(records);
    model = classifier.train(5, 2);
  }

  @Override
  public void test(String testFileName) {

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
