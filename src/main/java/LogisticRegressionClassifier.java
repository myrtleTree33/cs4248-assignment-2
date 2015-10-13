import java.util.List;

/**
 * Created by joel on 10/13/15.
 */
public class LogisticRegressionClassifier implements Classifier {

  List<Record> records;

  public LogisticRegressionClassifier() {
  }

  public void loadDataset(List<Record> records) {
    this.records = records;
  }

  public void train() {

  }

  public void train(float minThreshold) {

  }

  public double test() {
    return 0;
  }
}
