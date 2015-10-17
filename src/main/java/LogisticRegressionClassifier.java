import java.util.Collections;
import java.util.List;

/**
 * Created by joel on 10/13/15.
 */
public class LogisticRegressionClassifier implements Classifier {

  List<Record> records;
  float minThreshold = 2;
  private double alpha;

  public LogisticRegressionClassifier() {
  }

  public void loadDataset(List<Record> records) {
    this.records = records;
    this.alpha = 2;
  }

  /**
   * Heaviside function used to calculate y result
   *
   * @param weights
   * @param vectors
   * @return
   */
  public static int heaviside(double raw) {
    if (raw >= 0) {
      return 1;
    } else {
      return 0;
    }
  }

  private int getDimen() {
    if (records.size() < 1) {
      return 0;
    }
    return records.get(0).getDimen();
  }

  public Model train() {
    // init weights to zero
    Collections.shuffle(records); // shuffle the collection
    Vector weights = Vector.zero(getDimen());

    for (Record r : records) {
      // use stochastic GA
      trainWeightStochastic(r, weights, alpha);
    }

    return new Model(weights);
  }

  private void trainWeightStochastic(Record r, Vector existingWeights, double alpha) {
    // for each weight
    for (int i = 0; i < getDimen(); i++) {
      double actualX = r.getVectors().get(i);
      double newWeight = existingWeights.get(i) + alpha * actualX * (r.getLabel() - 1 / (1 + Math.exp(-1 * existingWeights.dot(r.getVectors()))));
      existingWeights.set(i, newWeight);
    }
  }

  public Model train(float minThreshold, double alpha) {
    this.minThreshold = minThreshold;
    this.alpha = alpha;
    return train();
  }

  public double test() {
    return 0;
  }
}
