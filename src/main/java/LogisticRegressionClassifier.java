import java.util.Date;
import java.util.List;

/**
 * Created by joel on 10/13/15.
 */
public class LogisticRegressionClassifier implements Classifier {

  public static final long NO_TIMEOUT = -1;

  List<Record> records;
  float minThreshold = 2;
  private double alpha;
  private double learningDecay;
  private double terminationThreshold;
  private long timeoutPerDimen;

  public LogisticRegressionClassifier() {
  }

  public void loadDataset(List<Record> records) {
    this.records = records;
    this.alpha = 2;
    this.learningDecay = 0.8;
    this.terminationThreshold = 0.000000001;
    this.timeoutPerDimen = NO_TIMEOUT;
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
    Vector weights = Vector.zero(getDimen());
    return train(weights);
  }

  public Model train(Vector weights) {
    // init weights to zero

    // use stochastic GA
    trainWeightStochastic(records, weights, alpha, learningDecay, terminationThreshold, timeoutPerDimen);
    return new Model(weights);
  }

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

  public Model train(float minThreshold, double alpha, double learningDecay, double terminationThreshold, long timeoutPerDimen) {
    this.minThreshold = minThreshold;
    this.alpha = alpha;
    this.learningDecay = learningDecay;
    this.terminationThreshold = terminationThreshold;
    this.timeoutPerDimen = timeoutPerDimen;
    return train();
  }

  public Model train(Vector initialWeights, float minThreshold, double alpha, double learningDecay, double terminationThreshold, long timeoutPerDimen) {
    this.minThreshold = minThreshold;
    this.alpha = alpha;
    this.learningDecay = learningDecay;
    this.terminationThreshold = terminationThreshold;
    this.timeoutPerDimen = timeoutPerDimen;
    return train(initialWeights);
  }

  public double test() {
    return 0;
  }
}
