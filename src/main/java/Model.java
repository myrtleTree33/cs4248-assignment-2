import java.util.List;

/**
 * Created by joel on 10/14/15.
 */
public class Model {
  private Vector weights;

  public Model(Vector weights) {
    this.weights = weights;
  }

  public Vector getWeights() {
    return weights;
  }

  public void setWeights(Vector weights) {
    this.weights = weights;
  }

  public int evaluate(Vector vectors) {
    return LogisticRegressionClassifier.heaviside(weights.dot(vectors));
  }

  @Override
  public String toString() {
    return "Model{" +
        "weights=" + weights +
        '}';
  }

  /**
   * A low-level native method used to test accuracy
   *
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
