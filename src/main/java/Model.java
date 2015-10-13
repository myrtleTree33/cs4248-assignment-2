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
}
