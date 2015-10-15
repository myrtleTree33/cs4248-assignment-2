import java.util.Iterator;
import java.util.Map;

/**
 * Created by joel on 10/15/15.
 */
public class PredictionResult {

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

  public void incTotal() {
    total++;
  }

  public void incCorrect() {
    correct++;
  }

  public double getAccuracy() {
    return ((double) correct) / total;
  }

  public static void printResults(Map<String, PredictionResult> results) {
    StringBuffer sb = new StringBuffer();
    sb.append("--- Results ---\n");
    Iterator it = results.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<String, PredictionResult> curr = (Map.Entry<String, PredictionResult>) it.next();
      sb.append("Label=" + curr.getKey() + " Accuracy=" + curr.getValue().getAccuracy() + "\n");

    }
    sb.append("--- /Results ---\n");
    System.out.println(sb.toString());
  }


}
