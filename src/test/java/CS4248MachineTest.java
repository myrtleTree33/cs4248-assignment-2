import org.junit.Ignore;
import org.junit.Test;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  @Ignore
  @Test
  public void testTrain() throws Exception {
    CS4248Machine machine = new CS4248Machine();
    machine.setParam(0.2, 5, 30, -3, -1);
    machine.train(ROOT_PATH + "adapt_adopt.train", ROOT_PATH + "stopwd.txt");
    PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));
  }

  @Test
  public void testTrainGenerative() throws Exception {
    CS4248Machine machine = new CS4248Machine();

    double[] learningRates = new double[]{0.2};
    float[] learningMinThresholds = new float[]{5};
    int[] wordDiffMinThresholds = new int[]{30};
    int[] stopWordsStart = new int[]{
        -3,-4,
    };
    int[] stopWordsEnd = new int[]{
        -1,-1
    };

    for (int a = 0; a < learningRates.length; a++) {
      for (int b = 0; b < learningMinThresholds.length; b++) {
        for (int c = 0; c < wordDiffMinThresholds.length; c++) {
          for (int d = 0; d < stopWordsStart.length; d++) {
            for (int e = 0; e < stopWordsEnd.length; e++) {
              machine.setParam(
                  learningRates[a],
                  learningMinThresholds[b],
                  wordDiffMinThresholds[c],
                  stopWordsStart[d],
                  stopWordsEnd[e]);
              machine.train(ROOT_PATH + "adapt_adopt.train", ROOT_PATH + "stopwd.txt");
              System.out.println(
                  "LearningRate=" + learningRates[a] + " " +
                      "LearningMinThresholds=" + learningMinThresholds[b] + " " +
                      "WordDiffMinThresholds=" + wordDiffMinThresholds[c] + " " +
                      "stopWordsStart=" + stopWordsStart[d] + " " +
                      "stopWordsEnd=" + stopWordsEnd[e]
              );
              PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));

            }
          }
        }
      }
    }

  }


}