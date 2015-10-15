import org.junit.Ignore;
import org.junit.Test;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  class Pair {
    int a;
    int b;

    public Pair(int a, int b) {
      this.a = a;
      this.b = b;
    }
  }

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
    Pair[] stopWordsRef = new Pair[]{
        new Pair(-3, -1),
        new Pair(-4, -2),
        new Pair(-5, -3),
        new Pair(-6, -4),
        new Pair(-7, -5),
        new Pair(-8, -6),
        new Pair(-9, -7)
    };

    for (int a = 0; a < learningRates.length; a++) {
      for (int b = 0; b < learningMinThresholds.length; b++) {
        for (int c = 0; c < wordDiffMinThresholds.length; c++) {
          for (int d = 0; d < stopWordsRef.length; d++) {
            machine.setParam(
                learningRates[a],
                learningMinThresholds[b],
                wordDiffMinThresholds[c],
                stopWordsRef[d].a,
                stopWordsRef[d].b);
            machine.train(ROOT_PATH + "adapt_adopt.train", ROOT_PATH + "stopwd.txt");
            System.out.println(
                "LearningRate=" + learningRates[a] + " " +
                    "LearningMinThresholds=" + learningMinThresholds[b] + " " +
                    "WordDiffMinThresholds=" + wordDiffMinThresholds[c] + " " +
                    "stopWordsStart=" + stopWordsRef[d].a + " " +
                    "stopWordsEnd=" + stopWordsRef[d].b
            );
            PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));

          }
        }
      }
    }

  }


}