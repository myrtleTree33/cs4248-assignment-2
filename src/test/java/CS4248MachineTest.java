import org.junit.Ignore;
import org.junit.Test;

import java.util.Date;
import java.util.Map;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  @Ignore
  @Test
  public void testTrain() throws Exception {
    CS4248Machine machine = new CS4248Machine();
    machine.setParam(0.1, 0.8, 0.00000000001, LogisticRegressionClassifier.NO_TIMEOUT, 5, 30, -3, -1, 3, 2);
    machine.train(ROOT_PATH + "adapt_adopt.train");
    PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));
  }

  //  @Ignore
  @Test
  public void testTrainGenerative() throws Exception {
    CS4248Machine machine = new CS4248Machine();
    int numFolds = 5;
//    int[] nGramSize = new int[]{2,3,4};
    int[] nGramSize = new int[]{3};
    double[] learningRates = new double[]{0.15,0.2};
    double learningDecay = 0.75;
    double terminationThreshold = 0.0000000001;
    long timeoutPerDimen = LogisticRegressionClassifier.NO_TIMEOUT;
    float[] learningMinThresholds = new float[]{5};
//    int[] wordDiffMinThresholds = new int[]{5,10,15,20,40};
    int[] wordDiffMinThresholds = new int[]{7};
    Util.Pair[] stopWordsRef = new Util.Pair[]{
//        new Pair(-3, -1),
//        new Pair(-4, -2),
//        new Pair(-5, -3),
//        new Pair(-6, -4),
//        new Pair(-7, -5),
//        new Pair(-8, -6),
//        new Pair(-9, -7)

//        new Pair(1, 3),
//        new Pair(2,4),
//        new Pair(3,5),
//        new Pair(4,6),
//        new Pair(5,7),
//        new Pair(6,8),
//        new Pair(7,9)

//        new Pair(1, 2),
//        new Pair(2,3),
//        new Pair(3,4),
//        new Pair(4,5),
//        new Pair(5,6),
//        new Pair(6,7),
//        new Pair(7,8)

//        new Pair(1, 7),
//        new Pair(2, 8),
//        new Pair(3, 9),
//        new Pair(4, 10),
//        new Pair(5, 11),
//        new Pair(6, 12),
//        new Pair(7, 13)

//        new Pair(-2, 2),
        new Util.Pair(-2, 2),
//        new Pair(-4, 4),

    };

//    String trainFilePath = ROOT_PATH + "adapt_adopt.train";
//    String stopWordFilePath = ROOT_PATH + "stopwd.txt";
//    String testFilePath = ROOT_PATH + "adapt_adopt.test";
//    String testAnswerFilePath = ROOT_PATH + "adapt_adopt.answer";

    String trainFilePath = ROOT_PATH + "bought_brought.train";
    String stopWordFilePath = ROOT_PATH + "stopwd.txt";
    String testFilePath = ROOT_PATH + "bought_brought.test";
    String testAnswerFilePath = ROOT_PATH + "bought_brought.answer";

    for (int a = 0; a < learningRates.length; a++) {
      for (int b = 0; b < learningMinThresholds.length; b++) {
        for (int c = 0; c < wordDiffMinThresholds.length; c++) {
          for (int d = 0; d < stopWordsRef.length; d++) {
            for (int e = 0; e < nGramSize.length; e++) {
              long startTime = new Date().getTime();
              machine.setParam(
                  learningRates[a],
                  learningDecay,
                  terminationThreshold,
                  timeoutPerDimen,
                  learningMinThresholds[b],
                  wordDiffMinThresholds[c],
                  stopWordsRef[d].a,
                  stopWordsRef[d].b,
                  nGramSize[e],
                  numFolds
              );

              machine.train(trainFilePath);
              long timeDiff = new Date().getTime() - startTime;
              System.out.println(
                  "LearningRate=" + learningRates[a] + " " +
                      "LearningMinThresholds=" + learningMinThresholds[b] + " " +
                      "WordDiffMinThresholds=" + wordDiffMinThresholds[c] + " " +
                      "stopWordsStart=" + stopWordsRef[d].a + " " +
                      "stopWordsEnd=" + stopWordsRef[d].b + " " +
                      "NGramSize=" + nGramSize[e] + " " +
                      "TimeTaken=" + timeDiff
              );
              PredictionResult.printResults(machine.test(testFilePath, testAnswerFilePath));

            }
          }
        }
      }
    }
  }

  // TODO remove from code
  @Deprecated
  private StringBuffer printCsv(double learningRate,
                                int wordDiffThreshold,
                                int collocationStart,
                                int collocationEnd,
                                Map<String,
                                    PredictionResult> result) {
    StringBuffer sb = new StringBuffer();
    sb.append(learningRate + ",")
        .append(wordDiffThreshold + ",")
        .append(collocationStart + ",")
        .append(collocationEnd + ",");

    return sb;

  }


}