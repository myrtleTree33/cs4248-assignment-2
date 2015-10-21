import org.junit.Ignore;
import org.junit.Test;

import java.util.Date;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  @Ignore
  @Test
  public void testTrain() throws Exception {
    App.CS4248Machine machine = new App.CS4248Machine();
    machine.setParam(0.1, 0.8, 0.00000000001, App.LogisticRegressionClassifier.NO_TIMEOUT, 5, 30, -3, -1, 3, 2);
    machine.train(ROOT_PATH + "adapt_adopt.train");
    App.PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));
  }

  //  @Ignore
  @Test
  public void testTrainGenerative() throws Exception {

    class Dataset {
      public String trainFilepath;
      public String testFilepath;
      public String answerFilepath;

      public Dataset(String trainFilepath, String testFilepath, String answerFilepath) {
        this.trainFilepath = trainFilepath;
        this.testFilepath = testFilepath;
        this.answerFilepath = answerFilepath;
      }
    }

    App.CS4248Machine machine = new App.CS4248Machine();
    int numFolds = 3;
//    int[] nGramSize = new int[]{2,3,4};
    int[] nGramSize = new int[]{3};
    double[] learningRates = new double[]{2, 1.7, 1.5, 1.2, 1.1};
//    double[] learningDecay = new double[]{0.75};
    double[] learningDecay = new double[]{0.75};
    double[] terminationThreshold = new double[]{
        0.0000000001,
        0.000000000001
    };
    long timeoutPerDimen = App.LogisticRegressionClassifier.NO_TIMEOUT;
    float[] learningMinThresholds = new float[]{2};
//    int[] wordDiffMinThresholds = new int[]{5,10,15,20,40};
    int[] wordDiffMinThresholds = new int[]{20};
    App.Util.Pair[] stopWordsRef = new App.Util.Pair[]{
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
        new App.Util.Pair(-3, 3),
//        new App.Util.Pair(-5, 5),
//        new App.Util.Pair(-6, 6),
//        new App.Util.Pair(-7, 7),
//        new App.Util.Pair(-10, 10),
//        new Pair(-4, 4),

    };

    Dataset[] datasets = new Dataset[]{
        new Dataset(
            "bought_brought.train",
            "bought_brought.test",
            "bought_brought.answer"
        ),
        new Dataset(
            "adapt_adopt.train",
            "adapt_adopt.test",
            "adapt_adopt.answer"
        ),
        new Dataset(
            "peace_piece.train",
            "peace_piece.test",
            "peace_piece.answer"
        )
    };

//    String trainFilePath = ROOT_PATH + "adapt_adopt.train";
//    String stopWordFilePath = ROOT_PATH + "stopwd.txt";
//    String testFilePath = ROOT_PATH + "adapt_adopt.test";
//    String testAnswerFilePath = ROOT_PATH + "adapt_adopt.answer";

    for (int h = 0; h < datasets.length; h++) {
      for (int b = 0; b < learningMinThresholds.length; b++) {
        for (int c = 0; c < wordDiffMinThresholds.length; c++) {
          for (int d = 0; d < stopWordsRef.length; d++) {
            for (int e = 0; e < nGramSize.length; e++) {
              for (int g = 0; g < terminationThreshold.length; g++) {
                for (int f = 0; f < learningDecay.length; f++) {
                  for (int a = 0; a < learningRates.length; a++) {

                    String trainFilePath = ROOT_PATH + datasets[h].trainFilepath;
                    String testFilePath = ROOT_PATH + datasets[h].testFilepath;
                    String testAnswerFilePath = ROOT_PATH + datasets[h].answerFilepath;


                    long startTime = new Date().getTime();
                    machine.setParam(
                        learningRates[a],
                        learningDecay[f],
                        terminationThreshold[g],
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
                        "Filename=" + trainFilePath + " " +
                        "LearningRate=" + learningRates[a] + " " +
                            "LearningMinThresholds=" + learningMinThresholds[b] + " " +
                            "WordDiffMinThresholds=" + wordDiffMinThresholds[c] + " " +
                            "stopWordsStart=" + stopWordsRef[d].a + " " +
                            "stopWordsEnd=" + stopWordsRef[d].b + " " +
                            "NGramSize=" + nGramSize[e] + " " +
                            "TimeTaken=" + (timeDiff / 1000) + "s"
                    );
                    App.PredictionResult.printResults(machine.test(testFilePath, testAnswerFilePath));

                  }
                }
              }
            }
          }
        }
      }
    }
  }

}