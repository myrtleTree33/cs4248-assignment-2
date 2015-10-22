import org.junit.Ignore;
import org.junit.Test;

import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  @Ignore
  @Test
  public void testTrain() throws Exception {
    App.CS4248Machine machine = new App.CS4248Machine();
    machine.setParam(0.1, 0.8, 0.00000000000001, App.LogisticRegressionClassifier.NO_TIMEOUT, 5, 30, -3, -1, 3, 2, 2);
    machine.train(ROOT_PATH + "adapt_adopt.train");
    App.PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));
  }

    @Ignore
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
    int[] featureCountMin = new int[]{3,4}; // 4 is best
    int numFolds[] = new int[]{3};
    int[] nGramSize = new int[]{3};
//    double[] learningRates = new double[]{1.5};
    double[] learningRates = new double[]{2,2.5};
    double[] learningDecay = new double[]{.8};
    double[] terminationThreshold = new double[]{
//        0.0000001,
//        0.00000001,
          0.0000000001,
//        0.000000001,
//        0.0000000001, // best
//        0.00000000001,
    };
    long timeoutPerDimen = App.LogisticRegressionClassifier.NO_TIMEOUT;
    float[] learningMinThresholds = new float[]{2};// deprecated
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
//        new App.Util.Pair(-3, 3), // best
//        new App.Util.Pair(-5, 5),
//        new App.Util.Pair(-6, 6),
//        new App.Util.Pair(-10, 10),
        new App.Util.Pair(-4, 4),
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

    StringBuffer csvDump = new StringBuffer();
    csvDump.append(
        "Filename," +
            "LearningRate," +
            "LearningMinThreshold," +
            "WordDiffMinThresholds," +
            "stopWordsStart," +
            "stopWordsEnd," +
            "NGramSize," +
            "TerminationThreshold," +
            "FeatureCountMin," +
            "NumFolds," +
            "TimeTaken," +
            "Word1Accuracy," +
            "Word2Accuracy," +
            "TotalAccuracy" +
            "\n"
    );

    for (int h = 0; h < datasets.length; h++) {
      for (int b = 0; b < learningMinThresholds.length; b++) {
        for (int c = 0; c < wordDiffMinThresholds.length; c++) {
          for (int d = 0; d < stopWordsRef.length; d++) {
            for (int e = 0; e < nGramSize.length; e++) {
              for (int g = 0; g < terminationThreshold.length; g++) {
                for (int f = 0; f < learningDecay.length; f++) {
                  for (int a = 0; a < learningRates.length; a++) {
                    for (int i = 0; i < featureCountMin.length; i++) {
                      for (int j = 0; j < numFolds.length; j++) {

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
                            numFolds[j],
                            featureCountMin[i]
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
                                "TerminationThreshold=" + terminationThreshold[g] + " " +
                                "FeatureCountMin=" + featureCountMin[i] + " " +
                                "NumFolds=" + numFolds[j] + " " +
                                "TimeTaken=" + (timeDiff / 1000) + "s"
                        );
                        Map<String, App.PredictionResult> results = machine.test(testFilePath, testAnswerFilePath);
                        App.PredictionResult.printResults(results);
                        List<String> labels = App.PredictionResult.getLabels(results);

                        double word1Accuracy = results.get(labels.get(0)).getAccuracy();
                        double word2Accuracy = results.get(labels.get(1)).getAccuracy();
                        double totalAccuracy = (word1Accuracy + word2Accuracy) / 2;


                        csvDump.append(
                            trainFilePath + "," +
                                learningRates[a] + "," +
                                learningMinThresholds[b] + "," +
                                wordDiffMinThresholds[c] + "," +
                                stopWordsRef[d].a + "," +
                                stopWordsRef[d].b + "," +
                                nGramSize[e] + "," +
                                terminationThreshold[g] + "," +
                                featureCountMin[i] + "," +
                                numFolds[j] + "," +
                                timeDiff + "," +
                                word1Accuracy + "," +
                                word2Accuracy + "," +
                                totalAccuracy +
                                "\n"
                        );
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

    System.out.println("--- CSV Dump ---");
    System.out.println(csvDump.toString());
    System.out.println("--- /CSV Dump ---");
  }

}