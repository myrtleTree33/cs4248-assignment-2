import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by joel on 10/20/15.
 */
public class SCTrainer {

  public String word1;
  public String word2;
  public String trainFile;
  public String modelFile;
  public CS4248Machine machine;

  public SCTrainer(String word1, String word2, String trainFile, String modelFile) {
    this.word1 = word1;
    this.word2 = word2;
    this.trainFile = trainFile;
    this.modelFile = modelFile;
    init();
  }

  private void init() {
    int numFolds = 5;
    int nGramSize = 3;
    double learningRate = 0.15;
    double learningDecay = 0.75;
    double terminationThreshold = 0.0000000001;
    long timeoutPerDimen = LogisticRegressionClassifier.NO_TIMEOUT;
    float learningMinThreshold = 5;
    int wordDiffMinThreshold = 7;
    Util.Pair stopWordsRef = new Util.Pair(-2,2);

    machine = new CS4248Machine();
    machine.setParam(
        learningRate,
        learningDecay,
        terminationThreshold,
        timeoutPerDimen,
        learningMinThreshold,
        wordDiffMinThreshold,
        stopWordsRef.a,
        stopWordsRef.b,
        nGramSize,
        numFolds
    );

  }

  public void train() throws FileNotFoundException {
    machine.train(trainFile);
  }

  public void write() throws IOException {
    machine.writeToFile(modelFile);
  }

}
