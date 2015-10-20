import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by joel on 10/20/15.
 */
public class SCTester {

  private CS4248Machine machine;

  private String testFile;
  private String answerFile;
  private String modelFile;

  public SCTester(String testFile, String answerFile, String modelFile) throws IOException {
    this.testFile = testFile;
    this.answerFile = answerFile;
    this.modelFile = modelFile;
    createModel();
  }

  private void createModel() throws IOException {
    machine = new CS4248Machine();
    machine.readFromFile(modelFile);
  }

  public void runTest() throws FileNotFoundException {
    PredictionResult.printResults(machine.test(testFile, answerFile));
  }

}
