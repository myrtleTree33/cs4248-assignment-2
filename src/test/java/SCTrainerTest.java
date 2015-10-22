import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.util.Date;

/**
 * Created by joel on 10/20/15.
 */
@Ignore
public class SCTrainerTest {

  private final String ROOT_PATH = "src/test/resources/";
  private App.SCTrainer scTrainer;
  private App.SCTester scTester;

//    private String confusionPair = "adapt_adopt";
//  private String confusionPair = "bought_brought";
  private String confusionPair = "peace_piece";

  @Test
  public void trainAndTest() throws Exception {
    for (int i = 0; i < 3; i++) {
      long startTime = new Date().getTime();
      train();
      runTest();
      long diff = (new Date().getTime() - startTime) / 1000;
      System.out.println("Ran for " + diff + "s");
    }
  }

  private void train() throws IOException {
    scTrainer = new App.SCTrainer(
        "adapt",
        "adopt",
        ROOT_PATH + confusionPair + ".train",
        ROOT_PATH + "testOutput-test.model.backup3");
    scTrainer.train();
    scTrainer.write();
  }

  private void runTest() throws IOException {
    scTester = new App.SCTester(
        ROOT_PATH + confusionPair + ".test",
        ROOT_PATH + confusionPair + ".answer",
        ROOT_PATH + "testOutput-test.model.backup3"
    );
    scTester.runTest();
  }


  @Before
  public void setUp() throws Exception {

  }

  @After
  public void tearDown() throws Exception {

  }
}