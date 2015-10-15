import org.junit.Test;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  @Test
  public void testTrain() throws Exception {
    CS4248Machine machine = new CS4248Machine();
    machine.train(ROOT_PATH + "adapt_adopt.train", ROOT_PATH + "stopwd.txt", 20, 5, 0.2);
    PredictionResult.printResults(machine.test(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer"));
  }


}