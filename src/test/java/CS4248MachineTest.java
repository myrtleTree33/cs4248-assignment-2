import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by joel on 10/14/15.
 */
public class CS4248MachineTest {

  private final String ROOT_PATH = "src/test/resources/";

  @Test
  public void testTrain() throws Exception {
    CS4248Machine machine = new CS4248Machine();
    machine.train(ROOT_PATH + "adapt_adopt.train");

  }
}