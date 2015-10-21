import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Created by joel on 10/20/15.
 */
@Ignore
public class SCTrainerTest {

  private final String ROOT_PATH = "src/test/resources/";
  private SCTrainer scTrainer;

  @Test
  public void testTrain() throws Exception {
    scTrainer = new SCTrainer(
        "adapt",
        "adopt",
        ROOT_PATH + "adapt_adopt.train",
        ROOT_PATH + "testOutput-test.model.backup");
    scTrainer.train();
    scTrainer.write();
  }

  @Before
  public void setUp() throws Exception {

  }

  @After
  public void tearDown() throws Exception {

  }
}