import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by joel on 10/21/15.
 */
//@Ignore
public class SCTesterTest {

  private final String ROOT_PATH = "src/test/resources/";
  private SCTester scTester;

  @Before
  public void setUp() throws Exception {

  }

  @After
  public void tearDown() throws Exception {

  }

  @Test
  public void testRunTest() throws Exception {
    scTester = new SCTester(
        ROOT_PATH + "adapt_adopt.test",
        ROOT_PATH + "adapt_adopt.answer",
        ROOT_PATH + "testOutput-test.model"
    );
    scTester.runTest();
  }

}