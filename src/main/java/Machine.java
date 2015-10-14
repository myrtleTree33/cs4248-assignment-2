import java.io.FileNotFoundException;

/**
 * Created by joel on 10/14/15.
 */
public interface Machine {

  void train(String datasetFileName) throws FileNotFoundException;
  void test(String testFileName);

}
