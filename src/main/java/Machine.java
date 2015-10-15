import java.io.FileNotFoundException;
import java.util.Map;

/**
 * Created by joel on 10/14/15.
 */
public interface Machine {

  void train(String datasetFileName, String stopWordsFileName) throws FileNotFoundException;

  Map<String, PredictionResult> test(String questionFilename, String answerFilename) throws FileNotFoundException;
}
