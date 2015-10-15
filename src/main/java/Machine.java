import java.io.FileNotFoundException;
import java.util.Map;

/**
 * Created by joel on 10/14/15.
 */
public interface Machine {

//  void train(String datasetFileName) throws FileNotFoundException;

  void train(String datasetFileName, String stopWordsFileName, int minThreshold, float learningMinThreshold, double alpha) throws FileNotFoundException;

  Map<String, PredictionResult> test(String questionFilename, String answerFilename) throws FileNotFoundException;
}
