import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by joel on 10/14/15.
 */
public class LogisticRegressionClassifierTest {

  List<App.Record> dataset;
  App.LogisticRegressionClassifier classifier;

  @Before
  public void setUp() throws Exception {
    dataset = new ArrayList<>();
    dataset.add(new App.Record(1, new App.Vector(new double[]{0.9, 0.9, 0.1})));
    dataset.add(new App.Record(1, new App.Vector(new double[]{0.8, 0.8, 0.2})));
    dataset.add(new App.Record(0, new App.Vector(new double[]{0.1, 0.1, 0.9})));
    dataset.add(new App.Record(0, new App.Vector(new double[]{0.15, 0.12, 0.85})));

    classifier = new App.LogisticRegressionClassifier();
  }

  @After
  public void tearDown() throws Exception {

  }

  @Test
  public void testTrain() throws Exception {
    classifier.loadDataset(dataset);
    App.Model model = classifier.train(5, 2, 0.8, 0.00000000001, App.LogisticRegressionClassifier.NO_TIMEOUT);
    for (int i = 0; i < dataset.size(); i++) {
      System.out.print("Label=" + dataset.get(i).getLabel() + " ");
      System.out.println(model.evaluate(dataset.get(i).getVectors()) + "");
    }

    // assert following labels are classified correctly
    assertEquals(1, model.evaluate(new App.Vector(new double[]{0.9, 0.8, 0.1})));
    assertEquals(0, model.evaluate(new App.Vector(new double[]{0.1, 0.3, 0.8})));

  }
}