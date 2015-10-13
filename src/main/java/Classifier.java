import java.util.List;

/**
 * Created by joel on 10/13/15.
 */
public interface Classifier {

  public void loadDataset(List<Record> records);
  public void train();
  public double test();
}
