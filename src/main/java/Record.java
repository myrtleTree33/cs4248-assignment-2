/**
 * Created by joel on 10/13/15.
 */
public class Record {

  private int label;      // Y
  private Vector vectors; // x

  public Record(int label, Vector vectors) {
    this.label = label;
    this.vectors = vectors;
  }

  public Record() {
  }

  public int getLabel() {
    return label;
  }

  public void setLabel(int label) {
    this.label = label;
  }

  public Vector getVectors() {
    return vectors;
  }

  public void setVectors(Vector vectors) {
    this.vectors = vectors;
  }
}
