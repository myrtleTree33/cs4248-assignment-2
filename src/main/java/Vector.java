import java.util.Arrays;

/**
 * Class for handling Vectors.
 *
 * Created by joel on 10/13/15.
 */
public class Vector {

  protected double[] vectors;

  /**
   * Initializer from varargs
   *
   * @param vectors
   */
  public Vector(Double... vectors) {
    this.vectors = new double[vectors.length];
    for (int i = 0; i < vectors.length; i++) {
      this.vectors[i] = vectors[i];
    }
  }

  /**
   * Initializer.
   *
   * @param vectors
   */
  public Vector(double[] vectors) {
    this.vectors = vectors;
  }

  public Vector clone() {
    double[] tempVec = Arrays.copyOf(this.vectors, this.vectors.length);
    return new Vector(tempVec);
  }

  /**
   * Gets at index.
   *
   * @param idx
   * @return
   */
  public double get(int idx) {
    return vectors[idx];
  }

  /**
   * Sets at index.
   *
   * @param idx
   * @param val
   */
  public void set(int idx, double val) {
    vectors[idx] = val;
  }

  /**
   * Returns size of the vector.
   *
   * @return
   */
  public int size() {
    return vectors.length;
  }

  /**
   * Calculates the dot product between 2 vectors
   *
   * @param b
   * @return
   */
  public double dot(Vector b) {
    double result = 0;
    for (int i = 0; i < this.size(); i++) {
      result += this.vectors[i] * b.vectors[i];
    }
    return result;
  }

  /**
   * Initializes a vector accordingly.
   *
   * @param size
   * @param initVal
   * @return
   */
  public static Vector init(int size, double initVal) {
    double[] vec = new double[size];
    for (int i = 0; i < size; i++) {
      vec[i] = initVal + 0;
    }
    return new Vector(vec);
  }

  /**
   * Returns a zero vector.
   *
   * @param size
   * @return
   */
  public static Vector zero(int size) {
    return init(size, 0);
  }

  /**
   * Adds another vector.
   *
   * @param b
   * @return
   */
  public Vector add(Vector b) {
    Vector result = Vector.init(b.size(), 0);
    for (int i = 0; i < this.size(); i++) {
      result.vectors[i] = this.vectors[i] + b.vectors[i];
    }
    return result;
  }

  /**
   * Subtracts another vector.
   *
   * @param b
   * @return
   */
  public Vector subtract(Vector b) {
    Vector result = Vector.init(b.size(), 0);
    for (int i = 0; i < this.size(); i++) {
      result.vectors[i] = this.vectors[i] - b.vectors[i];
    }
    return result;
  }

  /**
   * Calculates the normal of a vector
   *
   * @return
   */
  public double norm() {
    double result = 0;
    for (double vec : this.vectors) {
      result += vec * vec;
    }
    return Math.sqrt(result);
  }

  @Override
  public String toString() {
    return "Vector{" +
        "vectors=" + Arrays.toString(vectors) +
        '}';
  }
}
