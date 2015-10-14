import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

/**
 * Created by joel on 10/14/15.
 */
public class Util {
  public static Set<String> loadStopWords(String filename) throws FileNotFoundException {
    Set<String> result = new HashSet<>();
    Scanner scanner = new Scanner(new File(filename));
    while (scanner.hasNextLine()) {
      String word = scanner.nextLine();
      result.add(word);
    }
    return result;
  }
}
