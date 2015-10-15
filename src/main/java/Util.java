import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

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


  /**
   * Function to select distinct stop words, based on analysis of 2 individual corpuses.
   * If there are huge differences greater than minThreshold, then these words are used.
   * @param stopWords
   * @param corpus1
   * @param corpus2
   * @param minThreshold
   * @return
   */
  public static Set<String> selectDistinctStopWords(Set<String> stopWords, List<RawRecord> corpus1, List<RawRecord> corpus2, int minThreshold) {

    Map<String, Integer> distrCorpus1 = RawRecord.makeVocabulary(corpus1);
    Map<String, Integer> distrCorpus2 = RawRecord.makeVocabulary(corpus2);
    Set<String> result = new HashSet<>();

    for (String word : stopWords) {
      int count1, count2, diff;
      count1 = count2 = diff = 0;
      if (distrCorpus1.containsKey(word)) {
        count1 = distrCorpus1.get(word);
      }
      if (distrCorpus2.containsKey(word)) {
        count2 = distrCorpus2.get(word);
      }
      diff = Math.abs(count1 - count2);
      if (diff >= minThreshold) {
        result.add(word);
      }
    }

    return result;
  }

  /**
   * @deprecated
   * @param tokens
   * @param start
   * @param end
   * @param ref
   * @return
   */
  public static List<String> getNGrams(List<String> tokens, int start, int end, int ref) {
    List<String> collocations = new ArrayList<>();
    int max = ref + end;
    int min = ref - start;
    if (max < tokens.size()) {
      max = tokens.size() - end;
    }
    if (min > 0) {
      min = start;
    }
    for (int i = min; i < max; i++) {
      String collocation = Util.join(" ", tokens.subList(i - start, i + end));
      collocations.add(collocation);
    }
    return collocations;
  }

  public static String getCollocation(List<String> tokens, int start, int end, int ref) {
    int max = ref + end;
    int min = ref + start;
    if (max >= tokens.size()) {
      max = tokens.size() - 1;
    }
    if (min < 0) {
      min = 0;
    }
    if (min >= tokens.size()) {
      min = tokens.size() - 1;
    }
    if (max < 0) {
      max = 0;
    }
    String collocation = Util.join(" ", tokens.subList(min, max));
    return collocation;
  }

  public static String join(String separator, List<String> tokens) {
    if (tokens.size() < 1) return "";
    StringBuffer sb = new StringBuffer();
    sb.append(tokens.get(0));
    for (int i = 1; i < tokens.size(); i++) {
      sb.append(" ");
      sb.append(tokens.get(i));
    }
    return sb.toString();
  }


}
