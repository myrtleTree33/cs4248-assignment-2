import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Created by joel on 10/14/15.
 */
public class RawRecord {

  private String label;
  private String id;
  private List<String> tokens;
  private int idx;                // where label was found

  public RawRecord(String label, String id, String text) {
    this.label = label;
    this.id = id;
    this.tokens = tokenize(text);
  }

  public RawRecord(String label, String id, List<String> tokens) {
    this.label = label;
    this.id = id;
    this.tokens = tokens;
    this.idx = findIdx();
  }

  private static List<String> tokenize(String text) {
    return new ArrayList<>(Arrays.asList(text.split("[\\s']")));
  }

  public static List<RawRecord> parse(String filename) throws FileNotFoundException {
    List<RawRecord> result = new ArrayList<>();
    Scanner scanner = new Scanner(new File(filename));
    while (scanner.hasNextLine()) {
      List<String> currTokens = tokenize(scanner.nextLine());
      String currId = currTokens.remove(0);
      String foundLabel = getLabelAndClean(currTokens);
      result.add(new RawRecord(foundLabel, currId, currTokens));
    }
    return result;
  }

  /**
   * Finds the index of the label.
   *
   * @return
   */
  private int findIdx() {
    for (int i = 0; i < tokens.size(); i++) {
      String token = tokens.get(i);
      if (token.equals(label)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Cleans up string and formats it to nice specification.
   *
   * @param currTokens
   * @return
   */
  private static String getLabelAndClean(List<String> currTokens) {
    String foundLabel = "";
    List<String> detected = new ArrayList<>(2);
    for (int i = 0; i < currTokens.size(); i++) {
      String curr = currTokens.get(i).toLowerCase().replaceAll("[\"\':;,./?!@-_]","");
      currTokens.set(i, curr); // to lower case as well
      if (curr.equals(">>")) {
        foundLabel = currTokens.get(i + 1); // the label
//        detected.add(currTokens.get(i)); // >>
        detected.add(currTokens.get(i + 1)); // remove target word
//        detected.add(currTokens.get(i + 2)); // <<
      }
    }
    currTokens.removeAll(detected);
    return foundLabel;
  }

  public static void removeTokens(List<RawRecord> rawRecords, Set<String> stopWords) {
    for (RawRecord record : rawRecords) {
      record.removeTokens(stopWords);
    }
  }

  public static void removePunctuation(List<RawRecord> rawRecords) {
    for (RawRecord record : rawRecords) {
      record.removePunctuation();
    }
  }

  public static HashMap<String, Integer> makeVocabulary(List<RawRecord> records) {
    HashMap<String, Integer> vocabulary = new HashMap<>();
    for (RawRecord record : records) {
      List<String> tokens = record.getTokens();
      for (String token : tokens) {
        if (vocabulary.containsKey(token)) {
          vocabulary.put(token, vocabulary.get(token) + 1);
        } else {
          vocabulary.put(token, 1);
        }
      }
    }
    return vocabulary;
  }

  public static List<String> makeVocabularyList(List<RawRecord> records) {
    HashMap<String, Integer> vocabulary = makeVocabulary(records);
    List<String> result = new ArrayList<>();
    for (String key : vocabulary.keySet()) {
      result.add(key);
    }
    return result;
  }

  public void removePunctuation() {
    Set<String> punctuation = new HashSet<>(Arrays.asList(new String[]{
        "!",
        "@",
        "\'",
        "\"",
        ",",
        ".",
        ":",
        ";",
        "(",
        ")",
        "?",
        "!",
        "-",
        "--",
        "[",
        "]"
    }));
    removeTokens(punctuation);
  }

  public void removeTokens(Set<String> tokens) {
    List<String> found = new ArrayList();
    for (String token : this.tokens) {
      if (tokens.contains(token)) {
        found.add(token);
      }
    }
    this.tokens.removeAll(found);
  }

  public static List<RawRecord> splice(List<RawRecord> records, String label) {
    List<RawRecord> result = new ArrayList<>(records.size());
    for (RawRecord record : records) {
      if (record.getLabel().equals(label)) {
        result.add(record);
      }
    }
    return result;
  }

  public static void print(List<RawRecord> rawRecords) {
    StringBuffer sb = new StringBuffer();
    for (RawRecord record : rawRecords) {
      sb.append(record.toString() + "\n");
    }
    System.out.println("--- Record Dump ---");
    System.out.println(sb.toString());
    System.out.println("--- /Record Dump ---");
  }

  @Override
  public String toString() {
    return "RawRecord{" +
        "label='" + label + '\'' +
        ", id='" + id + '\'' +
        ", tokens=" + tokens +
        ", idx=" + idx +
        '}';
  }

  public String getLabel() {
    return label;
  }

  public void setLabel(String label) {
    this.label = label;
  }

  public String getId() {
    return id;
  }

  public void setId(String id) {
    this.id = id;
  }

  public List<String> getTokens() {
    return tokens;
  }

  public void setTokens(List<String> tokens) {
    this.tokens = tokens;
  }
}
