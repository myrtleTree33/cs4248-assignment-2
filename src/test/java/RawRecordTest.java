import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.util.*;

/**
 * Created by joel on 10/14/15.
 */
public class RawRecordTest {

  private final String ROOT_PATH = "src/test/resources/";
  private List<RawRecord> records;

  @Before
  public void setUp() throws Exception {
    records = RawRecord.parse(ROOT_PATH + "adapt_adopt.train");

  }

  @After
  public void tearDown() throws Exception {

  }

  @Ignore
  @Test
  public void testParse() throws Exception {
    RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testRemoveStopWords() throws Exception {
    Set<String> stopWords = new HashSet<>();
    stopWords.add("the");
    RawRecord.removeTokens(records, stopWords);
    RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testRemoveStopWordsFromFile() throws Exception {
    Set<String> stopWords = Util.loadStopWords(ROOT_PATH + "stopwd.txt");
    RawRecord.removeTokens(records, stopWords);
    RawRecord.removePunctuation(records);
    RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testMakeVocabulary() throws Exception {
    RawRecord.removePunctuation(records);
    List<RawRecord> recordsA = RawRecord.splice(records, "adapt");
    HashMap<String, Integer> vocabulary = RawRecord.makeVocabulary(recordsA);
    System.out.println("Size=" + vocabulary.size());
    StringBuffer sb = new StringBuffer();
    Iterator it = vocabulary.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<String, Integer> curr = (Map.Entry<String, Integer>) it.next();
      sb.append(curr.getKey() + "," + curr.getValue() + "\n");
    }
    System.out.println(sb.toString());
  }

  @Ignore
  @Test
  public void testParseAnswers() throws Exception {
    records = RawRecord.parse(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer");
    RawRecord.print(records);
  }
}