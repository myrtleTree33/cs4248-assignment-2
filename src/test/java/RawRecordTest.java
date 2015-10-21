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
  private List<App.RawRecord> records;

  @Before
  public void setUp() throws Exception {
    records = App.RawRecord.parse(ROOT_PATH + "adapt_adopt.train");

  }

  @After
  public void tearDown() throws Exception {

  }

  @Ignore
  @Test
  public void testParse() throws Exception {
    App.RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testRemoveStopWords() throws Exception {
    Set<String> stopWords = new HashSet<>();
    stopWords.add("the");
    App.RawRecord.removeTokens(records, stopWords);
    App.RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testRemoveStopWordsFromFile() throws Exception {
    Set<String> stopWords = App.Util.loadStopWords(ROOT_PATH + "stopwd.txt");
    App.RawRecord.removeTokens(records, stopWords);
    App.RawRecord.removePunctuation(records);
    App.RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testMakeVocabulary() throws Exception {
    App.RawRecord.removePunctuation(records);
    List<App.RawRecord> recordsA = App.RawRecord.splice(records, "adapt");
    HashMap<String, Integer> vocabulary = App.RawRecord.makeVocabulary(recordsA);
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
    records = App.RawRecord.parse(ROOT_PATH + "adapt_adopt.test", ROOT_PATH + "adapt_adopt.answer");
    App.RawRecord.print(records);
  }

  @Ignore
  @Test
  public void testGetAllCollocations() throws Exception {
    Set<String> collocationsNGram = App.RawRecord.getAllCollocationsAsNGram(3, records, -2, 2);
    for (String collocation : collocationsNGram) {
      System.out.println(collocation);
    }
  }
}