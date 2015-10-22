CS4248 Assignment 2
===============================

> TONG Haowen Joel
> me <at> joeltong <dot> org
> National University of Singapore
> CS4248
> Oct 23, 2015

## About

This assignment implements a Logistic Regression classifier to disambiguate a confusable word pair.

Details of implementation is found in the accompanying report, in this folder.

----------------------------------------------------------------------------------------

## Production

Both trainer and tester programs are found in `final/`.

### Training

    java sctrain <word1> <word2> <train_file> <model_file>

### Testing

    java sctest <word1> <word2> <test_file> <model_file> <answer_file>

----------------------------------------------------------------------------------------


## Methodology

### Training

- Texts are tokenized, punctuation stripped.
- Stop words are analyzed for distinctivensss.  Undistinct stop words are removed from tokens.
- Features are built.  The features used are:
    - Collocations around the surrounding confusable word
        - For collocations, N-grams are used.
    - Individual word tokens
- Feature reduction is performed, by removing sparse features.
- Trained model is output.


### Testing

- Feature recognition similar to above is performed.
- However, feature mining and arrangement is not needed.
- For each feature in model, tester searches for articles in test record.
- Resultant label is computed from earlier model weights and output.

----------------------------------------------------------------------------------------


## Classes of interest

- CS4248Machine
    - Packages classifier, training and testing into convenient package
- LogisticRegressionClassifier
    - Classifier used.
- Model
    - Model output
- PredictionResult
    - Used to show per-label accuracy results.
- RawRecord
    - Raw Record storing high-level record info and labels
- Record
    - Low-level vectorial equivalent of RawRecord
- SCTester
    - Top-level testing app
- SCTrainer
    - Top-level training app
- StopWords
    - Stop words used
- Util
    - Util library
- Vector
    - Used to compute low-level vectors.

- In addition, CS4248MachineTest.java contains a Hyper-parameter optimization and CSV generator.
- Other tests are written and commented out as well, in Junit style.

----------------------------------------------------------------------------------------


## Development

Development uses in-built Gradle wrapper and JDK 1.7.

### Compiling with sunfire

In the directory, run the following:

    $ gradlew tasks

To generate tasks.

Then run the respective tasks, for example `test`:

    $ gradlew test

----------------------------------------------------------------------------------------


## Excel Files

Excel files detailing graphs and tables used in permutation are as attached in `docs/`.
