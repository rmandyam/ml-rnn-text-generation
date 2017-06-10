package ml.examples.rnn;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.*;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class SenGenExampleIterator implements DataSetIterator {

    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private  int lineCount ;

    private final SentenceIterator iter ;

    private int cursor = 0;

    private final TokenizerFactory tokenizerFactory;




    public SenGenExampleIterator(int batchSize, int vectorSize, int truncateLength, boolean train) throws IOException {
        this.batchSize = batchSize;

        this.truncateLength = truncateLength;

        this.vectorSize = vectorSize ;

        // Split on white spaces in the line to get words
        tokenizerFactory = new DefaultTokenizerFactory();
        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());


        // Gets Path to Text file
        String filePath = new ClassPathResource("input.txt").getFile().getAbsolutePath();

        System.out.println("..Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        iter = new BasicLineIterator(filePath);


        System.out.println("Building model....");
        //NOTE : check params below
        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(1) //(5)
            .iterations(1)
            .layerSize(vectorSize)   //word vector size
            .seed(42)
            .windowSize(5)
            .iterate(iter)
            .tokenizerFactory(tokenizerFactory)
            .build();

        System.out.println("Fitting Word2Vec model....");
        vec.fit();

        this.wordVectors = vec ;
        //this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        System.out.println("wordVectors ..num of words.. ...." + wordVectors.vocab().numWords() );
        System.out.println("wordVectors ..vectorSize.." + vectorSize) ;
        System.out.println("wordVectors..deduced vectorSize.." + wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length );

        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        System.out.println("Closest Words:");

        Collection<String> lst2 = vec.wordsNearest("rnn", 10);
        System.out.println("10 Words closest to 'rnn': " + lst2);

        Collection<String> lst3 = vec.wordsNearest("gradient", 10);
        System.out.println("10 Words closest to 'gradient': " + lst3);

        Collection<String> lst4 = vec.wordsNearest("learn", 10);
        System.out.println("10 Words closest to 'learn': " + lst4);

        //Check if SENTENCE_START  and SENTENCE_END  tokens exist
        //NOTE : All tokens are converted to lower case by DefaultTokenizerFactory
        if (wordVectors.hasWord("sentence_start"))
            System.out.println("WordVectors has the word sentence_start") ;
        else
            System.out.println("WordVectors DOES NOT HAVE the word sentence_start") ;

        if (wordVectors.hasWord("sentence_end"))
            System.out.println("WordVectors has the word sentence_end") ;
        else
            System.out.println("WordVectors DOES NOT HAVE the word sentence_end") ;

        //Get line count
        iter.reset() ;
        lineCount = 0 ;
        while (iter.hasNext()){
            String s = iter.nextSentence() ;
            //System.out.println(s) ;
            lineCount++ ;
        }
        //reset for fresh use
        iter.reset() ;

        System.out.println("Line Count : " + lineCount) ;


    }

    @Override
    public DataSet next(int num) {
        //System.out.println("\nDataSet..called with num = " + num + "...Cursor is.." + cursor) ;
        if (cursor >= lineCount) throw new NoSuchElementException();
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {

        //First : Load sentences to String
        List<String> lines = new ArrayList<>(num);  //this is a subset of all lines, i.e the minibatch
        boolean[] positive = new boolean[num];
        String l ;
        for( int i=0; i<num && cursor<totalExamples(); i++ ){

            while(iter.hasNext())
            {
                l = iter.nextSentence() ;
                if (l.length() > 0) // i.e not a blank line
                {
                    lines.add(l);
                    break ;
                }else
                {
                    //System.out.println("Blank line..skipping..") ;
                    //NOTE : Cursor still has to be incremented..
                    cursor++ ;
                }
            }

            cursor++;
        }

        //System.out.println("nextDataSet...lines array..") ;
        //System.out.println(lines) ;
        System.out.println("Mini Batch Cursor is at .. " + cursor) ;


        //Second : tokenize sentences and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(lines.size());
        int maxLength = 0;
        //iter.reset();
        //while (iter.hasNext())
        for (String s : lines) {
            //String s = iter.nextSentence() ;
            //System.out.println("Input Sentence: " + s);
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            //System.out.println("Tokenized Sentence : " + tokens) ;
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            //System.out.println("Filtered Tokens : " + tokensFiltered) ;
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());

        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        //System.out.println("Truncate Len : " + truncateLength) ;
        //System.out.println("Max Len  : " + maxLength) ;

        //Create data for training
        //Here: we have lineCount  examples of varying lengths

        INDArray features = Nd4j.create(lines.size(), vectorSize, maxLength);

        INDArray labels = Nd4j.create(lines.size(), totalOutcomes(), maxLength);


        //Because we are dealing with lines of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(lines.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(lines.size(), maxLength);
        int[] temp = new int[2];

        for( int i=0; i<lines.size() ; i++ ) {
            List<String> tokens = allTokens.get(i);

            //List<String> feature_tokens = allTokens.get(i) ;  this just returns a pointer..we want a real copy..
            List<String> feature_tokens = new ArrayList<>(tokens) ;
            //NOTE : DefaultTokenizerFactory converts all tokens to lower case
            feature_tokens.remove("sentence_end") ;

            //List<String> label_tokens = allTokens.get(i); this just returns a pointer..we want a real copy..
            List<String> label_tokens = new ArrayList<>(tokens) ;
            //NOTE : DefaultTokenizerFactory converts all tokens to lower case
            label_tokens.remove("sentence_start") ;


            temp[0] = i;

            //System.out.println(tokens);
            if (tokens.size() > 1)
                //System.out.println("token 0 : " + tokens.get(0) + " last token : " + tokens.get(tokens.size()-1)) ;

            /*System.out.println("\nLine.." + i) ;
            System.out.println("Tokens : " + tokens) ;
            System.out.println("Feature Tokens : " + feature_tokens) ;
            System.out.println("Label Tokens : " + label_tokens) ;*/



            //FEATURES
            //Get word vectors for each word in sentence, and put them in the training data
            for (int j = 0; j < feature_tokens.size() && j < maxLength; j++) {
                String token = feature_tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }


            //LABELS
            //This is the hot-vector over all possible outcomes i.e the size of vocab..
            for (int j = 0; j < label_tokens.size() && j < maxLength; j++) {
                String token = label_tokens.get(j);
                //INDArray vector = wordVectors.getWordVectorMatrix(token);
                //labels.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                int idx = wordVectors.vocab().indexOf(token) ;
                int lastIdx = Math.min(label_tokens.size(), maxLength);
                labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
                labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);

                temp[1] = j;
                labelsMask.putScalar(temp, 1.0);



            }


        }

        return new DataSet(features,labels,featuresMask,labelsMask);
        //return new DataSet(features,labels);
    }



    @Override
    public int totalExamples() {

        return lineCount ;
        //return 5 ;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        //all possible words in dictionary

        return wordVectors.vocab().numWords() ;
        //return 100 ;
        //return 90 ;
    }

    @Override
    public void reset() {
        cursor = 0;
        iter.reset();  //reset it for next epoch
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }


    public WordVectors getMyWordVectors()
    {
        return wordVectors ;
    }



} //end class
