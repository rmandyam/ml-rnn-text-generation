package ml.examples.rnn;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Date;
import java.util.LinkedHashSet;
import java.util.Random;

/**
 *
 */
public class SenGenBasicRnnExample {

    // RNN dimensions
    public static final int HIDDEN_LAYER_WIDTH = 100  ; // 50;
    public static final int HIDDEN_LAYER_COUNT = 2;
    public static final Random r = new Random(7894);

    //public static final int vectorSize = 100;   //Size of the word vectors.
    //public static final int vocabSize = 1000 ;

    public static void main(String[] args)  throws Exception {

        //int lstmLayerSize = 200;
        int tbpttLength = 50;

        int batchSize = 50 ; //300 ; //300 ; //100 ; // 1 ; //64;     //Number of examples in each minibatch
        int vectorSize = 100;   //Size of the word vectors.
        int nEpochs = 50 ; // 5 ; // 1;        //Number of epochs (full passes of training data) to train on
        int truncateSentencesToLength = 150 ;  //Truncate sentences with length (# words) greater than this
        int nSamplesToGenerate = 25;					//Number of samples to generate after each training epoch
        int nWordsToSample = 20;				//Length of each sample to generate
        String generationInitialization = null;		//Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        Date d = new Date() ;
        System.out.println("Starting Process at ... " + d.toString() ) ;
        //System.out.println("\n\n") ;

        //DataSetIterators for training and testing respectively
        SenGenExampleIterator train = new SenGenExampleIterator( batchSize, vectorSize,  truncateSentencesToLength, true);
        int nOut = train.totalOutcomes() ;

        // some common parameters
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.iterations(10);
        builder.learningRate(0.001);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.seed(123);
        builder.biasInit(0);
        builder.miniBatch(false);
        builder.updater(Updater.RMSPROP);
        builder.weightInit(WeightInit.XAVIER);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        // first difference, for rnns we need to use GravesLSTM.Builder
        for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
            GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? vectorSize : HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
            // adopted activation function from GravesLSTMCharModellingExample
            // seems to work well with RNNs
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        // we need to use RnnOutputLayer for our RNN
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
        // softmax normalizes the output neurons, the sum of all outputs is 1
        // this is required for our sampleFromDistribution-function
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
        outputLayerBuilder.nOut(nOut);
        listBuilder.layer(HIDDEN_LAYER_COUNT, outputLayerBuilder.build());

        // finish builder
        listBuilder.pretrain(false);
        listBuilder.backprop(true);

        // create network
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        //Set up listener that prints the error score for the network every x iterations
        net.setListeners(new ScoreIterationListener(50));

        //Print the hyper parameters
        System.out.println("\n\n Hyper Parameters...") ;
        System.out.println(" Word Vector Size .. " + vectorSize) ;
        //System.out.println(" LSTM Layer Size .. " + lstmLayerSize) ;
        System.out.println(" Batch Size .. " + batchSize) ;
        System.out.println(" Epochs .. " + nEpochs) ;
        System.out.println(" Learning Rate 0.1, rmsDecay 0.95, l2 0.001 ") ;
        System.out.println(" Truncate BPTT Length .. " + tbpttLength) ;
        System.out.println(" Truncate Sentences to Length .. " + truncateSentencesToLength) ;
        System.out.println(" Samples to Generate .. " + nSamplesToGenerate) ;
        System.out.println(" Words per Sample .. " + nWordsToSample) ;

        //Print the  number of parameters in the network (and for each layer)

        System.out.println("Model Parameters...") ;
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for( int i=0; i<layers.length; i++ ){
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);


        System.out.println("\n\n ***************** Starting training ******************");
        for (int i = 0; i < nEpochs; i++) {
            d = new Date() ;
            System.out.println("\n\n*************** Starting Epoch " + i + " at " + d.toString()+ " *********************") ;
            net.fit(train);
            train.reset();
            d = new Date() ;
            System.out.println("Epoch " + i + " completed at ..." + d.toString());


            System.out.println("Sampling Words from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");

            for (int j= 0 ; j < nSamplesToGenerate ; j ++ ) {
                String sample = sampleWordsFromNetwork(generationInitialization, net, train, rng, nWordsToSample, 1);
                System.out.println("\n----- Sample " + j + " -----");
                System.out.println(sample) ;
            }


        }




        d = new Date() ;
        System.out.println("\n\nExiting Process at ... " + d.toString() ) ;

    } // end main


    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param numOfWordsToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     */
    //numOfWordsToSample : number of words per sentence
    //numSamples : number of sentences to generate
    private static String sampleWordsFromNetwork(String initialization, MultiLayerNetwork net,
                                                 SenGenExampleIterator iter, Random rng, int numOfWordsToSample, int numSamples ){

        //Set up initialization. If no initialization: use <SENTENCE_START>
        if( initialization == null ){
            //initialization = String.valueOf(iter.getRandomCharacter());
            //NOTE : DefaultTokenizerFactory converts all tokens to lower case
            initialization = "sentence_start" ; //start with begin of sentence
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), 1);
        INDArray vector = iter.getMyWordVectors().getWordVectorMatrix(initialization);

        for( int j=0; j<numSamples; j++ ){
            initializationInput.put(new INDArrayIndex[]{NDArrayIndex.point(j), NDArrayIndex.all(),NDArrayIndex.point(0)}, vector);
        }


        StringBuilder sb = new StringBuilder(initialization) ;
        //net.rnnClearPreviousState(); 
        INDArray output = net.rnnTimeStep(initializationInput);
        int timeSeriesLength = output.size(2);
        INDArray lastTimeStepProbabilities = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength-1));

        for( int i=0; i<numOfWordsToSample; i++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns(), 1);
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            double[] outputProbDistribution = new double[iter.totalOutcomes()];
            for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(j);
            int sampledWordIdx = sampleFromDistribution(outputProbDistribution,rng);

            String sampledWord = iter.getMyWordVectors().vocab().wordAtIndex(sampledWordIdx) ;
            INDArray nextVector = iter.getMyWordVectors().getWordVectorMatrix(sampledWord);

            nextInput.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(),NDArrayIndex.point(0)}, vector); //Prepare next time step input
            //nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
            //sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)


            sb.append(" " + sampledWord);	//Add sampled word to StringBuilder (human readable output)
            if(sampledWord.equals("sentence_end"))
                break ;

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
            lastTimeStepProbabilities = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength-1));
        }

        return (sb.toString()) ;


    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }



} // end class
