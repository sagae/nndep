#include <ctime>
#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string/join.hpp>
# include <boost/interprocess/managed_shared_memory.hpp>
# include <boost/interprocess/allocators/allocator.hpp>
# include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/math/distributions/binomial.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "maybe_omp.h"
#include <tclap/CmdLine.h>

#include "model.h"
#include "propagator.h"
#include "param.h"
#include "neuralClasses.h"
#include "graphClasses.h"
#include "util.h"
#include "multinomial.h"
//#include "gradientCheck.h"

//#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace TCLAP;
using namespace Eigen;
using namespace boost;
using namespace boost::random;

using namespace nplm;

namespace ip = boost::interprocess;
typedef unordered_map<Matrix<int,Dynamic,1>, double> vector_map;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;


typedef long long int data_size_t; // training data can easily exceed 2G instances

int main(int argc, char** argv)
{ 
    ios::sync_with_stdio(false);
    bool use_mmap_file, randomize;
    param myParam;
    try {
      // program options //
      CmdLine cmd("Trains a two-layer neural probabilistic language model.", ' ' , "0.2\n",
          "Changes since V0.1: Addition of biases");

      // The options are printed in reverse order

      ValueArg<string> unigram_probs_file("", "unigram_probs_file", "Unigram model (deprecated and ignored)." , false, "", "string", cmd);

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);

      ValueArg<double> final_momentum("", "final_momentum", "Final value of momentum. Default: 0.9.", false, 0.9, "double", cmd);
      ValueArg<double> initial_momentum("", "initial_momentum", "Initial value of momentum. Default: 0.9.", false, 0.9, "double", cmd);
      ValueArg<bool> use_momentum("", "use_momentum", "Use momentum (hidden layer weights only). 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

      ValueArg<double> normalization_init("", "normalization_init", "Initial normalization parameter. Default: 0.", false, 0.0, "double", cmd);
      ValueArg<bool> normalization("", "normalization", "Learn individual normalization factors during training. 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

      ValueArg<bool> mmap_file("", "mmap_file", "Use memory mapped files. This is useful if the entire data cannot fit in memory. prepareNeuralLM can generate memory mapped files", false, 0, "bool", cmd);

      ValueArg<bool> arg_randomize("", "randomize", "Randomize training instances for better training. 1 = yes, 0 = no. Default: 1.", false, true, "bool", cmd);

      ValueArg<int> num_noise_samples("", "num_noise_samples", "Number of noise samples for noise-contrastive estimation. Default: 100.", false, 100, "int", cmd);

      ValueArg<double> L2_reg("", "L2_reg", "L2 regularization strength (hidden layer weights only). Default: 0.", false, 0.0, "double", cmd);

      ValueArg<double> learning_rate("", "learning_rate", "Learning rate for stochastic gradient ascent. Default: 1.", false, 1., "double", cmd);

      ValueArg<double> conditioning_constant("", "conditioning_constant", "Constant to condition the RMS of the expected square of the gradient in ADADELTA. Default: 10E-3.", false, 10E-3, "double", cmd);

      ValueArg<double> decay("", "decay", "Decay for ADADELTA. Default: 0.95", false, 0.95, "double", cmd);
      ValueArg<double> adagrad_epsilon("", "adagrad_epsilon", "Constant to initialize the L2 squared norm of the gradients with.\
          Default: 10E-3", false, 10E-3, "double", cmd);
      ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "Minibatch size for validation. Default: 64.", false, 64, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 1000.", false, 1000, "int", cmd);

      ValueArg<int> num_epochs("", "num_epochs", "Number of epochs. Default: 10.", false, 10, "int", cmd);

      ValueArg<double> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.01", false, 0.01, "double", cmd);
      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0.", false, 0, "bool", cmd);

      ValueArg<string> loss_function("", "loss_function", "Loss function (log, nce). Default: nce.", false, "nce", "string", cmd);
      ValueArg<string> activation_function("", "activation_function", "Activation function (identity, rectifier, tanh, hardtanh). Default: rectifier.", false, "rectifier", "string", cmd);
      ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 100.", false, 100, "int", cmd);
	  ValueArg<int> num_second_hidden("", "num_second_hidden", "Number of hidden nodes in the second hidden layer. Default: 100.", false, 100, "int", cmd);

      ValueArg<bool> share_embeddings("", "share_embeddings", "Share input and output embeddings. 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);
      ValueArg<int> output_embedding_dimension("", "output_embedding_dimension", "Number of output embedding dimensions. Default: 50.", false, 50, "int", cmd);
      ValueArg<int> input_embedding_dimension("", "input_embedding_dimension", "Number of input embedding dimensions. Default: 50.", false, 50, "int", cmd);
      ValueArg<int> embedding_dimension("", "embedding_dimension", "Number of input and output embedding dimensions. Default: none.", false, -1, "int", cmd);
	  ValueArg<string> input_embeddings_file("","input_embeddings_file", "Read the input embeddings from the specified file. Default: none", false,"","string",cmd);
	  ValueArg<int> context_vector_size("", "context_vector_size", "Size of the context vector. Default: 128.", false, 128, "int", cmd);
      ValueArg<int> vocab_size("", "vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<int> input_vocab_size("", "input_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<int> output_vocab_size("", "output_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<int> ngram_size("", "ngram_size", "Size of n-grams. Default: auto.", false, 0, "int", cmd);

      ValueArg<string> model_prefix("", "model_prefix", "Prefix for output model files." , false, "", "string", cmd);
      ValueArg<string> words_file("", "words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> parameter_update("", "parameter_update", "parameter update type.\n Stochastic Gradient Descent(SGD)\n \
          ADAGRAD(ADA)\n \
          ADADELTA(ADAD)" , false, "SGD", "string", cmd);
      ValueArg<string> input_words_file("", "input_words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> output_words_file("", "output_words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> validation_file("", "validation_file", "Validation data (one numberized example per line)." , false, "", "string", cmd);
      ValueArg<string> train_file("", "train_file", "Training data (one numberized example per line)." , true, "", "string", cmd);

      ValueArg<string> model_file("", "model_file", "Model file.", false, "", "string", cmd);
      ValueArg<double> hidden1_dropout_value("", "hidden1_dropout_value", "The probability of drop out for hidden layer 1.\
          Default: 0.9", false, 0.9, "double", cmd);
      ValueArg<double> hidden2_dropout_value("", "hidden2_dropout_value", "The probability of drop out for hidden layer 2.\
          Default: 0.9", false, 0.9, "double", cmd);
      ValueArg<double> hidden3_dropout_value("", "hidden3_dropout_value", "The probability of drop out for hidden layer 3.\
          Default: 0.9", false, 0.9, "double", cmd);
	  ValueArg<string> train_context_vectors_file("", "train_context_vectors_file", "Training context vectors file." 
		  , false, "", "string", cmd);
	  ValueArg<string> validation_context_vectors_file("", "validation_context_vectors_file", "Validation context vectors file." 
		  , false, "", "string", cmd);


      cmd.parse(argc, argv);

      // define program parameters //
      use_mmap_file = mmap_file.getValue();
      randomize = arg_randomize.getValue();
      myParam.model_file = model_file.getValue();
      myParam.train_file = train_file.getValue();
      myParam.validation_file = validation_file.getValue();
      myParam.input_words_file = input_words_file.getValue();
      myParam.output_words_file = output_words_file.getValue();
      if (words_file.getValue() != "")
	      myParam.input_words_file = myParam.output_words_file = words_file.getValue();

      myParam.model_prefix = model_prefix.getValue();

      myParam.ngram_size = ngram_size.getValue();
      myParam.vocab_size = vocab_size.getValue();
      myParam.input_vocab_size = input_vocab_size.getValue();
      myParam.output_vocab_size = output_vocab_size.getValue();
      if (vocab_size.getValue() >= 0) {
	      myParam.input_vocab_size = myParam.output_vocab_size = vocab_size.getValue();
      }
      myParam.num_hidden = num_hidden.getValue();
	  myParam.num_second_hidden = num_second_hidden.getValue();
      myParam.activation_function = activation_function.getValue();
      myParam.loss_function = loss_function.getValue();

      myParam.num_threads = num_threads.getValue();

      myParam.num_noise_samples = num_noise_samples.getValue();

      myParam.input_embedding_dimension = input_embedding_dimension.getValue();
      myParam.output_embedding_dimension = output_embedding_dimension.getValue();
      if (embedding_dimension.getValue() >= 0) {
	      myParam.input_embedding_dimension = myParam.output_embedding_dimension = embedding_dimension.getValue();
      }
	  myParam.input_embeddings_file = input_embeddings_file.getValue();
	  
	  myParam.context_vector_size = context_vector_size.getValue();
	  
      myParam.minibatch_size = minibatch_size.getValue();
      myParam.validation_minibatch_size = validation_minibatch_size.getValue();
      myParam.num_epochs= num_epochs.getValue();
      myParam.learning_rate = learning_rate.getValue();
      myParam.conditioning_constant = conditioning_constant.getValue();
      myParam.decay = decay.getValue();
      myParam.adagrad_epsilon = adagrad_epsilon.getValue();
      myParam.use_momentum = use_momentum.getValue();
      myParam.share_embeddings = share_embeddings.getValue();
      myParam.normalization = normalization.getValue();
      myParam.initial_momentum = initial_momentum.getValue();
      myParam.final_momentum = final_momentum.getValue();
      myParam.L2_reg = L2_reg.getValue();
      myParam.init_normal= init_normal.getValue();
      myParam.init_range = init_range.getValue();
      myParam.normalization_init = normalization_init.getValue();
      myParam.parameter_update = parameter_update.getValue();
	  myParam.hidden1_dropout_value = hidden1_dropout_value.getValue();
	  myParam.hidden2_dropout_value = hidden2_dropout_value.getValue();
	  myParam.hidden3_dropout_value = hidden3_dropout_value.getValue();
	  
	  myParam.train_context_vectors_file = train_context_vectors_file.getValue();
	  myParam.validation_context_vectors_file = validation_context_vectors_file.getValue();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");
      cerr << train_file.getDescription() << sep << train_file.getValue() << endl;
      cerr << validation_file.getDescription() << sep << validation_file.getValue() << endl;
      cerr << input_words_file.getDescription() << sep << input_words_file.getValue() << endl;
      cerr << output_words_file.getDescription() << sep << output_words_file.getValue() << endl;
      cerr << model_prefix.getDescription() << sep << model_prefix.getValue() << endl;

      cerr << ngram_size.getDescription() << sep << ngram_size.getValue() << endl;
      cerr << input_vocab_size.getDescription() << sep << input_vocab_size.getValue() << endl;
      cerr << output_vocab_size.getDescription() << sep << output_vocab_size.getValue() << endl;
      cerr << mmap_file.getDescription() << sep << mmap_file.getValue() << endl;

      if (embedding_dimension.getValue() >= 0)
      {
	      cerr << embedding_dimension.getDescription() << sep << embedding_dimension.getValue() << endl;
      }
      else
      {
	      cerr << input_embedding_dimension.getDescription() << sep << input_embedding_dimension.getValue() << endl;
	      cerr << output_embedding_dimension.getDescription() << sep << output_embedding_dimension.getValue() << endl;
      }
      cerr << share_embeddings.getDescription() << sep << share_embeddings.getValue() << endl;
      if (share_embeddings.getValue() && input_embedding_dimension.getValue() != output_embedding_dimension.getValue())
      {
	      cerr << "error: sharing input and output embeddings requires that input and output embeddings have same dimension" << endl;
	      exit(1);
      }

      cerr << num_hidden.getDescription() << sep << num_hidden.getValue() << endl;

      if (string_to_activation_function(activation_function.getValue()) == InvalidFunction)
      {
	      cerr << "error: invalid activation function: " << activation_function.getValue() << endl;
	      exit(1);
      }
      cerr << activation_function.getDescription() << sep << activation_function.getValue() << endl;

      if (string_to_loss_function(loss_function.getValue()) == InvalidLoss)
      {
	      cerr << "error: invalid loss function: " << loss_function.getValue() << endl;
	      exit(1);
      }
      cerr << loss_function.getDescription() << sep << loss_function.getValue() << endl;

      cerr << init_normal.getDescription() << sep << init_normal.getValue() << endl;
      cerr << init_range.getDescription() << sep << init_range.getValue() << endl;

      cerr << num_epochs.getDescription() << sep << num_epochs.getValue() << endl;
      cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
      if (myParam.validation_file != "") {
	     cerr << validation_minibatch_size.getDescription() << sep << validation_minibatch_size.getValue() << endl;
      }
      cerr << learning_rate.getDescription() << sep << learning_rate.getValue() << endl;
      cerr << L2_reg.getDescription() << sep << L2_reg.getValue() << endl;

      cerr << num_noise_samples.getDescription() << sep << num_noise_samples.getValue() << endl;

      cerr << normalization.getDescription() << sep << normalization.getValue() << endl;
      if (myParam.normalization){
	      cerr << normalization_init.getDescription() << sep << normalization_init.getValue() << endl;
      }

      cerr << use_momentum.getDescription() << sep << use_momentum.getValue() << endl;
      if (myParam.use_momentum)
      {
        cerr << initial_momentum.getDescription() << sep << initial_momentum.getValue() << endl;
        cerr << final_momentum.getDescription() << sep << final_momentum.getValue() << endl;
      }

      cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;

      if (unigram_probs_file.getValue() != "")
      {
	      cerr << "Note: --unigram_probs_file is deprecated and ignored." << endl;
      }
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    myParam.num_threads = setup_threads(myParam.num_threads);
    int save_threads;

    //unsigned seed = std::time(0);
	//cerr<<"Seed is "<<seed<<endl;
    unsigned seed = 1234; //for testing only
    mt19937 rng(seed);

    /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Read training data

    vector<int> training_data_flat;
    vec * training_data_flat_mmap;
    data_size_t training_data_size; //num_tokens;
    ip::managed_mapped_file mmap_file;
    if (use_mmap_file == false) {
      cerr<<"Reading data from regular text file "<<endl;
      readDataFile(myParam.train_file, myParam.ngram_size, training_data_flat, myParam.minibatch_size);
      training_data_size = training_data_flat.size()/myParam.ngram_size;
    } else {
      cerr<<"Using mmaped file"<<endl;
      mmap_file = ip::managed_mapped_file(ip::open_only,myParam.train_file.c_str());
      training_data_flat_mmap = mmap_file.find<vec>("vector").first;
      cerr<<"Size of mmaped vector is "<<training_data_flat_mmap->size()<<endl;
      training_data_size = training_data_flat_mmap->size()/myParam.ngram_size;
      //randomly shuffle the data for better learning. The shuffling will 
      //be different for a standard stl vector
      // Randomly shuffle training data to improve learning
      if (randomize == true) {
        cerr<<"Randomly shuffling data...";
        data_size_t counter =0;
        while (counter < training_data_size) {
          data_size_t upper_limit = counter+5000000;
          long int vector_size = 5000000;
          if (counter + 10000000 >= training_data_size) {
            upper_limit = training_data_size;
            vector_size = training_data_size - counter;
          }
          vector<int> temp(vector_size*myParam.ngram_size,0);
          for (int i=0;i<vector_size;i++){
           for (int k=0;k<myParam.ngram_size;k++) {
             temp[i*myParam.ngram_size+k] = training_data_flat_mmap->at((i+counter)*myParam.ngram_size+k);
           }
          }
          /*
          for (data_size_t i=upper_limit; i>counter; i--)
          {
            if (i %500000 == 0) {
              cerr<<"Shuffled "<<training_data_size-1<<" instances...";
            }
            data_size_t j = uniform_int_distribution<data_size_t>(0, i-1)(rng);
            for (int k=0;k<myParam.ngram_size;k++) {
              int temp_val = training_data_flat_mmap->at(i*myParam.ngram_size+k);
              training_data_flat_mmap->at(i*myParam.ngram_size+k) =
                training_data_flat_mmap->at(j*myParam.ngram_size+k);
              training_data_flat_mmap->at(j*myParam.ngram_size+k) = temp_val;
            }
          }
          */
          for (data_size_t i=vector_size-1; i>0; i--)
          {
            if (i %500000 == 0) {
              cerr<<"Shuffled "<<training_data_size-1<<" instances...";
            }
            data_size_t j = uniform_int_distribution<data_size_t>(0, i-1)(rng);
            for (int k=0;k<myParam.ngram_size;k++) {
              int temp_val = temp.at(i*myParam.ngram_size+k);
              temp.at(i*myParam.ngram_size+k) =
                temp.at(j*myParam.ngram_size+k);
              temp.at(j*myParam.ngram_size+k) = temp_val;
            }
          }
          //Putting it back
          for (int i=0;i<vector_size;i++){
           for (int k=0;k<myParam.ngram_size;k++) {
             training_data_flat_mmap->at((i+counter)*myParam.ngram_size+k) = temp[i*myParam.ngram_size+k];
           }
          }
          counter = upper_limit;
        }
        /*
        for (data_size_t i=training_data_size-1; i>0; i--)
        {
          if (i %500000 == 0) {
            cerr<<"Shuffled "<<training_data_size-1<<" instances...";
          }
          data_size_t j = uniform_int_distribution<data_size_t>(0, i-1)(rng);
          for (int k=0;k<myParam.ngram_size;k++) {
            int temp_val = training_data_flat_mmap->at(i*myParam.ngram_size+k);
            training_data_flat_mmap->at(i*myParam.ngram_size+k) =
              training_data_flat_mmap->at(j*myParam.ngram_size+k);
            training_data_flat_mmap->at(j*myParam.ngram_size+k) = temp_val;
          }
        }
        */
      cerr<<endl;
      }
    }
    //cerr<<"Num tokens "<<num_tokens<<endl;
    //data_size_t training_data_size = num_tokens / myParam.ngram_size;
    cerr << "Number of training instances: "<< training_data_size << endl;
    
    Matrix<int,Dynamic,Dynamic> training_data;
    //(training_data_flat.data(), myParam.ngram_size, training_data_size);
    
    #ifdef MAP
    cerr<<"Setting up eigen map"<<endl;
    if (use_mmap_file == false) {
      training_data = Map< Matrix<int,Dynamic,Dynamic> >(training_data_flat.data(), myParam.ngram_size, training_data_size);
    } else {
      training_data = Map< Matrix<int,Dynamic,Dynamic> >(training_data_flat_mmap->data().get(), myParam.ngram_size, training_data_size);
    }
    cerr<<"Created eigen map"<<endl;
    #else 
    if (use_mmap_file == false) {
      training_data = Map< Matrix<int,Dynamic,Dynamic> >(training_data_flat.data(), myParam.ngram_size, training_data_size);
    }
    #endif 
    // If neither --input_vocab_size nor --input_words_file is given, set input_vocab_size to the maximum word index
    if (myParam.input_vocab_size == 0 and myParam.input_words_file == "")
    {
        myParam.input_vocab_size = training_data.topRows(myParam.ngram_size-1).maxCoeff()+1;
    }

    // If neither --output_vocab_size nor --output_words_file is given, set output_vocab_size to the maximum word index
    if (myParam.output_vocab_size == 0 and myParam.output_words_file == "")
    {
        myParam.output_vocab_size = training_data.row(myParam.ngram_size-1).maxCoeff()+1;
    }
 

    // Read validation data
    vector<int> validation_data_flat;
    int validation_data_size = 0;
    
    if (myParam.validation_file != "")
    {
      readDataFile(myParam.validation_file, myParam.ngram_size, validation_data_flat);
      validation_data_size = validation_data_flat.size() / myParam.ngram_size;
      cerr << "Number of validation instances: " << validation_data_size << endl;
    }

	// Read train context vectors
	vector<double> training_context_vectors_flat;
	//int context_size = 128;
    if (myParam.train_context_vectors_file != "")
    {
      readContextVectorsFile(myParam.train_context_vectors_file, myParam.context_vector_size, training_context_vectors_flat);
      int train_context_vectors_flat_size  = training_context_vectors_flat.size() / myParam.context_vector_size;
	  if (train_context_vectors_flat_size != training_data_size){
		  cerr<<"The nubmer of training context vectors was not equal to the training data!"<<endl;
		  exit(1);
	  }
    }

 
	// Read validation context vectors
	vector<double> validation_context_vectors_flat;
    if (myParam.validation_context_vectors_file != "")
    {
      readContextVectorsFile(myParam.validation_context_vectors_file, myParam.context_vector_size, validation_context_vectors_flat);
      int validation_context_vectors_flat_size  = validation_context_vectors_flat.size() / myParam.context_vector_size;
	  if (validation_context_vectors_flat_size != validation_data_size){
		  cerr<<"The nubmer of validation context vectors was not equal to the validationing data!"<<endl;
		  exit(1);
	  }
    }
	
	Map< Matrix<double,Dynamic,Dynamic> > validation_context_vectors(validation_context_vectors_flat.data(),
																	myParam.context_vector_size, validation_data_size);

	Map< Matrix<double,Dynamic,Dynamic> > training_context_vectors(training_context_vectors_flat.data(),
																	myParam.context_vector_size, training_data_size);

    Map< Matrix<int,Dynamic,Dynamic> > validation_data(validation_data_flat.data(), myParam.ngram_size, validation_data_size);

    if (use_mmap_file == false && randomize == true) {
      cerr<<"Randomly shuffling data..."<<endl;
      // Randomly shuffle training data to improve learning. Shuffling the context vectors as well
      for (data_size_t i=training_data_size-1; i>0; i--)
      {
        data_size_t j = uniform_int_distribution<data_size_t>(0, i-1)(rng);
        training_data.col(i).swap(training_data.col(j));
		//training_context_vectors.col(i).swap(training_context_vectors.col(j));
      }
    }
		
    ///// Read in vocabulary file. We don't actually use it; it just gets reproduced in the output file

    vector<string> input_words;
    if (myParam.input_words_file != "")
    {
        readWordsFile(myParam.input_words_file, input_words);
	if (myParam.input_vocab_size == 0)
	    myParam.input_vocab_size = input_words.size();
    }

    vector<string> output_words;
    if (myParam.output_words_file != "")
    {
        readWordsFile(myParam.output_words_file, output_words);
	if (myParam.output_vocab_size == 0)
	    myParam.output_vocab_size = output_words.size();
    }

    ///// Construct unigram model and sampler that will be used for NCE

    vector<data_size_t> unigram_counts(myParam.output_vocab_size);
    for (data_size_t train_id=0; train_id < training_data_size; train_id++)
    {
        int output_word;
        if (use_mmap_file == false) {
          output_word = training_data(myParam.ngram_size-1, train_id);
        } else {
	      //cerr<<"mmap word is "<<training_data_flat_mmap->at((train_id+1)*myParam.ngram_size - 1)<<endl;
          output_word = training_data_flat_mmap->at((train_id+1)*myParam.ngram_size - 1);
        }
		//cerr<<"output word is "<<output_word<<endl;
	    unigram_counts[output_word] += 1;
    }
    multinomial<data_size_t> unigram (unigram_counts);

    ///// Create and initialize the neural network and associated propagators.
    model nn;
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL
    //if (myParam.model_file != ""){
    //nn.read(myParam.model_file);
    // cerr<<"reading the model"<<endl;
    
	//} else {
      nn.resize(myParam.ngram_size,
          myParam.input_vocab_size,
          myParam.output_vocab_size,
          myParam.input_embedding_dimension,
          myParam.num_hidden,
		  myParam.num_second_hidden,
          myParam.output_embedding_dimension,
		  myParam.context_vector_size);

      nn.initialize(rng,
          myParam.init_normal,
          myParam.init_range,
          -log(myParam.output_vocab_size),
          myParam.parameter_update,
          myParam.adagrad_epsilon);
      nn.set_activation_function(string_to_activation_function(myParam.activation_function));
	  // If the input embeddings file has been specified then read from the input embeddings files
	  if (myParam.input_embeddings_file != ""){
		  cerr<<" Reading the input embeddings from file"<<myParam.input_embeddings_file<<endl;
		  nn.input_layer.read(myParam.input_embeddings_file);
	  }
      if (myParam.model_file != ""){
        nn.read(myParam.model_file);
        cerr<<"reading the model"<<endl;
      } 
    //}
    loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

    propagator prop(nn, myParam.minibatch_size);
    propagator prop_validation(nn, myParam.validation_minibatch_size);
    SoftmaxNCELoss<multinomial<data_size_t> > softmax_loss(unigram);
    // normalization parameters
    vector_map c_h, c_h_running_gradient;
    
    ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;
    cerr<<"Number of training minibatches: "<<num_batches<<endl;

    int num_validation_batches = 0;
    if (validation_data_size > 0)
    {
        num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size+1;
	cerr<<"Number of validation minibatches: "<<num_validation_batches<<endl;
    } 

    double current_momentum = myParam.initial_momentum;
    double momentum_delta = (myParam.final_momentum - myParam.initial_momentum)/(myParam.num_epochs-1);
    double current_learning_rate = myParam.learning_rate;
    double current_validation_ll = 0.0;
	double current_validation_accuracy = 0.0;

    int ngram_size = myParam.ngram_size;
    int input_vocab_size = myParam.input_vocab_size;
    int output_vocab_size = myParam.output_vocab_size;
    int minibatch_size = myParam.minibatch_size;
    int validation_minibatch_size = myParam.validation_minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;

    if (myParam.normalization)
    {
      for (data_size_t i=0;i<training_data_size;i++)
      {
          Matrix<int,Dynamic,1> context = training_data.block(0,i,ngram_size-1,1);
          if (c_h.find(context) == c_h.end())
          {
              c_h[context] = -myParam.normalization_init;
          }
      }
    }

	//Setting up the dropout sampler
	boost::random::uniform_real_distribution<> real_01(0, 1);
	 
    for (int epoch=0; epoch<myParam.num_epochs; epoch++)
    { 
		double train_correct_labels = 0.;
        cerr << "Epoch " << epoch+1 << endl;
        cerr << "Current learning rate: " << current_learning_rate << endl;
		
        if (myParam.use_momentum) 
	    cerr << "Current momentum: " << current_momentum << endl;
	else
            current_momentum = -1;

	cerr << "Training minibatches: ";

	double log_likelihood = 0.0;

	int num_samples = 0;
	if (loss_function == LogLoss)
	    num_samples = output_vocab_size;
	else if (loss_function == NCELoss)
	    num_samples = 1+num_noise_samples;

	Matrix<double,Dynamic,Dynamic> minibatch_weights(num_samples, minibatch_size);
	Matrix<int,Dynamic,Dynamic> minibatch_samples(num_samples, minibatch_size);
	Matrix<double,Dynamic,Dynamic> scores(num_samples, minibatch_size);
	Matrix<double,Dynamic,Dynamic> probs(num_samples, minibatch_size);
	Matrix<double,Dynamic,Dynamic> first_hidden_dropout_mask(myParam.num_hidden,minibatch_size);
	Matrix<double,Dynamic,Dynamic> second_hidden_dropout_mask(myParam.output_embedding_dimension,minibatch_size);
	

  for(data_size_t batch=0;batch<num_batches;batch++)
        {
            if (batch > 0 && batch % 10000 == 0)
            {
	        cerr << batch <<"...";
            } 

            data_size_t minibatch_start_index = minibatch_size * batch;

      int current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), training_data_size - minibatch_start_index);
      #ifdef MAP
	    Matrix<int,Dynamic,Dynamic> minibatch = training_data.middleCols(minibatch_start_index, current_minibatch_size);
      #else 
      //ALTERNATIVE OPTION IF YOU'RE NOT USING eigen map interface on the mmapped file
	    Matrix<int,Dynamic,Dynamic> minibatch;// = training_data.middleCols(minibatch_start_index, current_minibatch_size);
		//cerr<<"Minibatch start index "<<minibatch_start_index<<endl;
		//cerr<<"Minibatch size "<<current_minibatch_size<<endl;
            if (use_mmap_file == true) {
            minibatch.setZero(ngram_size,current_minibatch_size);
            //now reading the ngrams from the mmaped file
              for (int k=0; k<ngram_size; k++){
                for (data_size_t index = 0 ; index<current_minibatch_size; index++) {
				  data_size_t current_index = index + minibatch_start_index;
				  //cerr<<"the value in the mmap file "<<index<<" "<<k<<" is "<<training_data_flat_mmap->at(current_index*ngram_size+k)<<endl;
                  minibatch(k,index) = training_data_flat_mmap->at(current_index*ngram_size+k);
                }
              }
            } else {
              minibatch = training_data.middleCols(minibatch_start_index, current_minibatch_size);
            }
      #endif 
            double adjusted_learning_rate = current_learning_rate/current_minibatch_size;
            //cerr<<"Adjusted learning rate: "<<adjusted_learning_rate<<endl;

            /*
            if (batch == rand() % num_batches)
            {
                cerr<<"we are checking the gradient in batch "<<batch<<endl;
                /////////////////////////CHECKING GRADIENTS////////////////////////////////////////
                gradientChecking(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,
                              shuffled_training_data,c_h,unif_real_vector,eng_real_vector,unif_int_vector,eng_int_vector,unigram_probs_vector,
                              q_vector,J_vector,D_prime);
            }
            */

		// Generating the dropout mask
		for(int q=0; q<current_minibatch_size; q++){
			for (int p=0; p<myParam.num_hidden; p++){
				first_hidden_dropout_mask(p,q) = ( real_01(rng) <1.0 - myParam.hidden1_dropout_value ) ? 0: 1; 
			}
			for (int p=0; p<myParam.output_embedding_dimension; p++){
				second_hidden_dropout_mask(p,q) = ( real_01(rng) < 1.0 - myParam.hidden2_dropout_value) ? 0: 1; 
			}

		}
		//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
		//cerr<<"Shape of dropout mask is "<<second_hidden_dropout_mask.rows()<<" "<<second_hidden_dropout_mask.cols()<<endl;
		//cerr<<"Shape of dropout mask is "<<first_hidden_dropout_mask.rows()<<" "<<first_hidden_dropout_mask.cols()<<endl;
		//cerr<<"First hidden dropout mask is "<<first_hidden_dropout_mask<<endl;
		//cerr<<"dropout mas is "<<second_hidden_dropout_mask<<endl;
		//getchar();

			
        ///// Forward propagation
        prop.fPropDropout(minibatch.topRows(ngram_size-1),
					first_hidden_dropout_mask,
					second_hidden_dropout_mask);

	    if (loss_function == NCELoss)
	    {
	      ///// Noise-contrastive estimation

	      // Generate noise samples. Gather positive and negative samples into matrix.

	      start_timer(3);

        minibatch_samples.block(0, 0, 1, current_minibatch_size) = minibatch.bottomRows(1);
        
        for (int sample_id = 1; sample_id < num_noise_samples+1; sample_id++)
            for (int train_id = 0; train_id < current_minibatch_size; train_id++)
                minibatch_samples(sample_id, train_id) = unigram.sample(rng);
          
        stop_timer(3);

        // Final forward propagation step (sparse)
        start_timer(4);
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix,
                    minibatch_samples, scores);
        stop_timer(4);

        // Apply normalization parameters
        if (myParam.normalization)
        {
            for (int train_id = 0;train_id < current_minibatch_size;train_id++)
            {
          Matrix<int,Dynamic,1> context = minibatch.block(0, train_id, ngram_size-1, 1);
          scores.col(train_id).array() += c_h[context];
            }
        }

        double minibatch_log_likelihood;
        start_timer(5);
        softmax_loss.fProp(scores.leftCols(current_minibatch_size), 
               minibatch_samples,
               probs, 
			   minibatch_log_likelihood);
        stop_timer(5);
        log_likelihood += minibatch_log_likelihood;

        ///// Backward propagation

        start_timer(6);
        softmax_loss.bProp(probs, minibatch_weights);
        stop_timer(6);
        
        // Update the normalization parameters
        
        if (myParam.normalization)
        {
          for (int train_id = 0;train_id < current_minibatch_size;train_id++)
          {
            Matrix<int,Dynamic,1> context = minibatch.block(0, train_id, ngram_size-1, 1);
            c_h[context] += adjusted_learning_rate * minibatch_weights.col(train_id).sum();
          }
        }

        // Be careful of short minibatch
        prop.bProp(minibatch.topRows(ngram_size-1),
             minibatch_samples.leftCols(current_minibatch_size), 
             minibatch_weights.leftCols(current_minibatch_size),
			 first_hidden_dropout_mask.leftCols(current_minibatch_size),
			 second_hidden_dropout_mask.leftCols(current_minibatch_size),
             adjusted_learning_rate, 
             current_momentum,
             myParam.L2_reg,
             myParam.parameter_update,
             myParam.conditioning_constant,
             myParam.decay);
	    }
	    else if (loss_function == LogLoss)
	    {
	      ///// Standard log-likelihood
	      start_timer(4);
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
        stop_timer(4);

        double minibatch_log_likelihood;
        start_timer(5);
        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
                   minibatch.row(ngram_size-1), 
                   probs, 
                   minibatch_log_likelihood);
        stop_timer(5);
        log_likelihood += minibatch_log_likelihood;

        ///// Backward propagation
        
        start_timer(6);
        SoftmaxLogLoss().bProp(minibatch.row(ngram_size-1).leftCols(current_minibatch_size), 
                   probs.leftCols(current_minibatch_size), 
                   minibatch_weights);
        stop_timer(6);
        //Computing the training accuracy
   		for (int minibatch_instance=0; minibatch_instance < current_minibatch_size; minibatch_instance++){
   			Matrix<double,1,Dynamic>::Index max_index;
   			probs.col(minibatch_instance).maxCoeff(&max_index);
   			if (max_index == 
   				minibatch.row(ngram_size-1).leftCols(current_minibatch_size)(minibatch_instance)){
   					train_correct_labels += 1;
   				}
   		}		
    	

        prop.bProp(minibatch.topRows(ngram_size-1).leftCols(current_minibatch_size),
             minibatch_weights,
			 first_hidden_dropout_mask,
			 second_hidden_dropout_mask,
             adjusted_learning_rate,
             current_momentum,
             myParam.L2_reg,
             myParam.parameter_update,
             myParam.conditioning_constant,
             myParam.decay);
          }
      }
	  
	cerr << "done." << endl;

	if (loss_function == LogLoss)
	{
	    cerr << "Training log-likelihood: " << log_likelihood << endl;
        cerr << "         perplexity:     "<< exp(-log_likelihood/training_data_size) << endl;
		cerr << "         accuracy:       "<< train_correct_labels/training_data_size <<endl;
	}
	else if (loss_function == NCELoss)
	    cerr << "Training NCE log-likelihood: " << log_likelihood << endl;

        current_momentum += momentum_delta;

	#ifdef USE_CHRONO
	cerr << "Propagation times:";
	for (int i=0; i<timer.size(); i++)
	  cerr << " " << timer.get(i);
	cerr << endl;
	#endif
	
	//Scaled the model before writing
	nn.scale(myParam.hidden1_dropout_value,
			myParam.hidden2_dropout_value,
			myParam.hidden3_dropout_value);
	if (myParam.model_prefix != "")
	{
	    cerr << "Writing model" << endl;
	    if (myParam.input_words_file != "")
	        nn.write(myParam.model_prefix + "." + lexical_cast<string>(epoch+1), input_words, output_words);
	    else
	        nn.write(myParam.model_prefix + "." + lexical_cast<string>(epoch+1));
	}

        if (epoch % 1 == 0 && validation_data_size > 0)
        {
		    //Matrix<int,Dynamic,1> validation_argmaxes;
			//validation_argmaxes.setZero(validation_data_size);
			vector<int> validation_argmaxes;
			string argmax_file = "argmax."+lexical_cast<string>(epoch+1);
            //////COMPUTING VALIDATION SET PERPLEXITY///////////////////////
            ////////////////////////////////////////////////////////////////

            double log_likelihood = 0.0;

	    Matrix<double,Dynamic,Dynamic> scores(output_vocab_size, validation_minibatch_size);
	    Matrix<double,Dynamic,Dynamic> output_probs(output_vocab_size, validation_minibatch_size);
	    Matrix<int,Dynamic,Dynamic> minibatch(ngram_size, validation_minibatch_size);
		double validation_correct_labels  =0;
            for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
            {
                int validation_minibatch_start_index = validation_minibatch_size * validation_batch;
		int current_minibatch_size = min(validation_minibatch_size,
						 validation_data_size - validation_minibatch_start_index);
		minibatch.leftCols(current_minibatch_size) = validation_data.middleCols(validation_minibatch_start_index, 
											current_minibatch_size);
		prop_validation.fPropContext(minibatch.topRows(ngram_size-1));

		// Do full forward prop through output word embedding layer
		start_timer(4);
		prop_validation.output_layer_node.param->fProp(prop_validation.second_hidden_activation_node.fProp_matrix, scores);
		stop_timer(4);

		// And softmax and loss. Be careful of short minibatch
		double minibatch_log_likelihood;
		start_timer(5);
		SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
				       minibatch.row(ngram_size-1),
				       output_probs,
				       minibatch_log_likelihood);
		stop_timer(5);
		log_likelihood += minibatch_log_likelihood;
	   		for (int minibatch_instance=0; minibatch_instance < current_minibatch_size; minibatch_instance++){
	   			Matrix<double,1,Dynamic>::Index max_index;
	   			output_probs.col(minibatch_instance).maxCoeff(&max_index);
				validation_argmaxes.push_back(max_index);
	   			if (max_index == 
	   				minibatch.row(ngram_size-1)(minibatch_instance)){
	   					validation_correct_labels += 1;
	   				}
	   		}		
	    }
		   /*
           //CREATING AN EIGEN MATRIX OUT OF THE ARGMAXES
		   Matrix<int,Dynamic,1> argmaxes_vector(validation_data_size);
		   for (data_size_t index=0; index<=validation_data_size; validation_data_size++){
		     argmaxes_vector(index) = validation_argmaxes[index];
		   }
		   */
		   //argmaxes_vector = Map< Matrix<int,Dynamic,1> >(validation_argmaxes.data(),validation_data_size,1);
            cerr << "Validation log-likelihood: "<< log_likelihood << endl;
            cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;
			cerr << "           accuracy:       "<< validation_correct_labels/validation_data_size <<endl;
            writeVector(validation_argmaxes,argmax_file);
	    // If the validation perplexity decreases, halve the learning rate.

            //if (epoch > 0 && validation_correct_labels/validation_data_size < current_validation_accuracy && myParam.parameter_update != "ADA" && current_momentum < 0.)
			//
            if (epoch > 0 && log_likelihood < current_validation_ll && myParam.parameter_update != "ADA" && current_momentum < 0.)
            { 
                current_learning_rate /= 2;
            }
            current_validation_ll = log_likelihood;
			current_validation_accuracy = validation_correct_labels/validation_data_size;
			
		}
		//Scaled the model back after validation
		nn.scale(1.0/myParam.hidden1_dropout_value,
			1.0/myParam.hidden2_dropout_value,
			1.0/myParam.hidden3_dropout_value);
    }
    return 0;
}

