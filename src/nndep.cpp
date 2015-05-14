// ksdep.cpp
// A shift-reduce dependency parser with best-first search

#include <algorithm>
#include <fstream>
#include <map>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "param.h"

#include "neuralTM.h"
#include "neuralLM.h"

#include "ksdep.h"

#define DEBUG 0
#define PRINTFEATS 1

// global variables for parsing
// (Initial values are irrelevant and will
// be overwritten in main())

double LENBEAMFACTOR = 0;
double ACTCUTOFF = 0;
int NUMACTCUTOFF = 0;
int MAXSTATES = 0;
int NUMTHREADS =1;
int RRFEAT = 0;

int TRAIN = 0;
item dummyitem;
int NUMITER;

int sentcnt = 0;

using namespace std;
using namespace boost;
using namespace Eigen;

using namespace nplm;

// simple discretizer for numerical features
int discr( int n ) {
  if( n > 6 ) return 7;
  if( n > 3 ) return 4;
  return n;
}

// input: a parser state, a vector that will contain features
//
// This fills the given vector with features corresponding
// to the given parser state
int makefeats( parserstate& pst, vector<string>& fv ) {
  fv.clear();

  const item *s1 = pst.getst( 1 );
  const item *s2 = pst.getst( 2 );
  const item *s3 = pst.getst( 3 );
  
  const item *q1 = pst.getq( 1 );
  const item *q2 = pst.getq( 2 );
  const item *q3 = pst.getq( 3 );

  int dist1 = s1->idx - s2->idx;
  int dist2 = q1->idx - s1->idx;

  if( dist1 > 7 ) {
    dist1 = 7;
  }
  if( dist2 > 7 ) {
    dist2 = 7;
  }
  /*
  fv.push_back( s1->pos );
  fv.push_back( s2->pos );
  fv.push_back( s3->pos );
  fv.push_back( q1->pos );
  fv.push_back( q2->pos );
  fv.push_back( q3->pos );
  fv.push_back( s1->word );
  fv.push_back( s2->word );
  fv.push_back( q1->word );
  fv.push_back( q2->word );
  fv.push_back( pst.inputq[ s1->lch ].pos );
  fv.push_back( pst.inputq[ s1->rch ].pos );
  fv.push_back( pst.inputq[ s2->lch ].pos );
  fv.push_back( pst.inputq[ s2->rch ].pos );
  
  */
  // distance between s1 and s2
  fv.push_back( "NUM=" + toString( dist1 ) );
  fv.push_back( "NUM=" + toString( dist2 ) );

  fv.push_back( "NUM=" + toString( s1->nch ) );
  fv.push_back( "NUM=" + toString( s1->nrch ) );
  fv.push_back( "NUM=" + toString( s1->nlch ) );

  fv.push_back( "NUM=" + toString( s2->nch ) );
  fv.push_back( "NUM=" + toString( s2->nrch ) );
  fv.push_back( "NUM=" + toString( s2->nlch ) );

  // pos between s1 and s2, if any
  if( dist1 > 1 ) {
    fv.push_back( "POS=" + pst.inputq[s1->idx - 1 ].pos );
  }
  else {
    fv.push_back( "POS=NONE" );
  }

  if( dist1 > 2 ) {
    fv.push_back( "POS=" + pst.inputq[ s2->idx + 1 ].pos );
  }
  else {
    fv.push_back( "POS=NONE" );
  }

  fv.push_back( "ACT=" + pst.prevact );

  fv.push_back( "WRD=" + s1->word );
  fv.push_back( "POS=" + s1->pos );
  //fv.push_back( s1->label );
  fv.push_back( "POS=" + pst.inputq[ s1->lch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s1->lch ].word );
  //fv.push_back( pst.inputq[ s1->lch ].label );
  fv.push_back( "POS=" + pst.inputq[ s1->rch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s1->rch ].word );
  //fv.push_back( pst.inputq[ s1->rch ].label );
  fv.push_back( "POS=" + pst.inputq[ s1->lch2 ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s1->lch2 ].word );
  //fv.push_back( pst.inputq[ s1->lch ].label );
  fv.push_back( "POS=" + pst.inputq[ s1->rch2 ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s1->rch2 ].word );
  //fv.push_back( pst.inputq[ s1->rch ].label );
  fv.push_back( "POS=" + pst.inputq[ s1->lgch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s1->lgch ].word );
  //fv.push_back( pst.inputq[ s1->lch ].label );
  fv.push_back( "POS=" + pst.inputq[ s1->rgch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s1->rgch ].word );
  //fv.push_back( pst.inputq[ s1->rch ].label );
   
  fv.push_back( "WRD=" + s2->word );
  fv.push_back( "POS=" + s2->pos );
  //fv.push_back( s2->label );
  fv.push_back( "POS=" + pst.inputq[ s2->lch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s2->lch ].word );
  //fv.push_back( pst.inputq[ s2->lch ].label );
  fv.push_back( "POS=" + pst.inputq[ s2->rch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s2->rch ].word );
  //fv.push_back( pst.inputq[ s2->rch ].label );
  fv.push_back( "POS=" + pst.inputq[ s2->lch2 ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s2->lch2 ].word );
  //fv.push_back( pst.inputq[ s2->lch ].label );
  fv.push_back( "POS=" + pst.inputq[ s2->rch2 ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s2->rch2 ].word );
  //fv.push_back( pst.inputq[ s2->rch ].label );
  fv.push_back( "POS=" + pst.inputq[ s2->lgch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s2->lgch ].word );
  //fv.push_back( pst.inputq[ s2->lch ].label );
  fv.push_back( "POS=" + pst.inputq[ s2->rgch ].pos );
  fv.push_back( "WRD=" + pst.inputq[ s2->rgch ].word );
  //fv.push_back( pst.inputq[ s2->rch ].label );

  fv.push_back( "WRD=" + s3->word );
  fv.push_back( "POS=" + s3->pos );
  //fv.push_back( s3->label );

  fv.push_back( "WRD=" + q1->word );
  fv.push_back( "POS=" + q1->pos );

  fv.push_back( "WRD=" + q2->word );
  fv.push_back( "POS=" + q2->pos );

  fv.push_back( "WRD=" + q3->word );
  fv.push_back( "POS=" + q3->pos );

  /*
  int n = fv.size();

  for( int i = 0; i < n; i++ ) {
    fv.push_back( toString( fv.size() ) + "~" + fv[i] + "~" + s1->pos );
    fv.push_back( toString( fv.size() ) + "~" + fv[i] + "~" + s2->pos );
    fv.push_back( toString( fv.size() ) + "~" + fv[i] + "~" + q1->pos );
  }

  fv.push_back( toString( fv.size() ) + "~" + q1->pos + s1->pos + s2->pos );
  fv.push_back( toString( fv.size() ) + "~" + s1->pos + s2->pos + s3->pos );
  fv.push_back( toString( fv.size() ) + "~" + q1->pos + q2->pos + s1->pos + s2->pos );
  fv.push_back( toString( fv.size() ) + "~" + s1->pos + s2->pos + s3->pos + q1->pos );

  fv.push_back( toString( fv.size() ) + "~" + pst.prevact + s1->pos );
  fv.push_back( toString( fv.size() ) + "~" + pst.prevact + s1->pos + s2->pos );
  fv.push_back( toString( fv.size() ) + "~" + pst.prevact + q1->pos );
  fv.push_back( toString( fv.size() ) + "~" + pst.prevact + q1->pos + s1->pos);
  */
  return 0;
}

// input: a vector with the input items, a model, and 
// number of desired output parses
//
// This is where the parsing is done
int parse( vector<item>& inputq, vector<string> actions, int nparses,
	   neuralTM &lm, map<string, int> &syms ) {
 
  int n = inputq.size() - 1;
  vector<double> lenbeam( ( n * 2 ) + 10, 0.0 ); 
  vector<parserstate> psv;
  priority_queue<parserstate> pspq;
  parserstate newpst( inputq );
  newpst.score = 1;
  newpst.shift();
  pspq.push( newpst );

  parserstate currpst;

  while( pspq.size() > 0 ) {

    if( ( pspq.top().i >= n ) && ( pspq.top().s.size() <= 1 ) ) {
      psv.push_back( pspq.top() );
      if( psv.size() >= nparses ) {
	break;
      }
    }

    if( MAXSTATES && ( pspq.size() > ( MAXSTATES * 2 ) ) ) {
      priority_queue<parserstate> tmppq;
      for( int tt = 0; tt < MAXSTATES; tt++ ) {
	tmppq.push( pspq.top() );
	pspq.pop();
      }
      pspq = tmppq;
    }

    currpst = pspq.top();
    pspq.pop();

    if( DEBUG ) cout << "\nGOT PARSER STATE: " << currpst.score << endl;

    string act = "S";
    string label = "NONE";

    // Get the items on top of stack
    item* s1 = currpst.getst( 1 );
    item* s2 = currpst.getst( 2 );

    if( DEBUG ) cout << "S: " << currpst.s.size() << " "
		     << "Q: " << currpst.i << endl;

    if( DEBUG ) cout << "S1: " << s1->word << " IDX: " 
		     << " " << s1->idx << " GOLDIN: " << s1->goldin 
		     << " LINK: " << s1->goldlink << endl;
    if( DEBUG ) cout << "S2: " << s2->word << " IDX: "
		     << " " << s2->idx << " GOLDIN: " << s2->goldin 
		     << " LINK: " << s2->goldlink << endl;


    // Get features from parser state
    vector<string> feats;
    makefeats( currpst, feats );
    vector<double> vp(actions.size());

    // Choose action
    if( TRAIN ) {

      // is s1 a dependent of s2?
      if( ( s1->goldlink == s2->idx ) && ( s1->goldin == 0 ) && currpst.s.size() > 1 ) {
	act = "LEFT"; // + s1->goldlabel;
	label = s1->goldlabel;
      }

      // is s2 a dependent of s1?
      else if( ( s2->goldlink == s1->idx ) && ( s2->goldin == 0 ) && currpst.s.size() > 1 ) {
	act = "RIGHT"; // + s2->goldlabel;
	label = s2->goldlabel;
      }
      else if( currpst.i < n ) {
	act = "SHIFT";
      }
      else {
	act = "F";
      }

      //cout << currpst.score << endl;
      if( currpst.err ) {
	act = "ERR";
      }

      // add training example
      if( !( act == "F" ) ) {
	// would add example here
      }
	
      if( PRINTFEATS ) {
	cout << act;
	
	for( int i = 0; i < feats.size(); i++ ) {
	  cout << " " << feats[i];
	}
	cout << endl;
      }
    }

    if( !TRAIN ) {
      // classify action
      vector<int> feats2;
      map<string, int>::iterator it;
      for(int i = 0; i < feats.size(); i++) {
	string fstr = feats[i].substr(4);
	int f = 0;
	it = syms.find(fstr);
	if(it != syms.end()) {
	  f = it->second;
	}
	feats2.push_back(f);
      }
      feats2.push_back(0);
      lm.lookup_ngram(feats2, vp);
      for( int i = 0; i < vp.size(); i++ ) {
	vp[i] = pow(10, vp[i]);
      }
    }
    
    // put all possible actions in a priority queue
    priority_queue<pact> actpq;

    for( int i = 0; i < vp.size(); i++ ) {
      if( ACTCUTOFF && ( vp[i] < ACTCUTOFF ) ) {
	continue;
      }
      if( TRAIN ) {
	if( act != "ERR" ) {
	  if( actions[i] == act ) {
	    actpq.push( pact( 0.86, act ) );
	  }
	  else {
	    actpq.push( pact( 0.85, actions[i] ) );
	  }
	}
      }
      else {
	if( actions[i] != "ERR" ) {
	  actpq.push( pact( vp[i], actions[i] ) );
	}
      }
    }

    int actcnt = 0;
    while( actpq.size() ) {
      actcnt++;
      
      if( NUMACTCUTOFF && ( actcnt == NUMACTCUTOFF ) ) {
	break;
      }
      act = actpq.top().label;
      double score = actpq.top().score;
      actpq.pop();

      parserstate npst( currpst );

      if( TRAIN && score == 0.85 ) {
	npst.err = 1;
      }

      if( DEBUG ) cout << "   " << act << " : " << score << endl;

      if( act[0] == 'L' ) {
	if( DEBUG ) cout << ">>> LEFT\n";
	label = "_"; //act.substr( 2 );
	if( !npst.reduceleft( label ) ) {
	  act = "F";
	  if( DEBUG) cout << "NO!\n";
	}
	else {
	  npst.score *= score;
	}
      }
      else if( act[0] == 'R' ) {
	if( DEBUG ) cout << ">>> RIGHT\n";
	label = "_"; //act.substr( 2 );
	if( !npst.reduceright( label ) ) {
	  act = "F";
	  if( DEBUG) cout << "NO!\n";
	}
	else {
	  npst.score *= score;
	}
      }
      else if( act[0] == 'S' ) {
	if( DEBUG ) cout << ">>> SHIFT\n";
	if( !npst.shift() ) {
	  act = "F";
	  if( DEBUG) cout << "NO!\n";
	}
	else {
	  npst.score *= score;
	}
      }
      else {
	act = "F";
      }
      
      if( act == "F" ) {
	if( DEBUG ) cout << ">>> ACT-FAIL\n";
	continue;
      }
      
      if( DEBUG ) cout << "PUSH NPST\n";
      if( npst.score > ( lenbeam[npst.numacts] * LENBEAMFACTOR ) ) {
	pspq.push( npst );
	if( npst.score > ( lenbeam[npst.numacts] ) ) {
	  lenbeam[npst.numacts] = npst.score;
	}
      }
    }
  }
  
  int errflg = 0;

  // if we are training, don't print out a parse
  if( !TRAIN ) {
    // if we are doing 1-best parsing
    // just use the CoNLL-X output format
    if( nparses == 1 ) {
      for( int i = 1; i < n; i++ ) {
	if( pspq.size() > 0 ) {
	  pspq.top().inputq[i].print();
	}
	else {
	  errflg = 1;
	  currpst.inputq[i].print();
	}
      }
      cout << endl;
    }
    // n-best output
    else if( !RRFEAT ) {
      cout << psv.size() << " parses\n";
      for( int j = 0; j < psv.size(); j++ ) {
	cout << psv[j].score << endl;
	for( int i = 1; i < psv[j].inputq.size() - 1; i++ ) {
	  psv[j].inputq[i].print();
	}
	cout << endl;
      }
    }
    else {
      // print the n-best list to stdout
      for( int j = 0; j < psv.size(); j++ ) {
	cout << psv[j].score << endl;
	for( int i = 1; i < psv[j].inputq.size() - 1; i++ ) {
	  psv[j].inputq[i].print2();
	}
	cout << endl;
      }
    }
  }

  sentcnt++;

  // we didn't find a complete parse
  if( errflg ) {
    return -1;
  }

  // we found at least one complete parse
  return 0;
}
    
int main( int argc, char **argv ) {

  string modelname;
  int nparses;
  int heldout;
  int fcutoff;
  int helpmsg;
  double ineq;

  // read options
  ksopts opt( "rrfeat 0 :threads 1 :maxst 100 :numactcut 0 :actcut 0 :b 0 :m tmp.mod :it 400 t 0 :n 1 :h 0 :f 0 :i 1.0 help 0", argc, argv );
  opt.optset( "help", helpmsg, "Print this help message and exit" );
  opt.optset( "m", modelname, "(string) Model name" );
  opt.optset( "it", NUMITER, "(int) Number of maxent iterations for training" );
  opt.optset( "t", TRAIN, "Training mode.  The parser runs in parsing mode by default" );
  opt.optset( "n", nparses, "(int) Number of parses to output (n-best output)" );
  opt.optset( "h", heldout, "(int) Use this many actions as heldout data in maxent training" );
  opt.optset( "f", fcutoff, "(int) In training, discard features that appear less than this many times" );
  opt.optset( "i", ineq, "(float) Regularization parameter for training" );
  opt.optset( "b", LENBEAMFACTOR, "(float) Beam factor (between 0 and 1, where 0 is no beam, and 1 keeps only locally best path)" );
  opt.optset( "numactcut", NUMACTCUTOFF, "(int) Consider at most this many actions per iteration" );
  opt.optset( "actcut", ACTCUTOFF, "(float) Consider only actions with probability higher or equal to this value" );
  opt.optset( "maxst", MAXSTATES, "(int) Maximum number of parser states in the priority queue" );
  opt.optset( "threads", NUMTHREADS, "(int) number of threads. (for now only during training)" );
  opt.optset( "rrfeat", RRFEAT, "Generate features for reranking n-best lists" );


  if( helpmsg ) {
    cerr << "USAGE: ksdep [--help][-t][-m MODELNAME][-it NUMITERATIONS][-n NUMPARSES][-h NUMHELDOUT][-f FEATCUTOFF][-i INEQPARAM][-b BEAMFACTOR][-numactcut NUMACTS][-actcut ACTPROB][-maxst MAXSTATES]\n\n";
    cerr << "By default, the parser uses sensible pruning for efficient accurate parsing.  Most of the options are not necessary for training and parsing.  The crucial options are -m (set the model name) and -t (to train the parser, if desired).\n\n";

    opt.printhelp();
    exit( 0 );
  }
  
  string line;
  neuralTM lm;
  int id = 0;
  map<string, int> syms;
  vector<string> actions;

  if( !TRAIN ) {
    lm.read(modelname);
    lm.set_normalization(true);
    lm.set_log_base(10);
    lm.set_cache(0);
 
    ifstream modf(modelname.c_str());
    while(getline(modf, line)) {
      if(line == "\\input_vocab") {
	break;
      }
    }
    while(getline(modf, line)) {
      if( line == "\\output_vocab" ) {
	break;
      }
      syms[line] = id;
      id++;
    }
    int actionid = 0;
    while(getline(modf, line)) {
      if( line == "" ) {
	break;
      }
      actions.push_back(line);
    }
    
    modf.close();
    cerr << "Loaded model." << endl;
  }

  if( TRAIN ) {
    LENBEAMFACTOR = 0;
    ACTCUTOFF = 0;
    NUMACTCUTOFF = 0;

    actions.push_back("ERR");
    actions.push_back("LEFT");
    actions.push_back("RIGHT");
    actions.push_back("SHIFT");
  }

  // open the input file...
  ifstream infile;
  if( opt.args.size() > 0 ) {
    infile.open( opt.args[0].c_str() );
    if( !infile.is_open() ) {
      cerr << "File not found: " << opt.args[0] << endl;
      exit( 1 );
    }
  }

  // ... or use stdin
  istream *istrptr;
  if( infile.is_open() ) {
     istrptr = &infile;
  }
  else {
     istrptr = &std::cin;
  }

  string str;
  int linenum = 0;
  int sentnum = 0;
  int errnum = 0;

  vector<item> q;
  q.push_back( item() );
  q.back().word = "<s>";
  q.back().pos = "<s>";
  q.back().idx = 0;
  q.back().goldlink = 0;
  q.back().goldlabel = "LW";
  q.back().link = -1;
  q.back().label = "*NONE*";

  // main loop
  while( getline( *istrptr, str ) ) {
    
    // increase line counter
    linenum++;
    
    // remove new line and carriage return characters
    if( str.find( "\r", 0 ) != string::npos ) {
      str.erase( str.find( "\r", 0 ), 1 );
    }

    if( str.find( "\n", 0 ) != string::npos ) {
      str.erase( str.find( "\n", 0 ), 1 );
    }

    vector<string> tokens;
    Tokenize( str, tokens, " \t" );

    if( tokens.size() < 3 ) {

      if( q.size() <= 1 ) {
	continue;
      }
      
      // process sentence
      
      sentnum++;

      //if( TRAIN ) {
	if( !( sentnum % 100 ) ) {
	  cerr << sentnum << "... ";
	}
	//}
      
      // insert the right wall
      q.push_back( item() );
      q.back().word = "</s>";
      q.back().pos = "</s>";
      q.back().idx = q.size() - 1;
      q.back().goldlink = 0;
      q.back().goldlabel = "RW";
      q.back().link = -1;
      q.back().label = "*NONE*";
      
      for( int i = 0; i < q.size(); i++ ) {
        q[q[i].goldlink].goldin++;
	q[q[i].idx].goldout++;
      }
      
      if( parse( q, actions, nparses, lm, syms ) == -1 ) {
	cerr << "Sentence " << sentnum << ": parse failed.\n";
	errnum++;
      }

      // clear the queue, and initialize
      q.clear();

      q.push_back( item() );
      q.back().word = "<s>";
      q.back().pos = "<s>";
      q.back().idx = 0;
      q.back().goldlink = 0;
      q.back().goldlabel = "LW";
      q.back().link = -1;
      q.back().label = "*NONE*";
      
      continue;
    }
  
    // insert the current word in the queue
    string word = tokens[1];
    string pos = tokens[4];
    string label = tokens[7];
    int link = stringTo<int>( tokens[6] );
    int idx = stringTo<int>( tokens[0] );
    
    q.push_back( item() );
    q.back().word = word;
    q.back().pos = pos;
    q.back().idx = idx;
    q.back().goldlink = link;
    q.back().goldlabel = label;
    q.back().link = -1;
    q.back().label = "*NONE*";
    
  } // while( getline(*istrptr, str) );
  
   if( TRAIN ) {
     cerr << endl << "Finshed processing file with " << sentnum << " sententes and " <<
       errnum << " errors.";
   }

   cerr << endl;

   if( infile.is_open() ) {
     infile.close();
   }
 
   return 0; 
}
