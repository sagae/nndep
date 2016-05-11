# nndep: a neural network dependency parser

This is the neural network parser described in

Vaswani & Sagae, 2016. Efficient Structured Inference for Transition-Based Parsing with Neural Networks and Error States. TACL.

The parser uses locally normalized models with error states (see paper above), a straightforward way to perform structured prediction with principled search without the need for full decoding during training.
