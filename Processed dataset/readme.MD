Four SOTA baseline models were implemented on our dataset, as well as current benchmark dataset.
Due to limitations of current state-of-the-art baseline models (predict explicit start and position,do not work with implicit /empyty triples) the  dataset was processed as followed for a valid comparison across all baselines and with the benchmark dataset:
 - removing segments not yielding any triples, or empty triples
 - removing segments contatining implicit labels
 - replacing the aspects and opinion with their positional token numbers (or of their lemmatized/nearest version) in the text transcript segment, as required by constraints of current baseline models

Results reported in the paper
