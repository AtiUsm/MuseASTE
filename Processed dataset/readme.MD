Four SOTA baseline models were implemented on our dataset, as well as current benchmark dataset.

Due to limitations of current state-of-the-art baseline models (predict explicit start and position,do not work with implicit /empyt triples) that emobody the limitations of benchmark dataset (0 Implicit Aspects and Opinions), the  dataset was processed as followed for a valid comparison across all baselines and with the benchmark dataset:
 - removing segments not yielding any triples, or empty triples to run the SOTA models
 - removing segments contatining implicit labels to run SOTA models that predict start and end positions of aspects and opinions
 - replacing the aspects and opinion with their positional token numbers (or of their lemmatized/nearest token) in the text transcript segment, as required by constraints of current baseline models

Results reported in the paper for different experiments and datasets (full, processed, benchmark, comparison), with detailed explanations of experimental settings. 
# Instructions
1. *Format*-The correct final format to run the baselines is shown is sample(format).txt file for a sample of 25 rows. 
2. *Samples*- Sample train test and dev files in correct format are shown.
3. *Annotation Files*- Full and Complete train, dev and test annotation files are provided for each text transcript segment indicated by their id and segment_id.
4.  First obtain the muse text transcripts.
5. Replace the transcript ids with the actual segment text in all annotation files(as shown in sample(format)) and save each file as train.txt, dev.txt, and text.txt
6. After completing steps 4 & 5, *the final processed dataset with text transcripts is in 'Muse ASTE Full Processed Dataset with transcripts' folder.*
