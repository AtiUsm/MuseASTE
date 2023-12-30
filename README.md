# MuseASTE
Aspect Sentiment Triplet Extraction Annotations for the MuSe-Car Dataset - A multi-modal dataset consisting of many hours of video footage from YouTube and transcripts of reviewing automotive vehicles, mainly in English language. 
# Task Description
Aspect Sentiment Triplet Extraction (ASTE) within the automotive review domain. ASTE as a task was introduced by Peng et al. [1], which is one of the tasks among the 7 sub-tasks of aspect-based sentiment analysis (ABSA). It gives a complete picture or story about a product by extracting triplets (a,s,o) from review sentences. These triplets are of the form <a,o,s> consist of an aspect a, an opinion o, and a sentiment s.   For example, from the sentence “the gearbox is rubbish”, the triplet (gearbox, rubbish, NEG) is extracted. 
# Instructions
1.	Access the Aspect Sentiment Triplet AExtraction Annotations provided here.
2.	Then go to the primary dataset MuseCar-2020 [2] ([MuSe 2020 - ACM MM 2020 (google.com)](https://sites.google.com/view/muse2020), and acquire the Muse-Topic dataset to get access to the original transcript texts.
3.	The id, segment_id, label_topic columns in the train and devel files of our annotation dataset match with the id, segment_id, and label_topic columns of  the train and devel files in the original MuSe-Topic dataset.
4.	The code for generating figures, tables and charts is in Dataset Statistical Analysis.ipynb
5.	The code for Sampling is in Sampling.ipynb
6.	Other folders contain supplementary material, like post-processed annotations, example annotator feedback, and sampled subset. 

#  Acknowledgement
[1] Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu, and Luo Si. Knowing what, how and why: A near complete solution for aspect-based sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8600–8607, 2020.

[2]Stappen, Lukas, Alice Baird, Lea Schumann, and Schuller Bjorn. "The multimodal sentiment analysis in car reviews (muse-car) dataset: Collection, insights and improvements." IEEE Transactions on Affective Computing (2021).

[3] Muse 2020 - ACM MM 2020 (no date) MuSe 2020 - ACM MM 2020. Available at: https://sites.google.com/view/muse2020 (Accessed: 07 November 2023).
