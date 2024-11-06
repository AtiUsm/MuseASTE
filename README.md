**Code and Data** for the following paper [MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos](https://doi.org/10.1016/j.eswa.2024.125695) 

Usmani, A., Hamood Alsamhi, S., Jaleed Khan, M., Breslin, J., Curry, E., MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos, Expert Systems with Applications (2024), doi: https://doi.org/10.1016/j.eswa.2024.125695

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
6.	The Simplest version OF USECASE demo figure that creates ASTE knowledge graphs is in USECASE_DEMO_FIGURE_SIMPLEST_NOTEBOOK_VERSION.ipynb
7.	Other folders contain supplementary material, like post-processed annotations, example annotator feedback, and sampled subset.
8.	Code for baselines is taken from the original repositories and adapted to our datatset, the orginial repositories are cited in respective readme files
#  Acknowledgement
[1] Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu, and Luo Si. Knowing what, how and why: A near complete solution for aspect-based sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8600–8607, 2020.

[2] Stappen, Lukas, Alice Baird, Lea Schumann, and Schuller Bjorn. "The multimodal sentiment analysis in car reviews (muse-car) dataset: Collection, insights and improvements." IEEE Transactions on Affective Computing (2021).

[3] Muse 2020 - ACM MM 2020 (no date) MuSe 2020 - ACM MM 2020. Available at: https://sites.google.com/view/muse2020 (Accessed: 07 November 2023).

[4] Yan, H., Dai, J., Ji, T., Qiu, X., & Zhang, Z. (2021, August). A Unified Generative Framework for Aspect-based Sentiment Analysis. In C. Zong, F. Xia, W. Li, & R. Navigli (Eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 2416–2429). doi:10.18653/v1/2021.acl-long.188

[5] Xu, L., Chia, Y. K., & Bing, L. (2021, August). Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction. In C. Zong, F. Xia, W. Li, & R. Navigli (Eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 4755–4766). doi:10.18653/v1/2021.acl-long.367

[6] Chen, S., Wang, Y., Liu, J., & Wang, Y. (2021, May). Bidirectional machine reading comprehension for aspect sentiment triplet extraction. In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 14, pp. 12666-12674).

[7] Zhang, W., Li, X., Deng, Y., Bing, L., & Lam, W. (2021, August). Towards generative aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 504-510).

