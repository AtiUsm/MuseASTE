# MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos
[![MuseASTE](https://img.shields.io/badge/MuseASTE-blueviolet?style=flat-square&logo=Github&label=Github&color=blueviolet
)](https://github.com/AtiUsm/MuseASTE/tree/main)
[![ElsevierLink](https://img.shields.io/badge/Page-green?style=flat-square&logo=Elsevier&logoColor=White&label=Elsevier&labelColor=White&color=green
)](https://www.sciencedirect.com/science/article/pii/S0957417424025624)
[![ResearchGateLink](https://img.shields.io/badge/pdf-gray?style=flat-square&logo=researchgate&label=ReserachGate&color=red
)](https://www.researchgate.net/profile/Atiya-Usmani/publication/385563953_MuSe-CarASTE_A_comprehensive_dataset_for_aspect_sentiment_triplet_extraction_in_automotive_review_videos/links/676869d0e74ca64e1f25492e/MuSe-CarASTE-A-comprehensive-dataset-for-aspect-sentiment-triplet-extraction-in-automotive-review-videos.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)
[![Demo](https://img.shields.io/badge/Demo%20%5BOnline%5D-darkgreen?label=Visit%20Here)](https://museaste-t5fkvvgrvagkl9soq7uu5j.streamlit.app/)
[![PwC](https://img.shields.io/badge/MuseASTE-blue?style=flat-square&logo=Papers%20with%20Code&label=Papers%20with%20Code&color=blue
)](https://paperswithcode.com/dataset/musecar-aste)
[![Slides](https://img.shields.io/badge/MuseASTE-orange?style=flat-square&logo=SlideShare&label=ppt&color=orange)](https://github.com/AtiUsm/MuseASTE/blob/main/Muse-CAR%20ASTE%20dataset%20(1).pptx)
[![Poster](https://img.shields.io/badge/Poster-%230D98BA?style=flat-square&label=Insight&labelColor=%2363E2C6&color=%236EF9F5)](https://github.com/AtiUsm/MuseASTE/blob/main/Muse%20Poster2.pdf)





**Code and Data** for the following paper  [MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos](https://doi.org/10.1016/j.eswa.2024.125695) 


# MuseASTE
Aspect Sentiment Triplet Extraction (ASTE) Annotations for the MuSe-Car Dataset - A multi-modal dataset consisting of many hours of video footage from YouTube and transcripts of reviewing automotive vehicles, mainly in English language. The largest dataset repository curated for ASTE. 

![Screenshot (155)](https://github.com/user-attachments/assets/b12acf69-cd8f-47a8-8926-bb6a3ac55752)


# Task Description
Aspect Sentiment Triplet Extraction (ASTE) within the automotive review domain. ASTE as a task was introduced by Peng et al. [1], which is one of the tasks among the 7 sub-tasks of aspect-based sentiment analysis (ABSA). It gives a complete picture or story about a product by extracting triplets from review sentences. These tripleS are of the form <a,o,s> consisting of an aspect a, an opinion o, and a sentiment s.   For example, from the sentence *“the gearbox is rubbish”*, the triple *(gearbox, rubbish, NEG)* is extracted. Sometimes, an additionaly aspect category is also predicted.
![image](https://github.com/user-attachments/assets/f0fa7462-cb25-4ec4-8e26-9c075661a058)

# Key Aspects
Benchmark- Current Benchmark for Aspect Sentiment Triplet Extraction  i.e., ASTE-V2[9]  Sem-Eval datasets(14lap, 14 res, 15res,16res) all four datasets combined.

![Screenshot (149)](https://github.com/user-attachments/assets/897c9bc9-c83d-4e26-9ed6-855ed68b93e0)


# Domain
This dataset can be applied to:

Aspect Sentiment Triplet Extraction (ASTE)

Aspect Based Sentiment Analysis (ABSA)

Target Aspect Sentiment Detection (TASD)

Aspect Category Classification (ACC)

Sentiment Classification, Opinion Mining

# Research Citation
***Cite the following paper** if you use this dataset, also star our repo, and **follow the instructions** mentioned below and **cite additional** relevant citations :*

**Main paper:**
1. Muse-ASTE:
```
@article{USMANI2025125695,
title = {MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos},
journal = {Expert Systems with Applications},
volume = {262},
pages = {125695},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.125695},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424025624},
author = {Atiya Usmani and Saeed {Hamood Alsamhi} and Muhammad {Jaleed Khan} and John Breslin and Edward Curry}
}
```
**Additional citations:** 

2. Original Muse-Dataset and MuSe-challenge:
```
@article{stappen2021multimodal,
  title={The multimodal sentiment analysis in car reviews (muse-car) dataset: Collection, insights and improvements},
  author={Stappen, Lukas and Baird, Alice and Schumann, Lea and Schuller, Bj{\"o}rn},
  journal={IEEE Transactions on Affective Computing},
  volume={14},
  number={2},
  pages={1334--1350},
  year={2021},
  publisher={IEEE}
}
@inproceedings{stappen2020muse,
  title={Muse 2020 challenge and workshop: Multimodal sentiment analysis, emotion-target engagement and trustworthiness detection in real-life media: Emotional car reviews in-the-wild},
  author={Stappen, Lukas and Baird, Alice and Rizos, Georgios and Tzirakis, Panagiotis and Du, Xinchen and Hafner, Felix and Schumann, Lea and Mallol-Ragolta, Adria and Schuller, Bj{\"o}rn W and Lefter, Iulia and others},
  booktitle={Proceedings of the 1st International on Multimodal Sentiment Analysis in Real-life Media Challenge and Workshop},
  pages={35--44},
  year={2020}
}

```
3.If you use any baseline code - BMRC[7], BART-ABSA[5], Span-ASTE[6], GAS[8], then cite them too, if relevant.


# Instructions
1.	The main gold dataset and annotations are in the dataset folder. The processed dataset for comparison with baselines is in the processed dataset folder.
2.	The text transcripts are already added after seeking permission from MuSe [2]. **Please manditorily cite our paper[10] (provided under Research Citation), and also cite the Muse-Dataset[2] and challenge[4].**
3.	Code for baselines is adapted from the original repositories and adapted to our datatset, the orginial repositories are cited in respective readme files. **Provided you use any baseline code (see Baselines) then cite them.** We also provide our experimental settings and environment file.
4.	Additionally, in case you also want to do supervised topic modelling or ACC (Aspect Category Classification), go to the primary dataset MuseCar-2020 [2,3] (Link: [MuSeTopic 2020 - ACM MM 2020 (google.com)](https://sites.google.com/view/muse2020/challenge/get-data?authuser=0), and acquire the Muse-Topic dataset to get access to topic/category label that maybe of interest to you.
5.	The baselines run on both our and SemEval dataset, hence all 4 Sem-Eval datasets [8] are also contained in the repository.

# Baselines
-**BRMC** Machine Comprehension based [7].

-**GAS** Large Language Model (T5) based generative approach [8].

-**BART-ABSA** Pointer based indices generation/prediction approach. Predicts start and end positions of a tag [5].

-**Span-ASTE** Tagging based span prediction method [6].

# Demo
*Added November 11*
Demo gives you a sneak-peek in to one of our ASTE knowledge graphs and allows you to play with it. We created topic and ASTE labels for one car in our dataset (alternatively you can substitute segment-wise topic labels from primary dataset for more), and implemented a demo online graph inspection tool using streamlit. It also gives an insight into the aspect, sentiment, opinion annotations.
![Screenshot (166)](https://github.com/user-attachments/assets/eb01bc87-051f-4cc1-ae06-3a71d327bbad)


Requirements:
```
pip install pandas
pip install matplotlib
pip install streamlit
pip install networkx
pip install streamit-extras
pip install scipy
```

To run the (demo) code:

```
streamlit run demo/demo.py

```

# Usecase

![Screenshot_7-11-2024_6133_](https://github.com/user-attachments/assets/e5464294-606d-43fd-b64b-8bbe8204c61a)

# Contact

[![mail](https://github.com/user-attachments/assets/16a603cf-0e51-450a-9e79-07147f26ceb8)
](mailto:museaste@gmail.com)  **museaste@gmail.com**

#  Acknowledgement
[1] Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu, and Luo Si. Knowing what, how and why: A near complete solution for aspect-based sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8600–8607, 2020.

[2] Stappen, Lukas, Alice Baird, Lea Schumann, and Schuller Bjorn. "The multimodal sentiment analysis in car reviews (muse-car) dataset: Collection, insights and improvements." IEEE Transactions on Affective Computing (2021).

[3] Muse 2020 - ACM MM 2020 (no date) MuSe 2020 - ACM MM 2020. Available at: https://sites.google.com/view/muse2020 (Accessed: 07 November 2023).

[4] Stappen, L., Baird, A., Rizos, G., Tzirakis, P., Du, X., Hafner, F., Schumann, L., Mallol-Ragolta, A., Schuller, B.W., Lefter, I. and Cambria, E., 2020, October. Muse 2020 challenge and workshop: Multimodal sentiment analysis, emotion-target engagement and trustworthiness detection in real-life media: Emotional car reviews in-the-wild. In Proceedings of the 1st International on Multimodal Sentiment Analysis in Real-life Media Challenge and Workshop (pp. 35-44).

[5] Yan, H., Dai, J., Ji, T., Qiu, X., & Zhang, Z. (2021, August). A Unified Generative Framework for Aspect-based Sentiment Analysis. In C. Zong, F. Xia, W. Li, & R. Navigli (Eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 2416–2429). doi:10.18653/v1/2021.acl-long.188

[6] Xu, L., Chia, Y. K., & Bing, L. (2021, August). Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction. In C. Zong, F. Xia, W. Li, & R. Navigli (Eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 4755–4766). doi:10.18653/v1/2021.acl-long.367

[7] Chen, S., Wang, Y., Liu, J., & Wang, Y. (2021, May). Bidirectional machine reading comprehension for aspect sentiment triplet extraction. In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 14, pp. 12666-12674).

[8] Zhang, W., Li, X., Deng, Y., Bing, L., & Lam, W. (2021, August). Towards generative aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 504-510).

[9] xuuuluuu. (2020). GitHub - xuuuluuu/SemEval-Triplet-data: Aspect Sentiment Triplet Extraction (ASTE) dataset in AAAI 2020, EMNLP 2020 and ACL 2021. GitHub. (https://github.com/xuuuluuu/SemEval-Triplet-data) . Accessed 12 Mar. 2024.

[10] Usmani, A., Alsamhi, S.H., Khan, M.J., Breslin, J. and Curry, E., 2025. MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos. Expert Systems with Applications, 262, p.125695.

### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc/4.0/). This work is licensed under a [Creative Commons Attribution-NonCommercial-4.0-International License](http://creativecommons.org/licenses/by-nc/4.0/) for Noncommercial (academic & research) purposes only and must not be used for any other purpose without the author's explicit permission.
