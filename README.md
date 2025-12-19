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
4.	Additionally, in case you also want to access video footage, audio files, faces, or segment-wise topic labels, go to the primary dataset MuseCar-2020 [2,3] (Link: [MuSeTopic 2020 - ACM MM 2020 (google.com)](https://sites.google.com/view/muse2020/challenge/get-data?authuser=0), and acquire the Muse-Topic dataset to get access.
5.	The baselines run on both our and SemEval dataset, hence all 4 Sem-Eval datasets [8] are also contained in the repository.

# Baselines
-**BRMC** Machine Comprehension based [7].

-**GAS** Large Language Model (T5) based generative approach [8].

-**BART-ABSA** Pointer based indices generation/prediction approach. Predicts start and end positions of a tag [5].

-**Span-ASTE** Tagging based span prediction method [6].

# Results

For final baseline results, consult the paper and the corrigendum below:

The highest **triple-F1** is **0.25**.

[Paper](https://www.sciencedirect.com/science/article/pii/S0957417424025624)

[Corrigendum](https://pdf.sciencedirectassets.com/271506/AIP/1-s2.0-S0957417425029938/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCICNWPuXxHThjjlXrF7Q3zCNBrHrwvXFzr7hU8OHKMbZkAiEAz9uy3Opk24757nv1BFs3ahaBpdM2VQvJnRnK85dIk0sqsgUILBAFGgwwNTkwMDM1NDY4NjUiDNjnpdv7BpbTJWZB7iqPBXJdEC0k2Txd5gnRN0Y3R0YRiPUe5yjHRb4nP6CR9AFjiXFPx31dFQBHCZHwjM%2FJ%2FBrKh0satLzKo0C7lvAPzE%2FsC6R0iqL20GKRlh2whH75TlqYmIlsx%2FOdLI5O%2BYMuXExcvnS1eB4Hv2FRTXsuliOCNjOz2T3BFYylVDZ5GYv%2FFwhLA49hjkGVfaHg%2Beig6rCsWTog49P4%2B%2FqVXKIpq0lxj2gN%2BLejTrrfbayFpUEM9IcpHQS2uoUIjLSY6GQw%2FPN8YcwdhEJaV7wDZgVQfP6pq74%2Fznz%2Bmhr0kNRToIUGiPpo%2BFB9CUFN41Ilu87e%2B68K%2BB7ULRS7gOWYQ5sg5E%2BYAQZ7m3PY9jQODD4aSmo%2FSagAET3DdZl6UTj2YKICFGi65Vd3K%2Bkjv0umeJPvF5%2BU1lKhkoS1NT74CaxXNXZZVXFT6v09RR0kOiVkwMCjqnQp%2Bq1XcGyc3U9YowSKaOjEE5LLvc45cRqgERP0MN%2BN%2BHJBXWlU5D%2F03truFHpNbgAwVPxxcq44lzlFSpkh0J7Lb40ebbqQS0OSJRMPB7SyaKP45SquWv5ZEotJiIitJQPUZE9wY4hxY2vN6pzhmWsXBcic7qVAx55DB8NynMxqo0IQ2LwjzR8J4t%2BeZHAbuHrAvbU0Xn%2BQLBXKpNpDxnquw0C%2BUSjnCOqjfmfWDaQMuc7fvdXcZQMKet%2B77df6eegrohWONw0zKW0H3EMv33hB3BYlRi2uOWKF7tL5JVj%2Ful6QLPhwe0N3oibRNEDUFJLJfph%2BBNFLVyoM%2FOtYBxbpWTi9qq2DEmuuBA7APADnkC3YtDjc8IDQRUWvShKAulbKXnAwCGVNV0Dt7GrtzHZG%2FbWpaQlGycH5SVyOJgsw0KbbxQY6sQG1CxXwu%2B9h1OrYt8ZDCkSEYoUEYZzbXfRdfcVoXE7QyOxx1JCDBQwrggkrYmMTwmI0f9dMPnP4tWWPmwuofO%2F1izIoMJSPvkkTCqK9gbi5HZPDDSoK%2FxlHVM1%2BFMubXJt84mq2%2FjLUFWJnVZ4vxWhTCHiw3qRsmeOOoiB97zoJvUs4AXqRKzKAgcKymbCMkM5%2Bd3ms%2FimxrPxDMxgt0yLL99Z7lE3NymgzjAoBU5jyUB0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250902T120020Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZQJ2VENO%2F20250902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b41a3a2ef2aac2f19eb1863ab530df9bef3bb11ccfc4ef13f3546bd6b6b5eb1f&hash=10ad8169b39da01aa805975cb23af19285fac3626f5f11dccd5f14b97e027c5a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0957417425029938&tid=spdf-6b013c7a-02d0-4a0a-a923-69c7664fecdf&sid=bc5f09617e3be842b44998847b16e9e7f5bdgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=13065a575f0c015a02&rr=978cd8ab28365759&cc=ie)
# Demo
Demo gives you a sneak-peek in to one of our ASTE knowledge graphs and allows you to play with it. We created topic and ASTE labels for one car in our dataset (alternatively you can substitute segment-wise topic labels from primary dataset for more), and implemented a demo online graph inspection tool using streamlit. It also gives an insight into the aspect, sentiment, opinion annotations.
![Screenshot (166)](https://github.com/user-attachments/assets/eb01bc87-051f-4cc1-ae06-3a71d327bbad)
This demo gives you a sneak-peek in to one of our cars of our dataset. We created topic and ASTE labels for one car (example_demp.csv), and implemented this demo using streamlit.

![Screenshot (167)](https://github.com/user-attachments/assets/e083d459-0a4a-46d9-bb4b-7073d7e55a02)


Requirements:
@@ -129,45 +11,17 @@ pip install matplotlib
pip install streamlit
pip install networkx
pip install streamit-extras
pip install st-annotated-text
pip install scipy
```

To run the (demo) code:
Or:

```
streamlit run demo/demo.py

pip install -r requirements.txt
```
To run the code:

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
```
streamlit run demo.py

### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc/4.0/). This work is licensed under a [Creative Commons Attribution-NonCommercial-4.0-International License](http://creativecommons.org/licenses/by-nc/4.0/) for Noncommercial (academic & research) purposes only and must not be used for any other purpose without the author's explicit permission.
```