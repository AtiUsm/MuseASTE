The contents of the folder is described below:

- **Main dataset:** train.csv, devel.csv, contain transcripts and gold annotations.

- **Full Dataset Folder:** Full dataset with transcripts, annotations, and  post-prcossed annotations

- **ASTE Annotations:** Full annotations. The id, segment_id, columns in the train and devel files of our annotation dataset match with the id, segment_id, and label_topic columns of  the train and devel files in the original MuSe-Topic dataset.
  
- **Category Annotations:** Created our own fine grained category annotation which is based on each triple, instead of segment wise like Muse. The aspect and topic labels are in AspectCategoryLabel.csv, and TopicCategoryLabels.csv. Car Dataset Categories Readme.pdf contains a description of these categories.
  
- **Supplementary Files** : Folder contain supplementary code files, like sampling, statistical analysis (creates figures and charts on the dataset), simple usecase and annotator feedback.
  - The code for generating figures, tables and charts is in Dataset Statistical Analysis.ipynb
  - The code for Sampling is in Sampling.ipynb
  - The Simplest version OF USECASE demo figure that creates ASTE knowledge graphs is in USECASE_DEMO_FIGURE_SIMPLEST_NOTEBOOK_VERSION.ipynb
