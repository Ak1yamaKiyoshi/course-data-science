### Task descriptions 

1. In this task, we need to train a named entity recognition (NER) model for the identification of
mountain names inside the texts. For this purpose you need:
● Find / create a dataset with labeled mountains.
● Select the relevant architecture of the model for NER solving.
● Train / finetune the model.
● Prepare demo code / notebook of the inference results.

2. In this task, you will work on the algorithm (or model) for matching satellite images. For the
dataset creation, you can download Sentinel-2 images from the official source here or use our
dataset from Kaggle. Your algorithm should work with images from different seasons. For this
purpose you need:
● Prepare a dataset for keypoints detection and image matching (in case of using the ML
approach).
● Build / train the algorithm.
● Prepare demo code / notebook of the inference results.

### Structure 
- [task1_bert_alternative_implementation](https://github.com/Ak1yamaKiyoshi/quantummobile-tasks/tree/main/task1_bert_alternative_implementation)
<br> First implementation of first task using pytorch and Hugging face transformers pre-trained BERT model.
<br> [model weights](https://drive.google.com/file/d/1nw-9f-EGTuTuZ4TlIew4ny8YFI20JcMf/view?usp=sharing)
- [task1_nlp_ner](https://github.com/Ak1yamaKiyoshi/quantummobile-tasks/tree/main/task1_nlp_ner)
Main, right one implementation of first task. Used SpaCy and some other utilites.
Compared to the first attempt, the dataset, how it is generated and used,
has changed significantly. The model and the way it is trained have also changed.
The training time has improved. The number of parameters is much smaller compared to BERT.
- [task2_sift_feature_registration](https://github.com/Ak1yamaKiyoshi/quantummobile-tasks/tree/main/task2_sift_feature_registration)
Second task implementation using cv2 library for feature recognizing.
