# CRC-MRC
code for paper 《CRC-MRC: Reader Comments Augmented Machine Reading Comprehension for Social Emotion Prediction》

It is a Clustering-based Reader Comments Augmented Machine Reading Comprehension framework~({CRC-MRC}) to comprehensively model the reading process from the readers' perspective while browsing news and comments.

## ✈️ Getting Started

### Installation

**Make sure you have Python >= 3.9**  
Install the required packages using pip  

```bash
pip install -r requirements.txt
```
download the pretrain model and put it on pretrained_model_path_/
### Usage
Set up OPENAI key in src/args.py

Run the run.sh to start
```bash
cd src/scripts
bash run.sh
```

Or ```bash cd src/MRC```
Then Run the main.py to start

```python
python main.py \
    --model_name ISEAR_BERT_MRC_A+C \
    --dataset_name ISEAR_NEW \
    --device 'cuda:2' \
    --DataSet_Name ISEAR_ChatGPT \
    --textname All_text_cluster\
    --batch_size 16 \
    --accum_iter 1 \
    --num_epoch  20\
    &
```

