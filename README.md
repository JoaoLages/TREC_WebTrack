# TREC WebTrack 
This is a repository used to employ Machine Learning models on the adhoc task, [TREC Web Track](https://trec.nist.gov/data/webmain.html).
Any issues, PRs or suggestions will be welcome.

To be more specific, these models are reranking models for query-document pairs.
Since the cost of computing the relevance score for every query-document pair is too 
high, the objective is to rerank the QL submissions of each 
year, that you can find in [here](https://github.com/trec-web/trec-web-2014/tree/master/data/runs/baselines).


**These models are capable of ordering a list of text documents according to their relevance to a particular query.**
**It is possible to use this repository to train your rerank models or use a pre-trained on custom data, i.e., 
a set of queries and documents.**


Currently, there are 2 models implemented, described in:

- Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[PACRR: A Position-Aware Neural IR Model for Relevance Matching](https://arxiv.org/pdf/1704.03940.pdf).
*In EMNLP, 2017.*

- Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
 [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval](https://arxiv.org/pdf/1706.10192.pdf). *In WSDM, 2018.*


Their implementation was adapted from the [official release](https://github.com/khui/repacrr).

### To install run (Python 3.5+):
    python setup.py develop

### Create a softlink to point to your stored data:
    ln -s path_to_your_data DATA

Under your DATA directory you'll need to have different data, depending whether:
- [You want to reproduce results on the TREC Web Track](#reproduce-trec-web-track-results)
- [You want to use a pre-trained model on your own data](#using-a-pre-trained-model-on-your-data)


## Reproduce TREC Web Track results
For now, I only provide scripts to  reproduce 2013 and 2014 results. However, you can change the bash and config files accordingly to run on other configurations.
Throughout the instructions, replace `gpu_device` with the CUDA_ID you want to run with (replace with `None` for running on CPU).

Under your `DATA` directory, download the official [similarity matrices](https://drive.google.com/file/d/0B3FrsWe6Y5YqdEtfSjI4N0h1LXM/view?usp=sharing), provided by the authors, and extract them using:

    cd DATA
    tar xvf simmat.tar.gz

Now you can either **train and test** or **test only** the PACRR model.
* For **train and test**, run any script under `bin/test13` or `bin/test14` as the following example: 
    
      bash bin/test1*/run_pacrr_1*val.sh gpu_device
      
    or, to run using a round-robin procedure (will take longer):

      bash bin/test1*/run_pacrr_test1*.sh gpu_device

* For **test only**, you'll need to download my [weights files](https://drive.google.com/open?id=14PaETJGPvNeLIo_YpcjaUak-gset0YLa) and extract them under DATA.
      
      cd DATA
      unzip model_outputs.zip
      
    Then comment the part of the `bin/test1*/` bash files that call `script/train.py` and run the same way as described for **train and test**.

## Using a pre-trained model on your data

You'll need to download my [weights files](https://drive.google.com/open?id=14PaETJGPvNeLIo_YpcjaUak-gset0YLa) and extract them under DATA.

      cd DATA
      unzip model_outputs.zip

and the pretrained [embeddings](https://www.dropbox.com/s/khi6aw48fijg14m/embeddings.zip?dl=0)
      cd DATA
      unzip embeddings.zip
      
      
Now, you will have to change the file `qrels/customdata.txt` according to your data. 
As you can see in the example file I have, the file is constructed with the following format:
```
query text
document text

(...)

query text
document text
```

Just change this file with your queries and document and run:

    bash bin/run_pacrr_customdata.sh

At the end, a message will be printed by that script saying where the `test.probs` file was saved.
That file contains the relevance scores of every query-document pair you inserted in `qrels/customdata.txt`.
     
