# TREC WebTrack 2013 and 2014
This is a repository used to employ Machine Learning models on the adhoc task of Web Track 2013 and 2014.

Since the cost of computing the relevance score for every query-document pair is too high, the method used to evaluate the models is to rerank the QL submissions of each year, that you can find [here](https://github.com/trec-web/trec-web-2014/tree/master/data/runs/baselines).

**This repo is still under development. Any issues, PRs or suggestions will be welcome.**
It currently only has an implementation of the model described in:

Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[PACRR: A Position-Aware Neural IR Model for Relevance Matching](https://arxiv.org/pdf/1704.03940.pdf).
*In EMNLP, 2017.*

Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[RE-PACRR: A Context and Density-Aware Neural Information Retrieval Model](https://arxiv.org/pdf/1706.10192.pdf).
*In Neu-IR workshop, 2017.*

Its implementation was adapted from the [official release](https://github.com/khui/repacrr).
### To install run (Python 3.5+):
    pip install -r requirements.txt
    python setup.py develop
* For CPU version:

        pip install tensorflow
* For GPU version:
    
        pip install tensorflow-gpu




### Create a softlink to point to your stored data:

    ln -s path_to_your_data DATA

From now on, it depends if you want to only reproduce REPACRR results or to run your own version of the model.

## Reproduce REPACRR results
For now, it is only possible to reproduce 2013 results (**and they still have to be optimized**).
Throughout the instructions, replace `*` with the year you want to use for validation (09, 10, 11, 12 or 14) and 
`gpu_device` with the CUDA_ID you want to run with (replace with `None` for running on CPU).

Under your `DATA` directory, download the official [similarity matrices](https://drive.google.com/file/d/0B3FrsWe6Y5YqdEtfSjI4N0h1LXM/view?usp=sharing), provided by the authors, and extract them using:

    cd DATA
    tar xvf simmat.tar.gz

Since the similarity matrices are not complete, I constructed new qrel files to be able to work with them 'properly' under `qrels/new_*`.

Now you can either **train and test** or **test only** the REPACRR model.
* For **test only**, you'll need to download my [weights files](https://drive.google.com/file/d/1g7xkoZ5cZgKWNyhxkN0CbZASaUPumA0G/view?usp=sharing) and extract them under DATA.
    
      cd DATA
      unzip model_outputs.zip
    Then run:
    
      bash bin/test13/eval_repacrr_*val.sh
* For **train and test**, run: 
    
      bash bin/test13/run_repacrr_*val.sh gpu_device



## (TODO) Run your own version of REPACRR (provide your data)

**You will need to have access to 
[ClueWeb09](https://lemurproject.org/clueweb09/) and [ClueWeb12](https://lemurproject.org/clueweb12/) datasets.**
Most of the code is available. I will post instructions on how to run it soon.

Nevertheless, if you can't wait, I'll be glad to instruct you directly how to run it.

# TODOs
- [ ] Get close to reported results of PACRR
- [ ] Finish REPACRR model
  - [ ] Add context
  - [ ] Add extra Conv layer with kernel len_query*len_query
  - [ ] Cascade k-max
- [ ] Add other models
- [ ] Add instructions to run different models with custom data
