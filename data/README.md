# KQA Pro version 1.0

KQA Pro is a large-scale dataset of complex question answering over knowledge base. The questions are very diverse and challenging, requiring multiple reasoning capabilities including compositional reasoning, multi-hop reasoning, quantitative comparison, set operations, and etc. Strong supervisions of SPARQL and program are provided for each question.
If you find our dataset is helpful in your work, please cite us by
```
@article{shi2020kqa,
    title={KQA Pro: A Large Diagnostic Dataset for Complex Question Answering over Knowledge Base},
    author={Shi, Jiaxin and Cao, Shulin and Pan, Liangming and Xiang, Yutong and Hou, Lei and Li, Juanzi and Zhang, Hanwang and He, Bin},
    journal={arXiv preprint arXiv:2007.03875},
    year={2020}
}
```

## Usage
There are four json files included in our dataset:

- `kb.json`, the target knowledge base used to answer questions, which is a dense subset of [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).
- `train.json`, the training set, including 94,376 QA pairs with annotations of SPARQL and program for each.
- `val.json`, the validation set, including 11,797 QA pairs with SPARQL and program.
- `test.json`, the test set, including 11,797 questions, with 10 candidate choices for each. You can submit your predictions and your performance will be shown in our leaderboard.

Following is the detailed formats

**train.json/val.json**

```
[
    {
        'question': str,
        'sparql': str, # executable in our virtuoso engine
        'program': 
        [
            {
                'function': str,  # function name
                'dependencies': [int],  # functional inputs, representing indices of the preceding functions
                'inputs': [str],  # textual inputs
            }
        ],
        'choices': [str],  # 10 answer choices
        'answer': str,  # golden answer
    }
]
```

**test.json**
```
[
    {
        'question': str,
        'choices': [str],  # 10 answer choices
    }
]
```
