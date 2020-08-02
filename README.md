Abductive Natural Language Inference (aNLI) is a commonsense benchmark
dataset designed to test an AI system’s capability to apply abductive reasoning
and common sense to form possible explanations for a given set of observations.
Formulated as a binary-classification task, the goal is to pick the most
plausible explanatory hypothesis given two observations from narrative
contexts.

The data in this archive was originally downloaded from
[here](https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip).
To find more details about the dataset, checkout
[this](https://arxiv.org/abs/1908.05739) paper.

Originally, the files are formatted as Jsonlines (each line is a json object, i.e. one instance per line).
Each instance in the dataset consists of the following fields:

```
story_id: Story ID
obs1: First observation in narrative order. (i.e. time order)
obs2: Second observation in narrative order.
hyp1: First hypothesis choice.
hyp2: Second hypothesis choice.
```
You can find the jsonl formatted files in the folder ``jsonl/``.

Example entry:

```
{
  "story_id": "005c14c3-27e6-45fe-8a1e-aa1a53ee6602-1",
  "obs1": "Jasper told his parents that he wanted a dog.",
  "obs2": "His parents decided not to give him a dog.",
  "hyp1": "Jasper asked his parents, but they were allergic to dogs.",
  "hyp2": "Jasper asked his parents, but they were allergic to rabbits."
}
```


The labels are available in the jsonl/{train/dev}-labels.lst file in the following format:
```
2
1
1
…
```

Labels 1 and 2 correspond to hyp1 or hyp2 (indicating which hypothesis correctly supports the observations).

We also provide you the dataset formated as TSV (fields are separated by tabs).
Find this dataset in the folder ``tsv/``.
In the tsv format we join the labels with the instances and the header is the following:

story_id	obs1	obs2	hyp1	hyp2	label

Example entry:
```
000731f7-c71d-49c8-b8cd-b6848933db99-1  Chad loves Barry Bonds. Chad ensured that he took a picture to remember the event.     Chad missed Barry Bonds.        Chad met Barry Bonds.   2
```

One can submit their predictions (here)[https://leaderboard.allenai.org/anli/submissions/get-started] .
