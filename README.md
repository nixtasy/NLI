# TeamLab NLP - Group 6 Tianxiang Wang & Zhe Fan
**N**atural **L**anguage **I**nference  - on the ART dataset


## Task Description


ART, a challenge dataset, that consists of over 20k
commonsense narrative contexts and 200k explanations. Based on this dataset, we
conceptualize a challenging task, namely "Abductive NLI": a multiple-choice question answering task for choosing the more likely explanation, where the goal is to pick the most plausible explanatory hypothesis
given two observations from narrative contexts.

Raw data format in the ART set:
```
{
"story_id": "005c14c3-27e6-45fe-8a1e-aa1a53ee6602-1",
"obs1": "Jasper told his parents that he wanted a dog.",
"obs2": "His parents decided not to give him a dog.",
"hyp1": "Jasper asked his parents, but they were allergic to dogs.",
"hyp2": "Jasper asked his parents, but they were allergic to rabbits.",
"label": 1
}
```

Our formatted (after pre-processing) version for training on a baseline model:
```
{
"obs1": "Jasper told his parents that he wanted a dog.",
"obs2": "His parents decided not to give him a dog.",
"hyp1": "Jasper asked his parents, but they were allergic to dogs.",
"label": 1 (Ture)
}
{
"obs1": "Jasper told his parents that he wanted a dog.",
"obs2": "His parents decided not to give him a dog.",
"hyp2": "Jasper asked his parents, but they were allergic to rabbits.",
"label": 0 (False)
}
```
## Architecture

 We will have 4 main functional classes, namely: 
 * KB(**K**nowledge **B**ase)
 * Perceptron 
 * Utils 
 * Learning
 
  ### @ KB:
  KB is the structrue stores each (training, dev ...) instance，also used for extracting features
  * self.o1
  * self.o2
  * self.h
  * self.label
  * self.F
  * featureExtraction() // for extracting feature within an instance
  * add_feature() //for adding cross-instance feature
 
  ### @ Perceptron:
  Percptron is the core of our model, in which the training and predicting process happen
  * self.weight
  * self.prediction
  * self.learning_rate // for setting update stride
  * self.max_iteration // setting iterations
  * feedforward() // to calculate wx, the linear combination
  * predict() // to generate predictions given a trained model
  * train() // training 


  ### @ Utils:
  For evaluating our model 
  * self.predictions
  * self.labels
  * self.tp // True Postive
  * self.tn // True Negative
  * self.fp // False Positive
  * self.fn // False Negative
  * self.P // Precision
  * self.R // Recall
  * self.A // Accuracy
  * self.F // F1 score
  * precision()
  * recall()
  * f1()
  * accuracy()
  * evaluation() // call precision, recall, f1, accuracy and print out the scores
  
  
  ### @ learning:
  The class learning contains the whole learning procedure 
  * self.train // stores the training instances of class KB 
  * self.dev // stores the dev instances of class KB 
  * ...
  * ingest_data() // read and re-format our data into tuple pairs, populate self.train and self.dev
  * generate_cross_features() // generate cross-instance features and populate them back into the feature list of each instance
  * get_labels() // get the true labels of all the training/dev instances
  * get_features() // get the feature of all the training/dev instances
  * mimic_predictions() // mimic predictions for comparing with our model

## Methdology
### Perceptron Classifier
### Feature Engineering

## Experiments

### Experimental Design
#### Feature engineering/Error analysis
#### Feature selection
#### Improvements in inference (sequential information, multi-label predictions)
#### More/other classifiers


### Results

## Next Steps
### Advanced Features
