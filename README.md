This repository contains code for the COLING 2018 paper "Document-level Multi-aspect Sentiment Classification by Jointly Modeling Users, Aspects, and Overall Ratings".

## About this code

This code was developed for python 3.6 and PyTorch 0.3.0.

## How to run

### Get the dataset

Download our dataset from [here](https://pan.baidu.com/s/191Nu55GksNkK7s1I4FwEvw) with password "u61k".

train, test, dev represent training set, testing set and development set. Each line in each file is a sample. Each line makes up of 5 elements, which are split by "\t\t". Element 1 is the sample ID, element 2 is the user ID, element 3 is the overall rating , element 4 is the aspect rating and element 5 is the review content. For example, the first line in test is "\_\_46906\_\_		0452FC4DD1569050B7AEF6880B4AE7EF		4		5 5 -1 -1 4 5 3		the staff is off the hook nice , they are very helpful , pour you a nice glass of water or bubbly when you arrive . <ssssss> the breakfast that was included was good , the lobby is very nice and the bar is also great . <ssssss> however , be aware that if you are an american you will think this is a shoebox . <ssssss> the room is tiny , maybe 250 square feet on a good day . <ssssss> that is the only negative about this place ."

sample ID --> \_\_46906\_\_

user ID --> 0452FC4DD1569050B7AEF6880B4AE7EF

overall rating --> 4

aspect rating --> 5 5 -1 -1 4 5 3 (service, cleanliness, business service, check in, value, location, room)

review content --> the staff is off the hook nice , they are very helpful.....

Aspect rating is from 1 to 5, -1 means the user doesn't give a rating for the specific aspect.

keywords of each aspects are given in "aspect.words" file.

### Run training

cd Codes

python main.py --config config_train.json

### Run testing

cd Codes

python main.py --config config_test.json


