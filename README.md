# MeTooUs
### Setup.py
- Read multiple input files. Function used `read_all_files_in_dir`
- Extract desired rows and append to a common output file. Function used `read_tweet_file`
### CalculateTopicProportion.py
- applying preprocessing to tweet data to remove commonly used words/chars in tweet ie "http", "@" & "metoo" etc. Function used `preprocessing`
- convert tweet data to tokens and remove STOP words. Function used `secondary_preprocess`
- apply stemming and lemmatization to tweet data. Function used `lemmatize_stemming`
- compute coherence values by using LDA model with different combinations of parameters (num of topics, alpha, beta). Function used `compute_coherence_values`
- used `calculate_topic_proportion` function for orchestration of above functions
### DataFilter.p'y
- used to generate tweet data subset based on filtering of `start_data`, `end_date` and `gender`.
- the output can be used as an input for `GenerateGraph.py`
### GenerateGraph.py
- aggregation of tweet data is done on a monthly basis. Function used `generate_monthly_graph`
- aggregation of tweet data is done on an yearly basis. Function used `generate_yearly_graph`
