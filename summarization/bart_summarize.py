from transformers import pipeline
import requests
#import pprint
import time
import os
from os.path import join as pj
import sys
import pandas as pd
import numpy as np
import argparse
#pp = pprint.PrettyPrinter(indent=14)


def main():
	parser = argparse.ArgumentParser()
	summarizer_bart = pipeline(task='summarization', model="bart-large-cnn")

	# Required parameters
    parser.add_argument(
        "--input_data_file", default=None, type=str, required=True, help="The input data file (a csv file)."
    )
    parser.add_argument(
        "--data_dir", default=None, type=str, required=True, help="The data directory to save"
    )

    args = parser.parse_args()
    fname = args.input_data_file
    data_dir = args.data_dir
    

    docs = pd.read_csv(pj(data_dir,fname))
    bart_summaries_df = docs.copy()

	bart_summaries = []
	for idx, row in docs.iterrows():
	    t0 = time.time() # timer
	    doc_summary = summarizer_bart(str(row.summary_tokenized), min_length=1, max_length=500) # change min_ and max_length for different output
	    print("Summarization took " + str(round((time.time() - t0) / 60, 2)) + " minutes.")
	    bart_summaries.append(doc_summary[0]['summary_text'])
	    if idx%20==0 or idx==docs.shape[0]-1: # save checkpoints in case of crash
        print(idx)
        with open("chkpt_"+str(idx)+".pkl",'wb') as f:
            pickle.dump(bart_summaries, f)
	
	bart_summaries_df["bart_summaries"] = bart_summaries

	bart_summaries_df.to_csv(pj(data_dir,"bart_"+fname),index=False)