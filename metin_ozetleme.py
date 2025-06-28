# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 18:29:50 2025

@author: omer_
"""

from transformers import pipeline


summarizer=pipeline("summarization")


text="""
Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention. Today, machine learning is used in a wide range of applications, from recommendation systems and image recognition to self-driving cars and natural language processing. One of the most important aspects of machine learning is the ability to continuously improve its performance as it is exposed to more data. However, building effective machine learning models requires careful selection of algorithms, proper data preprocessing, and rigorous evaluation to avoid bias and ensure accuracy.
"""

# metini ozetleme

summary=summarizer(
    text,
    max_length=90,
    min_length=45,
    do_sample=True)

# Özeti ekrana yazdır

print("Özet",summary[0]["summary_text"])