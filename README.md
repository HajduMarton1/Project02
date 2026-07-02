# Project02
Introduction to machine learning
Started on: 2026.04.16
Made by Fodor Gergely and Hajdu Márton

Project name: Music Genre Classification

This project is about classifying music into 10 genres. The program uses the 30-second song snippets contained in the GTZAN database
divided into equal user-defined lengths for training, validation and initial testing, but is also capable of processing user-imported (wav) audio files. 
The model in focus is the k-Nearest Neighbors model, but a Random Forest classifier is included for comparison.
To combat the target leakage resulting from the similarities of the possibly many snippets for each song,
the cross-validation is group-based and majority voting is used in the final prediction.
This project came from the project ideas (5th project idea), so a broader technical desciption can be found there.
Other than the MFCC features, the program can extract their first and second derivatives, the tonal distribution of the segments,
their spectral centroid and BPM for a more various feature composition.
