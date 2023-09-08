# Intelligent-Call-Prioritization-using-SER

To detect emotion from audio cues and based on prosody and transcribed text from audio to thereby prioritize the calls in the waiting queue of a call center.


![img1](https://github.com/deep0307/Intelligent-Call-Prioritization-using-SER/assets/52921002/33b0f3ca-47fe-4566-b5aa-746efe7b3dff)

Flowchart Diagram of Call Processing and Prioritization.


The project contains 3 modules- 


1) Audio Pre-Processer Module: This module receives the raw audio input and its objective is to clean the audio and extract the
features of the audio input which will further be used by the SER module for emotion detection.

2)Speech Emotion Recognition Module: In this module, the audio input’s extracted feature set is analyzed to detect the emotion of the caller.

3)Textual Emotional Analysis Module: The transcribed content of the call is utilised as input in this module, which is processed in three steps: tokenizing the text and counting unique tokens, padding, and finally converting labels to
integers.

4)Call Priortizer Module: This module feeds the detected emotions from the input calls into an algorithm that prioritises the
calls depending on the established use case.
