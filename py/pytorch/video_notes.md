===============
seqtoseq revised, network modules
Video file: 20170715-123731.mov => seqtoseq_revised_modules.mp4
Title: seqtoseq, revised v2, idiomatic pytorch modules
Description:

Upgrades to the code from my initial seqtoseq tutorial, with the network model upgraded to be a pytorch idiomatic network Module. So, we have a module for Encoder, another for Decoder.

The sourcecode is at https://github.com/hughperkins/pub-prototyping , file: papers/attention/seq2seq_noattention_trainbyparts.py  There are two versions:
- Original/old version: commit 0a6841a
- Revised version: commit b3d8f1

Original video, which includes some slides on concept of seqtoseq, is at: https://www.youtube.com/watch?v=DdWHQrECWqA

Tags: pytorch seqtoseq rnn dnn neuralnet
Start: 0:0:6
===============
seqtoseq revised, batch timesteps together
Video file: 20170715010437.mov => seqtoseq_revised_batchtimesteps.mp4
start: 0:0:4
Title: seqtoseq, revised v2, batch timesteps
This follows on from 'seqtoseq, revised v2, idiomatic pytorch modules', and just goes over the code for batching timesteps together a bit.

The sourcecode is at https://github.com/hughperkins/pub-prototyping , file: papers/attention/seq2seq_noattention_trainbyparts.py  There are two versions:
- Original/old version: commit 0a6841a
- Revised version: commit b3d8f1

Original video, which includes some slides on concept of seqtoseq, is at: https://www.youtube.com/watch?v=DdWHQrECWqA

Tags: pytorch seqtoseq rnn dnn neuralnet

============
seqtoseq v3, actual batching
start: 0:0:22
Video file: 20170716-084806.mov => seqtoseq_v3_fullbatching.mp4
Title: pytorch seqtoseq, revised v3, minibatches
Description:
Seqtoseq in pytorch, with idiomatic network modules, and example batching. Trains in ~60 seconds or so, for playing/prototyping.

Revised version of the original seqtoseq code I presented in https://www.youtube.com/watch?v=DdWHQrECWqA

Code for 'new version' described in this video: https://github.com/hughperkins/pub-prototyping/blob/1600bc70f034df1418faee4d65b100b4481009ca/papers/attention/seq2seq_noattention_trainbyparts.py
Code for 'old version' in this video: https://github.com/hughperkins/pub-prototyping/blob/b3d8f1304bb6bc3301fb3d303993eeef0aec59aa/papers/attention/seq2seq_noattention_trainbyparts.py

(Note that the encoding.py module has changed between these two versions too, from using numpy, to using torch)

Previous videos:
v1, presents seqtoseq concepts, original, stupid slow code, not very idiomatic pytorch: https://www.youtube.com/watch?v=DdWHQrECWqA
v2, upgrade to idiomatic pytorch Modules: https://www.youtube.com/watch?v=DuRux_2TjRU
v2, upgrade to timestep batching in encoder: https://www.youtube.com/watch?v=u32DOiB5rI8
v3, upgrade to minibatches, in both encoder and decoder: this one :-) https://youtu.be/EL3uVzAXO5Y
