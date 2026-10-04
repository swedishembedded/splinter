# 004. A replay union drawn uniformly starves the new data

Examples are drawn uniformly with replacement, one per step, from the union
of the dataset and its replay files. With 169 target records and 1,352 replay
records over 600 steps, each target record was drawn 0.39 times on average
and about two thirds were never drawn; nine steps in ten went to the replay
set, which was the one that had already taught the model to give one answer.

Weight replay by share (list the target data enough times that replay is
drawn the stated fraction of the time), measure steps in epochs of the target
data, and use an effective batch larger than one when records of very
different kinds alternate: single-example batches swung the loss between 0.35
and 4.3.
