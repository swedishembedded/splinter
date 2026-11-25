Measured counts behind every number of the paper; scripts/build_data.py derives the statistics, tables and figure data
from these files and nothing is typed into the paper by hand. scripts/collect.py rewrites exam.csv, train.csv,
behaviour.csv and curve.csv from splinter run directories (scripts/run-metrics.py of splinter).

exam.csv         model,arm,k1..k5   tasks of the 100 exams solved within k attempts (k=1 is pass@1; reruns of the failed)
paired.csv       model,arm,unit,wins,losses   paired sign test, adapter against its reference arm
train.csv        model,arm,passed,trials      practice-task attempts (228 tasks, 4 attempts each)
behaviour.csv    model,arm,set,loop_trials,trials,repeat_after_error   loop stops and repeated failed calls
curve.csv        model,arm,step,loss          held-out loss while training
data.csv         source,train,validation      the training set by the model that made each record
recovery.csv     model,sample,attempts,passes,continued,solved   round-two attempts and the teacher's continuations
integration.csv  name,value                   admission of the round-two candidates into the set
optimiser.csv    model,steps,kl_median,clip_median   policy-gradient optimiser statistics

Arms: base (the base model); full (adapter trained on the full set); first (an earlier measurement of the same recipe);
records, grown (adapters trained on the published records and on a smaller grown set); rtwo, rtworef (adapter trained on
the integrated set and the full-set adapter measured in the same run); pg, pgref (policy gradient and the adapter it
started from, same run).
