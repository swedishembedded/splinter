Measured counts behind every number of the paper; scripts/build_data.py derives the statistics, tables and figure data
from these files and nothing is typed into the paper by hand. scripts/collect.py rewrites exam.csv, train.csv,
behaviour.csv and curve.csv from splinter run directories (scripts/run-metrics.py of splinter).

exam.csv         model,arm,k1..k5   tasks of the 100 exams solved within k attempts (k=1 is pass@1; k>1 adds the four
                 reruns of every task that failed its first attempt)
paired.csv       model,arm,unit,wins,losses   paired sign test on solved within five, arm against its reference arm
train.csv        model,arm,passed,trials      training-task attempts (228 practice and composite tasks, 4 attempts each)
behaviour.csv    model,arm,set,loop_trials,trials,repeat_after_error   attempts the loop watchdog ended, of all attempts
                 of the arm (exam: first attempts and reruns); share of calls directly after a Vim error that repeated
                 the failed call (exam: first attempts only)
curve.csv        model,arm,step,loss          validation loss during supervised training (step 0: the base model)
data.csv         source,train,validation      the integrated set by the model that made each record
recovery.csv     model,sample,attempts,passes,continued,solved   second-round attempts at the training tasks (main: the
                 evaluation sampling settings; hot: temperature 1.0) and the teacher's continuations of the failed ones
integration.csv  name,value                   admission of the second-round candidates into the set
optimiser.csv    model,steps,kl_median,clip_median,loop_percent,repeat_percent   policy-gradient optimiser statistics and
                 the two penalty weights in percent of a pass's reward (qwen3-4b-strong: the doubled-penalty variant)
uncounted.csv    model,arm,slots              exam slots that never ended with a grade after the retries
replication.csv  name,value                   final training loss of the 4B round-one adapter, first and rebuilt

Arms: base (the base model); full (the round-one adapter, trained on the round-one set); first (a replicate adapter:
the same recipe and records, trained and measured separately); records, grown (adapters trained on the published
teacher records and on a smaller grown set); rtwo, rtworef (the round-two adapter, trained on the integrated set, and
the round-one adapter measured in the same run); pg, pgref (policy gradient with the loop penalty and the round-one
adapter it started from, same run); pgd, pgdref (the same with both penalty weights doubled, and its reference).

Derived by build_data.py (never typed): Wilson intervals, sign-test p values, percentages, the differences between the
round-one adapter and its replicate, the range of solved-within-five over all measurements of the round-one recipe
(arms full, first, rtworef, pgref, pgdref), the records added by integration, and the number of failed attempts.
