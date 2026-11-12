# 132. A deadline object read from a lateness limit fails every task that asks for it

The proposer turned "If the rent is more than 20 days late, the landlord may terminate the lease" into a deadline `rent_payment` (due date + 20 days), and
the exam asked for it: on the three tenancy chained-deadline tasks the 27B got the real deadline right and `rent_payment` wrong every time (credit
0.50 each, replayed with transcripts), because the material does not say that date. Admission now requires a deadline's quote to count a period from an
event ("within N days", "at least N days before", "N days after the invoice date") and refuses one preceded by "more than", "less than", "over" or "under":
a limit on lateness is a threshold. The check is the same in the Rust and the consumer's admission (9 of 41 deadline objects in the 76 document pack
went); traceability of the number had passed, the meaning had not.
