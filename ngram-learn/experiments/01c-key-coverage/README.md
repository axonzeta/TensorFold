# 01c — Key coverage (CounterFact, first 5,000 cases)

`tools/coverage.py`. Written keys = bigram/trigram n-grams at the subject's token positions in the training
expressions; test = each case's generation prompts (14,722 subject occurrences). Equal n-grams → equal rows on all
8 heads of that order. Prompts tokenised as plain text (a subject at the start has no leading space, as after a chat
turn opener).

| Written from | Subject length | tests | any n-gram | subject-last bigram | trigram | inner bigram |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| statement | 1 token | 527 | 35.5% | 35.5% | 34.0% | 20.9% |
| statement | 2 tokens | 2,836 | 58.4% | 58.4% | 51.3% | 58.4% |
| statement | 3+ tokens | 11,359 | 99.4% | 99.1% | 83.2% | 99.4% |
| + 2 paraphrases | 1 token | 527 | 38.7% | 38.7% | 34.2% | 20.9% |
| + 2 paraphrases | 2 tokens | 2,836 | 88.3% | 88.3% | 52.6% | 88.3% |
| + 2 paraphrases | 3+ tokens | 11,359 | 99.9% | 99.7% | 95.3% | 99.9% |

Neighbourhood prompts (other subjects, same relation) sharing any of the statement's subject n-grams: 7.0%.

Reading: multi-token subjects are reached almost always, through bigrams inside the subject (independent of the
words around it). Single-token subjects depend on the preceding token and are reached about a third of the time;
two-token subjects need paraphrases. Raw token ids make case and leading-space variants separate keys
("My sister" and "my sister" share no row). Caveat: CounterFact test prompts repeat the subject verbatim; real
questions may use pronouns or other names.
