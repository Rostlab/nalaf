## Annotation guidelines for natural language mentions

### definition of natural language mutation mentions as subset of mutation mentions
- MUST: minimum 3 spaces
- MUST: minimum 1x **connecting** word e.g. of, at, placed
- MUST: **indicative** word e.g. duplication, insertion, ...
- **helping indicative** words e.g. downstream, ... (but not e.g. nucleotide)

### graphical tree representation
TODO

### notes
- when having multiple variants possible using the *minimal set*
- possibly trying out **maximal set**
- **whitelists** and **blacklists** for "indicative", "connecting" and "helping indicative" words

### programming notes
- maybe weight matrices on correlation between words. --> patterns /pre-processing includes svm? in some way... @unclear
- filter out important words "**substitution of histidine** by tyrosine **at position 452**"
- weight important words
