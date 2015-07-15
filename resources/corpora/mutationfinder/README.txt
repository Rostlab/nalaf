List and Description of Supplementary Materials

raw_patterns.txt 
The full set of raw patterns generated in Step 1. These are the 256,196 unique raw patterns observed in MEDLINE, and are presented in order of decreasing frequency (i.e., the most frequent pattern is first). Each line contains a integer followed by the raw pattern. The integer represents the number of times the exact pattern was observed in MEDLINE. Care should be taken when interpreting these values, because the full pattern occurances are counted rather than the occurances of each individual mutation pattern. For example, the first pattern, 'RESPOSRES' is counted as occurring 3885 times. However, the pattern 'RESPOSRES and RESPOSRES' also occurs 430 times. The count of the former doesn't inlcude the count of the latter, which actually represents an addition 860 (430 x 2) counts of the simpler pattern. Also, note that single letter occurances of mutation mentions are not included as we do not idenifty raw patterns using single-letter abbreviated mutation mentions. 

mutation_patterns.txt
The set of mutation patterns (i.e., patterns with WRES, MRES, and PPOS semantic tags assigned). These are the patterns used to generate the automatically generated regular expressions.

regex.txt 
The set of regular expressions generated from mutation_pattern_nr.txt. Comments (lines beginning with #) present the mutation pattern being matched by the regular expression on the following line. Since these are the automatically generated regular expressions, the WRESPPOSMRES pattern using single-letter abbreviations is not included in this data. If using the regular expressions to match mutations, for best results you should always use the regext.txt file included with the most recent MutationFinder distribution, available at http://mutationfinder.sourceforge.net.

If automatically processing any of the above files, lines beginning with # should be treated as comments.


full_text_analysis:
Files used in our full-text analysis. 
final_content_1.txt	final_content_2.txt	final_content_3.txt	final_content_4.txt	final_content_5.txt	final_content_6.txt
MutationFinder-formatted input. These contain the full text compiled from the journal articles.

final_answers_1.txt	final_answers_2.txt	final_answers_3.txt	final_answers_4.txt	final_answers_5.txt	final_answers_6.txt
The corresponding gold standard (based on human annotations) for the input files.

final_content_1.txt.mf	final_content_2.txt.mf	final_content_3.txt.mf	final_content_4.txt.mf	final_content_5.txt.mf	final_content_6.txt.mf
The results of applying MutationFinder to the input files.


Supplementary Figure 1
Schematic representation of the mutation ontology for annotating mutation events.

annotation_guidelines.pdf
The annotation guidelines developed for the mutation annotation project.


Please direct any questions about this data to gregcaporaso@gmail.com.

Thanks for looking at this data, and for using MutationFinder. The latest information about the MutationFinder project is always available at http://mutationfinder.sourceforge.net.
