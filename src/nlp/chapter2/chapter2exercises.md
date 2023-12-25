
page 30

### 2.1
Write regular expressions for the following languages.
1. the set of all alphabetic strings;
2. the set of all lower case alphabetic strings ending in a b;
3. the set of all strings from the alphabet a,b such that each a is immedi-
ately preceded by and immediately followed by a b;

### 2.2 
Write regular expressions for the following languages. By “word”, we mean
an alphabetic string separated from other words by whitespace, any relevant
punctuation, line breaks, and so forth.
1. the set of all strings with two consecutive repeated words (e.g., “Hum-
bert Humbert” and “the the” but not “the bug” or “the big bug”);
2. all strings that start at the beginning of the line with an integer and that
end at the end of the line with a word;
3. all strings that have both the word grotto and the word raven in them
(but not, e.g., words like grottos that merely contain the word grotto);
4. write a pattern that places the first word of an English sentence in a
register. Deal with punctuation

### 2.3 
Implement an ELIZA-like program, using substitutions such as those described
on page 13. You might want to choose a different domain than a Rogerian psy-
chologist, although keep in mind that you would need a domain in which your
program can legitimately engage in a lot of simple repetition

### 2.4 
Compute the edit distance (using insertion cost 1, deletion cost 1, substitution
cost 1) of “leda” to “deal”. Show your work (using the edit distance grid).

### 2.5 
Figure out whether drive is closer to brief or to divers and what the edit dis-
tance is to each. You may use any version of distance that you like

### 2.6 
Now implement a minimum edit distance algorithm and use your hand-computed
results to check your code

### 2.7
Augment the minimum edit distance algorithm to output an alignment; you
will need to store pointers and add a stage to compute the backtrace