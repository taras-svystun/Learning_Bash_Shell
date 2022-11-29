# Learning the Bash Shell


This the summary of the book "Learning the Bash Shell" 3rd edition, by Cameron Newbam and Bill Rosenblatt.


## Chapter 1. Bash basics

1. It's a well known fact, but still I didn't know it: the shell translates the user's commands into OS instructions.
2. BSD stands for Berkeley Software Distribution.
3 `echo $SHELL` - to see the shell.
4. `lp -d lp1 -h myfile` -- sends file "myfile" to printer "lp1".
5. `cd -` -- to switch to previous directiry (whatever it was).
6. Wildcard for any character not in set `[!abc]`.
7. Brace expansion, e.g. `t{ara, a, ere}s` will match: *taras*, *tas*, *teres*. It's also possible to nest the braces, like `t{a{ra, ta}, o}s` will yield *taras*, *tatas* and *tos*. Or even ta{1..3}ras is possible: *ta1ras, ta2ras, ta3ras*.
8. UNIX filtering utilities:
   - `cat` -- copy input to output;
   - `grep` -- searches for strings in the input;
   - `sort` -- sort lines of the input;
   - `cut -d -f -s -w` -- extract columns from input; delimiter, field num, suppress lines without delims, use whitespaces as delims;
   - `sed` -- perform editing operations on input;
   - `tr` -- translate characters in the input to other characters.
9. `date > now` will write date command result into file *now*.
10. If `cp` is broken, the `cat < f1 > f2` will work the same as `cp f1 f2` would.
11. `ls -l dir | more` would print output of `ls -l dir` but page by page.
12. `cut -d, -f3 -s numbers.csv | sort -n` will select numbers from 3rd field, whose rows contain delim and sort numerically.
13. `uncompress gcc.tar &` will run the command in the background. One can type `jobs` to list all bg jobs.
14. `diff` -- to see the difference brtween two files.
15. Useful CTRL keys:
    - `^U` -- erase all input;
    - `^W` -- werase, word erase;
    - `^R` -- find previous commands;
    - `^O` = `^M` = `Return` -- enter the command;
    - `^C` -- interupt;
    - `^D` -- eof.


## Chapter 2. Command-Line Editing

#### VIM

1. `h` - move left one character;
2. `l` - move right one character;
3. `j` - move down;
4. `k` - move up;
5. `w` - move forward one word;
6. `b` - move backward one word;
7. `e` - move to end of current word;
8. `^` - move to beginning of line;
9. `$` - move to end of line;
10. `W` - move forward one non-blank word;
11. `B` - move backward one non-blank word.

#### Command-line editing.

Press `esc` to enter control mode. And then one can use **vim** control commands to navigate over the input command.
