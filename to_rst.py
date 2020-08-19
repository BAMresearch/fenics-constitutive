import sys

"""
Assumption: 
    1) Everything that starts with a comment is reStructuredText and
       and the comment "#" is simply stripped.
    2) Everything else is a codeblock and a "::" is prepended.
    
    To ensure 2), we use the variable "in_codeblock".
"""

with open(sys.argv[1], "r") as f:
    in_codeblock = True

    for line in f:
        is_comment = line.startswith("#")
        if is_comment:
            in_codeblock=False
            print(line[2:].rstrip(), end="\n")

        else:
            if not in_codeblock:
                in_codeblock = True
                print("\n::", end="")

            print("  "+line, end="")

