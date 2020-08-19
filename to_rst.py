import sys

"""
Assumption: 
    1) Everything that starts with a comment is reStructuredText and
       and the comment "#" is simply stripped.
    2) Everything else is a codeblock and a "::" is prepended.
    
    To ensure 2), we use the variable "in_codeblock".
"""



with open(sys.argv[1], "r") as f:
    
    in_commentblock = False
    in_codeblock = False

    for line in f:
        if line.startswith("#") and not in_commentblock:
            in_codeblock = False
            print(line[2:].rstrip())
            continue
        
        if line.startswith("\"\"\""):
            if not in_commentblock:
                in_commentblock = True
                in_codeblock = False
            else:
                in_commentblock = False

            print(line[3:])
            continue

        if in_commentblock:
            print(line, end="")
            continue


        if not in_codeblock:
            in_codeblock = True
            print("\n::\n")

        print("  "+line, end="")

