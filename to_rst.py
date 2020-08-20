import sys
import re

"""
Assumption: 
    1) Everything that starts with a comment is reStructuredText and
       and the comment "#" is simply stripped.
    2) Everything else is a codeblock and a "::" is prepended.
    
    To ensure 2), we use the variable "in_codeblock".
"""

def fix_inline_math(f):
    """
    In the code, I prefer to use inline math like        $\int \bm x \mathrm dx$,
    so we have to convert it to rst inline math    :math:`\int \boldsymbol x \mathrm dx` 

    This is done by using a regex to find groups of $something$ and 
    replace the second one with `.
    Afterwards, we replace the remaining $ with :math:`
    """
    f = re.sub(r"\\bm", r"\\boldsymbol", f)
    f = re.sub(r"(\$[^\$]*)\$", r"\1`", f)
    return re.sub(r"\$", r":math:`", f)

f = open(sys.argv[1], "r").read()
f = fix_inline_math(f)
lines = f.split("\n")
   

in_commentblock = False
in_codeblock = False

for line in lines:
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

        print(line.replace("\"", ""))
        continue

    if in_commentblock:
        print(line)
        continue


    if not in_codeblock:
        in_codeblock = True
        print("\n::\n")

    print("  "+line)

