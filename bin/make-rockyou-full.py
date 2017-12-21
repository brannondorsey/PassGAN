from __future__ import print_function
# utility script that creates the full rockyou leak using the word count
# redirect this output to a file to save it. Once you do that it is recommended
# to 'sort -R' to randomize it.
with open('../data/rockyou-withcount.txt', 'r') as f:
    for line in f:
        line = line.strip() # remove leading spaces
        try:
            count, password = line.split(' ', 1)
            for i in range(int(count)):
                # print the password to stdout once for each time it showed up in
                # the original leak
                print(password)
        except:
            # 11 of the number keys don't have values for some reason...
            # ignore them
            pass
