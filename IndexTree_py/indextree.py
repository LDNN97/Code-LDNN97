import os

# get path and open file we want to write
path = os.path.dirname(os.path.realpath(__file__))
file = open("README.md", "w")
file.write('```\n')

# begin from the current path
for root, dirs, files in os.walk(path):
    depth = root.replace(path, '').count(os.sep)

    # ignore some folders
    if os.path.basename(root) in ['Android', 'Learning Material', 'Tex', '.git', '.idea',
                                  '__pycache__', 'cmake-build-debug']:
        dirs[:] = []
        continue

    # set the indent
    dir_indent = "|   " * (depth - 1) + "|-- "
    file_indent = "|   " * depth + "|-- "

    # print the index tree
    if not depth:
        file.write('.\n')
    else:
        file.write('{}{}\n'.format(dir_indent, os.path.basename(root)))
    for f in files:
        # ignore some files
        if f in ['.DS_Store', 'CMakeLists.txt']:
            continue
        file.write('{}{}\n'.format(file_indent, f))

file.write('```\n')
# close file
file.close()
